"""
Memcache bandwidth benchmark script.

Scenario: one copy of KV data stored in memcache, all NPUs read the same data
simultaneously via batch_copy. Measures per-NPU read bandwidth under concurrent
access.

Each block is allocated as a separate tensor so blocks are NON-CONTIGUOUS
in NPU memory, simulating real KV cache pool allocation patterns.

Single NPU:
  python benchmark_memcache.py --block-sizes 131072,32768,16384 --num-blocks 256

16 NPUs reading the same data (measure per-NPU read bandwidth):
  torchrun --nproc_per_node=16 benchmark_memcache.py --block-sizes 131072,32768,16384 --num-blocks 256

  Rank 0 allocates and writes data to memcache, then broadcasts GVA addresses
  to all ranks. All ranks simultaneously read the same data via batch_copy.
  Rank 0 collects and prints per-rank results.
"""

import argparse
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MmcDirect,
    MemcacheBackend,
)


@dataclass
class BlockType:
    block_len: int
    num_blocks: int


@dataclass
class BenchmarkResult:
    direction: str
    total_bytes: int
    elapsed_s: float
    throughput_gib_s: float
    num_blocks: int
    num_threads: int
    block_types: list[BlockType]
    layer_elapsed_s: list[float]


def _alloc_scattered_npu_buffers(
    block_types: list[BlockType],
    num_blocks: int,
    num_layers: int,
    device: str,
) -> tuple[list[torch.Tensor], np.ndarray, list[int], list[int]]:
    num_bt = len(block_types)
    block_addrs = np.zeros((num_blocks, num_layers, num_bt), dtype=np.int64)
    tensors: list[torch.Tensor] = []
    ptrs: list[int] = []
    lengths: list[int] = []

    for block_idx in range(num_blocks):
        for layer_id in range(num_layers):
            for bt_idx, bt in enumerate(block_types):
                buf = torch.zeros(bt.block_len, dtype=torch.uint8, device=device)
                tensors.append(buf)
                addr = buf.data_ptr()
                block_addrs[block_idx, layer_id, bt_idx] = addr
                ptrs.append(addr)
                lengths.append(bt.block_len)

    return tensors, block_addrs, ptrs, lengths


def _generate_key(block_idx: int, layer_id: int) -> str:
    return (
        f"bench_model@pcp0@dcp0@head_or_tp_rank:0"
        f"@pp_rank:0@bench_hash_{block_idx}@{layer_id}"
    )


def _alloc_gvas_for_blocks(
    backend: MemcacheBackend,
    num_blocks: int,
    num_layers: int,
    page_size_bytes: int,
    keys_per_block_hash: int,
) -> dict[tuple[int, int], int]:
    alloc_size = page_size_bytes * keys_per_block_hash
    gva_map: dict[tuple[int, int], int] = {}

    for layer_id in range(num_layers):
        for block_idx in range(num_blocks):
            key = _generate_key(block_idx, layer_id)
            gvas = backend.batch_alloc([key], [alloc_size])
            if not gvas or gvas[0] <= 0:
                raise RuntimeError(
                    f"batch_alloc failed for block {block_idx} "
                    f"layer {layer_id}: gvas={gvas}"
                )
            gva_map[(block_idx, layer_id)] = gvas[0]

    return gva_map


def _gva_map_to_tensor(
    gva_map: dict[tuple[int, int], int],
    num_blocks: int,
    num_layers: int,
    device: str,
) -> torch.Tensor:
    flat = np.zeros(num_blocks * num_layers, dtype=np.int64)
    for layer_id in range(num_layers):
        for block_idx in range(num_blocks):
            flat[layer_id * num_blocks + block_idx] = gva_map[(block_idx, layer_id)]
    return torch.from_numpy(flat).to(device=device)


def _tensor_to_gva_map(
    tensor: torch.Tensor,
    num_blocks: int,
    num_layers: int,
) -> dict[tuple[int, int], int]:
    flat = tensor.cpu().numpy()
    gva_map: dict[tuple[int, int], int] = {}
    for layer_id in range(num_layers):
        for block_idx in range(num_blocks):
            gva_map[(block_idx, layer_id)] = int(flat[layer_id * num_blocks + block_idx])
    return gva_map


def _build_block_inner_offsets(block_types: list[BlockType]) -> np.ndarray:
    block_lens = np.array([bt.block_len for bt in block_types], dtype=np.int64)
    offsets = np.zeros(len(block_types), dtype=np.int64)
    offsets[1:] = np.cumsum(block_lens[:-1])
    return offsets


def _write_data_to_memcache(
    backend: MemcacheBackend,
    block_addrs: np.ndarray,
    block_types: list[BlockType],
    num_blocks: int,
    num_layers: int,
) -> None:
    num_bt = len(block_types)
    for layer_id in range(num_layers):
        addrs = []
        sizes = []
        keys = []
        for block_idx in range(num_blocks):
            addrs.append([int(block_addrs[block_idx, layer_id, bt_idx]) for bt_idx in range(num_bt)])
            sizes.append([block_types[bt_idx].block_len for bt_idx in range(num_bt)])
            keys.append(_generate_key(block_idx, layer_id))
        backend.put(keys, addrs, sizes)


def _build_batch_copy_arrays(
    block_addrs: np.ndarray,
    block_types: list[BlockType],
    num_blocks: int,
    gva_map: dict[tuple[int, int], int],
    layer_id: int,
    my_key_index: int,
    num_ranks_per_layer: int,
    page_size_bytes: int,
) -> tuple[list[int], list[int], list[int]]:
    num_bt = len(block_types)
    inner_offsets = _build_block_inner_offsets(block_types)
    rank_layer_offset = (
        layer_id * num_ranks_per_layer + my_key_index
    ) * page_size_bytes

    addr_list: list[int] = []
    size_list: list[int] = []
    gva_list: list[int] = []

    for bt_idx in range(num_bt):
        for block_idx in range(num_blocks):
            addr_list.append(int(block_addrs[block_idx, layer_id, bt_idx]))
            size_list.append(block_types[bt_idx].block_len)
            base_gva = gva_map[(block_idx, layer_id)]
            gva_list.append(base_gva + rank_layer_offset + int(inner_offsets[bt_idx]))

    return addr_list, size_list, gva_list


def _run_batch_copy_benchmark(
    backend: MemcacheBackend,
    block_addrs: np.ndarray,
    block_types: list[BlockType],
    num_blocks: int,
    num_layers: int,
    num_threads: int,
    iterations: int,
    my_key_index: int,
    num_ranks_per_layer: int,
    page_size_bytes: int,
    gva_map: dict[tuple[int, int], int],
    rank: int,
    direction_value: int,
    direction_name: str,
) -> list[BenchmarkResult]:
    results = []

    total_bytes_per_layer = num_blocks * sum(bt.block_len for bt in block_types)
    total_bytes = total_bytes_per_layer * num_layers

    for iter_idx in range(iterations):
        layer_elapsed: list[float] = []
        start_time = time.perf_counter()

        if num_threads == 1:
            for layer_id in range(num_layers):
                layer_start = time.perf_counter()
                addr_arr, size_arr, gva_arr = _build_batch_copy_arrays(
                    block_addrs, block_types, num_blocks,
                    gva_map, layer_id,
                    my_key_index, num_ranks_per_layer, page_size_bytes,
                )
                res = backend.store.batch_copy(
                    gva_arr, addr_arr, size_arr,
                    direction_value,
                )
                layer_elapsed.append(time.perf_counter() - layer_start)
                if res != 0:
                    print(f"[rank {rank}] batch_copy failed at layer {layer_id}, res={res}")
        else:
            barrier = threading.Barrier(num_threads)
            errors: list[Exception | None] = [None] * num_threads
            layer_times = [[0.0] * num_layers for _ in range(num_threads)]

            def thread_batch_copy(thread_id: int):
                try:
                    barrier.wait()
                    for layer_id in range(num_layers):
                        addr_arr, size_arr, gva_arr = _build_batch_copy_arrays(
                            block_addrs, block_types, num_blocks,
                            gva_map, layer_id,
                            my_key_index, num_ranks_per_layer, page_size_bytes,
                        )
                        n = len(addr_arr)
                        chunk_size = n // num_threads
                        start = thread_id * chunk_size
                        end = (
                            start + chunk_size
                            if thread_id < num_threads - 1
                            else n
                        )
                        if start < end:
                            t0 = time.perf_counter()
                            res = backend.store.batch_copy(
                                gva_arr[start:end],
                                addr_arr[start:end],
                                size_arr[start:end],
                                direction_value,
                            )
                            layer_times[thread_id][layer_id] = time.perf_counter() - t0
                            if res != 0:
                                errors[thread_id] = RuntimeError(
                                    f"batch_copy GET failed at layer {layer_id}, "
                                    f"res={res}"
                                )
                except Exception as e:
                    errors[thread_id] = e

            threads = []
            for t in range(num_threads):
                th = threading.Thread(target=thread_batch_copy, args=(t,))
                threads.append(th)
                th.start()
            for th in threads:
                th.join()
            for err in errors:
                if err is not None:
                    raise err
            for layer_id in range(num_layers):
                layer_elapsed.append(max(lt[layer_id] for lt in layer_times))

        elapsed = time.perf_counter() - start_time
        throughput = total_bytes / elapsed / (1024 ** 3)
        results.append(BenchmarkResult(
            direction=f"BATCH_COPY {direction_name}",
            total_bytes=total_bytes,
            elapsed_s=elapsed,
            throughput_gib_s=throughput,
            num_blocks=num_blocks,
            num_threads=num_threads,
            block_types=block_types,
            layer_elapsed_s=layer_elapsed,
        ))

    return results


def _print_results(results: list[BenchmarkResult], title: str, rank: int = 0) -> None:
    prefix = f"[rank {rank}] " if rank >= 0 else ""
    print(f"\n{prefix}{'=' * 70}")
    print(f"{prefix}  {title}")
    print(f"{prefix}{'=' * 70}")

    if not results:
        print(f"{prefix}  No results.")
        return

    block_desc = ", ".join(
        f"{bt.block_len}B*{bt.num_blocks}" for bt in results[0].block_types
    )
    print(f"{prefix}  Block layout   : {block_desc}")
    print(f"{prefix}  Num blocks     : {results[0].num_blocks}")
    print(f"{prefix}  Num threads    : {results[0].num_threads}")
    print(f"{prefix}  Total bytes    : {results[0].total_bytes} ({results[0].total_bytes / (1024**3):.3f} GiB)")
    print(f"{prefix}  Iterations     : {len(results)}")
    print(f"{prefix}  {'-' * 66}")

    throughputs = []
    for i, r in enumerate(results):
        layer_detail = ""
        if r.layer_elapsed_s:
            layer_detail = "  |  layers: " + ", ".join(
                f"L{j}={e * 1000:.2f}ms" for j, e in enumerate(r.layer_elapsed_s)
            )
        print(
            f"{prefix}  Iter {i}: {r.elapsed_s * 1000:.2f}ms  |  "
            f"{r.throughput_gib_s:.3f} GiB/s"
            f"{layer_detail}"
        )
        throughputs.append(r.throughput_gib_s)

    avg = np.mean(throughputs)
    std = np.std(throughputs)
    print(f"{prefix}  {'-' * 66}")
    print(f"{prefix}  Average : {avg:.3f} GiB/s  |  Std: {std:.3f} GiB/s")
    print(f"{prefix}{'=' * 70}")


def _print_multi_rank_summary(
    all_throughputs: dict[int, float],
    all_elapsed: dict[int, float],
    direction: str,
    block_types: list[BlockType],
    num_blocks: int,
    num_layers: int,
    total_bytes: int,
) -> None:
    print(f"\n{'=' * 70}")
    print(f"  MULTI-RANK SUMMARY: {direction}")
    print(f"  (All NPUs reading the SAME data from memcache)")
    print(f"{'=' * 70}")

    block_desc = ", ".join(
        f"{bt.block_len}B*{bt.num_blocks}" for bt in block_types
    )
    print(f"  Block layout   : {block_desc}")
    print(f"  Num blocks     : {num_blocks}")
    print(f"  Num layers     : {num_layers}")
    print(f"  Total bytes    : {total_bytes} ({total_bytes / (1024**3):.3f} GiB)")
    print(f"  {'-' * 66}")

    ranks = sorted(all_throughputs.keys())
    values = [all_throughputs[r] for r in ranks]

    for r in ranks:
        elapsed = all_elapsed.get(r, 0.0)
        print(f"  Rank {r:2d}: {all_throughputs[r]:.3f} GiB/s  |  avg elapsed: {elapsed * 1000:.2f}ms")

    avg = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    total_throughput = sum(values)
    elapsed_values = [all_elapsed[r] for r in ranks]
    avg_elapsed = np.mean(elapsed_values)

    print(f"  {'-' * 66}")
    print(f"  Per-NPU avg : {avg:.3f} GiB/s  |  Std: {std:.3f} GiB/s")
    print(f"  Per-NPU min : {min_val:.3f} GiB/s  |  Max: {max_val:.3f} GiB/s")
    print(f"  Aggregate   : {total_throughput:.3f} GiB/s ({len(ranks)} NPUs)")
    print(f"  Avg elapsed : {avg_elapsed * 1000:.2f}ms")
    print(f"{'=' * 70}")


def _init_distributed(rank: int, local_rank: int, world_size: int) -> None:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="hccl",
            world_size=world_size,
            rank=rank,
            init_method="tcp://127.0.0.1:29500",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Memcache bandwidth benchmark: one copy, all NPUs read via batch_copy"
    )
    parser.add_argument(
        "--block-sizes",
        type=str,
        default="131072,32768,16384",
        help="Comma-separated block sizes in bytes. Each size creates a block type. "
             "E.g., '131072,32768,16384' means three block types with 128KB, 32KB, 16KB.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=256,
        help="Number of blocks per block type per layer.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of layers to simulate.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of concurrent threads within each rank.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--num-readers",
        type=int,
        default=0,
        help="Number of NPUs that participate in reading. 0 means all NPUs read. "
             "Useful for testing how bandwidth scales with concurrent readers. "
             "E.g., torchrun --nproc_per_node=16 with --num-readers=8 means "
             "only 8 of 16 NPUs read, the rest just synchronize.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations (not counted).",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="G2L",
        choices=["L2G", "G2L", "G2H", "H2G"],
        help="Transfer direction: L2G (NPU->memcache, D2H), G2L (memcache->NPU, H2D), "
             "G2H (memcache->Host), H2G (Host->memcache). Default: G2L.",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", local_rank))

    num_readers = args.num_readers if args.num_readers > 0 else world_size
    is_reader = rank < num_readers

    block_size_values = [int(x.strip()) for x in args.block_sizes.split(",")]
    block_types = [BlockType(block_len=bs, num_blocks=args.num_blocks) for bs in block_size_values]
    num_layers = args.num_layers
    num_threads = args.num_threads
    iterations = args.iterations

    page_size_bytes = sum(bt.block_len for bt in block_types)
    my_key_index = 0
    num_ranks_per_layer = 1
    keys_per_block_hash = len(block_types) * num_layers
    direction_value = MmcDirect[f"COPY_{args.direction}"].value

    device = f"npu:{local_rank}"
    print(f"[rank {rank}] Initializing on {device} (world_size={world_size})...")
    torch.npu.set_device(device)

    if world_size > 1:
        print(f"[rank {rank}] Initializing distributed process group...")
        _init_distributed(rank, local_rank, world_size)

    from vllm.config import ParallelConfig
    parallel_config = ParallelConfig()
    backend = MemcacheBackend(parallel_config, local_rank=local_rank, init_bm=True)

    print(f"[rank {rank}] Allocating SCATTERED NPU buffers (each block is a separate tensor)...")
    tensors, block_addrs, ptrs, lengths = _alloc_scattered_npu_buffers(
        block_types, args.num_blocks, num_layers, device,
    )

    print(f"[rank {rank}] Registering {len(ptrs)} buffers with memcache...")
    backend.register_buffer(ptrs, lengths)

    if world_size > 1:
        torch.distributed.barrier()

    print(f"[rank {rank}] Benchmark configuration:")
    print(f"  Block types     : {[(bt.block_len, bt.num_blocks) for bt in block_types]}")
    print(f"  Num layers      : {num_layers}")
    print(f"  Num threads     : {num_threads}")
    print(f"  World size      : {world_size}")
    print(f"  Num readers     : {num_readers}")
    print(f"  This rank is    : {'READER' if is_reader else 'IDLE (barrier only)'}")
    print(f"  Iterations      : {iterations}")
    print(f"  Page size bytes : {page_size_bytes}")
    print(f"  Memory layout   : SCATTERED (non-contiguous blocks)")
    print(f"  Total tensors   : {len(tensors)}")

    if rank == 0:
        print("[rank 0] Allocating GVA space and writing data to memcache...")
    gva_map: dict[tuple[int, int], int] = {}
    if rank == 0:
        gva_map = _alloc_gvas_for_blocks(
            backend, args.num_blocks, num_layers, page_size_bytes,
            keys_per_block_hash,
        )
        _write_data_to_memcache(
            backend, block_addrs, block_types, args.num_blocks, num_layers,
        )

    if world_size > 1:
        if rank == 0:
            gva_tensor = _gva_map_to_tensor(gva_map, args.num_blocks, num_layers, device)
        else:
            gva_tensor = torch.zeros(args.num_blocks * num_layers, dtype=torch.int64, device=device)
        torch.distributed.broadcast(gva_tensor, src=0)
        if rank != 0:
            gva_map = _tensor_to_gva_map(gva_tensor, args.num_blocks, num_layers)
        torch.distributed.barrier()

    if is_reader:
        print(f"[rank {rank}] Warming up BATCH_COPY {args.direction}...")
        _run_batch_copy_benchmark(
            backend, block_addrs, block_types, args.num_blocks,
            num_layers, num_threads, args.warmup,
            my_key_index, num_ranks_per_layer,
            page_size_bytes, gva_map, rank,
            direction_value, args.direction,
        )
    if world_size > 1:
        torch.distributed.barrier()
    if is_reader:
        print(f"[rank {rank}] Running BATCH_COPY benchmark ({args.direction})...")
        batch_copy_results = _run_batch_copy_benchmark(
            backend, block_addrs, block_types, args.num_blocks,
            num_layers, num_threads, iterations,
            my_key_index, num_ranks_per_layer,
            page_size_bytes, gva_map, rank,
            direction_value, args.direction,
        )
        avg_throughput = float(np.mean([r.throughput_gib_s for r in batch_copy_results]))
        avg_elapsed = float(np.mean([r.elapsed_s for r in batch_copy_results]))
        direction_label = f"BATCH_COPY {args.direction}"
        per_rank_throughput = {direction_label: avg_throughput}
        per_rank_elapsed = {direction_label: avg_elapsed}
        _print_results(batch_copy_results, f"{direction_label} Bandwidth", rank=rank)
    else:
        per_rank_throughput = {}
        per_rank_elapsed = {}
    if world_size > 1:
        torch.distributed.barrier()

    total_bytes_per_layer = args.num_blocks * sum(bt.block_len for bt in block_types)
    total_bytes = total_bytes_per_layer * num_layers

    if world_size > 1:
        for direction, throughput in per_rank_throughput.items():
            elapsed = per_rank_elapsed.get(direction, 0.0)
            throughput_tensor = torch.tensor(
                [throughput if is_reader else 0.0],
                dtype=torch.float64,
                device=device,
            )
            elapsed_tensor = torch.tensor(
                [elapsed if is_reader else 0.0],
                dtype=torch.float64,
                device=device,
            )
            gathered_tput = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(world_size)]
            gathered_elapsed = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_tput, throughput_tensor)
            torch.distributed.all_gather(gathered_elapsed, elapsed_tensor)
            if rank == 0:
                all_rank_throughputs = {
                    r: gathered_tput[r].item()
                    for r in range(num_readers)
                    if gathered_tput[r].item() > 0
                }
                all_rank_elapsed = {
                    r: gathered_elapsed[r].item()
                    for r in range(num_readers)
                    if gathered_elapsed[r].item() > 0
                }
                _print_multi_rank_summary(
                    all_rank_throughputs, all_rank_elapsed, direction,
                    block_types, args.num_blocks, num_layers, total_bytes,
                )

    if rank == 0:
        print("\nDone.")

    backend.store.close()
    del tensors

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
