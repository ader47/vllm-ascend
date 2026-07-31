"""KV full-block dump correctness and effective-bandwidth measurements."""

import gc
import os

import numpy as np
import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

_BLOCK_SIZE = 128
_NOPE_DIM = 512
_ROPE_DIM = 64
_WARMUP_ITERS = 20
_MEASURE_ITERS = 100
_DRAM_ROTATION_SLOTS = 32
_PCIE_THEORETICAL_GBPS = float(
    os.getenv("DSA_DUMP_PCIE_THEORETICAL_GBPS", "63.0")
)


def _swapped_arena(
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        pytest.skip("torch_npu swapped-memory API is unavailable")
    return torch_npu.empty_with_swapped_memory(
        shape,
        dtype=dtype,
        device=torch.device(f"npu:{torch.npu.current_device()}"),
    )


def _read_swapped_arena(tensor: torch.Tensor) -> torch.Tensor:
    """Stage mapped host memory through an ordinary NPU tensor."""

    staging = torch.empty(
        tuple(tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    staging.fill_(1).mul_(tensor)
    torch.npu.synchronize()
    return staging.cpu()


def _make_sources(
    block_count: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_values = torch.arange(
        block_count,
        dtype=torch.float32,
        device="npu",
    ).to(dtype)
    nope = (
        block_values.view(block_count, 1, 1)
        .expand(block_count, _BLOCK_SIZE, _NOPE_DIM)
        .contiguous()
    )
    rope = (
        (block_values + 100)
        .view(block_count, 1, 1)
        .expand(block_count, _BLOCK_SIZE, _ROPE_DIM)
        .contiguous()
    )
    return nope, rope


def _make_metadata_variants(
    *,
    active_blocks: int,
    row_count: int,
    rotation_slots: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    src_ids = torch.full(
        (row_count,),
        999,
        dtype=torch.int32,
        device="npu",
    )
    active_rows = torch.empty(0, dtype=torch.int64, device="npu")
    if active_blocks:
        if active_blocks == 1 and row_count > 1:
            active_rows = torch.tensor(
                [row_count - 1],
                dtype=torch.int64,
                device="npu",
            )
        else:
            active_rows = torch.arange(
                active_blocks,
                dtype=torch.int64,
                device="npu",
            )
        src_ids[active_rows] = torch.arange(
            active_blocks,
            dtype=torch.int32,
            device="npu",
        )

    dst_variants: list[torch.Tensor] = []
    for rotation in range(rotation_slots):
        dst_ids = torch.full(
            (row_count,),
            -1,
            dtype=torch.int32,
            device="npu",
        )
        if active_blocks:
            dst_ids[active_rows] = (
                torch.arange(
                    active_blocks,
                    dtype=torch.int32,
                    device="npu",
                )
                + rotation * active_blocks
            )
        dst_variants.append(dst_ids)
    return src_ids, dst_variants


def _run_dump(
    source_nope: torch.Tensor,
    source_rope: torch.Tensor,
    dram_nope: torch.Tensor,
    dram_rope: torch.Tensor,
    src_ids: torch.Tensor,
    dst_ids: torch.Tensor,
) -> None:
    torch.ops._C_ascend.kv_cache_full_block_dump(
        source_nope,
        source_rope,
        dram_nope,
        dram_rope,
        src_ids,
        dst_ids,
    )


def _measure_case(
    *,
    active_blocks: int,
    row_count: int,
) -> dict[str, float | int]:
    dtype = torch.bfloat16
    source_blocks = max(active_blocks, 1)
    destination_blocks = max(
        active_blocks * _DRAM_ROTATION_SLOTS,
        1,
    )
    source_nope, source_rope = _make_sources(source_blocks, dtype)
    dram_nope = _swapped_arena(
        (destination_blocks, _BLOCK_SIZE, _NOPE_DIM),
        dtype,
    )
    dram_rope = _swapped_arena(
        (destination_blocks, _BLOCK_SIZE, _ROPE_DIM),
        dtype,
    )
    src_ids, dst_variants = _make_metadata_variants(
        active_blocks=active_blocks,
        row_count=row_count,
        rotation_slots=_DRAM_ROTATION_SLOTS,
    )

    for iteration in range(_WARMUP_ITERS):
        _run_dump(
            source_nope,
            source_rope,
            dram_nope,
            dram_rope,
            src_ids,
            dst_variants[iteration % _DRAM_ROTATION_SLOTS],
        )
    torch.npu.synchronize()

    event_pairs: list[tuple[torch.npu.Event, torch.npu.Event]] = []
    for iteration in range(_MEASURE_ITERS):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        _run_dump(
            source_nope,
            source_rope,
            dram_nope,
            dram_rope,
            src_ids,
            dst_variants[iteration % _DRAM_ROTATION_SLOTS],
        )
        end.record()
        event_pairs.append((start, end))
    torch.npu.synchronize()

    samples_us = np.asarray(
        [
            start.elapsed_time(end) * 1000.0
            for start, end in event_pairs
        ],
        dtype=np.float64,
    )

    sustained_start = torch.npu.Event(enable_timing=True)
    sustained_end = torch.npu.Event(enable_timing=True)
    sustained_start.record()
    for iteration in range(_MEASURE_ITERS):
        _run_dump(
            source_nope,
            source_rope,
            dram_nope,
            dram_rope,
            src_ids,
            dst_variants[iteration % _DRAM_ROTATION_SLOTS],
        )
    sustained_end.record()
    torch.npu.synchronize()

    payload_bytes = (
        active_blocks
        * _BLOCK_SIZE
        * (_NOPE_DIM + _ROPE_DIM)
        * source_nope.element_size()
    )
    p50_us = float(np.median(samples_us))
    min_us = float(np.min(samples_us))
    p95_us = float(np.percentile(samples_us, 95))
    p50_gbps = (
        payload_bytes / (p50_us * 1e-6) / 1e9
        if payload_bytes
        else 0.0
    )
    peak_gbps = (
        payload_bytes / (min_us * 1e-6) / 1e9
        if payload_bytes
        else 0.0
    )
    sustained_total_us = sustained_start.elapsed_time(
        sustained_end
    ) * 1000.0
    sustained_avg_us = sustained_total_us / _MEASURE_ITERS
    sustained_gbps = (
        payload_bytes
        * _MEASURE_ITERS
        / (sustained_total_us * 1e-6)
        / 1e9
        if payload_bytes
        else 0.0
    )
    pcie_util_percent = (
        p50_gbps / _PCIE_THEORETICAL_GBPS * 100.0
        if payload_bytes and _PCIE_THEORETICAL_GBPS > 0
        else 0.0
    )
    sustained_pcie_util_percent = (
        sustained_gbps / _PCIE_THEORETICAL_GBPS * 100.0
        if payload_bytes and _PCIE_THEORETICAL_GBPS > 0
        else 0.0
    )
    return {
        "active_blocks": active_blocks,
        "row_count": row_count,
        "payload_bytes": payload_bytes,
        "min_us": min_us,
        "p50_us": p50_us,
        "p95_us": p95_us,
        "p50_gbps": p50_gbps,
        "peak_gbps": peak_gbps,
        "pcie_util_percent": pcie_util_percent,
        "sustained_avg_us": sustained_avg_us,
        "sustained_gbps": sustained_gbps,
        "sustained_pcie_util_percent": sustained_pcie_util_percent,
    }


@torch.inference_mode()
def test_kv_cache_full_block_dump_single_block_swapped() -> None:
    """One production-shaped block must reach both swapped-memory planes."""

    dtype = torch.bfloat16
    source_nope, source_rope = _make_sources(1, dtype)
    dram_nope = _swapped_arena((2, _BLOCK_SIZE, _NOPE_DIM), dtype)
    dram_rope = _swapped_arena((2, _BLOCK_SIZE, _ROPE_DIM), dtype)
    dram_nope.fill_(-1)
    dram_rope.fill_(-1)
    src_ids = torch.tensor([0], dtype=torch.int32, device="npu")
    dst_ids = torch.tensor([1], dtype=torch.int32, device="npu")

    _run_dump(
        source_nope,
        source_rope,
        dram_nope,
        dram_rope,
        src_ids,
        dst_ids,
    )
    torch.npu.synchronize()

    nope_cpu = _read_swapped_arena(dram_nope)
    rope_cpu = _read_swapped_arena(dram_rope)
    torch.testing.assert_close(nope_cpu[1], source_nope[0].cpu())
    torch.testing.assert_close(rope_cpu[1], source_rope[0].cpu())
    assert torch.all(nope_cpu[0] == -1)
    assert torch.all(rope_cpu[0] == -1)


@torch.inference_mode()
def test_kv_cache_full_block_dump_many_blocks_swapped() -> None:
    """Every row in the 24-block bandwidth shape must reach swapped DRAM."""

    block_count = 24
    dtype = torch.bfloat16
    source_nope, source_rope = _make_sources(block_count, dtype)
    dram_nope = _swapped_arena(
        (block_count + 1, _BLOCK_SIZE, _NOPE_DIM),
        dtype,
    )
    dram_rope = _swapped_arena(
        (block_count + 1, _BLOCK_SIZE, _ROPE_DIM),
        dtype,
    )
    dram_nope.fill_(-1)
    dram_rope.fill_(-1)
    src_ids = torch.arange(block_count, dtype=torch.int32, device="npu")
    dst_ids = torch.arange(
        block_count - 1,
        -1,
        -1,
        dtype=torch.int32,
        device="npu",
    )

    _run_dump(
        source_nope,
        source_rope,
        dram_nope,
        dram_rope,
        src_ids,
        dst_ids,
    )
    torch.npu.synchronize()

    nope_cpu = _read_swapped_arena(dram_nope)
    rope_cpu = _read_swapped_arena(dram_rope)
    torch.testing.assert_close(
        nope_cpu[:block_count],
        source_nope.flip(0).cpu(),
    )
    torch.testing.assert_close(
        rope_cpu[:block_count],
        source_rope.flip(0).cpu(),
    )
    assert torch.all(nope_cpu[block_count] == -1)
    assert torch.all(rope_cpu[block_count] == -1)


@torch.inference_mode()
def test_kv_cache_full_block_dump_bandwidth() -> None:
    """Report effective HBM-to-swapped-DRAM payload bandwidth.

    NPU Events bracket only the custom operator. Tensor construction, Python
    metadata preparation and synchronization are outside the measured range.
    Bandwidth counts the useful NOPE+ROPE bytes written to mapped host memory;
    PCIe utilization is relative to a configurable unidirectional theoretical
    rate and is not a hard performance assertion.
    """

    cases = (
        (0, 24),
        (1, 1),
        (1, 24),
        (4, 4),
        (8, 8),
        (16, 16),
        (24, 24),
    )
    results = []
    for active_blocks, row_count in cases:
        results.append(
            _measure_case(
                active_blocks=active_blocks,
                row_count=row_count,
            )
        )
        gc.collect()

    bytes_per_block = (
        _BLOCK_SIZE
        * (_NOPE_DIM + _ROPE_DIM)
        * torch.tensor([], dtype=torch.bfloat16).element_size()
    )
    print(
        "\nKV_CACHE_FULL_BLOCK_DUMP_BANDWIDTH "
        f"bytes_per_block={bytes_per_block} "
        f"({bytes_per_block / 1024:.1f} KiB) "
        f"pcie_theoretical={_PCIE_THEORETICAL_GBPS:.1f} GB/s "
        f"warmup={_WARMUP_ITERS} iterations={_MEASURE_ITERS}"
    )
    print(
        "active/rows | payload_KiB | min_us | p50_us | p95_us | "
        "p50_GB/s | peak_GB/s | avg_us | sustained_GB/s | "
        "p50_util | sustained_util"
    )
    for result in results:
        payload_kib = int(result["payload_bytes"]) / 1024
        print(
            f"{int(result['active_blocks']):>2}/"
            f"{int(result['row_count']):<2}         | "
            f"{payload_kib:>11.1f} | "
            f"{float(result['min_us']):>6.2f} | "
            f"{float(result['p50_us']):>6.2f} | "
            f"{float(result['p95_us']):>6.2f} | "
            f"{float(result['p50_gbps']):>8.2f} | "
            f"{float(result['peak_gbps']):>9.2f} | "
            f"{float(result['sustained_avg_us']):>6.2f} | "
            f"{float(result['sustained_gbps']):>14.2f} | "
            f"{float(result['pcie_util_percent']):>8.2f}% | "
            f"{float(result['sustained_pcie_util_percent']):>13.2f}%"
        )

    assert all(float(result["p50_us"]) > 0 for result in results)
    assert all(float(result["sustained_avg_us"]) > 0 for result in results)
