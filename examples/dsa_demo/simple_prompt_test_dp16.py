#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DSA 稀疏卸载最小离线冒烟脚本。

测试时只修改下方“用户配置”区。单机直接运行：

    python examples/dsa_demo/simple_prompt_test_dp16.py

双机使用相同脚本和固定端口，例如：

    # 141.61.141.22
    python examples/dsa_demo/simple_prompt_test_dp16.py \
        --node-size 2 --node-rank 0 \
        --master-addr 141.61.141.22 --master-port 29500 \
        --sync-port 29600

    # 141.61.141.26
    python examples/dsa_demo/simple_prompt_test_dp16.py \
        --node-size 2 --node-rank 1 \
        --master-addr 141.61.141.22 --master-port 29500 \
        --sync-port 29600

``disabled`` 用于验证基线隔离，``cache-init`` 只验证双平面 KV cache
初始化，``eager`` 和 ``graph`` 分别验证 DSA eager 与 FULL decode graph。
DP>1 时每台机器的父进程只创建本机离线前端进程，并通过 ``VLLM_DP_*``
建立全局 SPMD 协调；graph 模式下 target 一张图，三轮 MTP 合成独立一张图。
``ENABLE_MTP_GRAPH=False`` 可回退到 target graph + drafter eager 做精度/性能对照。
默认短 prompt 主要覆盖 DENSE；验证真正的 sparse/ENTER 路径时，应换成
token 长度超过 ``DSA_SPARSE_ACTIVATION_TOKENS`` 的文本。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import socket
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from novel_dataset import chinese_20k

LONG_PROMPTS = chinese_20k * 16

# =========================
# 用户配置
# =========================

MODEL_PATH = "/home/models/GLM-5.2-w4a4c8-mxfp4"
RUN_MODE = "cache-init"  # disabled / cache-init / eager / graph

# 第一轮默认验收步骤 2：双机 DP16TP1EP16、DSA cache-init、无 MTP、eager。
# 硬件与 cache-init 通过后，只将 RUN_MODE 改为 "eager" 即进入步骤 3。
TENSOR_PARALLEL_SIZE = 1
DATA_PARALLEL_SIZE = 16

# 每个 DP rank 使用同一输入，便于步骤 3 直接逐 token 比较跨 rank 一致性。
SHORT_PROMPTS = ["请用一句话介绍你自己。"] * DATA_PARALLEL_SIZE
PROMPTS = SHORT_PROMPTS

# 全局总 NPU 数 = TP * DP；每机 NPU 数 = TP * (DP / node-size)。
DP_PROCESS_TIMEOUT_SECONDS = 3600
# max_num_seqs 按 DP rank 计算；16 条短 prompt 在 DP16 下每个 rank 1 条。
MAX_NUM_SEQS = 1
MAX_MODEL_LEN = 4096
# 关闭 chunked prefill 时 token budget 不得小于 max_model_len；步骤 6 将两者一起改为 24576。
MAX_NUM_BATCHED_TOKENS = 4096
ENABLE_CHUNKED_PREFILL = False
MAX_TOKENS = 32
GPU_MEMORY_UTILIZATION = 0.82
QUANTIZATION = "ascend"
ENABLE_EXPERT_PARALLEL = True
ENABLE_MTP = False
MTP_NUM_SPECULATIVE_TOKENS = 3
# 仅在 RUN_MODE="graph" 且 ENABLE_MTP=True 时生效；支持 A5 DSA TP1 或 TP8DP1。
ENABLE_MTP_GRAPH = True

# A5/950 上的 DSA 首版只支持 LI C8 与 SFA C8 同时开启；A3/910C 保持 False。
# disabled 模式也会保留这个 vLLM-Ascend 原生物理布局开关，便于用同一 C8
# cache 编码公平对照“原生 C8”与“DSA C8”。
ENABLE_A5_PACKED_C8_DSA = True

ENABLE_PROFILE = False
PROFILE_DIR = "/home/w00916487/vllm_profile/tp8_offload_seq20k_bs16/"
RESULT_JSON: str | None = None

DSA_PROMPT_BUDGET_THRESHOLDS = [32768, 65536]
DSA_INDEXER_MLA_BLOCK_RATIO = 3
DSA_MAX_ACTIVE_REQS = 256
DSA_HOT_CPU_BLOCK_MULTIPLE = 1.0
DSA_GRAPH_CAPTURE_SIZES = [1, 2, 4, 8]
DSA_TRACE_POINTS = {
    # 当前仅解析预留合同，尚无稳定日志 consumer；验收默认关闭。
    "enabled": False,
    "points": ["first_sample"],
    "ranks": [0],
}

NATIVE_RUNTIME_ENV_OVERRIDES = {
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_BUFFSIZE": "200",
    "OMP_NUM_THREADS": "10",
    "OMP_PROC_BIND": "false",
    "PYTHONHASHSEED": "114514",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "VLLM_ASCEND_ENABLE_FUSED_MC2": "0",
    "VLLM_ASCEND_ENABLE_MLAPO": "1",
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}

for key, value in NATIVE_RUNTIME_ENV_OVERRIDES.items():
    os.environ[key] = value


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline DSA data-parallel smoke test"
    )
    parser.add_argument(
        "--node-size",
        type=int,
        default=1,
        help="total number of nodes",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="zero-based rank of this node",
    )
    parser.add_argument(
        "--master-addr",
        default="",
        help="IP address of node 0; required when node-size > 1",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=0,
        help="fixed vLLM DP rendezvous port; required when node-size > 1",
    )
    parser.add_argument(
        "--sync-port",
        type=int,
        default=0,
        help="fixed cross-node frontend barrier port; keep it separate from master-port",
    )
    return parser.parse_args()


def resolve_node_launch(
    args: argparse.Namespace,
) -> tuple[str, int, int | None, list[tuple[int, int]]]:
    """Resolve this node's local-to-global DP rank mapping."""

    if args.node_size < 1:
        raise ValueError("node-size must be positive")
    if not 0 <= args.node_rank < args.node_size:
        raise ValueError(
            f"node-rank must be in [0, {args.node_size}), got {args.node_rank}"
        )
    if DATA_PARALLEL_SIZE % args.node_size != 0:
        raise ValueError(
            f"DATA_PARALLEL_SIZE={DATA_PARALLEL_SIZE} must be divisible by "
            f"node-size={args.node_size}"
        )

    dp_per_node = DATA_PARALLEL_SIZE // args.node_size
    first_global_rank = args.node_rank * dp_per_node
    rank_map = [
        (local_rank, first_global_rank + local_rank)
        for local_rank in range(dp_per_node)
    ]
    if args.node_size == 1:
        return "127.0.0.1", get_open_port(), None, rank_map

    master_addr = args.master_addr.strip()
    if not master_addr:
        raise ValueError("master-addr is required when node-size > 1")
    for name, port in (
        ("master-port", args.master_port),
        ("sync-port", args.sync_port),
    ):
        if not 1 <= port <= 65535:
            raise ValueError(
                f"{name} must be in [1, 65535] when node-size > 1, got {port}"
            )
    if args.master_port == args.sync_port:
        raise ValueError("sync-port must differ from master-port")
    return master_addr, args.master_port, args.sync_port, rank_map


def create_frontend_store(
    dp_rank: int,
    dp_master_ip: str,
    sync_port: int,
) -> Any:
    """Create a CPU TCPStore used only for cross-node frontend barriers."""

    from torch.distributed import TCPStore

    return TCPStore(
        dp_master_ip,
        sync_port,
        DATA_PARALLEL_SIZE,
        dp_rank == 0,
        timedelta(seconds=DP_PROCESS_TIMEOUT_SECONDS),
        True,
    )


def wait_frontend_barrier(store: Any, phase: str, dp_rank: int) -> None:
    store.set(f"{phase}/{dp_rank}", "1")
    store.wait(
        [f"{phase}/{rank}" for rank in range(DATA_PARALLEL_SIZE)]
    )


def build_dsa_config(enable_graph: bool) -> dict[str, Any]:
    sparse_activation_tokens = (
        10240 if ENABLE_MTP else 6144
    )
    resident_budget_tokens = (
        [10240, 10240, 10240]
        if ENABLE_MTP
        else [10240, 10240, 10240]
    )
    return {
        "enabled": True,
        "split_indexer_cache": True,
        "indexer_mla_block_ratio": DSA_INDEXER_MLA_BLOCK_RATIO,
        "sparse_activation_tokens": sparse_activation_tokens,
        "prompt_budget_thresholds": DSA_PROMPT_BUDGET_THRESHOLDS,
        "resident_budget_tokens": resident_budget_tokens,
        "max_active_reqs": DSA_MAX_ACTIVE_REQS,
        "hot_cpu_block_multiple": DSA_HOT_CPU_BLOCK_MULTIPLE,
        "enable_row_mode_decode_graph": enable_graph,
        "trace_points": DSA_TRACE_POINTS,
    }


def build_llm_kwargs(dp_rank: int, multi_node: bool) -> dict[str, Any]:
    if DATA_PARALLEL_SIZE < 1:
        raise ValueError("DATA_PARALLEL_SIZE must be positive")
    if DATA_PARALLEL_SIZE > 1 and RUN_MODE != "disabled":
        if not ENABLE_EXPERT_PARALLEL:
            raise ValueError("DSA DP requires ENABLE_EXPERT_PARALLEL=True")
    if ENABLE_MTP and not ENABLE_A5_PACKED_C8_DSA:
        raise ValueError(
            "DSA compromise MTP3 requires ENABLE_A5_PACKED_C8_DSA=True"
        )
    if ENABLE_MTP and MTP_NUM_SPECULATIVE_TOKENS != 3:
        raise ValueError(
            "DSA compromise MTP requires MTP_NUM_SPECULATIVE_TOKENS=3"
        )
    graph_enabled = RUN_MODE == "graph"
    kwargs: dict[str, Any] = {
        "model": MODEL_PATH,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "pipeline_parallel_size": 1,
        "quantization": QUANTIZATION,
        "seed": 1024,
        "enable_expert_parallel": ENABLE_EXPERT_PARALLEL,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": ENABLE_CHUNKED_PREFILL,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "block_size": 128,
        "async_scheduling": False,
        "enforce_eager": not graph_enabled,
        "disable_log_stats": False,
    }
    additional_config: dict[str, Any] = {}
    if DATA_PARALLEL_SIZE > 1:
        # Ascend 950 上 CPU DP metadata all_reduce 可能返回脏数据。
        additional_config["dp_allreduce_on_npu"] = True
    if multi_node:
        # Ascend 950 跨机使用已验证的默认 MC2 算法，不强制 hierarchy。
        additional_config["enable_mc2_hierarchy_comm"] = False
    if ENABLE_A5_PACKED_C8_DSA:
        additional_config.update(
            {
                "enable_sparse_sfa_c8": ENABLE_A5_PACKED_C8_DSA,
                "enable_sparse_li_c8": ENABLE_A5_PACKED_C8_DSA,
            }
        )
    if RUN_MODE != "disabled":
        additional_config.update(
            {
                "dsa_sparse_config": build_dsa_config(graph_enabled),
            }
        )
    if additional_config:
        kwargs["additional_config"] = additional_config
    if ENABLE_MTP:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": MTP_NUM_SPECULATIVE_TOKENS,
            "draft_tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            # 只约束 drafter；target 仍由顶层配置进入 FULL decode graph。
            "enforce_eager": not (graph_enabled and ENABLE_MTP_GRAPH),
        }
    if graph_enabled:
        decode_query_len = (
            1 + MTP_NUM_SPECULATIVE_TOKENS if ENABLE_MTP else 1
        )
        capture_sizes = sorted(
            {
                size * decode_query_len
                for size in (*DSA_GRAPH_CAPTURE_SIZES, MAX_NUM_SEQS)
                if size <= MAX_NUM_SEQS
            }
        )
        kwargs["compilation_config"] = {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": capture_sizes,
        }
        if ENABLE_MTP:
            # Keep identical collective/hidden layouts for MTP graph and its
            # eager control, independently of the chosen TP/DP topology.
            additional_config.update(
                enable_flashcomm1=False,
                enable_shared_expert_dp=False,
                enable_flashcomm2_parallel_size=0,
                enable_reduce_sample=False,
            )
            kwargs["compilation_config"]["pass_config"] = {"enable_sp": False}
    if ENABLE_PROFILE:
        from vllm.config import ProfilerConfig

        profile_dir = Path(PROFILE_DIR).resolve() / f"dp_rank_{dp_rank}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(profile_dir),
            torch_profiler_with_stack=True,
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=True,
        )
    return kwargs


def _rank_result_path(dp_rank: int) -> Path | None:
    if RESULT_JSON is None:
        return None
    path = Path(RESULT_JSON)
    if DATA_PARALLEL_SIZE > 1:
        path = path.with_name(f"{path.stem}.dp{dp_rank}{path.suffix}")
    return path


def write_result(payload: dict[str, Any], dp_rank: int) -> None:
    path = _rank_result_path(dp_rank)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[dsa-smoke] wrote result: {path}")


def shard_prompts(dp_rank: int) -> list[tuple[int | None, str]]:
    """Return this rank's contiguous prompt shard.

    An empty rank still receives a placeholder so every MoE DP/EP rank enters
    the coordinated forward loop.
    """

    floor, remainder = divmod(len(PROMPTS), DATA_PARALLEL_SIZE)

    def start(rank: int) -> int:
        return rank * floor + min(rank, remainder)

    begin = start(dp_rank)
    end = start(dp_rank + 1)
    prompt_items = list(enumerate(PROMPTS[begin:end], start=begin))
    return prompt_items or [(None, "Placeholder")]


def configure_dp_rank(
    *,
    dp_rank: int,
    local_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
) -> None:
    """Configure vLLM's offline SPMD DP rank before constructing ``LLM``."""

    os.environ.update(
        {
            "VLLM_DP_RANK": str(dp_rank),
            "VLLM_DP_RANK_LOCAL": str(local_dp_rank),
            "VLLM_DP_SIZE": str(DATA_PARALLEL_SIZE),
            "VLLM_DP_MASTER_IP": dp_master_ip,
            "VLLM_DP_MASTER_PORT": str(dp_master_port),
        }
    )


def run_dp_rank(
    dp_rank: int,
    local_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    shutdown_barrier: Any | None,
    sync_port: int | None,
) -> None:
    configure_dp_rank(
        dp_rank=dp_rank,
        local_dp_rank=local_dp_rank,
        dp_master_ip=dp_master_ip,
        dp_master_port=dp_master_port,
    )

    frontend_store = None
    llm = None
    try:
        if sync_port is not None:
            frontend_store = create_frontend_store(
                dp_rank,
                dp_master_ip,
                sync_port,
            )
            wait_frontend_barrier(frontend_store, "ready", dp_rank)
            if dp_rank == 0:
                print(
                    "[dsa-smoke] all cross-node frontend ranks connected "
                    f"at {dp_master_ip}:{sync_port}"
                )

        # DP 环境变量必须在首次导入 vLLM 前设置。
        from vllm import LLM, SamplingParams

        prompt_items = shard_prompts(dp_rank)
        local_prompts = [prompt for _, prompt in prompt_items]
        kwargs = build_llm_kwargs(
            dp_rank,
            multi_node=sync_port is not None,
        )
        print(
            "[dsa-smoke] "
            f"rank={dp_rank}/{DATA_PARALLEL_SIZE} local_rank={local_dp_rank} "
            f"mode={RUN_MODE} model={MODEL_PATH!r} prompts={len(local_prompts)} "
            f"tp={TENSOR_PARALLEL_SIZE} ep={TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE} "
            f"max_model_len={MAX_MODEL_LEN} max_num_seqs={MAX_NUM_SEQS} "
            f"a5_packed_c8={ENABLE_A5_PACKED_C8_DSA} mtp={ENABLE_MTP} "
            f"mtp_graph_requested={RUN_MODE == 'graph' and ENABLE_MTP and ENABLE_MTP_GRAPH}"
        )
        llm = LLM(**kwargs)

        if RUN_MODE == "cache-init":
            print(
                f"[dsa-smoke][dp={dp_rank}] split Indexer/MLA cache initialized; "
                "verify one capacity report per DP replica in the complete process log"
            )
            write_result(
                {
                    "mode": RUN_MODE,
                    "model": MODEL_PATH,
                    "dp_rank": dp_rank,
                    "status": "passed",
                    "validated": "LLM construction completed with DSA split cache enabled",
                },
                dp_rank,
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                min_tokens=0,
                ignore_eos=False,
            )
            try:
                if ENABLE_PROFILE:
                    llm.start_profile()
                outputs = llm.generate(
                    local_prompts,
                    sampling_params,
                    use_tqdm=False,
                )
            finally:
                if ENABLE_PROFILE:
                    llm.stop_profile()

            records = []
            print(f"\nGenerated Outputs (DP rank {dp_rank}):\n" + "-" * 60)
            for prompt_item, output in zip(prompt_items, outputs, strict=True):
                request_index, _ = prompt_item
                completion = output.outputs[0]
                record = {
                    "request_index": request_index,
                    "dp_rank": dp_rank,
                    "prompt": output.prompt,
                    "token_ids": list(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "text": completion.text,
                }
                records.append(record)
                print(
                    f"DP {dp_rank} Req {request_index}: "
                    f"token_ids={record['token_ids']} "
                    f"finish_reason={record['finish_reason']!r}"
                )
                print(f"Output: {record['text']!r}")
                print("-" * 60)

            write_result(
                {
                    "mode": RUN_MODE,
                    "model": MODEL_PATH,
                    "dp_rank": dp_rank,
                    "status": "passed",
                    "outputs": records,
                },
                dp_rank,
            )

        # Do not tear down one rank while another rank is still in a DP/EP
        # collective. Multi-node runs use a global TCPStore barrier; the
        # existing multiprocessing barrier is retained for single-node runs.
        if frontend_store is not None:
            wait_frontend_barrier(frontend_store, "shutdown", dp_rank)
        else:
            assert shutdown_barrier is not None
            shutdown_barrier.wait(timeout=DP_PROCESS_TIMEOUT_SECONDS)
    except BaseException:
        if shutdown_barrier is not None:
            shutdown_barrier.abort()
        raise
    finally:
        if llm is not None:
            del llm


def stop_processes(processes: list[multiprocessing.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)


def run_node_dp(args: argparse.Namespace) -> None:
    dp_master_ip, dp_master_port, sync_port, rank_map = (
        resolve_node_launch(args)
    )
    context = multiprocessing.get_context("spawn")
    shutdown_barrier = (
        context.Barrier(DATA_PARALLEL_SIZE)
        if args.node_size == 1
        else None
    )

    global_ranks = [global_rank for _, global_rank in rank_map]
    print(
        "[dsa-smoke] "
        f"node={args.node_rank}/{args.node_size} global topology "
        f"TP={TENSOR_PARALLEL_SIZE} DP={DATA_PARALLEL_SIZE} "
        f"EP={TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE}; "
        f"local_dp_ranks={len(rank_map)} global_dp_ranks={global_ranks} "
        f"required_local_npus={TENSOR_PARALLEL_SIZE * len(rank_map)} "
        f"dp_master={dp_master_ip}:{dp_master_port} "
        f"sync_port={sync_port}"
    )

    if DATA_PARALLEL_SIZE == 1:
        local_dp_rank, global_dp_rank = rank_map[0]
        run_dp_rank(
            global_dp_rank,
            local_dp_rank,
            dp_master_ip,
            dp_master_port,
            shutdown_barrier,
            sync_port,
        )
        return

    processes = [
        context.Process(
            target=run_dp_rank,
            args=(
                global_dp_rank,
                local_dp_rank,
                dp_master_ip,
                dp_master_port,
                shutdown_barrier,
                sync_port,
            ),
            name=f"dsa-dp-{global_dp_rank}",
        )
        for local_dp_rank, global_dp_rank in rank_map
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + DP_PROCESS_TIMEOUT_SECONDS
    try:
        while True:
            failed = [
                process
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed:
                details = ", ".join(
                    f"{process.name}={process.exitcode}" for process in failed
                )
                raise RuntimeError(f"offline DSA DP rank failed: {details}")
            if all(process.exitcode == 0 for process in processes):
                for process in processes:
                    process.join()
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "offline DSA DP did not finish within "
                    f"{DP_PROCESS_TIMEOUT_SECONDS} seconds"
                )
            time.sleep(1)
    finally:
        if any(process.is_alive() for process in processes):
            if shutdown_barrier is not None:
                shutdown_barrier.abort()
            stop_processes(processes)


def main() -> None:
    args = parse_args()
    if RUN_MODE not in {"disabled", "cache-init", "eager", "graph"}:
        raise ValueError(f"RUN_MODE must be disabled/cache-init/eager/graph, got {RUN_MODE!r}")
    if TENSOR_PARALLEL_SIZE < 1:
        raise ValueError("TENSOR_PARALLEL_SIZE must be positive")
    if DATA_PARALLEL_SIZE < 1:
        raise ValueError("DATA_PARALLEL_SIZE must be positive")
    max_prompts_per_rank = max(
        1,
        (len(PROMPTS) + DATA_PARALLEL_SIZE - 1)
        // DATA_PARALLEL_SIZE,
    )
    if max_prompts_per_rank > MAX_NUM_SEQS:
        raise ValueError(
            f"each DP rank receives up to {max_prompts_per_rank} prompts, "
            f"which exceeds MAX_NUM_SEQS={MAX_NUM_SEQS}"
        )
    if DATA_PARALLEL_SIZE > 1 and RUN_MODE != "disabled":
        if not ENABLE_EXPERT_PARALLEL:
            raise ValueError("DSA DP requires ENABLE_EXPERT_PARALLEL=True")
    run_node_dp(args)


if __name__ == "__main__":
    main()
