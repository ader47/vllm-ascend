#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DSA 稀疏卸载最小离线冒烟脚本。

测试时只修改下方“用户配置”区，然后直接运行：

    python examples/dsa_demo/simple_prompt_test_dp.py

``disabled`` 用于验证基线隔离，``cache-init`` 只验证双平面 KV cache
初始化，``eager`` 和 ``graph`` 分别验证 DSA eager 与 FULL decode graph。
DP>1 时由父进程创建一个离线前端进程/DP rank，并通过 ``VLLM_DP_*``
建立 SPMD 协调；MTP drafter 保持 eager，target model 使用 FULL decode graph。
默认短 prompt 主要覆盖 DENSE；验证真正的 sparse/ENTER 路径时，应换成
token 长度超过 ``DSA_SPARSE_ACTIVATION_TOKENS`` 的文本。
"""

from __future__ import annotations

import json
import multiprocessing
import os
import socket
import time
from pathlib import Path
from typing import Any

from novel_dataset import chinese_20k

PROMPTS = [(chinese_20k[0]*3)[:92800],] * (4 * 16)

# =========================
# 用户配置
# =========================

MODEL_PATH = "/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4"
RUN_MODE = "graph"  # disabled / cache-init / eager / graph

# 单机总 NPU 数 = TP * DP。8 卡上用 TP4DP2；TP8DP2 需要 16 卡。
TENSOR_PARALLEL_SIZE = 2
DATA_PARALLEL_SIZE = 4
DP_PROCESS_TIMEOUT_SECONDS = 3600
# max_num_seqs 按 DP rank 计算；16 条 prompt 在 DP2 下每个 rank 处理 8 条。
MAX_NUM_SEQS = 20
MAX_MODEL_LEN = 131072
MAX_NUM_BATCHED_TOKENS = 4096
ENABLE_CHUNKED_PREFILL = True
MAX_TOKENS = 256
GPU_MEMORY_UTILIZATION = 0.94
QUANTIZATION = "ascend"
ENABLE_EXPERT_PARALLEL = True
ENABLE_MTP = True
MTP_NUM_SPECULATIVE_TOKENS = 3

# A5/950 上的 DSA 首版只支持 LI C8 与 SFA C8 同时开启；A3/910C 保持 False。
# disabled 模式也会保留这个 vLLM-Ascend 原生物理布局开关，便于用同一 C8
# cache 编码公平对照“原生 C8”与“DSA C8”。
ENABLE_A5_PACKED_C8_DSA = True

ENABLE_PROFILE = True
PROFILE_DIR = "/home/w00916487/vllm_profile/dp4tp2_offload_seq64k_bs64/"
RESULT_JSON: str | None = None

DSA_PROMPT_BUDGET_THRESHOLDS = [32768, 65536]
DSA_INDEXER_MLA_BLOCK_RATIO = 4
DSA_MAX_ACTIVE_REQS = 256
DSA_HOT_CPU_BLOCK_MULTIPLE = 1.0
DSA_GRAPH_CAPTURE_SIZES = [1, 2, 4, 8, 16]
DSA_TRACE_POINTS = {
    # 当前仅解析预留合同，尚无稳定日志 consumer；验收默认关闭。
    "enabled": False,
    "points": ["first_sample"],
    "ranks": [0],
}

NATIVE_RUNTIME_ENV_OVERRIDES = {
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_BUFFSIZE": "512",
    "OMP_NUM_THREADS": "10",
    "OMP_PROC_BIND": "false",
    "PYTHONHASHSEED": "114514",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
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


def build_dsa_config(enable_graph: bool) -> dict[str, Any]:
    sparse_activation_tokens = (
        12288 if ENABLE_MTP else 12288
    )
    resident_budget_tokens = (
        [12288, 12288, 12288]
        if ENABLE_MTP
        else [12288, 12288, 12288]
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


def build_llm_kwargs(dp_rank: int) -> dict[str, Any]:
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
            # 只约束 drafter；target 仍由顶层配置进入 FULL decode graph。
            "enforce_eager": True,
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
    shutdown_barrier: Any,
) -> None:
    configure_dp_rank(
        dp_rank=dp_rank,
        local_dp_rank=local_dp_rank,
        dp_master_ip=dp_master_ip,
        dp_master_port=dp_master_port,
    )

    # DP 环境变量必须在首次导入 vLLM 前设置。
    from vllm import LLM, SamplingParams

    prompt_items = shard_prompts(dp_rank)
    local_prompts = [prompt for _, prompt in prompt_items]
    kwargs = build_llm_kwargs(dp_rank)
    print(
        "[dsa-smoke] "
        f"rank={dp_rank}/{DATA_PARALLEL_SIZE} local_rank={local_dp_rank} "
        f"mode={RUN_MODE} model={MODEL_PATH!r} prompts={len(local_prompts)} "
        f"tp={TENSOR_PARALLEL_SIZE} ep={TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE} "
        f"max_model_len={MAX_MODEL_LEN} max_num_seqs={MAX_NUM_SEQS} "
        f"a5_packed_c8={ENABLE_A5_PACKED_C8_DSA} mtp={ENABLE_MTP}"
    )
    llm = LLM(**kwargs)

    try:
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
        # collective. The parent aborts this barrier if a rank fails.
        shutdown_barrier.wait(timeout=DP_PROCESS_TIMEOUT_SECONDS)
    except BaseException:
        shutdown_barrier.abort()
        raise
    finally:
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


def run_single_node_dp() -> None:
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    context = multiprocessing.get_context("spawn")
    shutdown_barrier = context.Barrier(DATA_PARALLEL_SIZE)

    if DATA_PARALLEL_SIZE == 1:
        run_dp_rank(
            0,
            0,
            dp_master_ip,
            dp_master_port,
            shutdown_barrier,
        )
        return

    processes = [
        context.Process(
            target=run_dp_rank,
            args=(
                dp_rank,
                dp_rank,
                dp_master_ip,
                dp_master_port,
                shutdown_barrier,
            ),
            name=f"dsa-dp-{dp_rank}",
        )
        for dp_rank in range(DATA_PARALLEL_SIZE)
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
            shutdown_barrier.abort()
            stop_processes(processes)


def main() -> None:
    if RUN_MODE not in {"disabled", "cache-init", "eager", "graph"}:
        raise ValueError(f"RUN_MODE must be disabled/cache-init/eager/graph, got {RUN_MODE!r}")
    if len(PROMPTS) > MAX_NUM_SEQS:
        print(
            f"[dsa-smoke] PROMPTS has {len(PROMPTS)} rows; DP sharding applies "
            f"MAX_NUM_SEQS={MAX_NUM_SEQS} independently to each rank"
        )
    if TENSOR_PARALLEL_SIZE < 1:
        raise ValueError("TENSOR_PARALLEL_SIZE must be positive")
    if DATA_PARALLEL_SIZE < 1:
        raise ValueError("DATA_PARALLEL_SIZE must be positive")
    if DATA_PARALLEL_SIZE > 1 and RUN_MODE != "disabled":
        if not ENABLE_EXPERT_PARALLEL:
            raise ValueError("DSA DP requires ENABLE_EXPERT_PARALLEL=True")
    print(
        "[dsa-smoke] "
        f"single-node topology TP={TENSOR_PARALLEL_SIZE} "
        f"DP={DATA_PARALLEL_SIZE} EP={TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE}; "
        f"required_npus={TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE}"
    )
    run_single_node_dp()


if __name__ == "__main__":
    main()
