#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DSA 稀疏卸载最小离线冒烟脚本。

测试时只修改下方“用户配置”区，然后直接运行：

    python examples/dsa_demo/simple_prompt_test.py

``disabled`` 用于验证基线隔离，``cache-init`` 只验证双平面 KV cache
初始化，``eager`` 和 ``graph`` 分别验证 DSA eager 与 FULL decode graph。
默认使用 novel_dataset.chinese_20k，在单机 8 卡上验证 TP8 DSA offload
与 MTP3；实际 prompt token 长度以模型 tokenizer 为准。
DP 验收支持单机 DP + TP + EP；固定 MTP3 支持 target FULL decode graph，
GLM drafter 保持 eager。
"""

from __future__ import annotations

import json
import multiprocessing
import os
import socket
from pathlib import Path
from typing import Any

from novel_dataset import chinese_20k

# =========================
# 用户配置
# =========================

MODEL_PATH = "/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4"
RUN_MODE = "eager"  # disabled / cache-init / eager / graph
PROMPTS = chinese_20k

TENSOR_PARALLEL_SIZE = 8
DATA_PARALLEL_SIZE = 1
MAX_NUM_SEQS = 1
MAX_MODEL_LEN = 25600
MAX_NUM_BATCHED_TOKENS = 25600
ENABLE_CHUNKED_PREFILL = False
MAX_TOKENS = 64
GPU_MEMORY_UTILIZATION = 0.85
QUANTIZATION = "ascend"
ENABLE_EXPERT_PARALLEL = True
ENABLE_MTP = True
MTP_NUM_SPECULATIVE_TOKENS = 3

# A5/950 上的 DSA 首版只支持 LI C8 与 SFA C8 同时开启；A3/910C 保持 False。
# disabled 模式也会保留这个 vLLM-Ascend 原生物理布局开关，便于用同一 C8
# cache 编码公平对照“原生 C8”与“DSA C8”。
ENABLE_A5_PACKED_C8_DSA = True

ENABLE_PROFILE = False
PROFILE_DIR = "/home/data/vllm_profile/dsa_smoke"
RESULT_JSON: str | None = None

DSA_SPARSE_ACTIVATION_TOKENS = 6144
DSA_PROMPT_BUDGET_THRESHOLDS = [32768, 65536]
DSA_RESIDENT_BUDGET_TOKENS = [6144, 10240, 12288]
DSA_INDEXER_MLA_BLOCK_RATIO = 3
DSA_MAX_ACTIVE_REQS = 256
DSA_HOT_CPU_BLOCK_MULTIPLE = 3.0
DSA_GRAPH_CAPTURE_SIZES = [1, 2, 4, 8]
DP_WORKER_TIMEOUT_SECONDS = 1800
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
    "VLLM_ASCEND_ENABLE_MLAPO": "1",
    "VLLM_LOGGING_LEVEL": "INFO",
}

for key, value in NATIVE_RUNTIME_ENV_OVERRIDES.items():
    os.environ[key] = value


def build_dsa_config(enable_graph: bool) -> dict[str, Any]:
    sparse_activation_tokens = 8192 if ENABLE_MTP else DSA_SPARSE_ACTIVATION_TOKENS
    resident_budget_tokens = [8192, 10240, 12288] if ENABLE_MTP else DSA_RESIDENT_BUDGET_TOKENS
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
        raise ValueError("DSA compromise MTP3 requires ENABLE_A5_PACKED_C8_DSA=True")
    if ENABLE_MTP and MTP_NUM_SPECULATIVE_TOKENS != 3:
        raise ValueError("DSA compromise MTP requires MTP_NUM_SPECULATIVE_TOKENS=3")
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
        decode_query_len = 1 + MTP_NUM_SPECULATIVE_TOKENS if ENABLE_MTP else 1
        capture_sizes = sorted(
            {size * decode_query_len for size in (*DSA_GRAPH_CAPTURE_SIZES, MAX_NUM_SEQS) if size <= MAX_NUM_SEQS}
        )
        kwargs["compilation_config"] = {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": capture_sizes,
        }
    if ENABLE_PROFILE:
        from vllm.config import ProfilerConfig

        profile_dir = Path(PROFILE_DIR).resolve()
        if DATA_PARALLEL_SIZE > 1:
            profile_dir /= f"dp_rank_{dp_rank}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(profile_dir),
            torch_profiler_with_stack=True,
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=True,
        )
    return kwargs


def prompt_indices_for_rank(dp_rank: int) -> list[int]:
    floor, remainder = divmod(len(PROMPTS), DATA_PARALLEL_SIZE)
    start = dp_rank * floor + min(dp_rank, remainder)
    stop = start + floor + (dp_rank < remainder)
    return list(range(start, stop))


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_result(payload: dict[str, Any], dp_rank: int) -> None:
    if RESULT_JSON is None:
        return
    path = Path(RESULT_JSON)
    if DATA_PARALLEL_SIZE > 1:
        path = path.with_name(f"{path.stem}.dp_rank_{dp_rank}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[dsa-smoke] wrote result: {path}")


def main(dp_rank: int = 0, dp_master_port: int = 0) -> None:
    if RUN_MODE not in {"disabled", "cache-init", "eager", "graph"}:
        raise ValueError(f"RUN_MODE must be disabled/cache-init/eager/graph, got {RUN_MODE!r}")
    max_local_prompts = (len(PROMPTS) + DATA_PARALLEL_SIZE - 1) // DATA_PARALLEL_SIZE
    if max_local_prompts > MAX_NUM_SEQS:
        raise ValueError(f"Each DP rank may receive {max_local_prompts} prompts but MAX_NUM_SEQS={MAX_NUM_SEQS}")

    if DATA_PARALLEL_SIZE > 1:
        os.environ.update(
            {
                "VLLM_DP_RANK": str(dp_rank),
                "VLLM_DP_RANK_LOCAL": str(dp_rank),
                "VLLM_DP_SIZE": str(DATA_PARALLEL_SIZE),
                "VLLM_DP_MASTER_IP": "127.0.0.1",
                "VLLM_DP_MASTER_PORT": str(dp_master_port),
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            }
        )

    # DP 环境变量必须在首次导入 vLLM 前设置。
    from vllm import LLM, SamplingParams

    prompt_indices = prompt_indices_for_rank(dp_rank)
    local_prompts = [PROMPTS[index] for index in prompt_indices]
    if DATA_PARALLEL_SIZE > 1 and not local_prompts:
        local_prompts = ["Placeholder"]

    kwargs = build_llm_kwargs(dp_rank)
    print(
        "[dsa-smoke] "
        f"mode={RUN_MODE} model={MODEL_PATH!r} prompts={len(local_prompts)} "
        f"max_model_len={MAX_MODEL_LEN} max_num_seqs={MAX_NUM_SEQS} "
        f"dp={DATA_PARALLEL_SIZE} dp_rank={dp_rank} "
        f"a5_packed_c8={ENABLE_A5_PACKED_C8_DSA} mtp={ENABLE_MTP}"
    )
    llm = LLM(**kwargs)

    if RUN_MODE == "cache-init":
        print(
            "[dsa-smoke] split Indexer/MLA cache initialized; "
            "verify exactly one capacity report in the complete process log"
        )
        write_result(
            {
                "mode": RUN_MODE,
                "model": MODEL_PATH,
                "status": "passed",
                "validated": ("LLM construction completed with DSA split cache enabled"),
            },
            dp_rank,
        )
        del llm
        return

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
    for request_index, output in zip(prompt_indices, outputs):
        completion = output.outputs[0]
        record = {
            "request_index": request_index,
            "prompt": output.prompt,
            "token_ids": list(completion.token_ids),
            "finish_reason": completion.finish_reason,
            "text": completion.text,
        }
        records.append(record)
        print(
            f"DP {dp_rank} Req {request_index}: token_ids={record['token_ids']} "
            f"finish_reason={record['finish_reason']!r}"
        )
        print(f"Output: {record['text']!r}")
        print("-" * 60)

    write_result(
        {
            "mode": RUN_MODE,
            "model": MODEL_PATH,
            "status": "passed",
            "outputs": records,
        },
        dp_rank,
    )
    del llm


if __name__ == "__main__":
    if DATA_PARALLEL_SIZE < 1:
        raise ValueError("DATA_PARALLEL_SIZE must be positive")
    if DATA_PARALLEL_SIZE == 1:
        main()
    else:
        context = multiprocessing.get_context("spawn")
        master_port = get_open_port()
        processes = [
            context.Process(target=main, args=(rank, master_port))
            for rank in range(DATA_PARALLEL_SIZE)
        ]
        for process in processes:
            process.start()

        exit_code = 0
        for process in processes:
            process.join(timeout=DP_WORKER_TIMEOUT_SECONDS)
            if process.is_alive():
                print(f"Killing DP worker {process.pid} after timeout")
                process.kill()
                process.join()
                exit_code = 1
            elif process.exitcode:
                exit_code = process.exitcode
        raise SystemExit(exit_code)
