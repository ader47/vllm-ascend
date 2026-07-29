#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DSA 稀疏卸载最小离线冒烟脚本。

测试时只修改下方“用户配置”区，然后直接运行：

    python examples/dsa_demo/simple_prompt_test.py

``disabled`` 用于验证基线隔离，``cache-init`` 只验证双平面 KV cache
初始化，``eager`` 和 ``graph`` 分别验证 DSA eager 与 FULL decode graph。
默认短 prompt 主要覆盖 DENSE；验证真正的 sparse/ENTER 路径时，应换成
token 长度超过 ``DSA_SPARSE_ACTIVATION_TOKENS`` 的文本。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# =========================
# 用户配置
# =========================

MODEL_PATH = "/mnt/kv_dpc/weight/GLM-5.1-w4a8"
RUN_MODE = "eager"  # disabled / cache-init / eager / graph
PROMPTS = [
    "你好，请用一句话介绍你自己。",
    "请只回答数字：一加一等于多少？",
]

TENSOR_PARALLEL_SIZE = 16
DATA_PARALLEL_SIZE = 1
MAX_NUM_SEQS = 2
MAX_MODEL_LEN = 8192
MAX_NUM_BATCHED_TOKENS = 8192
MAX_TOKENS = 32
GPU_MEMORY_UTILIZATION = 0.90
QUANTIZATION = "ascend"
ENABLE_EXPERT_PARALLEL = True

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

# 环境变量必须在首次导入 vLLM 前设置，平台插件会在 import/LLM 构造期间发现。
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import ProfilerConfig  # noqa: E402


def build_dsa_config(enable_graph: bool) -> dict[str, Any]:
    return {
        "enabled": True,
        "split_indexer_cache": True,
        "indexer_mla_block_ratio": DSA_INDEXER_MLA_BLOCK_RATIO,
        "sparse_activation_tokens": DSA_SPARSE_ACTIVATION_TOKENS,
        "prompt_budget_thresholds": DSA_PROMPT_BUDGET_THRESHOLDS,
        "resident_budget_tokens": DSA_RESIDENT_BUDGET_TOKENS,
        "max_active_reqs": DSA_MAX_ACTIVE_REQS,
        "hot_cpu_block_multiple": DSA_HOT_CPU_BLOCK_MULTIPLE,
        "enable_row_mode_decode_graph": enable_graph,
        "trace_points": DSA_TRACE_POINTS,
    }


def build_llm_kwargs() -> dict[str, Any]:
    graph_enabled = RUN_MODE == "graph"
    kwargs: dict[str, Any] = {
        "model": MODEL_PATH,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "pipeline_parallel_size": 1,
        "data_parallel_size": DATA_PARALLEL_SIZE,
        "quantization": QUANTIZATION,
        "seed": 1024,
        "enable_expert_parallel": ENABLE_EXPERT_PARALLEL,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "block_size": 128,
        "async_scheduling": False,
        "enforce_eager": not graph_enabled,
        "disable_log_stats": False,
    }
    if RUN_MODE != "disabled":
        kwargs["additional_config"] = {
            "dsa_sparse_config": build_dsa_config(graph_enabled),
        }
    if graph_enabled:
        capture_sizes = sorted({size for size in (*DSA_GRAPH_CAPTURE_SIZES, MAX_NUM_SEQS) if size <= MAX_NUM_SEQS})
        kwargs["compilation_config"] = {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": capture_sizes,
        }
    if ENABLE_PROFILE:
        profile_dir = Path(PROFILE_DIR).resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(profile_dir),
            torch_profiler_with_stack=True,
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=True,
        )
    return kwargs


def write_result(payload: dict[str, Any]) -> None:
    if RESULT_JSON is None:
        return
    path = Path(RESULT_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[dsa-smoke] wrote result: {path}")


def main() -> None:
    if RUN_MODE not in {"disabled", "cache-init", "eager", "graph"}:
        raise ValueError(f"RUN_MODE must be disabled/cache-init/eager/graph, got {RUN_MODE!r}")
    if len(PROMPTS) > MAX_NUM_SEQS:
        raise ValueError(f"PROMPTS has {len(PROMPTS)} rows but MAX_NUM_SEQS={MAX_NUM_SEQS}")

    kwargs = build_llm_kwargs()
    print(
        "[dsa-smoke] "
        f"mode={RUN_MODE} model={MODEL_PATH!r} prompts={len(PROMPTS)} "
        f"max_model_len={MAX_MODEL_LEN} max_num_seqs={MAX_NUM_SEQS}"
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
            }
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
            PROMPTS,
            sampling_params,
            use_tqdm=False,
        )
    finally:
        if ENABLE_PROFILE:
            llm.stop_profile()

    records = []
    print("\nGenerated Outputs:\n" + "-" * 60)
    for request_index, output in enumerate(outputs):
        completion = output.outputs[0]
        record = {
            "request_index": request_index,
            "prompt": output.prompt,
            "token_ids": list(completion.token_ids),
            "finish_reason": completion.finish_reason,
            "text": completion.text,
        }
        records.append(record)
        print(f"Req {request_index}: token_ids={record['token_ids']} finish_reason={record['finish_reason']!r}")
        print(f"Output: {record['text']!r}")
        print("-" * 60)

    write_result(
        {
            "mode": RUN_MODE,
            "model": MODEL_PATH,
            "status": "passed",
            "outputs": records,
        }
    )
    del llm


if __name__ == "__main__":
    main()
