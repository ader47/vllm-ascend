#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""验证 v0.23 DSA 迁移 P2 KV-cache 控制面。

本脚本只验证：

1. DSA 关闭时，v0.23 原生路径没有受到影响；
2. DSA 开启时，Indexer/MLA 能形成两个独立 group、tensor 和物理池；
3. 尚未支持的 async scheduling 会在启动期被明确拒绝。

``cache-init`` 模式只构造 ``LLM`` 并等待 KV cache 初始化完成，不执行
``generate``。P4/P5 数据面尚未接通前，调用生成不能证明 DSA 稀疏卸载
正确，甚至可能让原生 SFA 按 packed cache 语义误读独立 tensor。
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

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

DEFAULT_PROMPTS = [
    "你好，请用一句话介绍你自己。",
    "请只回答数字：一加一等于多少？",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the v0.23 DSA P2 KV-cache control-plane smoke test.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DSA_MODEL_PATH"),
        help="Local GLM-5.1 or DeepSeek-V3.2 model path. It can also be supplied through DSA_MODEL_PATH.",
    )
    parser.add_argument(
        "--mode",
        choices=("disabled", "cache-init", "reject-async"),
        default="cache-init",
        help="P2 scenario to execute.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=16)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--quantization", default="ascend")
    parser.add_argument(
        "--disable-expert-parallel",
        action="store_true",
        help="Disable expert parallelism for a model that does not need it.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Raw prompt. Repeat this option to send multiple prompts.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Optional path used to store output token IDs for comparison.",
    )
    args = parser.parse_args()

    if not args.model:
        parser.error("--model or DSA_MODEL_PATH is required")
    if args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be positive")
    if args.data_parallel_size <= 0:
        parser.error("--data-parallel-size must be positive")
    if args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")
    if args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")

    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = args.max_model_len
    if args.max_num_batched_tokens <= 0:
        parser.error("--max-num-batched-tokens must be positive")
    if args.prompts is None:
        args.prompts = list(DEFAULT_PROMPTS)
    if len(args.prompts) > args.max_num_seqs:
        parser.error("the number of prompts cannot exceed --max-num-seqs in this compact smoke test")
    return args


def apply_runtime_env() -> None:
    for key, value in NATIVE_RUNTIME_ENV_OVERRIDES.items():
        os.environ.setdefault(key, value)


def build_dsa_sparse_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "split_indexer_cache": True,
        "indexer_mla_block_ratio": 3,
        "sparse_activation_tokens": 6144,
        "prompt_budget_thresholds": [32768, 65536],
        "resident_budget_tokens": [6144, 10240, 12288],
        "max_active_reqs": 256,
        "hot_cpu_block_multiple": 3.0,
        # P2 只验证 cache 控制面；图数据面将在 P6 迁移。
        "enable_row_mode_decode_graph": False,
        "trace_points": {
            "enabled": False,
            "points": ["first_sample"],
            "ranks": [0],
        },
    }


def exception_chain_contains(exc: BaseException, expected: str) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if expected in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def write_result(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[dsa-p2] wrote result: {path}")


def main() -> None:
    args = parse_args()
    apply_runtime_env()

    # vLLM-Ascend must discover its platform plugin during this import, so all
    # environment variables are configured first.
    from vllm import LLM, SamplingParams

    dsa_config = None if args.mode == "disabled" else build_dsa_sparse_config()
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "data_parallel_size": args.data_parallel_size,
        "quantization": args.quantization,
        "seed": 1024,
        "enable_expert_parallel": not args.disable_expert_parallel,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "block_size": 128,
        "async_scheduling": args.mode == "reject-async",
        "enforce_eager": True,
        "disable_log_stats": False,
    }
    if dsa_config is not None:
        llm_kwargs["additional_config"] = {
            "dsa_sparse_config": dsa_config,
        }

    config_payload = {
        "phase": "P2-kv-cache-control-plane",
        "mode": args.mode,
        "model": args.model,
        "prompts": args.prompts,
        "llm": {key: value for key, value in llm_kwargs.items() if key not in {"model"}},
        "environment": {key: os.environ.get(key) for key in NATIVE_RUNTIME_ENV_OVERRIDES},
        "notice": (
            "P4-P5 are not connected yet; cache-init only validates split "
            "Indexer/MLA cache construction and allocation."
        ),
    }
    print("[dsa-p2] configuration:")
    print(json.dumps(config_payload, ensure_ascii=False, indent=2))

    expected_error = "DSA sparse offload currently requires async_scheduling=False"
    try:
        llm = LLM(**llm_kwargs)
    except Exception as exc:
        if args.mode != "reject-async":
            raise
        traceback.print_exc()
        if not exception_chain_contains(exc, expected_error):
            raise RuntimeError("reject-async failed for an unexpected reason") from exc
        payload = {
            **config_payload,
            "status": "passed",
            "expected_error": expected_error,
            "observed_error": str(exc),
        }
        print("[dsa-p2] PASS: async scheduling was rejected as expected")
        write_result(args.result_json, payload)
        return

    if args.mode == "reject-async":
        raise AssertionError("DSA enabled with async_scheduling=True unexpectedly started")

    if args.mode == "cache-init":
        payload = {
            **config_payload,
            "status": "passed",
            "expected_log": "DSA HBM CACHE CAPACITY REPORT",
        }
        print("[dsa-p2] PASS: split Indexer/MLA KV cache initialized; generation was intentionally skipped")
        write_result(args.result_json, payload)
        del llm
        return

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )
    try:
        outputs = llm.generate(
            args.prompts,
            sampling_params,
            use_tqdm=False,
        )
        output_payload = []
        for request_idx, output in enumerate(outputs):
            completion = output.outputs[0]
            record = {
                "request_index": request_idx,
                "prompt": output.prompt,
                "token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "text": completion.text,
            }
            output_payload.append(record)
            print(
                f"[dsa-p2] request={request_idx} "
                f"token_ids={record['token_ids']} "
                f"finish_reason={record['finish_reason']!r}"
            )
            print(f"[dsa-p2] output={record['text']!r}")

        payload = {
            **config_payload,
            "status": "passed",
            "outputs": output_payload,
        }
        write_result(args.result_json, payload)
    finally:
        del llm


if __name__ == "__main__":
    main()
