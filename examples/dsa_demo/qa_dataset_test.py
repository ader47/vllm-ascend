#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""运行 LongBench 风格的 DSA 精度回归。

测试同事通常只需要修改“用户配置”区，然后直接执行：

    python examples/dsa_demo/qa_dataset_test.py

脚本将预测保存为 LongBench 兼容 JSONL。分别把 ``RUN_MODE`` 设为
``disabled``、``eager`` 和 ``graph``，即可在相同数据切片上做基线、
DSA eager 与 DSA FULL graph 对照。
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
DATASET_FILE = "/home/data/longbench/multifieldqa_zh.jsonl"
DATASET_START = 0
DATASET_LIMIT = 100
RESULT_DIR = "LongBenchResult/glm51_dsa"

# disabled: 原生 vLLM-Ascend；eager: DSA eager；graph: DSA FULL decode graph。
RUN_MODE = "eager"
BATCH_SIZE = 4
MAX_MODEL_LEN = 131072
MAX_NUM_BATCHED_TOKENS = 131072
TENSOR_PARALLEL_SIZE = 16
GPU_MEMORY_UTILIZATION = 0.90
MAX_TOKENS = 32
MIN_TOKENS = 0

ENABLE_PROFILE = False
PROFILE_DIR = "/home/data/vllm_profile/dsa_dataset"

DSA_SPARSE_ACTIVATION_TOKENS = 6144
DSA_PROMPT_BUDGET_THRESHOLDS = [32768, 65536]
DSA_RESIDENT_BUDGET_TOKENS = [6144, 10240, 12288]
DSA_INDEXER_MLA_BLOCK_RATIO = 3
DSA_MAX_ACTIVE_REQS = 256
DSA_HOT_CPU_BLOCK_MULTIPLE = 3.0
DSA_GRAPH_CAPTURE_SIZES = [1, 2, 4, 8]

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

PROMPT_TEMPLATE = """请基于下面给定的上下文，简洁准确地回答问题。只输出最终答案，不要解释，不要重复题目或上下文。

问题：
{question}

上下文：
{context}

答案："""

SCRIPT_DIR = Path(__file__).resolve().parent


def apply_runtime_env() -> None:
    for key, value in NATIVE_RUNTIME_ENV_OVERRIDES.items():
        os.environ.setdefault(key, value)


apply_runtime_env()

# 环境变量必须在首次导入 vLLM 前设置，平台插件会在 import/LLM 构造期间发现。
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import ProfilerConfig  # noqa: E402
from vllm.tokenizers import get_tokenizer  # noqa: E402


def validate_user_config() -> None:
    if RUN_MODE not in {"disabled", "eager", "graph"}:
        raise ValueError(f"RUN_MODE must be one of disabled/eager/graph, got {RUN_MODE!r}")
    if BATCH_SIZE <= 0:
        raise ValueError(f"BATCH_SIZE must be positive, got {BATCH_SIZE}")
    if DATASET_START < 0 or DATASET_LIMIT <= 0:
        raise ValueError("DATASET_START must be non-negative and DATASET_LIMIT positive")
    if MAX_MODEL_LEN <= MAX_TOKENS:
        raise ValueError("MAX_MODEL_LEN must leave room for generated tokens")
    if MAX_NUM_BATCHED_TOKENS <= 0:
        raise ValueError("MAX_NUM_BATCHED_TOKENS must be positive")


def normalize_answers(raw_answers: Any) -> list[str]:
    if raw_answers is None:
        return []
    if isinstance(raw_answers, list):
        return [str(item) for item in raw_answers]
    return [str(raw_answers)]


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_index, line in enumerate(source):
            if line_index < DATASET_START:
                continue
            if len(records) >= DATASET_LIMIT:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"dataset line {line_index + 1} is not a JSON object")
            records.append(record)

    if not records:
        raise ValueError(f"no records loaded from {path}; start={DATASET_START}, limit={DATASET_LIMIT}")
    return records


def build_prompt(record: dict[str, Any], sample_index: int) -> str:
    question = str(record.get("input", "")).strip()
    context = str(record.get("context", "")).strip()
    if not question:
        raise ValueError(f"sample {sample_index} has empty 'input'")
    if not context:
        raise ValueError(f"sample {sample_index} has empty 'context'")
    return PROMPT_TEMPLATE.format(question=question, context=context)


def load_prompts() -> tuple[list[str], list[dict[str, Any]]]:
    records = load_records(Path(DATASET_FILE))
    prompts: list[str] = []
    sample_infos: list[dict[str, Any]] = []
    for offset, record in enumerate(records):
        sample_index = DATASET_START + offset
        prompts.append(build_prompt(record, sample_index))
        sample_infos.append(
            {
                "sample_index": sample_index,
                "sample_id": record.get("_id"),
                "question": str(record.get("input", "")).strip(),
                "answers": normalize_answers(record.get("answers")),
                "all_classes": record.get("all_classes", []),
                "dataset_length": record.get("length"),
            }
        )
    return prompts, sample_infos


def get_prompt_token_lengths(prompts: list[str]) -> list[int]:
    tokenizer = get_tokenizer(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    return [len(tokenizer.encode(prompt, add_special_tokens=True)) for prompt in prompts]


def filter_oversized_prompts(
    prompts: list[str],
    sample_infos: list[dict[str, Any]],
    prompt_lengths: list[int],
) -> tuple[list[str], list[dict[str, Any]], list[int]]:
    max_prompt_tokens = MAX_MODEL_LEN - MAX_TOKENS
    kept = [
        (prompt, info, length)
        for prompt, info, length in zip(
            prompts,
            sample_infos,
            prompt_lengths,
        )
        if length <= max_prompt_tokens
    ]
    if not kept:
        raise ValueError("all prompts exceed MAX_MODEL_LEN after reserving output tokens")
    dropped = len(prompts) - len(kept)
    if dropped:
        print(f"[dsa-dataset] skipped {dropped} oversized prompts; required prompt_tokens <= {max_prompt_tokens}")
    return (
        [item[0] for item in kept],
        [item[1] for item in kept],
        [item[2] for item in kept],
    )


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
        # 当前仅解析预留合同，正式精度/性能回归保持关闭。
        "trace_points": {
            "enabled": False,
            "points": ["first_sample"],
            "ranks": [0],
        },
    }


def build_llm() -> LLM:
    graph_enabled = RUN_MODE == "graph"
    llm_kwargs: dict[str, Any] = {
        "model": MODEL_PATH,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "quantization": "ascend",
        "seed": 1024,
        "enable_expert_parallel": True,
        "max_num_seqs": BATCH_SIZE,
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
        llm_kwargs["additional_config"] = {
            "dsa_sparse_config": build_dsa_config(graph_enabled),
        }
    if graph_enabled:
        capture_sizes = sorted({size for size in (*DSA_GRAPH_CAPTURE_SIZES, BATCH_SIZE) if size <= BATCH_SIZE})
        llm_kwargs["compilation_config"] = {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": capture_sizes,
        }
    if ENABLE_PROFILE:
        profile_dir = Path(PROFILE_DIR).resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        llm_kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(profile_dir),
            torch_profiler_with_stack=True,
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=True,
        )
    return LLM(**llm_kwargs)


def result_file() -> Path:
    result_dir = Path(RESULT_DIR)
    if not result_dir.is_absolute():
        result_dir = SCRIPT_DIR / result_dir
    return result_dir / RUN_MODE / f"{Path(DATASET_FILE).stem}.jsonl"


def print_config(prompt_lengths: list[int]) -> None:
    payload = {
        "mode": RUN_MODE,
        "model": MODEL_PATH,
        "dataset": DATASET_FILE,
        "dataset_slice": [DATASET_START, DATASET_START + DATASET_LIMIT],
        "result_file": str(result_file()),
        "batch_size": BATCH_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "prompt_count": len(prompt_lengths),
        "prompt_tokens_min": min(prompt_lengths),
        "prompt_tokens_max": max(prompt_lengths),
        "prompt_tokens_sum": sum(prompt_lengths),
        "dsa": (None if RUN_MODE == "disabled" else build_dsa_config(RUN_MODE == "graph")),
        "environment": {key: os.environ.get(key) for key in NATIVE_RUNTIME_ENV_OVERRIDES},
    }
    print("[dsa-dataset] configuration:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def score_record(
    prediction: str,
    info: dict[str, Any],
    prompt_tokens: int,
) -> dict[str, Any]:
    dataset_length = info["dataset_length"]
    return {
        "pred": prediction,
        "answers": info["answers"],
        "all_classes": info["all_classes"],
        "length": (prompt_tokens if dataset_length is None else dataset_length),
        "sample_id": info["sample_id"],
        "sample_index": info["sample_index"],
        "prompt_tokens": prompt_tokens,
    }


def batch_ranges(total: int):
    for start in range(0, total, BATCH_SIZE):
        yield start, min(start + BATCH_SIZE, total)


def main() -> None:
    validate_user_config()
    prompts, sample_infos = load_prompts()
    prompt_lengths = get_prompt_token_lengths(prompts)
    prompts, sample_infos, prompt_lengths = filter_oversized_prompts(
        prompts,
        sample_infos,
        prompt_lengths,
    )
    print_config(prompt_lengths)

    output_path = result_file()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        min_tokens=MIN_TOKENS,
        ignore_eos=False,
    )

    llm = build_llm()
    try:
        if ENABLE_PROFILE:
            llm.start_profile()
        with output_path.open("w", encoding="utf-8") as result:
            for start, end in batch_ranges(len(prompts)):
                batch_prompts = prompts[start:end]
                batch_infos = sample_infos[start:end]
                batch_lengths = prompt_lengths[start:end]
                total_prompt_tokens = sum(batch_lengths)
                print(
                    "\n[dsa-dataset] batch "
                    f"[{start}, {end}) bsz={end - start} "
                    f"lengths={batch_lengths} "
                    f"sum={total_prompt_tokens}"
                )
                if total_prompt_tokens > MAX_NUM_BATCHED_TOKENS:
                    print(
                        "[dsa-dataset] note: this generate batch exceeds "
                        "max_num_batched_tokens; the scheduler may admit "
                        "multiple prefill waves because chunked prefill is off"
                    )

                outputs = llm.generate(
                    batch_prompts,
                    sampling_params,
                    use_tqdm=False,
                )
                for offset, output in enumerate(outputs):
                    completion = output.outputs[0]
                    info = batch_infos[offset]
                    prompt_tokens = batch_lengths[offset]
                    record = score_record(
                        completion.text,
                        info,
                        prompt_tokens,
                    )
                    result.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(
                        f"[dsa-dataset] req={start + offset} "
                        f"sample_index={info['sample_index']} "
                        f"prompt_tokens={prompt_tokens} "
                        f"output_tokens={len(completion.token_ids)} "
                        f"finish_reason={completion.finish_reason!r}"
                    )
                    print(f"Output: {completion.text!r}")
                result.flush()
    finally:
        if ENABLE_PROFILE:
            llm.stop_profile()
        del llm

    print(f"[dsa-dataset] wrote LongBench JSONL: {output_path}")
    print(
        "[dsa-dataset] score with: "
        "python examples/dsa_demo/eval_dataset_acc_score.py "
        f"--result-path {output_path.parent}"
    )


if __name__ == "__main__":
    main()
