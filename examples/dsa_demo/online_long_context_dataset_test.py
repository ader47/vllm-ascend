#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""通过 OpenAI 兼容在线服务运行长上下文数据集回归。

脚本面向测试同事，通常只需修改下方“用户配置”区：

1. 把 ``DATASET_ROOT`` 指向解压后的数据包；
2. 确认 ``API_BASE`` 和 ``MODEL_NAME`` 与已拉起服务一致；
3. 直接执行本文件。

脚本按数据集原始顺序处理样本，使用与服务一致的本地 tokenizer 做长度预检，
并以正式响应中的 ``usage.prompt_tokens`` 复核。数据集自带的字符数或其他
模型 token 数仅作为辅助信息，不会用于决定 resident budget。
"""

from __future__ import annotations

import json
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

# =========================
# 用户配置
# =========================

API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "glm-5.1-dsa"

# 默认从服务端 /v1/models 返回的 root 加载同一 tokenizer。服务使用独立
# --tokenizer 路径时，在这里显式填写对应路径；只加载 tokenizer，不加载权重。
LOCAL_TOKENIZER_PATH: str | None = None

# 把 C:\downloads\dsa_long_context_dataset_pack 上传并解压到该目录。
DATASET_ROOT = "/home/data/dsa_long_context_dataset_pack"

# 已有 LongBench 中文数据可作为额外短序列来源；文件不存在时自动跳过。
EXTRA_LONGBENCH_FILE: str | None = "/home/data/longbench/multifieldqa_zh.jsonl"

# 修改标签即可保留 baseline/eager/graph 多次在线测试结果。
RUN_LABEL = "dsa-online"
RESULT_DIR = "OnlineLongContextResult/glm51_dsa"

# 默认完整处理每个数据文件。仅在本地快速调试脚本时，可改成正整数限制
# 每个文件读取的样本数；该限制不参与按长度档选样。
MAX_SAMPLES_PER_FILE: int | None = None
REQUEST_CONCURRENCY = 4
MAX_TOKENS = 64
TEMPERATURE = 0.0

# GLM-5.1 的 chat template 默认开启思考。数据集短答案回归应关闭思考，
# 否则较小的 MAX_TOKENS 可能全部消耗在 reasoning，最终 content 为空。
# 该配置直接用于正式请求；分桶采用响应里的实际 prompt token 数。
ENABLE_THINKING = False

# 数据集精度回归默认使用非流式响应，结果更容易收集。设为 True 可额外
# 验证 SSE 流，但并发请求不会逐 token 打印，以免多条输出交错。
STREAM_RESPONSES = False
REQUEST_TIMEOUT_SECONDS = 600.0
REQUEST_RETRIES = 1
RESUME_EXISTING_RESULT = True
PRINT_OUTPUT_CHARS = 300
PRINT_TASK_CHARS = 300

# DSA 当前配置的实际边界。区间均按最终 chat prompt token 数划分。
LENGTH_BUCKETS = (
    ("dense_0_6k", 0, 6145),
    ("sparse_6k_16k", 6145, 16384),
    ("sparse_16k_32k", 16384, 32768),
    ("budget_32k_48k", 32768, 49152),
    ("budget_48k_64k", 49152, 65536),
    ("budget_64k_96k", 65536, 98304),
    ("budget_96k_plus", 98304, None),
)

LV_EVAL_PROMPT = (
    "请阅读以下文章并用中文回答问题，问题和答案只与其中一篇文章有关。"
    "只需要直接给出问题的答案，不要输出其他任何解释和证据。\n\n"
    "文章：{context}\n\n"
    "请基于上面的文章回答下面的问题，问题和答案只与其中一篇文章有关。"
    "只需要直接给出问题的答案，不要输出其他任何解释和证据。\n\n"
    "问题：{question}\n"
    "回答："
)

CLONG_STORY_PROMPT = (
    "下面是一部小说的节选。请阅读该小说节选，并尽可能用简洁的短语"
    "（或短句）回答给定的问题，不要提供任何解释。\n"
    "小说节选如下：\n\n"
    "{context}\n\n"
    "请尽可能简洁地回答下列问题，不要提供任何解释。\n"
    "问题：{question}\n"
    "答案："
)

CLONG_RETRIEVAL_PROMPT = """请提取下面 JSON 对象中指定键对应的值。只输出对应键的值，不输出任何其他文字。Json数据如下：

{context}

键：{question}
值："""

LONGBENCH_PROMPT = """请基于下面给定的上下文，简洁准确地回答问题。只输出最终答案，不要解释，不要重复题目或上下文。

问题：
{question}

上下文：
{context}

答案："""

TASK_METADATA = {
    "lveval": (
        "multifield_qa",
        "从多篇混合文章中定位相关内容，回答目标问题，只输出答案。",
    ),
    "longbench": (
        "long_context_qa",
        "根据给定长上下文回答问题，只输出简洁答案。",
    ),
    "clong_story": (
        "long_story_qa",
        "根据小说节选回答问题，只输出简短答案。",
    ),
    "clong_retrieval": (
        "key_value_retrieval",
        "从 JSON 对象中提取指定键对应的值，只输出该值。",
    ),
}

SCRIPT_DIR = Path(__file__).resolve().parent
CHINESE_PUNCTUATION = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀"
    "｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟"
    "〰〾〿–—‘’‛“”„‟…‧﹏"
)
PUNCTUATION = set(string.punctuation + CHINESE_PUNCTUATION)


def validate_user_config() -> None:
    if MAX_SAMPLES_PER_FILE is not None and MAX_SAMPLES_PER_FILE <= 0:
        raise ValueError("MAX_SAMPLES_PER_FILE must be positive or None")
    if REQUEST_CONCURRENCY <= 0:
        raise ValueError("REQUEST_CONCURRENCY must be positive")
    if MAX_TOKENS <= 0:
        raise ValueError("MAX_TOKENS must be positive")
    if not RUN_LABEL.strip():
        raise ValueError("RUN_LABEL must not be empty")


def normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def dataset_specs(root: Path) -> list[dict[str, str]]:
    specs = [
        {
            "dataset": "clongeval_long_story_qa",
            "subset": "small",
            "adapter": "clong_story",
            "path": str(root / "clongeval/long_story_qa/small.jsonl"),
        },
        {
            "dataset": "clongeval_long_story_qa",
            "subset": "medium",
            "adapter": "clong_story",
            "path": str(root / "clongeval/long_story_qa/medium.jsonl"),
        },
        {
            "dataset": "clongeval_long_story_qa",
            "subset": "large",
            "adapter": "clong_story",
            "path": str(root / "clongeval/long_story_qa/large.jsonl"),
        },
        {
            "dataset": "clongeval_key_passage_retrieval",
            "subset": "small",
            "adapter": "clong_retrieval",
            "path": str(root / "clongeval/key_passage_retrieval/small.jsonl"),
        },
        {
            "dataset": "clongeval_key_passage_retrieval",
            "subset": "medium",
            "adapter": "clong_retrieval",
            "path": str(root / "clongeval/key_passage_retrieval/medium.jsonl"),
        },
        {
            "dataset": "clongeval_key_passage_retrieval",
            "subset": "large",
            "adapter": "clong_retrieval",
            "path": str(root / "clongeval/key_passage_retrieval/large.jsonl"),
        },
        {
            "dataset": "lveval_multifieldqa_zh_mixup",
            "subset": "16k",
            "adapter": "lveval",
            "path": str(root / "lveval/multifieldqa_zh_mixup_16k.jsonl"),
        },
        {
            "dataset": "lveval_multifieldqa_zh_mixup",
            "subset": "32k",
            "adapter": "lveval",
            "path": str(root / "lveval/multifieldqa_zh_mixup_32k.jsonl"),
        },
        {
            "dataset": "lveval_multifieldqa_zh_mixup",
            "subset": "64k",
            "adapter": "lveval",
            "path": str(root / "lveval/multifieldqa_zh_mixup_64k.jsonl"),
        },
        {
            "dataset": "lveval_multifieldqa_zh_mixup",
            "subset": "128k",
            "adapter": "lveval",
            "path": str(root / "lveval/multifieldqa_zh_mixup_128k.jsonl"),
        },
    ]
    if EXTRA_LONGBENCH_FILE is not None:
        specs.insert(
            0,
            {
                "dataset": "longbench_multifieldqa_zh",
                "subset": "default",
                "adapter": "longbench",
                "path": EXTRA_LONGBENCH_FILE,
            },
        )
    return specs


def build_sample(
    record: dict[str, Any],
    spec: dict[str, str],
    line_index: int,
) -> dict[str, Any]:
    adapter = spec["adapter"]
    task_metadata = TASK_METADATA.get(adapter)
    if task_metadata is None:
        raise ValueError(f"unknown dataset adapter: {adapter}")
    task_type, task_requirement = task_metadata

    if adapter in {"lveval", "longbench"}:
        context = str(record.get("context", "")).strip()
        question = str(record.get("input", "")).strip()
        answers = normalize_answers(record.get("answers"))
        prompt_template = LV_EVAL_PROMPT if adapter == "lveval" else LONGBENCH_PROMPT
        sample_id = record.get("_id") or (f"{spec['dataset']}:{spec['subset']}:{line_index}")
        source_length = record.get("length")
    elif adapter in {"clong_story", "clong_retrieval"}:
        context = str(record.get("context", "")).strip()
        question = str(record.get("query", "")).strip()
        answers = normalize_answers(record.get("answer"))
        prompt_template = CLONG_STORY_PROMPT if adapter == "clong_story" else CLONG_RETRIEVAL_PROMPT
        sample_id = record.get("id") or (f"{spec['dataset']}:{spec['subset']}:{line_index}")
        source_length = record.get("chatglm_length") or record.get("qwen_length") or record.get("internlm2_length")
    else:
        raise AssertionError(f"unhandled dataset adapter: {adapter}")

    if not context or not question or not answers:
        raise ValueError(
            f"invalid record in {spec['path']} line {line_index + 1}: context, question and answer are required"
        )

    return {
        "sample_key": (f"{spec['dataset']}/{spec['subset']}/{sample_id}"),
        "sample_id": sample_id,
        "sample_index": line_index,
        "dataset": spec["dataset"],
        "subset": spec["subset"],
        "task_type": task_type,
        "task_requirement": task_requirement,
        "source_file": spec["path"],
        "source_length": source_length,
        "question": question,
        "answers": answers,
        "gold_ans": record.get("gold_ans"),
        "prompt": prompt_template.format(
            context=context,
            question=question,
        ),
    }


def iter_file_samples(spec: dict[str, str]):
    path = Path(spec["path"])
    if not path.is_file():
        print(f"[dsa-online-dataset] skip missing source: {path}")
        return

    yielded = 0
    with path.open("r", encoding="utf-8") as source:
        for line_index, line in enumerate(source):
            if MAX_SAMPLES_PER_FILE is not None and yielded >= MAX_SAMPLES_PER_FILE:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"dataset line is not an object: {path}:{line_index + 1}")
            yield build_sample(record, spec, line_index)
            yielded += 1


def bucket_name(prompt_tokens: int) -> str | None:
    for name, lower, upper in LENGTH_BUCKETS:
        if prompt_tokens < lower:
            continue
        if upper is None or prompt_tokens < upper:
            return name
    return None


def server_root_url() -> str:
    base = API_BASE.rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return base


def server_tokenize_prompt(prompt: str) -> tuple[int, int]:
    response = httpx.post(
        f"{server_root_url()}/tokenize",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {
                "enable_thinking": ENABLE_THINKING,
            },
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise RuntimeError(f"POST /tokenize failed with HTTP {response.status_code}: {response.text}") from error
    payload = response.json()
    return int(payload["count"]), int(payload["max_model_len"])


def load_local_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    print(f"[dsa-online-dataset] loading local tokenizer from {model_path}")
    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )


def count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=ENABLE_THINKING,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise RuntimeError(f"local tokenizer returned an unexpected batched result: batch={len(token_ids)}")
        token_ids = token_ids[0]
    return len(token_ids)


def prepare_samples_in_dataset_order(
    tokenizer: Any,
    model_card_max_len: int | None,
) -> tuple[list[dict[str, Any]], int]:
    samples: list[dict[str, Any]] = []
    server_max_model_len: int | None = None
    calibration_complete = False

    for spec in dataset_specs(Path(DATASET_ROOT)):
        loaded_count = 0
        runnable_count = 0
        oversized_count = 0
        bucket_counts: Counter[str] = Counter()
        for sample in iter_file_samples(spec):
            loaded_count += 1
            prompt_tokens = count_prompt_tokens(tokenizer, sample["prompt"])

            if not calibration_complete:
                server_count, server_max_model_len = server_tokenize_prompt(sample["prompt"])
                if prompt_tokens != server_count:
                    raise RuntimeError(
                        "local/server tokenizer count mismatch: "
                        f"local={prompt_tokens}, server={server_count}; "
                        "set LOCAL_TOKENIZER_PATH to the tokenizer used by the service"
                    )
                if model_card_max_len is not None and model_card_max_len != server_max_model_len:
                    raise RuntimeError(
                        "server max_model_len mismatch between /v1/models and "
                        f"/tokenize: {model_card_max_len} != {server_max_model_len}"
                    )
                calibration_complete = True
                print(
                    "[dsa-online-dataset] tokenizer calibration passed "
                    f"prompt_tokens={prompt_tokens} "
                    f"max_model_len={server_max_model_len}"
                )

            if server_max_model_len is None:
                raise RuntimeError("tokenizer calibration did not return max_model_len")
            if prompt_tokens + MAX_TOKENS > server_max_model_len:
                oversized_count += 1
                print(
                    "[dsa-online-dataset] skip oversized "
                    f"dataset={sample['dataset']} subset={sample['subset']} "
                    f"sample={sample['sample_id']} "
                    f"prompt_tokens={prompt_tokens} "
                    f"max_tokens={MAX_TOKENS} "
                    f"max_model_len={server_max_model_len}"
                )
                continue

            length_bucket = bucket_name(prompt_tokens)
            if length_bucket is None:
                raise RuntimeError(f"no length bucket for prompt_tokens={prompt_tokens}")
            sample["preflight_prompt_tokens"] = prompt_tokens
            sample["length_bucket"] = length_bucket
            samples.append(sample)
            runnable_count += 1
            bucket_counts[length_bucket] += 1

        print(
            "[dsa-online-dataset] prepared source "
            f"dataset={spec['dataset']} subset={spec['subset']} "
            f"loaded={loaded_count} runnable={runnable_count} "
            f"oversized={oversized_count} buckets={dict(bucket_counts)}"
        )

    if not calibration_complete or server_max_model_len is None:
        raise RuntimeError("no dataset samples were found; check DATASET_ROOT and source files")
    print(f"[dsa-online-dataset] preparation complete runnable_samples={len(samples)}")
    return samples, server_max_model_len


def normalize_text(text: str) -> str:
    return "".join(character.lower() for character in text if not character.isspace() and character not in PUNCTUATION)


def character_f1(prediction: str, answer: str) -> float:
    prediction_chars = list(normalize_text(prediction))
    answer_chars = list(normalize_text(answer))
    if not prediction_chars or not answer_chars:
        return float(prediction_chars == answer_chars)
    common = Counter(prediction_chars) & Counter(answer_chars)
    matches = sum(common.values())
    if matches == 0:
        return 0.0
    precision = matches / len(prediction_chars)
    recall = matches / len(answer_chars)
    return 2 * precision * recall / (precision + recall)


def score_prediction(
    prediction: str,
    answers: list[str],
) -> tuple[float | None, bool | None]:
    if not answers:
        return None, None
    normalized_prediction = normalize_text(prediction)
    best_f1 = max(character_f1(prediction, answer) for answer in answers)
    contains_answer = any(
        normalize_text(answer) in normalized_prediction for answer in answers if normalize_text(answer)
    )
    return best_f1, contains_answer


def consume_stream(
    stream: Any,
    request_started_at: float,
) -> tuple[str, str, Any, float | None, str | None]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    usage = None
    first_token_latency_ms = None
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason
        delta = choice.delta
        reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)
        if reasoning:
            reasoning_parts.append(reasoning)
        if content:
            if first_token_latency_ms is None:
                first_token_latency_ms = (time.perf_counter() - request_started_at) * 1000
            content_parts.append(content)

    return (
        "".join(content_parts),
        "".join(reasoning_parts),
        usage,
        first_token_latency_ms,
        finish_reason,
    )


def request_one(
    client: OpenAI,
    sample: dict[str, Any],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    last_error: Exception | None = None

    for attempt in range(REQUEST_RETRIES + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": sample["prompt"],
                    }
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": STREAM_RESPONSES,
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": ENABLE_THINKING,
                    },
                },
            }
            if STREAM_RESPONSES:
                request_kwargs["stream_options"] = {"include_usage": True}
            response = client.chat.completions.create(
                **request_kwargs,
            )
            if STREAM_RESPONSES:
                (
                    prediction,
                    reasoning,
                    usage,
                    first_token_latency_ms,
                    finish_reason,
                ) = consume_stream(response, started_at)
            else:
                choice = response.choices[0]
                prediction = choice.message.content or ""
                reasoning = (
                    getattr(choice.message, "reasoning", None)
                    or getattr(choice.message, "reasoning_content", None)
                    or ""
                )
                finish_reason = choice.finish_reason
                usage = response.usage
                first_token_latency_ms = None

            latency_ms = (time.perf_counter() - started_at) * 1000
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
            length_bucket = bucket_name(int(prompt_tokens)) or "unknown" if prompt_tokens is not None else "unknown"
            response_errors = []
            if prompt_tokens is None:
                response_errors.append("response usage.prompt_tokens is missing")
            elif int(prompt_tokens) != sample["preflight_prompt_tokens"]:
                response_errors.append(
                    f"preflight/response prompt token mismatch ({sample['preflight_prompt_tokens']} != {prompt_tokens})"
                )
            if prediction.strip():
                score, contains_answer = score_prediction(
                    prediction,
                    sample["answers"],
                )
            else:
                score, contains_answer = None, None
                response_errors.append(
                    "empty final content "
                    f"(reasoning_chars={len(reasoning)}, "
                    f"completion_tokens={completion_tokens}, "
                    f"finish_reason={finish_reason!r})"
                )
            return {
                "run_label": RUN_LABEL,
                "sample_key": sample["sample_key"],
                "sample_id": sample["sample_id"],
                "sample_index": sample["sample_index"],
                "dataset": sample["dataset"],
                "subset": sample["subset"],
                "task_type": sample["task_type"],
                "task_requirement": sample["task_requirement"],
                "source_file": sample["source_file"],
                "source_length": sample["source_length"],
                "length_bucket": length_bucket,
                "preflight_prompt_tokens": sample["preflight_prompt_tokens"],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason,
                "latency_ms": round(latency_ms, 3),
                "first_token_latency_ms": (
                    None if first_token_latency_ms is None else round(first_token_latency_ms, 3)
                ),
                "question": sample["question"],
                "answers": sample["answers"],
                "gold_ans": sample["gold_ans"],
                "prediction": prediction,
                "reasoning": reasoning,
                "character_f1": score,
                "contains_answer": contains_answer,
                "error": "; ".join(response_errors) or None,
            }
        except Exception as error:
            last_error = error
            if attempt < REQUEST_RETRIES:
                time.sleep(2**attempt)

    latency_ms = (time.perf_counter() - started_at) * 1000
    return {
        "run_label": RUN_LABEL,
        "sample_key": sample["sample_key"],
        "sample_id": sample["sample_id"],
        "sample_index": sample["sample_index"],
        "dataset": sample["dataset"],
        "subset": sample["subset"],
        "task_type": sample["task_type"],
        "task_requirement": sample["task_requirement"],
        "source_file": sample["source_file"],
        "source_length": sample["source_length"],
        "length_bucket": sample["length_bucket"],
        "preflight_prompt_tokens": sample["preflight_prompt_tokens"],
        "prompt_tokens": None,
        "completion_tokens": None,
        "finish_reason": None,
        "latency_ms": round(latency_ms, 3),
        "first_token_latency_ms": None,
        "question": sample["question"],
        "answers": sample["answers"],
        "gold_ans": sample["gold_ans"],
        "prediction": "",
        "reasoning": "",
        "character_f1": None,
        "contains_answer": None,
        "error": repr(last_error),
    }


def result_path() -> Path:
    directory = Path(RESULT_DIR)
    if not directory.is_absolute():
        directory = SCRIPT_DIR / directory
    return directory / f"{RUN_LABEL}.jsonl"


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not RESUME_EXISTING_RESULT or not path.is_file():
        return []
    results = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                result = json.loads(line)
                if "prompt_tokens" not in result:
                    result["prompt_tokens"] = result.get(
                        "usage_prompt_tokens",
                        result.get("selected_prompt_tokens"),
                    )
                result.setdefault(
                    "preflight_prompt_tokens",
                    result.get("selected_prompt_tokens", result["prompt_tokens"]),
                )
                result.setdefault("task_type", "legacy_unknown")
                result.setdefault(
                    "task_requirement",
                    "旧版结果未记录任务说明。",
                )
                results.append(result)
    return results


def format_task_value(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False)
    if len(rendered) <= PRINT_TASK_CHARS:
        return rendered
    return rendered[:PRINT_TASK_CHARS] + "..."


def run_requests(
    samples: list[dict[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    loaded_results = load_existing_results(output_path)
    existing = [
        result for result in loaded_results if result.get("error") is None and str(result.get("prediction", "")).strip()
    ]
    completed_keys = {result["sample_key"] for result in existing}
    pending = [sample for sample in samples if sample["sample_key"] not in completed_keys]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[dsa-online-dataset] requests "
        f"prepared={len(samples)} pending={len(pending)} "
        f"concurrency={REQUEST_CONCURRENCY} stream={STREAM_RESPONSES}"
    )
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    new_results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as output:
        for result in existing:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        for start in range(0, len(pending), REQUEST_CONCURRENCY):
            group = pending[start : start + REQUEST_CONCURRENCY]
            print(f"\n[dsa-online-dataset] concurrent group [{start}, {start + len(group)})")
            for row_offset, item in enumerate(group):
                input_label = "lookup_key" if item["task_type"] == "key_value_retrieval" else "question"
                print(
                    "  "
                    f"request={start + row_offset} "
                    f"row={row_offset} "
                    f"dataset={item['dataset']} "
                    f"subset={item['subset']} "
                    f"sample={item['sample_id']} "
                    f"prompt_tokens={item['preflight_prompt_tokens']} "
                    f"bucket={item['length_bucket']}"
                )
                print(f"    task={item['task_type']} requirement={format_task_value(item['task_requirement'])}")
                print(f"    {input_label}={format_task_value(item['question'])}")
                print(f"    golden_answers={format_task_value(item['answers'])}")
            with ThreadPoolExecutor(max_workers=len(group)) as executor:
                futures = [executor.submit(request_one, client, sample) for sample in group]
                group_results = [future.result() for future in futures]

            print("[dsa-online-dataset] completed group layout")
            for row_offset, result in enumerate(group_results):
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                new_results.append(result)
                print(
                    "  "
                    f"request={start + row_offset} "
                    f"row={row_offset} "
                    f"dataset={result['dataset']} "
                    f"subset={result['subset']} "
                    f"sample={result['sample_id']} "
                    f"bucket={result['length_bucket']} "
                    f"prompt_tokens={result['prompt_tokens']} "
                    f"completion={result['completion_tokens']} "
                    f"finish={result['finish_reason']!r} "
                    f"f1={result['character_f1']} "
                    f"latency_ms={result['latency_ms']} "
                    f"error={result['error']}"
                )
                prediction = result["prediction"]
                if prediction:
                    print(
                        "Output: "
                        f"{prediction[:PRINT_OUTPUT_CHARS]!r}"
                        + (" ..." if len(prediction) > PRINT_OUTPUT_CHARS else "")
                    )
                elif result["reasoning"]:
                    reasoning = result["reasoning"]
                    print(
                        "Reasoning-only: "
                        f"{reasoning[:PRINT_OUTPUT_CHARS]!r}" + (" ..." if len(reasoning) > PRINT_OUTPUT_CHARS else "")
                    )
                else:
                    print("Output: ''")

    return existing + new_results


def aggregate_rows(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)

    summary = {}
    for name, group in groups.items():
        successful = [row for row in group if row.get("error") is None]
        scores = [float(row["character_f1"]) for row in successful if row.get("character_f1") is not None]
        contains = [bool(row["contains_answer"]) for row in successful if row.get("contains_answer") is not None]
        latencies = [float(row["latency_ms"]) for row in successful]
        token_lengths = [int(row["prompt_tokens"]) for row in successful if row.get("prompt_tokens") is not None]
        summary[name] = {
            "count": len(group),
            "successful": len(successful),
            "errors": len(group) - len(successful),
            "prompt_tokens_min": (min(token_lengths) if token_lengths else None),
            "prompt_tokens_max": (max(token_lengths) if token_lengths else None),
            "character_f1_avg": (round(sum(scores) / len(scores), 6) if scores else None),
            "contains_answer_rate": (round(sum(contains) / len(contains), 6) if contains else None),
            "latency_ms_avg": (round(sum(latencies) / len(latencies), 3) if latencies else None),
        }
    return summary


def write_summary(
    output_path: Path,
    rows: list[dict[str, Any]],
    server_max_model_len: int | None,
    prepared_samples: list[dict[str, Any]],
) -> Path:
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "run_label": RUN_LABEL,
        "api_base": API_BASE,
        "model": MODEL_NAME,
        "dataset_root": DATASET_ROOT,
        "server_max_model_len": server_max_model_len,
        "max_tokens": MAX_TOKENS,
        "enable_thinking": ENABLE_THINKING,
        "max_samples_per_file": MAX_SAMPLES_PER_FILE,
        "request_concurrency": REQUEST_CONCURRENCY,
        "stream_responses": STREAM_RESPONSES,
        "prepared_samples": len(prepared_samples),
        "recorded_results": len(rows),
        "by_length_bucket": aggregate_rows(rows, "length_bucket"),
        "by_dataset": aggregate_rows(rows, "dataset"),
        "by_task_type": aggregate_rows(rows, "task_type"),
        "metric_notice": (
            "character_f1 and contains_answer are lightweight smoke "
            "metrics, not the official LV-Eval/CLongEval leaderboard score"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    validate_user_config()
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    models = client.models.list().data
    served_model = next((model for model in models if model.id == MODEL_NAME), None)
    if served_model is None:
        available_models = [model.id for model in models]
        raise RuntimeError(f"model {MODEL_NAME!r} is not served; available={available_models}")
    server_max_model_len = getattr(served_model, "max_model_len", None)
    tokenizer_path = LOCAL_TOKENIZER_PATH or getattr(served_model, "root", None)
    if not tokenizer_path:
        raise RuntimeError("the service did not expose its model/tokenizer path; set LOCAL_TOKENIZER_PATH explicitly")

    print(f"[dsa-online-dataset] connected api={API_BASE} model={MODEL_NAME} max_model_len={server_max_model_len}")
    tokenizer = load_local_tokenizer(str(tokenizer_path))
    samples, server_max_model_len = prepare_samples_in_dataset_order(
        tokenizer,
        server_max_model_len,
    )
    if not samples:
        raise RuntimeError("no runnable dataset samples")

    output_path = result_path()
    rows = run_requests(samples, output_path)
    summary_path = write_summary(
        output_path,
        rows,
        server_max_model_len,
        samples,
    )
    print(f"\n[dsa-online-dataset] results: {output_path}")
    print(f"[dsa-online-dataset] summary: {summary_path}")


if __name__ == "__main__":
    main()
