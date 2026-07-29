#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""对 ``qa_dataset_test.py`` 生成的 LongBench JSONL 进行评分。

示例：

    python examples/dsa_demo/eval_dataset_acc_score.py \
        --result-path examples/dsa_demo/LongBenchResult/glm51_dsa
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one LongBench JSONL file or a directory of files.")
    parser.add_argument(
        "--result-path",
        "-p",
        type=Path,
        required=True,
        help="A LongBench JSONL file or a directory containing JSONL files.",
    )
    parser.add_argument(
        "--count-limit",
        type=int,
        default=9999,
        help="Maximum number of samples read from each file.",
    )
    parser.add_argument(
        "--longbench-e",
        "-e",
        action="store_true",
        help="Report 0-4k, 4-8k and 8k+ length buckets.",
    )
    args = parser.parse_args()
    if args.count_limit <= 0:
        parser.error("--count-limit must be positive")
    return args


def require_optional_dependencies() -> tuple[Any, Any, Any]:
    try:
        import jieba
        from fuzzywuzzy import fuzz
        from rouge import Rouge
    except ImportError as exc:
        raise RuntimeError(
            "LongBench scoring requires jieba, fuzzywuzzy and rouge. "
            "Install the evaluator dependencies in the current environment."
        ) from exc
    return jieba, fuzz, Rouge


JIEBA: Any = None
FUZZ: Any = None
ROUGE_CLASS: Any = None


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(character for character in text if character not in set(string.punctuation))
    return " ".join(text.split())


def normalize_zh_answer(text: str) -> str:
    chinese_punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀"
        "｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰"
        "〾〿–—‘’‛“”„‟…‧﹏."
    )
    punctuation = set(string.punctuation + chinese_punctuation)
    return "".join(character for character in text.lower() if character not in punctuation and not character.isspace())


def f1_score(
    prediction_tokens: Sequence[str],
    ground_truth_tokens: Sequence[str],
) -> float:
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(
    prediction: str,
    ground_truth: str,
    **_: Any,
) -> float:
    return f1_score(
        normalize_answer(prediction).split(),
        normalize_answer(ground_truth).split(),
    )


def qa_f1_zh_score(
    prediction: str,
    ground_truth: str,
    **_: Any,
) -> float:
    prediction_tokens = [normalize_zh_answer(token) for token in JIEBA.cut(prediction, cut_all=False)]
    ground_truth_tokens = [normalize_zh_answer(token) for token in JIEBA.cut(ground_truth, cut_all=False)]
    return f1_score(
        [token for token in prediction_tokens if token],
        [token for token in ground_truth_tokens if token],
    )


def count_score(
    prediction: str,
    ground_truth: str,
    **_: Any,
) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    matches = sum(number == str(ground_truth) for number in numbers)
    return matches / len(numbers)


def retrieval_score(
    prediction: str,
    ground_truth: str,
    **_: Any,
) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    return count_score(prediction, matches[0])


def retrieval_zh_score(
    prediction: str,
    ground_truth: str,
    **_: Any,
) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    return count_score(prediction, matches[0])


def classification_score(
    prediction: str,
    ground_truth: str,
    *,
    all_classes: Sequence[str],
    **_: Any,
) -> float:
    matches = [class_name for class_name in all_classes if class_name in prediction]
    matches = [match for match in matches if match == ground_truth or match not in ground_truth]
    return 1.0 / len(matches) if ground_truth in matches else 0.0


def rouge_score(
    prediction: str,
    ground_truth: str,
    **_: Any,
) -> float:
    try:
        result = ROUGE_CLASS().get_scores(
            [prediction],
            [ground_truth],
            avg=True,
        )
    except (ValueError, IndexError, ZeroDivisionError):
        return 0.0
    return float(result["rouge-l"]["f"])


def rouge_zh_score(
    prediction: str,
    ground_truth: str,
    **_: Any,
) -> float:
    prediction = " ".join(JIEBA.cut(prediction, cut_all=False))
    ground_truth = " ".join(JIEBA.cut(ground_truth, cut_all=False))
    return rouge_score(prediction, ground_truth)


def code_sim_score(
    prediction: str,
    ground_truth: str,
    **_: Any,
) -> float:
    candidate = ""
    for line in prediction.lstrip("\n").splitlines():
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    return float(FUZZ.ratio(candidate, ground_truth) / 100)


Metric = Callable[..., float]

DATASET_TO_METRIC: dict[str, Metric] = {
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

FIRST_LINE_DATASETS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
}


def load_result_rows(
    path: Path,
    count_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if len(rows) >= count_limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"pred", "answers", "all_classes"} - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} misses fields {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"result file contains no samples: {path}")
    return rows


def score_one(
    dataset_name: str,
    row: dict[str, Any],
) -> float:
    metric = DATASET_TO_METRIC[dataset_name]
    prediction = str(row["pred"])
    # 与 LongBench 官方评测保持一致。QA/retrieval 任务必须保留完整输出，
    # 否则“首行正确、后续乱码或重复 prompt”的 DSA 精度故障会被掩盖。
    if dataset_name in FIRST_LINE_DATASETS:
        prediction = prediction.lstrip("\n").split("\n", maxsplit=1)[0]
    answers = row["answers"]
    if not isinstance(answers, list):
        answers = [answers]
    return max(
        (
            metric(
                prediction,
                str(answer),
                all_classes=row.get("all_classes", []),
            )
            for answer in answers
        ),
        default=0.0,
    )


def mean_percent(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(100 * sum(values) / len(values), 2)


def score_file(
    path: Path,
    *,
    count_limit: int,
    longbench_e: bool,
) -> dict[str, Any]:
    dataset_name = path.stem
    if dataset_name not in DATASET_TO_METRIC:
        raise ValueError(f"unsupported LongBench dataset {dataset_name!r}; supported={sorted(DATASET_TO_METRIC)}")
    rows = load_result_rows(path, count_limit)
    scores = [score_one(dataset_name, row) for row in rows]
    summary: dict[str, Any] = {
        "file": path.name,
        "dataset": dataset_name,
        "samples": len(rows),
        "mean_output_chars": round(
            sum(len(str(row["pred"])) for row in rows) / len(rows),
            2,
        ),
        "score": mean_percent(scores),
    }
    if longbench_e:
        buckets: dict[str, list[float]] = {
            "0-4k": [],
            "4-8k": [],
            "8k+": [],
        }
        for row, score in zip(rows, scores):
            length = int(row.get("length", 0))
            key = "0-4k" if length < 4000 else "4-8k" if length < 8000 else "8k+"
            buckets[key].append(score)
        summary["length_scores"] = {key: mean_percent(values) for key, values in buckets.items()}
    return summary


def result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if files:
            return files
        raise ValueError(f"directory contains no JSONL files: {path}")
    raise FileNotFoundError(f"result path not found: {path}")


def main() -> None:
    global JIEBA, FUZZ, ROUGE_CLASS
    args = parse_args()
    JIEBA, FUZZ, ROUGE_CLASS = require_optional_dependencies()
    for path in result_files(args.result_path):
        summary = score_file(
            path,
            count_limit=args.count_limit,
            longbench_e=args.longbench_e,
        )
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
