from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import jieba
from rouge_score import rouge_scorer

from ..config import (
    EXTERNAL_REPOS,
    LONG_BENCH_DATASET_TO_METRIC,
    LONG_BENCH_FALLBACK_PROMPTS,
    LONG_BENCH_MAX_NEW_TOKENS,
    LONG_BENCH_TASKS,
    OFFICIAL_RESOURCES,
    PROFILES,
)
from ..types import RunConfig
from ..utils import apply_model_prompt_suffix, detect_repetition, ensure_dir, normalize_space, write_json, write_jsonl

ARTICLES = {"a", "an", "the"}
PUNCT_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff\s]")
ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _normalize_en(text: str) -> str:
    text = PUNCT_RE.sub(" ", text.lower())
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens)


def _normalize_zh(text: str) -> str:
    return normalize_space(PUNCT_RE.sub(" ", text.lower()))


def _f1(prediction: str, answer: str, zh: bool = False) -> float:
    if zh:
        pred_tokens = [token for token in jieba.lcut(_normalize_zh(prediction)) if token.strip()]
        gold_tokens = [token for token in jieba.lcut(_normalize_zh(answer)) if token.strip()]
    else:
        pred_tokens = _normalize_en(prediction).split()
        gold_tokens = _normalize_en(answer).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = 0
    gold_counts = defaultdict(int)
    for token in gold_tokens:
        gold_counts[token] += 1
    for token in pred_tokens:
        if gold_counts[token] > 0:
            common += 1
            gold_counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _rouge_l(prediction: str, answer: str) -> float:
    return ROUGE.score(answer, prediction)["rougeL"].fmeasure


def _classification(prediction: str, answers: list[str]) -> float:
    normalized_pred = _normalize_en(prediction)
    for answer in answers:
        normalized_answer = _normalize_en(answer)
        if normalized_pred == normalized_answer or normalized_answer in normalized_pred:
            return 1.0
    return 0.0


def _edit_similarity(prediction: str, answer: str) -> float:
    return SequenceMatcher(None, prediction.strip(), answer.strip()).ratio()


def _score(metric: str, prediction: str, answers: list[str]) -> float:
    if metric == "qa_f1":
        return max(_f1(prediction, answer) for answer in answers)
    if metric == "qa_f1_zh":
        return max(_f1(prediction, answer, zh=True) for answer in answers)
    if metric == "rouge_l":
        return max(_rouge_l(prediction, answer) for answer in answers)
    if metric == "rouge_l_zh":
        return max(_rouge_l(" ".join(jieba.lcut(prediction)), " ".join(jieba.lcut(answer))) for answer in answers)
    if metric == "classification":
        return _classification(prediction, answers)
    if metric == "edit_similarity":
        return max(_edit_similarity(prediction, answer) for answer in answers)
    raise ValueError(f"Unsupported LongBench metric: {metric}")


def _load_prompt_map(config: RunConfig) -> dict[str, str]:
    prompt_path = config.external_root / EXTERNAL_REPOS["longbench"]["dir"] / "LongBench" / "config" / "dataset2prompt.json"
    if prompt_path.exists():
        return json.loads(prompt_path.read_text(encoding="utf-8"))
    return {}


def _load_rows(data_root: Path, task: str) -> list[dict]:
    for candidate in data_root.rglob(f"{task}.jsonl"):
        rows = []
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            return rows
    raise FileNotFoundError(f"LongBench data file for {task} not found under {data_root}")


def _build_prompt(prompt_map: dict[str, str], task: str, row: dict) -> str:
    template = prompt_map.get(task)
    if template:
        try:
            return template.format(**row)
        except KeyError:
            pass
    fallback = LONG_BENCH_FALLBACK_PROMPTS.get(task, LONG_BENCH_FALLBACK_PROMPTS["default"])
    return fallback.format(**row)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(config: RunConfig, client) -> dict:
    benchmark_dir = ensure_dir(config.output_root / config.benchmark)
    profile = PROFILES[config.profile]["longbench"]
    prompt_map = _load_prompt_map(config)
    data_root = config.data_root / "longbench"
    progress_path = benchmark_dir / "progress.json"
    predictions = []
    per_task_scores: dict[str, float] = {}
    selected_tasks = LONG_BENCH_TASKS[: profile.get("task_limit", len(LONG_BENCH_TASKS))]
    notes = [
        "Uses the official LongBench dataset and task split, while prediction and scoring run through the local adapted runner.",
        "If the model repeats or gets truncated in long-context settings, the result records `loop_risk` and `finish_reason`."
    ]

    for task_index, task in enumerate(selected_tasks):
        rows = _load_rows(data_root, task)
        sampled = rows[: profile["per_task"]]
        task_scores = []
        for row_index, row in enumerate(sampled):
            prompt = apply_model_prompt_suffix(config.model, _build_prompt(prompt_map, task, row))
            write_json(
                progress_path,
                {
                    "status": "requesting",
                    "updated_at": _iso_now(),
                    "task": task,
                    "task_index": task_index,
                    "row_index": row_index,
                    "completed_predictions": len(predictions),
                    "prompt_chars": len(prompt),
                },
            )
            try:
                response = client.chat_completion(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    seed=config.seed,
                    stream=True,
                    stream_idle_timeout_seconds=300,
                )
            except Exception as exc:  # noqa: BLE001
                write_jsonl(benchmark_dir / "predictions.jsonl", predictions)
                write_json(
                    progress_path,
                    {
                        "status": "failed",
                        "updated_at": _iso_now(),
                        "task": task,
                        "task_index": task_index,
                        "row_index": row_index,
                        "completed_predictions": len(predictions),
                        "prompt_chars": len(prompt),
                        "error": str(exc),
                    },
                )
                raise
            prediction = response["content"].strip()
            metric = LONG_BENCH_DATASET_TO_METRIC[task]
            answers = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]
            score = _score(metric, prediction, list(answers))
            task_scores.append(score)
            predictions.append(
                {
                    "task": task,
                    "task_index": task_index,
                    "row_index": row_index,
                    "id": row.get("_id"),
                    "prompt": prompt,
                    "metric": metric,
                    "score": score,
                    "answers": list(answers),
                    "finish_reason": response["finish_reason"],
                    "loop_risk": detect_repetition(prediction),
                    "usage": response["usage"],
                    "response_text": prediction,
                    "stream_debug": response.get("stream_debug"),
                }
            )
            write_jsonl(benchmark_dir / "predictions.jsonl", predictions)
            write_json(
                progress_path,
                {
                    "status": "received",
                    "updated_at": _iso_now(),
                    "task": task,
                    "task_index": task_index,
                    "row_index": row_index,
                    "completed_predictions": len(predictions),
                    "prompt_chars": len(prompt),
                    "usage": response["usage"],
                    "stream_debug": response.get("stream_debug"),
                },
            )
        per_task_scores[task] = sum(task_scores) / len(task_scores) if task_scores else 0.0

    write_jsonl(benchmark_dir / "predictions.jsonl", predictions)
    write_json(benchmark_dir / "per_task.json", per_task_scores)
    write_json(
        progress_path,
        {
            "status": "completed",
            "updated_at": _iso_now(),
            "completed_predictions": len(predictions),
            "per_task": per_task_scores,
        },
    )
    overall = sum(per_task_scores.values()) / len(per_task_scores) if per_task_scores else None
    return {
        "benchmark": config.benchmark,
        "profile": config.profile,
        "status": "completed",
        "model": config.model,
        "base_url": config.base_url,
        "sample_count": len(predictions),
        "metric": OFFICIAL_RESOURCES["longbench"]["metric"],
        "score": overall,
        "correct_count": None,
        "notes": notes,
        "repo": OFFICIAL_RESOURCES["longbench"]["repo"],
        "dataset": OFFICIAL_RESOURCES["longbench"]["dataset"],
        "per_task": per_task_scores,
    }
