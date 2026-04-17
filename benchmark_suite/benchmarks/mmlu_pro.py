from __future__ import annotations

import re
from collections import defaultdict

from datasets import load_dataset

from ..config import OFFICIAL_RESOURCES, PROFILES
from ..types import RunConfig
from ..utils import apply_model_prompt_suffix, detect_repetition, ensure_dir, grouped_sample, write_json, write_jsonl

CHOICE_LETTERS = "ABCDEFGHIJ"


def _answer_letter(record: dict) -> str:
    answer_index = record.get("answer_index")
    if isinstance(answer_index, int) and 0 <= answer_index < len(CHOICE_LETTERS):
        return CHOICE_LETTERS[answer_index]
    answer = str(record.get("answer", "")).strip().upper()
    match = re.search(r"[A-J]", answer)
    return match.group(0) if match else ""


def _format_question(record: dict) -> str:
    options = record["options"]
    lines = [record["question"].strip(), ""]
    for idx, option in enumerate(options):
        lines.append(f"({CHOICE_LETTERS[idx]}) {option}")
    return "\n".join(lines)


def _build_examples(validation_rows: list[dict], category: str, cot_shots: int) -> str:
    examples = []
    count = 0
    for row in validation_rows:
        if row["category"] != category:
            continue
        letter = _answer_letter(row)
        if not letter:
            continue
        answer_block = f"The answer is ({letter})."
        examples.append(f"Question:\n{_format_question(row)}\n\nAnswer:\n{answer_block}")
        count += 1
        if count >= cot_shots:
            break
    return "\n\n".join(examples)


def _extract_prediction(text: str) -> str:
    patterns = [
        r"answer is\s*\(?([A-J])\)?",
        r"the answer is\s*\(?([A-J])\)?",
        r"\(([A-J])\)",
    ]
    upper = text.upper()
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    trailing = re.findall(r"\b([A-J])\b", upper)
    return trailing[-1] if trailing else ""


def run(config: RunConfig, client) -> dict:
    profile = PROFILES[config.profile]["mmlu_pro"]
    benchmark_dir = ensure_dir(config.output_root / config.benchmark)
    dataset_cache = str(config.cache_root / "hf")
    test_rows = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test", cache_dir=dataset_cache))
    validation_rows = list(load_dataset("TIGER-Lab/MMLU-Pro", split="validation", cache_dir=dataset_cache))
    sampled = grouped_sample(test_rows, lambda row: row["category"], profile["per_category"], config.seed)
    category_limit = profile.get("category_limit")
    if category_limit is not None:
        allowed = sorted({row["category"] for row in sampled})[:category_limit]
        sampled = [row for row in sampled if row["category"] in allowed]
    predictions = []
    correct = 0
    category_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    notes = [
        "Initial validation may produce low scores; if the model loops or gets truncated, the raw behavior is preserved.",
        f"Uses a {profile['cot_shots']}-shot adapted prompt and requires the final output to use the option format directly.",
    ]

    for index, row in enumerate(sampled):
        category = row["category"]
        examples = _build_examples(validation_rows, category, profile["cot_shots"])
        prompt_parts = [
            f"The following are multiple-choice questions about {category}.",
            "Select the best option and end with exactly 'The answer is (X)'. Do not include a long chain of thought.",
        ]
        if examples:
            prompt_parts.append(examples)
        prompt_parts.append(f"Question:\n{_format_question(row)}\n\nAnswer:")
        prompt = apply_model_prompt_suffix(config.model, "\n\n".join(prompt_parts))
        response = client.chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            seed=config.seed,
            stream=True,
            stream_idle_timeout_seconds=300,
        )
        prediction = _extract_prediction(response["content"])
        gold = _answer_letter(row)
        is_correct = prediction == gold
        correct += int(is_correct)
        category_counts[category]["correct"] += int(is_correct)
        category_counts[category]["total"] += 1
        predictions.append(
            {
                "index": index,
                "category": category,
                "question_id": row.get("question_id"),
                "prompt": prompt,
                "gold": gold,
                "prediction": prediction,
                "correct": is_correct,
                "finish_reason": response["finish_reason"],
                "loop_risk": detect_repetition(response["content"]),
                "usage": response["usage"],
                "response_text": response["content"],
                "stream_debug": response.get("stream_debug"),
            }
        )

    write_jsonl(benchmark_dir / "predictions.jsonl", predictions)
    per_category = {
        category: counts["correct"] / counts["total"] if counts["total"] else None
        for category, counts in sorted(category_counts.items())
    }
    summary = {
        "benchmark": config.benchmark,
        "profile": config.profile,
        "status": "completed",
        "model": config.model,
        "base_url": config.base_url,
        "sample_count": len(sampled),
        "metric": OFFICIAL_RESOURCES["mmlu_pro"]["metric"],
        "score": correct / len(sampled) if sampled else None,
        "correct_count": correct,
        "notes": notes,
        "repo": OFFICIAL_RESOURCES["mmlu_pro"]["repo"],
        "dataset": OFFICIAL_RESOURCES["mmlu_pro"]["dataset"],
        "per_category": per_category,
    }
    write_json(benchmark_dir / "details.json", {"per_category": per_category})
    return summary
