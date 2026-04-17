from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ..config import EXTERNAL_REPOS, OFFICIAL_RESOURCES, PROFILES
from ..types import RunConfig
from ..utils import apply_model_prompt_suffix, detect_repetition, ensure_dir, sample_list, write_jsonl


def _load_rows(test_path: Path) -> list[dict]:
    rows = []
    with test_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _normalize_number(text: str) -> str | None:
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if number == number.to_integral():
        return str(number.quantize(Decimal("1")))
    return format(number.normalize(), "f").rstrip("0").rstrip(".")


def _extract_gold(answer: str) -> str | None:
    match = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", answer)
    return _normalize_number(match.group(1)) if match else None


def _extract_predicted(text: str) -> str | None:
    exact = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", text)
    if exact:
        return _normalize_number(exact.group(1))
    matches = re.findall(r"[-+]?[\d,]+(?:\.\d+)?", text)
    return _normalize_number(matches[-1]) if matches else None


def run(config: RunConfig, client) -> dict:
    benchmark_dir = ensure_dir(config.output_root / config.benchmark)
    profile = PROFILES[config.profile]["gsm8k"]
    repo_dir = config.external_root / EXTERNAL_REPOS["gsm8k"]["dir"]
    test_path = repo_dir / "grade_school_math" / "data" / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"GSM8K test file not found: {test_path}")
    sampled = sample_list(_load_rows(test_path), profile["limit"], config.seed)
    predictions = []
    correct = 0
    notes = [
        "Uses an adapted prompt and asks for a concise final numeric answer.",
        "If the model does not output ####, the scorer falls back to extracting the last number."
    ]

    for index, row in enumerate(sampled):
        prompt = apply_model_prompt_suffix(
            config.model,
            (
            "Solve the following grade school math problem.\n"
            "Keep the response concise and end with the exact format '#### <number>'.\n\n"
            f"Question: {row['question']}\n\nAnswer:"
            ),
        )
        response = client.chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            seed=config.seed,
            stream=True,
            stream_idle_timeout_seconds=300,
        )
        gold = _extract_gold(row["answer"])
        prediction = _extract_predicted(response["content"])
        is_correct = gold is not None and prediction == gold
        correct += int(is_correct)
        predictions.append(
            {
                "index": index,
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
    return {
        "benchmark": config.benchmark,
        "profile": config.profile,
        "status": "completed",
        "model": config.model,
        "base_url": config.base_url,
        "sample_count": len(sampled),
        "metric": OFFICIAL_RESOURCES["gsm8k"]["metric"],
        "score": correct / len(sampled) if sampled else None,
        "correct_count": correct,
        "notes": notes,
        "repo": OFFICIAL_RESOURCES["gsm8k"]["repo"],
        "dataset": OFFICIAL_RESOURCES["gsm8k"]["dataset"],
    }
