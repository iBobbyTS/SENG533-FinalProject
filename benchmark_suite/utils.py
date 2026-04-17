from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[object]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_list(items: Sequence[object], limit: int | None, seed: int) -> list[object]:
    if limit is None or limit >= len(items):
        return list(items)
    rng = random.Random(seed)
    indexes = list(range(len(items)))
    rng.shuffle(indexes)
    selected = sorted(indexes[:limit])
    return [items[index] for index in selected]


def grouped_sample(items: Sequence[object], key_fn, per_group: int, seed: int) -> list[object]:
    grouped: dict[str, list[object]] = {}
    for item in items:
        grouped.setdefault(str(key_fn(item)), []).append(item)
    rng = random.Random(seed)
    output: list[object] = []
    for key in sorted(grouped):
        group = list(grouped[key])
        rng.shuffle(group)
        output.extend(group[:per_group])
    return output


def proportional_sample(items: Sequence[object], key_fn, limit: int, seed: int) -> list[object]:
    grouped: dict[str, list[object]] = {}
    for item in items:
        grouped.setdefault(str(key_fn(item)), []).append(item)
    total = len(items)
    rng = random.Random(seed)
    selected: list[object] = []
    remaining = limit
    keys = sorted(grouped.keys())
    for position, key in enumerate(keys):
        group = list(grouped[key])
        rng.shuffle(group)
        if position == len(keys) - 1:
            take = min(len(group), remaining)
        else:
            take = round(limit * len(group) / total)
            take = min(len(group), take, remaining)
        selected.extend(group[:take])
        remaining -= take
    if remaining > 0:
        leftovers = [item for key in keys for item in grouped[key] if item not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])
    return selected


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def detect_repetition(text: str) -> bool:
    cleaned = normalize_space(text)
    if len(cleaned) < 80:
        return False
    tail = cleaned[-40:]
    return cleaned.count(tail) >= 3


def apply_model_prompt_suffix(model: str, prompt: str) -> str:
    return prompt


def apply_model_max_tokens(model: str, default_tokens: int, qwen35_a3b_cap: int | None = None) -> int:
    if model == "qwen/qwen3.5-35b-a3b" and qwen35_a3b_cap is not None:
        return min(default_tokens, qwen35_a3b_cap)
    return default_tokens


def markdown_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def mean_numeric(values: Iterable[Any]) -> float | None:
    numeric_values: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            numeric_values.append(float(value))
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def approx_prefill_tps(prompt_tokens: Any, ttft_seconds: Any) -> float | None:
    if not isinstance(prompt_tokens, (int, float)) or not isinstance(ttft_seconds, (int, float)):
        return None
    if ttft_seconds <= 0:
        return None
    return float(prompt_tokens) / float(ttft_seconds)


def summarize_prediction_performance(predictions: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    decode_tps_values = []
    prefill_tps_values = []
    for row in predictions:
        usage = row.get("usage", {})
        if not isinstance(usage, dict):
            continue
        decode_tps_values.append(usage.get("tokens_per_second"))
        prefill_tps_values.append(approx_prefill_tps(usage.get("prompt_tokens"), usage.get("time_to_first_token_seconds")))
    return {
        "mean_decode_tps": mean_numeric(decode_tps_values),
        "mean_approx_prefill_tps": mean_numeric(prefill_tps_values),
    }
