from __future__ import annotations

import base64
import json
import re
import urllib3
from pathlib import Path

import requests

from ..config import OFFICIAL_RESOURCES, PROFILES, VQA_DATA_URLS
from ..types import RunConfig
from ..utils import (
    apply_model_prompt_suffix,
    detect_repetition,
    ensure_dir,
    grouped_sample,
    proportional_sample,
    sample_list,
    write_json,
    write_jsonl,
)

ARTICLES = {"a", "an", "the"}
MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
PUNCT_RE = re.compile(r"[^\w\s]")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_path(data_root: Path, image_id: int) -> Path:
    return data_root / "images" / f"COCO_val2014_{image_id:012d}.jpg"


def _download_image(image_path: Path, image_id: int) -> None:
    if image_path.exists():
        return
    ensure_dir(image_path.parent)
    url = f"{VQA_DATA_URLS['image_base']}/COCO_val2014_{image_id:012d}.jpg"
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(url, timeout=120, verify=False)
    response.raise_for_status()
    image_path.write_bytes(response.content)


def _image_data_url(image_path: Path) -> str:
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = PUNCT_RE.sub(" ", text)
    tokens = []
    for token in text.split():
        mapped = MANUAL_MAP.get(token, token)
        if mapped not in ARTICLES:
            tokens.append(mapped)
    return " ".join(tokens)


def _score_answer(prediction: str, answers: list[str]) -> float:
    normalized_pred = _normalize_answer(prediction)
    matches = sum(1 for answer in answers if _normalize_answer(answer) == normalized_pred)
    return min(1.0, matches / 3.0)


def _pick_samples(items: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 3:
        return grouped_sample(items, lambda row: row["answer_type"], 1, seed)[:limit]
    if limit <= 60:
        per_group = max(1, limit // 3)
        return grouped_sample(items, lambda row: row["answer_type"], per_group, seed)[:limit]
    return proportional_sample(items, lambda row: row["answer_type"], limit, seed)


def run(config: RunConfig, client) -> dict:
    benchmark_dir = ensure_dir(config.output_root / config.benchmark)
    profile = PROFILES[config.profile]["vqa"]
    data_dir = config.data_root / "vqa_v2"
    questions_path = data_dir / "v2_OpenEnded_mscoco_val2014_questions.json"
    annotations_path = data_dir / "v2_mscoco_val2014_annotations.json"
    if not questions_path.exists() or not annotations_path.exists():
        raise FileNotFoundError("VQA metadata not found. Run scripts/setup_vqa.sh first.")

    questions = _load_json(questions_path)["questions"]
    annotations = {
        item["question_id"]: item
        for item in _load_json(annotations_path)["annotations"]
    }
    merged = []
    for question in questions:
        annotation = annotations.get(question["question_id"])
        if not annotation:
            continue
        merged.append(
            {
                "question_id": question["question_id"],
                "image_id": question["image_id"],
                "question": question["question"],
                "answer_type": annotation["answer_type"],
                "answers": [entry["answer"] for entry in annotation["answers"]],
            }
        )

    sampled = _pick_samples(merged, profile["limit"], config.seed)
    predictions = []
    score_total = 0.0
    breakdown = {"yes/no": [], "number": [], "other": []}
    notes = [
        "Uses the official VQA v2 validation questions and annotations, with images downloaded on demand.",
        "Uses the open-ended VQA accuracy formula, with a Python 3 answer normalization port."
    ]

    for index, row in enumerate(sampled):
        image_path = _image_path(data_dir, row["image_id"])
        _download_image(image_path, row["image_id"])
        prompt_text = apply_model_prompt_suffix(
            config.model,
            (
                "Answer the visual question using a short phrase, word, or number only.\n"
                f"Question: {row['question']}"
            ),
        )
        content = [
            {
                "type": "text",
                "text": prompt_text,
            },
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(image_path)},
            },
        ]
        response = client.chat_completion(
            [{"role": "user", "content": content}],
            temperature=0.0,
            seed=config.seed,
            stream=True,
            stream_idle_timeout_seconds=300,
        )
        prediction = response["content"].strip()
        score = _score_answer(prediction, row["answers"])
        score_total += score
        breakdown[row["answer_type"]].append(score)
        predictions.append(
            {
                "index": index,
                "question_id": row["question_id"],
                "image_id": row["image_id"],
                "answer_type": row["answer_type"],
                "prompt": f"{prompt_text}\nImage path: {image_path}",
                "score": score,
                "finish_reason": response["finish_reason"],
                "loop_risk": detect_repetition(prediction),
                "usage": response["usage"],
                "response_text": prediction,
                "answers": row["answers"],
                "stream_debug": response.get("stream_debug"),
            }
        )

    breakdown_summary = {
        key: (sum(values) / len(values) if values else None)
        for key, values in breakdown.items()
    }
    write_jsonl(benchmark_dir / "predictions.jsonl", predictions)
    write_json(benchmark_dir / "breakdown.json", breakdown_summary)
    return {
        "benchmark": config.benchmark,
        "profile": config.profile,
        "status": "completed",
        "model": config.model,
        "base_url": config.base_url,
        "sample_count": len(sampled),
        "metric": OFFICIAL_RESOURCES["vqa"]["metric"],
        "score": score_total / len(sampled) if sampled else None,
        "correct_count": None,
        "notes": notes,
        "repo": OFFICIAL_RESOURCES["vqa"]["repo"],
        "dataset": OFFICIAL_RESOURCES["vqa"]["dataset"],
        "breakdown": breakdown_summary,
    }
