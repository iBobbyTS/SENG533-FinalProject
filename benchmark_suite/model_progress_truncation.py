from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote


LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+summary\.json)\)")


@dataclass(frozen=True)
class ModelProgressRow:
    model_name: str
    params: str
    quantization: str
    notes: str
    latest_run_label: str
    summary_path: Path
    status: str
    completed: int
    failed: int
    failure_notes: str
    current_registry_quantization: str
    current_selected_variant: str


def split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def parse_model_progress(path: Path) -> list[ModelProgressRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    rows: list[ModelProgressRow] = []

    for line in lines:
        cells = split_markdown_row(line)
        if not cells:
            continue

        if "Latest Run" in cells and "Model Name" in cells:
            header = cells
            continue

        if header is None or set("".join(cells)) == {"-", ":"}:
            continue

        if len(cells) != len(header):
            continue

        row_map = dict(zip(header, cells, strict=True))
        latest_run = row_map.get("Latest Run", "")
        match = LINK_PATTERN.search(latest_run)
        if not match:
            continue

        summary_path = Path(unquote(match.group(2)))
        if not summary_path.is_absolute():
            summary_path = (path.parent / summary_path).resolve()

        rows.append(
            ModelProgressRow(
                model_name=row_map.get("Model Name", ""),
                params=row_map.get("Params", ""),
                quantization=row_map.get("Quantization", ""),
                notes=row_map.get("Notes", ""),
                latest_run_label=match.group(1),
                summary_path=summary_path,
                status=row_map.get("Status", ""),
                completed=int(row_map.get("Completed", "0") or 0),
                failed=int(row_map.get("Failed", "0") or 0),
                failure_notes=row_map.get("Failure Notes", ""),
                current_registry_quantization=row_map.get("Current Registry Quantization", ""),
                current_selected_variant=row_map.get("Current Selected Variant", ""),
            )
        )

    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_entries(summary: dict[str, Any]) -> list[dict[str, Any]]:
    benchmarks = summary.get("benchmarks", [])
    if isinstance(benchmarks, list):
        return benchmarks
    if isinstance(benchmarks, dict):
        out: list[dict[str, Any]] = []
        for benchmark_name, benchmark_summary in benchmarks.items():
            merged = dict(benchmark_summary)
            merged.setdefault("benchmark", benchmark_name)
            out.append(merged)
        return out
    return []


def is_token_truncated(record: dict[str, Any], context_length: int | None) -> tuple[bool, bool, bool]:
    finish_reason = record.get("finish_reason")
    explicit = finish_reason == "length"

    usage = record.get("usage") or {}
    total_tokens = usage.get("total_tokens")
    inferred = (
        not explicit
        and context_length is not None
        and isinstance(total_tokens, (int, float))
        and total_tokens >= context_length
    )
    return explicit or inferred, explicit, inferred


def analyze_predictions(predictions_path: Path, context_length: int | None) -> dict[str, Any]:
    if not predictions_path.exists():
        return {
            "prediction_rows": 0,
            "rows_with_usage": 0,
            "truncated_rows": 0,
            "explicit_length_rows": 0,
            "context_cap_hit_rows": 0,
        }

    prediction_rows = 0
    rows_with_usage = 0
    truncated_rows = 0
    explicit_length_rows = 0
    context_cap_hit_rows = 0

    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        prediction_rows += 1
        record = json.loads(line)
        usage = record.get("usage")
        if isinstance(usage, dict):
            rows_with_usage += 1
        truncated, explicit, inferred = is_token_truncated(record, context_length)
        if truncated:
            truncated_rows += 1
        if explicit:
            explicit_length_rows += 1
        if inferred:
            context_cap_hit_rows += 1

    return {
        "prediction_rows": prediction_rows,
        "rows_with_usage": rows_with_usage,
        "truncated_rows": truncated_rows,
        "explicit_length_rows": explicit_length_rows,
        "context_cap_hit_rows": context_cap_hit_rows,
    }


def analyze_model_progress(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for progress_row in parse_model_progress(path):
        summary = load_json(progress_row.summary_path)
        run_root = progress_row.summary_path.parent
        for benchmark in benchmark_entries(summary):
            benchmark_name = benchmark.get("benchmark", "")
            context_length = benchmark.get("context_length")
            predictions_path = run_root / benchmark_name / "predictions.jsonl"
            counts = analyze_predictions(predictions_path, context_length if isinstance(context_length, int) else None)
            results.append(
                {
                    "model_name": progress_row.model_name,
                    "params": progress_row.params,
                    "quantization": progress_row.quantization,
                    "notes": progress_row.notes,
                    "run_label": progress_row.latest_run_label,
                    "summary_path": str(progress_row.summary_path),
                    "benchmark": benchmark_name,
                    "benchmark_status": benchmark.get("status"),
                    "sample_count": benchmark.get("sample_count"),
                    "context_length": context_length,
                    **counts,
                }
            )
    return results


def render_markdown(path: Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Token Truncation Analysis",
        "",
        f"Source model progress file: `{path}`",
        "",
        "Counting rule:",
        "- `explicit_length_rows`: `finish_reason == \"length\"`.",
        "- `context_cap_hit_rows`: `usage.total_tokens >= context_length` when explicit finish reason is unavailable.",
        "- `truncated_rows`: union of the two rules above.",
        "",
        "| Model Name | Params | Quantization | Notes | Benchmark | Run | Benchmark Status | Context Length | Prediction Rows | Rows With Usage | Truncated Rows | Explicit `length` Rows | Context Cap Hit Rows |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| {model_name} | {params} | {quantization} | {notes} | {benchmark} | {run_label} | {benchmark_status} | {context_length} | {prediction_rows} | {rows_with_usage} | {truncated_rows} | {explicit_length_rows} | {context_cap_hit_rows} |".format(
                **{
                    **row,
                    "context_length": row.get("context_length", ""),
                }
            )
        )

    return "\n".join(lines) + "\n"

