from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import OFFICIAL_RESOURCES
from .utils import (
    approx_prefill_tps,
    ensure_dir,
    load_jsonl,
    markdown_percent,
    summarize_prediction_performance,
    write_json,
)


def write_benchmark_summary(summary_path: Path, summary: dict) -> None:
    write_json(summary_path, summary)


def _markdown_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _markdown_bytes(value: float | None) -> str:
    if value is None:
        return "N/A"
    size = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def _prediction_index(row: dict[str, Any]) -> str:
    if "index" in row:
        return str(row["index"])
    if "row_index" in row:
        return str(row["row_index"])
    if "task_index" in row:
        return str(row["task_index"])
    return "N/A"


def _category_column(predictions: list[dict[str, Any]]) -> tuple[str, list[str] | None]:
    if any("category" in row for row in predictions):
        return "Category", [str(row.get("category", "")) for row in predictions]
    if any("task" in row for row in predictions):
        return "Task", [str(row.get("task", "")) for row in predictions]
    if any("answer_type" in row for row in predictions):
        return "Category", [str(row.get("answer_type", "")) for row in predictions]
    return "", None


def enrich_benchmark_summary(summary: dict, benchmark_dir: Path, duration_seconds: float | None) -> tuple[dict, list[dict[str, Any]]]:
    predictions_path = benchmark_dir / "predictions.jsonl"
    predictions: list[dict[str, Any]] = []
    if predictions_path.exists():
        predictions = [row for row in load_jsonl(predictions_path) if isinstance(row, dict)]
    enriched = dict(summary)
    enriched.update(summarize_prediction_performance(predictions))
    enriched["time_total_seconds"] = duration_seconds
    return enriched, predictions


def write_benchmark_markdown(markdown_path: Path, summary: dict, predictions: list[dict[str, Any]] | None = None) -> None:
    ensure_dir(markdown_path.parent)
    resource = OFFICIAL_RESOURCES[summary["benchmark"]]
    lines = [
        f"# {summary['benchmark']}",
        "",
        f"- Status: `{summary['status']}`",
        f"- Profile: `{summary['profile']}`",
        f"- Model: `{summary['model']}`",
        f"- Context length: `{summary.get('context_length') if summary.get('context_length') is not None else 'N/A'}`",
        f"- Samples: `{summary['sample_count']}`",
        f"- Score: `{markdown_percent(summary.get('score'))}`",
        f"- Metric: `{summary['metric']}`",
        f"- Mean decode TPS: `{_markdown_float(summary.get('mean_decode_tps'))}`",
        f"- Mean approx. prefill TPS: `{_markdown_float(summary.get('mean_approx_prefill_tps'))}`",
        f"- Time total (s): `{_markdown_float(summary.get('time_total_seconds'))}`",
        f"- Memory Max: `{_markdown_bytes(summary.get('memory_max_bytes'))}`",
        f"- Memory Mean: `{_markdown_bytes(summary.get('memory_mean_bytes'))}`",
        f"- Virtual Max: `{_markdown_bytes(summary.get('virtual_max_bytes'))}`",
        f"- Virtual Mean: `{_markdown_bytes(summary.get('virtual_mean_bytes'))}`",
        f"- Swap Used Max: `{_markdown_bytes(summary.get('swap_used_max_bytes'))}`",
        f"- Swap Used Mean: `{_markdown_bytes(summary.get('swap_used_mean_bytes'))}`",
        f"- Mode: `{resource['mode']}`",
        f"- Repo: {resource['repo']}",
        f"- Dataset: {resource['dataset']}",
    ]
    notes = summary.get("notes", [])
    if notes:
        lines.append("- Notes:")
        for note in notes:
            lines.append(f"  - {note}")
    if predictions:
        category_header, category_values = _category_column(predictions)
        lines.append("")
        if category_values is None:
            lines.extend(
                [
                    "| Index | Prompt Tokens | Completion Tokens | Approx. Prefill TPS | Decode TPS |",
                    "| ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in predictions:
                usage = row.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
                prefill_tps = approx_prefill_tps(
                    usage.get("prompt_tokens") if isinstance(usage, dict) else None,
                    usage.get("time_to_first_token_seconds") if isinstance(usage, dict) else None,
                )
                decode_tps = usage.get("tokens_per_second") if isinstance(usage, dict) else None
                lines.append(
                    f"| {_prediction_index(row)} | {prompt_tokens if prompt_tokens is not None else 'N/A'} | "
                    f"{completion_tokens if completion_tokens is not None else 'N/A'} | "
                    f"{_markdown_float(prefill_tps)} | {_markdown_float(decode_tps)} |"
                )
        else:
            lines.extend(
                [
                    f"| Index | {category_header} | Prompt Tokens | Completion Tokens | Approx. Prefill TPS | Decode TPS |",
                    "| ---: | --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for row, category_value in zip(predictions, category_values, strict=False):
                usage = row.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
                prefill_tps = approx_prefill_tps(
                    usage.get("prompt_tokens") if isinstance(usage, dict) else None,
                    usage.get("time_to_first_token_seconds") if isinstance(usage, dict) else None,
                )
                decode_tps = usage.get("tokens_per_second") if isinstance(usage, dict) else None
                lines.append(
                    f"| {_prediction_index(row)} | {category_value or 'N/A'} | "
                    f"{prompt_tokens if prompt_tokens is not None else 'N/A'} | "
                    f"{completion_tokens if completion_tokens is not None else 'N/A'} | "
                    f"{_markdown_float(prefill_tps)} | {_markdown_float(decode_tps)} |"
                )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_markdown(markdown_path: Path, run_summary: dict) -> None:
    ensure_dir(markdown_path.parent)
    lines = [
        "# Benchmark Run Summary",
        "",
        f"- Run ID: `{run_summary['run_id']}`",
        f"- Profile: `{run_summary['profile']}`",
        f"- Model: `{run_summary['model']}`",
        f"- Base URL: `{run_summary['base_url']}`",
        f"- Time total (s): `{_markdown_float(run_summary.get('time_total_seconds'))}`",
        "",
        "| Benchmark | Context | Status | Score | Samples | Metric | Mean decode TPS | Mean approx. prefill TPS | Memory Max | Memory Mean | Virtual Max | Virtual Mean | Swap Used Max | Swap Used Mean | Power Max | Power Mean | GPU Power Max | GPU Power Mean | RAM Power Max | RAM Power Mean | Sys Power Max | Sys Power Mean | Time total (s) |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bench in run_summary["benchmarks"]:
        lines.append(
            f"| {bench['benchmark']} | {bench.get('context_length') if bench.get('context_length') is not None else 'N/A'} | "
            f"{bench['status']} | {markdown_percent(bench.get('score'))} | "
            f"{bench['sample_count']} | {bench['metric']} | "
            f"{_markdown_float(bench.get('mean_decode_tps'))} | "
            f"{_markdown_float(bench.get('mean_approx_prefill_tps'))} | "
            f"{_markdown_bytes(bench.get('memory_max_bytes'))} | "
            f"{_markdown_bytes(bench.get('memory_mean_bytes'))} | "
            f"{_markdown_bytes(bench.get('virtual_max_bytes'))} | "
            f"{_markdown_bytes(bench.get('virtual_mean_bytes'))} | "
            f"{_markdown_bytes(bench.get('swap_used_max_bytes'))} | "
            f"{_markdown_bytes(bench.get('swap_used_mean_bytes'))} | "
            f"{_markdown_float(bench.get('power_max_watts'), 2)} | "
            f"{_markdown_float(bench.get('power_mean_watts'), 2)} | "
            f"{_markdown_float(bench.get('gpu_power_max_watts'), 2)} | "
            f"{_markdown_float(bench.get('gpu_power_mean_watts'), 2)} | "
            f"{_markdown_float(bench.get('ram_power_max_watts'), 2)} | "
            f"{_markdown_float(bench.get('ram_power_mean_watts'), 2)} | "
            f"{_markdown_float(bench.get('sys_power_max_watts'), 2)} | "
            f"{_markdown_float(bench.get('sys_power_mean_watts'), 2)} | "
            f"{_markdown_float(bench.get('time_total_seconds'))} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
