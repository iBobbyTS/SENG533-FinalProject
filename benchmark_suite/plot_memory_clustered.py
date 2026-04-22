from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemorySeries:
    label: str
    model_dir: str
    rss_initial_bytes: int
    rss_final_bytes: int
    footprint_initial_bytes: int
    footprint_final_bytes: int
    swap_initial_bytes: int
    swap_final_bytes: int
    model_bucket_bytes: int
    context_initial_bytes: int
    context_final_bytes: int


@dataclass(frozen=True)
class PowerMemorySeries:
    label: str
    model_key: str
    source_run: str | None
    status: str
    completion_tokens: int | None
    memory: MemorySeries | None
    error: str | None = None


PLOT_MODELS: tuple[tuple[str, str], ...] = (
    ("4B\nQ4_K_M", "qwen3.5-4b"),
    ("9B\nQ4_K_M", "qwen_qwen3.5-9b"),
    ("27B\nQ4_K_M", "qwen3.5-27b_q4_k_m"),
    ("27B\nQ6_K", "qwen3.5-27b_q6_k"),
    ("35B-A3B\nQ4_K_M", "qwen_qwen3.5-35b-a3b"),
)

POWER_MEMORY_MODELS: tuple[tuple[str, str], ...] = (
    ("4B\nQ4_K_M", "qwen3.5-4b"),
    ("9B\nQ4_K_M", "qwen/qwen3.5-9b@q4_k_m"),
    ("9B\nQ6_K", "qwen/qwen3.5-9b@q6_k"),
    ("9B\nQ8_0", "qwen/qwen3.5-9b@q8_0"),
    ("27B\nQ4_K_M", "qwen3.5-27b@q4_k_m"),
    ("27B\nQ6_K", "qwen3.5-27b@q6_k"),
    ("35B-A3B\nQ4_K_M", "qwen/qwen3.5-35b-a3b@q4_k_m"),
)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024**3)


def _require_int(mapping: dict, *keys: str) -> int:
    current: object = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(keys)
            raise KeyError(f"Missing key: {joined}")
        current = current[key]
    if not isinstance(current, int):
        joined = ".".join(keys)
        raise TypeError(f"Expected int at {joined}, got {type(current).__name__}")
    return current


def _slugify(value: str) -> str:
    out = []
    for char in value:
        if char.isalnum() or char in "._-":
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _completion_tokens(row: dict) -> int | None:
    usage = row.get("first_run_usage")
    if not isinstance(usage, dict):
        return None
    completion_tokens = usage.get("completion_tokens")
    return completion_tokens if isinstance(completion_tokens, int) else None


def _memory_series_from_summary_row(label: str, row: dict) -> MemorySeries:
    first = row.get("first_snapshot")
    last = row.get("final_snapshot")
    if not isinstance(first, dict) or not isinstance(last, dict):
        raise TypeError(f"Unexpected memory snapshot summary format for {row.get('model_key')}")
    model_key = str(row.get("model_key") or "")
    return MemorySeries(
        label=label,
        model_dir=_slugify(model_key),
        rss_initial_bytes=_require_int(first, "rss_bytes"),
        rss_final_bytes=_require_int(last, "rss_bytes"),
        footprint_initial_bytes=_require_int(first, "footprint", "phys_footprint_bytes"),
        footprint_final_bytes=_require_int(last, "footprint", "phys_footprint_bytes"),
        swap_initial_bytes=_require_int(first, "memory_analysis", "swapped_sum_bytes"),
        swap_final_bytes=_require_int(last, "memory_analysis", "swapped_sum_bytes"),
        model_bucket_bytes=_require_int(first, "memory_analysis", "categories", "model", "resident_bytes"),
        context_initial_bytes=_require_int(
            first,
            "memory_analysis",
            "categories",
            "context_gpu",
            "resident_bytes",
        ),
        context_final_bytes=_require_int(
            last,
            "memory_analysis",
            "categories",
            "context_gpu",
            "resident_bytes",
        ),
    )


def load_memory_series(run_dir: Path, context_label: str = "ctx_32768") -> list[MemorySeries]:
    series: list[MemorySeries] = []
    for label, model_dir in PLOT_MODELS:
        snapshot_path = run_dir / model_dir / context_label / "run2_memory_snapshots.json"
        snapshots = load_json(snapshot_path)
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError(f"No snapshots found in {snapshot_path}")
        first = snapshots[0]
        last = snapshots[-1]
        if not isinstance(first, dict) or not isinstance(last, dict):
            raise TypeError(f"Unexpected snapshot format in {snapshot_path}")
        series.append(
            MemorySeries(
                label=label,
                model_dir=model_dir,
                rss_initial_bytes=_require_int(first, "rss_bytes"),
                rss_final_bytes=_require_int(last, "rss_bytes"),
                footprint_initial_bytes=_require_int(first, "footprint", "phys_footprint_bytes"),
                footprint_final_bytes=_require_int(last, "footprint", "phys_footprint_bytes"),
                swap_initial_bytes=_require_int(first, "memory_analysis", "swapped_sum_bytes"),
                swap_final_bytes=_require_int(last, "memory_analysis", "swapped_sum_bytes"),
                model_bucket_bytes=_require_int(first, "memory_analysis", "categories", "model", "resident_bytes"),
                context_initial_bytes=_require_int(
                    first,
                    "memory_analysis",
                    "categories",
                    "context_gpu",
                    "resident_bytes",
                ),
                context_final_bytes=_require_int(
                    last,
                    "memory_analysis",
                    "categories",
                    "context_gpu",
                    "resident_bytes",
                ),
            )
        )
    return series


def load_power_memory_series(
    results_root: Path,
    model_specs: tuple[tuple[str, str], ...] = POWER_MEMORY_MODELS,
) -> list[PowerMemorySeries]:
    latest_by_model: dict[str, tuple[str, dict]] = {}
    latest_success_by_model: dict[str, tuple[str, dict]] = {}

    for summary_path in sorted(results_root.glob("*/summary.json")):
        summary = load_json(summary_path)
        if not isinstance(summary, dict):
            continue
        memory_rows = summary.get("memory")
        if not isinstance(memory_rows, list):
            continue
        source_run = summary_path.parent.name
        for row in memory_rows:
            if not isinstance(row, dict):
                continue
            model_key = row.get("model_key")
            if not isinstance(model_key, str):
                continue
            latest_by_model[model_key] = (source_run, row)
            if row.get("status") == "completed":
                latest_success_by_model[model_key] = (source_run, row)

    series: list[PowerMemorySeries] = []
    for label, model_key in model_specs:
        if model_key in latest_success_by_model:
            source_run, row = latest_success_by_model[model_key]
            series.append(
                PowerMemorySeries(
                    label=label,
                    model_key=model_key,
                    source_run=source_run,
                    status="completed",
                    completion_tokens=_completion_tokens(row),
                    memory=_memory_series_from_summary_row(label, row),
                )
            )
        elif model_key in latest_by_model:
            source_run, row = latest_by_model[model_key]
            error = row.get("error")
            series.append(
                PowerMemorySeries(
                    label=label,
                    model_key=model_key,
                    source_run=source_run,
                    status=str(row.get("status") or "failed"),
                    completion_tokens=_completion_tokens(row),
                    memory=None,
                    error=error if isinstance(error, str) else None,
                )
            )
        else:
            series.append(
                PowerMemorySeries(
                    label=label,
                    model_key=model_key,
                    source_run=None,
                    status="missing",
                    completion_tokens=None,
                    memory=None,
                    error="No memory row found",
                )
            )
    return series
