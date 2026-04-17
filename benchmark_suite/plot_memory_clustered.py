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


PLOT_MODELS: tuple[tuple[str, str], ...] = (
    ("4B\nQ4_K_M", "qwen3.5-4b"),
    ("9B\nQ4_K_M", "qwen_qwen3.5-9b"),
    ("27B\nQ4_K_M", "qwen3.5-27b_q4_k_m"),
    ("27B\nQ6_K", "qwen3.5-27b_q6_k"),
    ("35B-A3B\nQ4_K_M", "qwen_qwen3.5-35b-a3b"),
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
