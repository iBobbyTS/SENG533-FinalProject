#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CACHE = PROJECT_ROOT / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "power_memory_test"
DEFAULT_OUT = DEFAULT_RESULTS_ROOT / "plots" / "power_system_components.png"

MODEL_ORDER = [
    "qwen3.5-4b",
    "qwen/qwen3.5-9b@q4_k_m",
    "qwen/qwen3.5-9b@q6_k",
    "qwen/qwen3.5-9b@q8_0",
    "qwen3.5-27b@q4_k_m",
    "qwen3.5-27b@q6_k",
    "qwen/qwen3.5-35b-a3b@q4_k_m",
]

MODEL_LABELS = {
    "qwen3.5-4b": "4B\nQ4_K_M",
    "qwen/qwen3.5-9b@q4_k_m": "9B\nQ4_K_M",
    "qwen/qwen3.5-9b@q6_k": "9B\nQ6_K",
    "qwen/qwen3.5-9b@q8_0": "9B\nQ8_0",
    "qwen3.5-27b@q4_k_m": "27B\nQ4_K_M",
    "qwen3.5-27b@q6_k": "27B\nQ6_K",
    "qwen/qwen3.5-35b-a3b@q4_k_m": "35B-A3B\nQ4_K_M",
}

BAR_ORDER = [
    ("sys_power", "System Power", "#E45756"),
    ("gpu_power", "GPU Power", "#F58518"),
    ("ram_power", "RAM Power", "#54A24B"),
    ("cpu_power", "CPU Power", "#4C78A8"),
]

LEGEND_ORDER = [
    ("sys_power", "System Power", "#E45756"),
    ("gpu_power", "GPU Power", "#F58518"),
    ("ram_power", "RAM Power", "#54A24B"),
    ("cpu_power", "CPU Power", "#4C78A8"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot power_memory_test power bars with System Power on the left of each cluster."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any) -> float | None:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def metric_ci(row: dict[str, Any], metric: str) -> tuple[float | None, float]:
    ci = row.get("metric_ci_over_run_means")
    if not isinstance(ci, dict):
        return None, 0.0
    metric_row = ci.get(metric)
    if not isinstance(metric_row, dict):
        return None, 0.0
    mean = finite_number(metric_row.get("mean"))
    half_width = finite_number(metric_row.get("ci_half_width")) or 0.0
    return mean, half_width


def collect_power_rows(results_root: Path) -> list[dict[str, Any]]:
    latest_completed_by_model: dict[str, tuple[str, dict[str, Any]]] = {}
    for summary_path in sorted(results_root.glob("*/summary.json")):
        summary = load_json(summary_path)
        if not isinstance(summary, dict):
            continue
        rows = summary.get("power")
        if not isinstance(rows, list):
            continue
        source_run = summary_path.parent.name
        for row in rows:
            if not isinstance(row, dict) or row.get("status") != "completed":
                continue
            model_key = row.get("model_key")
            if isinstance(model_key, str):
                latest_completed_by_model[model_key] = (source_run, row)

    output_rows: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        source_run, row = latest_completed_by_model.get(model_key, (None, {}))
        metrics = {}
        for metric, _, _ in BAR_ORDER:
            mean, half_width = metric_ci(row, metric)
            metrics[metric] = {
                "mean": mean,
                "ci_half_width": half_width,
            }
        output_rows.append(
            {
                "model_key": model_key,
                "label": MODEL_LABELS[model_key],
                "source_run": source_run,
                "status": "completed" if source_run else "missing",
                "metrics": metrics,
            }
        )
    return output_rows


def append_plot_to_readme(readme_path: Path, plot_name: str) -> None:
    if readme_path.exists():
        lines = readme_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = ["# Power Memory Plots", ""]
    item = f"- `{plot_name}`"
    if item not in lines:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(item)
        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_power_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_positions = np.arange(len(rows))
    width = 0.18
    offsets = [-0.30, -0.10, 0.10, 0.30]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(16, 8))

    for offset, (metric, label, color) in zip(offsets, BAR_ORDER, strict=True):
        means = []
        errors = []
        for row in rows:
            metric_row = row["metrics"][metric]
            means.append(metric_row["mean"] if metric_row["mean"] is not None else np.nan)
            errors.append(metric_row["ci_half_width"])
        bars = ax.bar(
            x_positions + offset,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=color,
            edgecolor="#333333",
            linewidth=0.7,
            label=label,
            zorder=3,
        )
        for bar, value in zip(bars, means, strict=True):
            if math.isfinite(float(value)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{float(value):.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_ylabel("Power (W)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([row["label"] for row in rows])
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    legend_handles = [
        Patch(facecolor=color, edgecolor="#333333", label=label)
        for _, label, color in LEGEND_ORDER
    ]
    ax.legend(handles=legend_handles, ncol=4, loc="upper left", frameon=True)
    ax.text(
        0.012,
        0.925,
        "Error bars: 95% CI",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="#444444",
    )
    ax.set_ylim(0, max(ax.get_ylim()[1], 120))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_plot_data(out_path: Path, rows: list[dict[str, Any]]) -> None:
    data_path = out_path.with_suffix(".json")
    data_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    if not results_root.exists():
        raise SystemExit(f"Results root not found: {results_root}")

    rows = collect_power_rows(results_root)
    out_path = args.out.resolve()
    plot_power_rows(rows, out_path)
    write_plot_data(out_path, rows)
    append_plot_to_readme(out_path.parent / "README.md", out_path.name)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
