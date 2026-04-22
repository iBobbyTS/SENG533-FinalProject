#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

from benchmark_suite.plot_memory_clustered import (  # noqa: E402
    PowerMemorySeries,
    bytes_to_gib,
    load_power_memory_series,
)

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "power_memory_test"
DEFAULT_OUT_DIR = DEFAULT_RESULTS_ROOT / "plots"
DEFAULT_CONTEXT_LENGTH = 32768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot clustered memory bars from power_memory_test memory summaries."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"power_memory_test results root. Default: {DEFAULT_RESULTS_ROOT}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help=f"Expected context length used to mark early stops. Default: {DEFAULT_CONTEXT_LENGTH}",
    )
    return parser.parse_args()


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


def write_quality_files(out_dir: Path, series: list[PowerMemorySeries], context_length: int) -> None:
    rows = []
    lines = [
        "# Power Memory Clustered Plot Quality",
        "",
        "| Model | Status | Source run | Completion tokens | Notes |",
        "| --- | --- | --- | ---: | --- |",
    ]
    early_stop_threshold = int(context_length * 0.9)
    for item in series:
        notes: list[str] = []
        if item.memory is None:
            notes.append(item.error or "No memory snapshot data")
        if item.completion_tokens is not None and item.completion_tokens < early_stop_threshold:
            notes.append("Early stop")
        if not notes:
            notes.append("OK")
        rows.append(
            {
                "label": item.label.replace("\n", " "),
                "model_key": item.model_key,
                "status": item.status,
                "source_run": item.source_run,
                "completion_tokens": item.completion_tokens,
                "notes": notes,
            }
        )
        lines.append(
            f"| {item.label.replace(chr(10), ' ')} | {item.status} | {item.source_run or ''} | "
            f"{item.completion_tokens if item.completion_tokens is not None else ''} | {'; '.join(notes)} |"
        )
    (out_dir / "power_memory_clustered_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "power_memory_clustered_data.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    if not results_root.exists():
        raise SystemExit(f"Results root not found: {results_root}")

    series = load_power_memory_series(results_root)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(17, 8))
    cluster_gap = 2.5
    bar_width = 0.28
    positions = [index * cluster_gap for index in range(len(series))]
    offsets = (-0.72, -0.24, 0.24, 0.72)

    rss_fill = "#4C6EF5"
    rss_edge = "#2B4ACB"
    footprint_fill = "#12B886"
    footprint_edge = "#0F8E68"
    swap_fill = "#E03131"
    swap_edge = "#AE1E1E"
    model_fill = "#5C677D"
    context_fill = "#F08C00"
    context_edge = "#C26B00"
    early_stop_threshold = int(args.context_length * 0.9)

    for x_base, item in zip(positions, series, strict=True):
        if item.memory is None:
            ax.text(x_base, 0.55, "failed", ha="center", va="bottom", rotation=90, color="#AE1E1E", fontsize=12)
            continue

        rss_x = x_base + offsets[0]
        footprint_x = x_base + offsets[1]
        bucket_x = x_base + offsets[2]
        swap_x = x_base + offsets[3]

        rss_initial = bytes_to_gib(item.memory.rss_initial_bytes)
        rss_final = bytes_to_gib(item.memory.rss_final_bytes)
        footprint_initial = bytes_to_gib(item.memory.footprint_initial_bytes)
        footprint_final = bytes_to_gib(item.memory.footprint_final_bytes)
        swap_initial = bytes_to_gib(item.memory.swap_initial_bytes)
        swap_final = bytes_to_gib(item.memory.swap_final_bytes)
        model_bucket = bytes_to_gib(item.memory.model_bucket_bytes)
        context_initial = bytes_to_gib(item.memory.context_initial_bytes)
        context_final = bytes_to_gib(item.memory.context_final_bytes)

        ax.bar(rss_x, rss_initial, width=bar_width, color=rss_fill, edgecolor=rss_fill, zorder=3)
        ax.bar(rss_x, rss_final, width=bar_width, fill=False, edgecolor=rss_edge, linestyle="--", linewidth=2.0, zorder=4)

        ax.bar(footprint_x, footprint_initial, width=bar_width, color=footprint_fill, edgecolor=footprint_fill, zorder=3)
        ax.bar(
            footprint_x,
            footprint_final,
            width=bar_width,
            fill=False,
            edgecolor=footprint_edge,
            linestyle="--",
            linewidth=2.0,
            zorder=4,
        )

        ax.bar(swap_x, swap_initial, width=bar_width, color=swap_fill, edgecolor=swap_fill, zorder=3)
        ax.bar(swap_x, swap_final, width=bar_width, fill=False, edgecolor=swap_edge, linestyle="--", linewidth=2.0, zorder=4)

        ax.bar(bucket_x, model_bucket, width=bar_width, color=model_fill, edgecolor=model_fill, zorder=3)
        ax.bar(
            bucket_x,
            context_initial,
            width=bar_width,
            bottom=model_bucket,
            color=context_fill,
            edgecolor=context_fill,
            zorder=3,
        )
        ax.bar(
            bucket_x,
            context_final,
            width=bar_width,
            bottom=model_bucket,
            fill=False,
            edgecolor=context_edge,
            linestyle="--",
            linewidth=2.0,
            zorder=4,
        )
        if item.completion_tokens is not None and item.completion_tokens < early_stop_threshold:
            ax.text(
                x_base,
                0.55,
                "early stop",
                ha="center",
                va="bottom",
                rotation=90,
                color=context_edge,
                fontsize=12,
            )

    ax.set_ylabel("Memory (GiB)")
    ax.set_xticks(positions)
    ax.set_xticklabels([item.label for item in series])
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    legend_handles_left = [
        Patch(facecolor=rss_fill, edgecolor=rss_fill, label="RSS (first snapshot)"),
        Patch(facecolor="none", edgecolor=rss_edge, linestyle="--", linewidth=2.0, label="RSS (final)"),
        Patch(facecolor=footprint_fill, edgecolor=footprint_fill, label="Footprint (first snapshot)"),
        Patch(facecolor="none", edgecolor=footprint_edge, linestyle="--", linewidth=2.0, label="Footprint (final)"),
        Patch(facecolor=swap_fill, edgecolor=swap_fill, label="Swap (first snapshot)"),
        Patch(facecolor="none", edgecolor=swap_edge, linestyle="--", linewidth=2.0, label="Swap (final)"),
    ]
    legend_handles_right = [
        Patch(facecolor=model_fill, edgecolor=model_fill, label="Model bucket"),
        Patch(facecolor=context_fill, edgecolor=context_fill, label="Context bucket (first snapshot)"),
        Patch(facecolor="none", edgecolor=context_edge, linestyle="--", linewidth=2.0, label="Context bucket (final)"),
    ]
    legend_left = ax.legend(
        handles=legend_handles_left,
        ncol=1,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.995),
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.6,
        labelspacing=0.45,
    )
    ax.add_artist(legend_left)
    ax.legend(
        handles=legend_handles_right,
        ncol=1,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.28, 0.995),
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.6,
        labelspacing=0.45,
    )

    out_path = out_dir / "power_memory_clustered.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    append_plot_to_readme(out_dir / "README.md", out_path.name)
    write_quality_files(out_dir, series, args.context_length)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
