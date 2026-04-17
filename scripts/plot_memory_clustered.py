#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from benchmark_suite.plot_memory_clustered import bytes_to_gib, load_memory_series

DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "20260404T065739Z_stream_perf_batch(on local)"
DEFAULT_CONTEXT_LABEL = "ctx_32768"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a clustered memory chart from run2_memory_snapshots.json files in a stream_perf_batch result directory."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Run directory to scan. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument(
        "--context-label",
        default=DEFAULT_CONTEXT_LABEL,
        help=f"Context directory name to read. Default: {DEFAULT_CONTEXT_LABEL}",
    )
    return parser.parse_args()


def append_plot_to_readme(readme_path: Path, plot_name: str) -> None:
    if readme_path.exists():
        lines = readme_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = ["# Stream TPS Plots", ""]
    item = f"- `{plot_name}`"
    if item not in lines:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(item)
        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    series = load_memory_series(run_dir, context_label=args.context_label)
    out_dir = run_dir / "plots"
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

    fig, ax = plt.subplots(figsize=(14, 7.5))
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

    for x_base, item in zip(positions, series, strict=True):
        rss_x = x_base + offsets[0]
        footprint_x = x_base + offsets[1]
        bucket_x = x_base + offsets[2]
        swap_x = x_base + offsets[3]

        rss_initial = bytes_to_gib(item.rss_initial_bytes)
        rss_final = bytes_to_gib(item.rss_final_bytes)
        footprint_initial = bytes_to_gib(item.footprint_initial_bytes)
        footprint_final = bytes_to_gib(item.footprint_final_bytes)
        swap_initial = bytes_to_gib(item.swap_initial_bytes)
        swap_final = bytes_to_gib(item.swap_final_bytes)
        model_bucket = bytes_to_gib(item.model_bucket_bytes)
        context_initial = bytes_to_gib(item.context_initial_bytes)
        context_final = bytes_to_gib(item.context_final_bytes)

        ax.bar(rss_x, rss_initial, width=bar_width, color=rss_fill, edgecolor=rss_fill, zorder=3)
        ax.bar(
            rss_x,
            rss_final,
            width=bar_width,
            fill=False,
            edgecolor=rss_edge,
            linestyle="--",
            linewidth=2.0,
            zorder=4,
        )

        ax.bar(
            footprint_x,
            footprint_initial,
            width=bar_width,
            color=footprint_fill,
            edgecolor=footprint_fill,
            zorder=3,
        )
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

        ax.bar(
            swap_x,
            swap_initial,
            width=bar_width,
            color=swap_fill,
            edgecolor=swap_fill,
            zorder=3,
        )
        ax.bar(
            swap_x,
            swap_final,
            width=bar_width,
            fill=False,
            edgecolor=swap_edge,
            linestyle="--",
            linewidth=2.0,
            zorder=4,
        )

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

    out_path = out_dir / "combined_non_claude_ctx_32768_memory_clustered.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    append_plot_to_readme(out_dir / "README.md", out_path.name)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
