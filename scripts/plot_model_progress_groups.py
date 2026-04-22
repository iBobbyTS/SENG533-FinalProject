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
import numpy as np  # noqa: E402

from benchmark_suite.model_progress_grouped_plots import (  # noqa: E402
    BENCHMARK_LABELS,
    BENCHMARK_ORDER,
    BenchmarkOverride,
    apply_benchmark_overrides,
    find_row,
    load_benchmark_overrides,
    load_benchmark_points,
    load_truncation_index,
    parse_model_progress_markdown,
    resolve_summary_path,
    transpose_group_series,
)

DEFAULT_MODEL_PROGRESS = PROJECT_ROOT / "results" / "20260330_model_progress.md"
DEFAULT_TRUNCATION = PROJECT_ROOT / "results" / "20260330_model_progress_truncation.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "inteligence_benchmark" / "grouped_plots"
DEFAULT_BENCHMARK_OVERRIDES = DEFAULT_OUT_DIR / "benchmark_overrides.json"
DEFAULT_SUMMARY_SEARCH_ROOTS = [PROJECT_ROOT / "results" / "inteligence_benchmark"]

GROUPS = [
    (
        "01_q4_model_size",
        "All Q4_K_M: 4B vs 9B vs 27B",
        [
            ("4B", "4B", "Q4_K_M", "Dense model"),
            ("9B", "9B", "Q4_K_M", "Dense model"),
            ("27B", "27B", "Q4_K_M", "Dense model"),
        ],
    ),
    (
        "02_9b_quantization",
        "9B Quantization Comparison",
        [
            ("Q4_K_M", "9B", "Q4_K_M", "Dense model"),
            ("Q6_K", "9B", "Q6_K", "Dense model"),
            ("Q8_0", "9B", "Q8_0", "Dense model"),
        ],
    ),
    (
        "03_dense_vs_moe",
        "27B Dense vs 35B-A3B MoE",
        [
            ("27B", "27B", "Q4_K_M", "Dense model"),
            ("35B-A3B", "35B-A3B", "Q4_K_M", "Official A3B MoE"),
        ],
    ),
    (
        "04_distillation",
        "27B Base vs 27B Claude Opus 4.6 Distilled",
        [
            ("Official", "27B", "Q4_K_M", "Dense model"),
            ("Opus", "27B", "Q4_K_M", "Claude 4.6 Opus reasoning distilled"),
        ],
    ),
    (
        "05_ultra_low_quantization",
        "27B Opus 4.6: Q4_K_M vs Q2_K",
        [
            ("Q2_K", "27B", "Q2_K", "Claude 4.6 Opus reasoning distilled"),
            ("Q4_K_M", "27B", "Q4_K_M", "Claude 4.6 Opus reasoning distilled"),
        ],
    ),
]

SCORE_COLORS = ["#4C6EF5", "#12B886", "#F08C00", "#7950F2", "#E03131", "#0B7285"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot grouped benchmark score charts from model progress and truncation files.")
    parser.add_argument("--model-progress", type=Path, default=DEFAULT_MODEL_PROGRESS)
    parser.add_argument("--truncation", type=Path, default=DEFAULT_TRUNCATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--benchmark-overrides",
        type=Path,
        default=DEFAULT_BENCHMARK_OVERRIDES,
        help="Optional JSON file with single-benchmark rerun summaries to use instead of the aggregate summary entry.",
    )
    parser.add_argument(
        "--summary-search-root",
        type=Path,
        action="append",
        help="Directory to search when model progress summary links point to moved result folders.",
    )
    return parser.parse_args()


def write_readme(
    out_dir: Path,
    files: list[tuple[str, str]],
    model_progress: Path,
    truncation: Path,
    overrides_path: Path,
    overrides: dict[tuple[str, str, str, str], BenchmarkOverride],
) -> None:
    lines = [
        "# Grouped Benchmark Score Plots",
        "",
        f"Source progress file: `{model_progress}`",
        f"Source truncation file: `{truncation}`",
    ]
    if overrides:
        lines.extend(
            [
                f"Benchmark override file: `{overrides_path}`",
                "",
                "Applied benchmark overrides:",
            ]
        )
        for override in overrides.values():
            model = f"{override.params} {override.quantization}"
            lines.append(f"- `{model}` `{override.benchmark}`: `{override.summary_path}`")
    lines.append("")
    for title, filename in files:
        lines.append(f"- `{filename}`: {title}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_group(
    title: str,
    output_path: Path,
    series: list[tuple[str, dict]],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 7.5))
    model_labels, benchmark_series = transpose_group_series(series)
    x = np.arange(len(BENCHMARK_ORDER))
    bar_width = 0.84 / len(BENCHMARK_ORDER)

    for idx, (label, points) in enumerate(series):
        score_offset = (idx - (len(series) - 1) / 2) * bar_width
        score_heights: list[float] = []
        missing_positions: list[tuple[float, str]] = []
        score_color = SCORE_COLORS[idx % len(SCORE_COLORS)]
        for bench_index, benchmark in enumerate(BENCHMARK_ORDER):
            point = points[benchmark]
            if point.score_percent is None:
                score_heights.append(0.0)
                missing_positions.append((x[bench_index] + score_offset, "NA"))
            else:
                score_heights.append(point.score_percent)

        score_bars = ax.bar(
            x + score_offset,
            score_heights,
            width=bar_width,
            color=score_color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        for bar, benchmark in zip(score_bars, BENCHMARK_ORDER, strict=True):
            point = points[benchmark]
            if point.score_percent is None:
                bar.set_facecolor("#E9ECEF")
                bar.set_edgecolor("#868E96")
                bar.set_hatch("//")
                text_y = 2.0
            else:
                text_y = bar.get_height() + 1.2
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#212529",
            )

        for xpos, text in missing_positions:
            ax.text(xpos, 0.35, text, ha="center", va="bottom", fontsize=9, color="#495057")

    ax.set_title(title)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_LABELS[name] for name in BENCHMARK_ORDER])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = parse_model_progress_markdown(args.model_progress)
    truncation_index = load_truncation_index(args.truncation)
    overrides = load_benchmark_overrides(args.benchmark_overrides)
    search_roots = args.summary_search_root or DEFAULT_SUMMARY_SEARCH_ROOTS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, str]] = []
    for slug, title, model_specs in GROUPS:
        series: list[tuple[str, dict]] = []
        for label, params, quantization, notes in model_specs:
            row = find_row(rows, params, quantization, notes)
            summary_path = resolve_summary_path(row.summary_path, search_roots)
            points = load_benchmark_points(summary_path, truncation_index, truncation_summary_path=row.summary_path)
            points = apply_benchmark_overrides(row, points, overrides, truncation_index)
            series.append((label, points))
        out_path = args.out_dir / f"{slug}.png"
        plot_group(title, out_path, series)
        written.append((title, out_path.name))
        print(out_path)

    write_readme(args.out_dir, written, args.model_progress, args.truncation, args.benchmark_overrides, overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
