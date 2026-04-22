#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
PROJECT_CACHE = PROJECT_ROOT / ".cache"

DEFAULT_OUTPUT_TPS_DIR = RESULTS_ROOT / "output_tps"
DEFAULT_INPUT_TTFT_DIR = RESULTS_ROOT / "input_ttft"

MODEL_LABEL_MAP = {
    "qwen3.5-4b": "4B Q4_K_M",
    "qwen_qwen3.5-4b": "4B Q4_K_M",
    "qwen/qwen3.5-9b": "9B Q4_K_M",
    "qwen/qwen3.5-9b@q4_k_m": "9B Q4_K_M",
    "qwen/qwen3.5-9b@q6_k": "9B Q6_K",
    "qwen/qwen3.5-9b@q8_0": "9B Q8_0",
    "qwen3.5-27b@q2_k": "27B Q2_K",
    "qwen3.5-27b@q4_k_m": "27B Q4_K_M",
    "qwen3.5-27b@q6_k": "27B Q6_K",
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q2_k": "27B Opus Q2_K",
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q4_k_m": "27B Opus Q4_K_M",
    "qwen/qwen3.5-35b-a3b": "35B-A3B Q4_K_M",
    "qwen/qwen3.5-35b-a3b@q4_k_m": "35B-A3B Q4_K_M",
}

MODEL_ORDER = {
    "4B Q4_K_M": 10,
    "9B Q4_K_M": 20,
    "9B Q6_K": 21,
    "9B Q8_0": 22,
    "27B Q2_K": 30,
    "27B Opus Q2_K": 31,
    "27B Q4_K_M": 40,
    "27B Opus Q4_K_M": 41,
    "27B Q6_K": 50,
    "35B-A3B Q4_K_M": 60,
}

MODEL_COLOR_MAP = {
    "4B Q4_K_M": "#4C6EF5",
    "9B Q4_K_M": "#15AABF",
    "9B Q6_K": "#12B886",
    "9B Q8_0": "#82C91E",
    "27B Q2_K": "#F08C00",
    "27B Opus Q2_K": "#E67700",
    "27B Q4_K_M": "#F03E3E",
    "27B Opus Q4_K_M": "#C92A2A",
    "27B Q6_K": "#AE3EC9",
    "35B-A3B Q4_K_M": "#7048E8",
}

FALLBACK_COLORS = [
    "#228BE6",
    "#40C057",
    "#FAB005",
    "#FA5252",
    "#BE4BDB",
    "#7950F2",
    "#12B886",
    "#FD7E14",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot consolidated output TPS and input TTFT charts from grouped final result folders."
    )
    parser.add_argument("--output-tps-dir", type=Path, default=DEFAULT_OUTPUT_TPS_DIR)
    parser.add_argument("--input-ttft-dir", type=Path, default=DEFAULT_INPUT_TTFT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any) -> float | None:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def ci_mean(row: dict[str, Any], key: str) -> float | None:
    ci = row.get(key)
    if not isinstance(ci, dict):
        return None
    return finite_number(ci.get("mean"))


def ci_half_width(row: dict[str, Any], key: str) -> float | None:
    ci = row.get(key)
    if not isinstance(ci, dict):
        return None
    return finite_number(ci.get("ci_half_width"))


def format_model_label(model_key: str) -> str:
    return MODEL_LABEL_MAP.get(model_key, model_key)


def format_axis_model_label(label: str) -> str:
    parts = label.split()
    if len(parts) < 2:
        return label
    return f"{' '.join(parts[:-1])}\n{parts[-1]}"


def model_sort_key(label: str, run_id: str = "") -> tuple[int, str, str]:
    return (MODEL_ORDER.get(label, 10_000), label, run_id)


def model_color(label: str) -> str:
    if label in MODEL_COLOR_MAP:
        return MODEL_COLOR_MAP[label]
    color_index = sum(ord(char) for char in label) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[color_index]


def output_row_key(row: dict[str, Any]) -> tuple[str, int | str]:
    model_key = str(row.get("model_key") or row.get("label") or "")
    context_length = row.get("context_length")
    return (model_key, context_length if isinstance(context_length, int) else "")


def prefer_output_row(current: dict[str, Any] | None, candidate: dict[str, Any]) -> bool:
    if current is None:
        return True
    return str(candidate.get("run_id") or "") >= str(current.get("run_id") or "")


def collect_output_tps_rows(output_tps_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows: dict[tuple[str, int | str], dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    for summary_path in sorted(output_tps_dir.glob("*/confidence_summary.json")):
        summary = load_json(summary_path)
        run_id = str(summary.get("run_id") or summary_path.parent.name)
        for result in summary.get("results", []):
            if not isinstance(result, dict):
                continue
            model_key = str(result.get("model_key") or "")
            mean_tps = ci_mean(result, "tokens_per_second_ci")
            if result.get("status") != "completed" or mean_tps is None:
                skipped.append(
                    {
                        "run_id": run_id,
                        "model_key": model_key,
                        "status": result.get("status"),
                        "reason": "not completed or missing tokens_per_second_ci.mean",
                        "summary_path": str(summary_path),
                    }
                )
                continue
            label = format_model_label(model_key)
            completion_mean = ci_mean(result, "completion_tokens_ci")
            result_json_path = resolve_result_json_path(summary_path, result)
            row = {
                "run_id": run_id,
                "model_key": model_key,
                "label": label,
                "context_length": result.get("context_length"),
                "mean_tps": mean_tps,
                "ci_half_width": ci_half_width(result, "tokens_per_second_ci") or 0.0,
                "completion_tokens_mean": completion_mean,
                "sample_count": (result.get("tokens_per_second_ci") or {}).get("count"),
                "summary_path": str(summary_path),
                "result_path": result.get("result_path"),
                "result_json_path": str(result_json_path) if result_json_path else None,
            }
            key = output_row_key(row)
            if prefer_output_row(selected_rows.get(key), row):
                selected_rows[key] = row
    rows = list(selected_rows.values())
    rows.sort(key=lambda item: model_sort_key(str(item["label"]), str(item["run_id"])))
    return rows, skipped


def resolve_result_json_path(summary_path: Path, result: dict[str, Any]) -> Path | None:
    result_path = result.get("result_path")
    if isinstance(result_path, str) and result_path:
        original = Path(result_path)
        if original.exists():
            return original
        run_dir_name = summary_path.parent.name
        if run_dir_name in original.parts:
            index = original.parts.index(run_dir_name)
            remapped = summary_path.parent.joinpath(*original.parts[index + 1 :])
            if remapped.exists():
                return remapped

    model_key = result.get("model_key")
    context_length = result.get("context_length")
    for candidate in sorted(summary_path.parent.rglob("result.json")):
        try:
            data = load_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("model") == model_key and data.get("context_length") == context_length:
            return candidate
    return None


def output_tps_curve_from_speed_runs(
    speed_runs: list[dict[str, Any]],
    *,
    max_points: int = 450,
) -> list[dict[str, float]]:
    valid_runs: list[tuple[list[float], float]] = []
    for run in speed_runs:
        raw_times = run.get("output_token_times_seconds")
        if not isinstance(raw_times, list):
            continue
        times = [float(item) for item in raw_times if finite_number(item) is not None]
        if len(times) < 2:
            continue
        first_output_seconds = finite_number(run.get("observed_first_output_seconds"))
        if first_output_seconds is None:
            first_output_seconds = times[0]
        valid_runs.append((times, first_output_seconds))

    if not valid_runs:
        return []

    max_common_tokens = min(len(times) for times, _ in valid_runs)
    if max_common_tokens < 2:
        return []
    if max_common_tokens <= max_points:
        token_indices = list(range(2, max_common_tokens + 1))
    else:
        token_indices = sorted(
            {
                round(2 + index * (max_common_tokens - 2) / (max_points - 1))
                for index in range(max_points)
            }
        )

    curve: list[dict[str, float]] = []
    for token_index in token_indices:
        samples: list[float] = []
        for times, first_output_seconds in valid_runs:
            elapsed = times[token_index - 1]
            duration = elapsed - first_output_seconds
            if duration > 0:
                samples.append(float(token_index) / duration)
        if samples:
            curve.append(
                {
                    "output_tokens": float(token_index),
                    "mean_tps": sum(samples) / len(samples),
                    "sample_count": float(len(samples)),
                }
            )
    return curve


def collect_output_tps_curves(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curves: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        result_json_path = row.get("result_json_path")
        if not isinstance(result_json_path, str) or not result_json_path:
            skipped.append(
                {
                    "run_id": row.get("run_id"),
                    "model_key": row.get("model_key"),
                    "reason": "missing result_json_path",
                }
            )
            continue
        result_path = Path(result_json_path)
        if not result_path.exists():
            skipped.append(
                {
                    "run_id": row.get("run_id"),
                    "model_key": row.get("model_key"),
                    "reason": "result_json_path not found",
                    "result_json_path": result_json_path,
                }
            )
            continue
        result_data = load_json(result_path)
        speed_runs = result_data.get("speed_runs")
        if not isinstance(speed_runs, list):
            skipped.append(
                {
                    "run_id": row.get("run_id"),
                    "model_key": row.get("model_key"),
                    "reason": "missing speed_runs",
                    "result_json_path": result_json_path,
                }
            )
            continue
        curve = output_tps_curve_from_speed_runs(speed_runs)
        if not curve:
            skipped.append(
                {
                    "run_id": row.get("run_id"),
                    "model_key": row.get("model_key"),
                    "reason": "no plottable output token time curve",
                    "result_json_path": result_json_path,
                }
            )
            continue
        curves.append(
            {
                "run_id": row.get("run_id"),
                "model_key": row.get("model_key"),
                "label": row.get("label"),
                "context_length": row.get("context_length"),
                "result_json_path": result_json_path,
                "points": curve,
            }
        )
    curves.sort(key=lambda item: model_sort_key(str(item["label"]), str(item["run_id"])))
    return curves, skipped


def collect_input_ttft_series(input_ttft_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    series: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for summary_path in sorted(input_ttft_dir.glob("*/input_ttft_summary.json")):
        summary = load_json(summary_path)
        run_id = str(summary.get("run_id") or summary_path.parent.name)
        for result in summary.get("results", []):
            if not isinstance(result, dict):
                continue
            model_key = str(result.get("model_key") or "")
            file_summaries = result.get("file_summaries")
            if result.get("status") != "completed" or not isinstance(file_summaries, list) or not file_summaries:
                skipped.append(
                    {
                        "run_id": run_id,
                        "model_key": model_key,
                        "status": result.get("status"),
                        "reason": "not completed or missing file_summaries",
                        "summary_path": str(summary_path),
                    }
                )
                continue

            points: list[dict[str, Any]] = []
            for item in file_summaries:
                if not isinstance(item, dict):
                    continue
                prompt_tokens = ci_mean(item, "prompt_tokens_ci")
                ttft_seconds = ci_mean(item, "ttft_minus_model_load_seconds_ci")
                if prompt_tokens is None or ttft_seconds is None:
                    continue
                points.append(
                    {
                        "file": item.get("file"),
                        "prompt_tokens_mean": prompt_tokens,
                        "ttft_seconds_mean": ttft_seconds,
                        "ttft_ci_half_width": ci_half_width(item, "ttft_minus_model_load_seconds_ci") or 0.0,
                    }
                )
            if not points:
                skipped.append(
                    {
                        "run_id": run_id,
                        "model_key": model_key,
                        "status": result.get("status"),
                        "reason": "no plottable input TTFT points",
                        "summary_path": str(summary_path),
                    }
                )
                continue
            label = format_model_label(model_key)
            points.sort(key=lambda point: (point["prompt_tokens_mean"], str(point["file"])))
            series.append(
                {
                    "run_id": run_id,
                    "model_key": model_key,
                    "label": label,
                    "context_length": result.get("context_length"),
                    "points": points,
                    "summary_path": str(summary_path),
                }
            )
    series.sort(key=lambda item: model_sort_key(str(item["label"]), str(item["run_id"])))
    return series, skipped


def collect_input_ttft_error_rows(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in series:
        points = item.get("points")
        if not isinstance(points, list) or not points:
            continue
        largest_input = max(points, key=lambda point: float(point["prompt_tokens_mean"]))
        rows.append(
            {
                "run_id": item.get("run_id"),
                "model_key": item.get("model_key"),
                "label": item.get("label"),
                "context_length": item.get("context_length"),
                "file": largest_input.get("file"),
                "prompt_tokens_mean": largest_input.get("prompt_tokens_mean"),
                "mean_ttft_seconds": largest_input.get("ttft_seconds_mean"),
                "ci_half_width": largest_input.get("ttft_ci_half_width") or 0.0,
            }
        )
    rows.sort(key=lambda item: model_sort_key(str(item["label"]), str(item["run_id"])))
    return rows


def ensure_matplotlib() -> Any:
    PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def write_output_tps_plot(output_tps_dir: Path, rows: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> Path:
    if not rows:
        raise RuntimeError(f"No completed output TPS rows found under {output_tps_dir}")

    output_tps_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_tps_dir / "output_tps_all_data.json"
    data_path.write_text(json.dumps({"rows": rows, "skipped": skipped}, indent=2), encoding="utf-8")

    plt = ensure_matplotlib()
    labels = [str(row["label"]) for row in rows]
    axis_labels = [format_axis_model_label(label) for label in labels]
    xs = list(range(len(rows)))
    means = [float(row["mean_tps"]) for row in rows]
    errors = [float(row.get("ci_half_width") or 0.0) for row in rows]

    fig_width = max(10, len(rows) * 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
    colors = [model_color(label) for label in labels]
    ax.bar(xs, means, yerr=errors, capsize=5, color=colors, edgecolor="#263238", linewidth=0.8)
    for x, mean in zip(xs, means, strict=False):
        ax.text(x, mean + max(means) * 0.015, f"{mean:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Output Tokens/s (mean)", fontsize=13)
    ax.set_xticks(xs)
    ax.set_xticklabels(axis_labels, rotation=0, ha="center", fontsize=11)
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        0.98,
        "Error bars: 95% CI",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        alpha=0.75,
    )

    out_path = output_tps_dir / "output_tps_all.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def write_output_tps_curve_plot(
    output_tps_dir: Path,
    curves: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> Path:
    if not curves:
        raise RuntimeError(f"No completed output TPS curves found under {output_tps_dir}")

    output_tps_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_tps_dir / "output_tps_over_output_tokens_data.json"
    data_path.write_text(json.dumps({"curves": curves, "skipped": skipped}, indent=2), encoding="utf-8")

    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    for item in curves:
        points = item["points"]
        xs = [point["output_tokens"] for point in points]
        ys = [point["mean_tps"] for point in points]
        label = str(item["label"])
        ax.plot(xs, ys, linewidth=1.7, label=label, color=model_color(label))

    ax.set_xlabel("Output Tokens", fontsize=13)
    ax.set_ylabel("Cumulative Output Tokens/s", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, ncols=2)

    out_path = output_tps_dir / "output_tps_over_output_tokens.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def write_input_ttft_plot(input_ttft_dir: Path, series: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> Path:
    if not series:
        raise RuntimeError(f"No completed input TTFT series found under {input_ttft_dir}")

    input_ttft_dir.mkdir(parents=True, exist_ok=True)
    data_path = input_ttft_dir / "input_ttft_all_data.json"
    data_path.write_text(json.dumps({"series": series, "skipped": skipped}, indent=2), encoding="utf-8")

    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    for item in series:
        points = item["points"]
        xs = [point["prompt_tokens_mean"] for point in points]
        ys = [point["ttft_seconds_mean"] for point in points]
        label = str(item["label"])
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4.2, label=label, color=model_color(label))

    ax.set_xlabel("Input Tokens", fontsize=13)
    ax.set_ylabel("Mean TTFT Minus Model Load (s)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, ncols=2)

    out_path = input_ttft_dir / "input_ttft_all.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def write_input_ttft_error_bar_plot(input_ttft_dir: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise RuntimeError(f"No completed input TTFT error-bar rows found under {input_ttft_dir}")

    input_ttft_dir.mkdir(parents=True, exist_ok=True)
    data_path = input_ttft_dir / "input_ttft_error_bar_data.json"
    data_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    plt = ensure_matplotlib()
    labels = [str(row["label"]) for row in rows]
    axis_labels = [format_axis_model_label(label) for label in labels]
    xs = list(range(len(rows)))
    means = [float(row["mean_ttft_seconds"]) for row in rows]
    errors = [float(row.get("ci_half_width") or 0.0) for row in rows]

    fig_width = max(10, len(rows) * 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
    colors = [model_color(label) for label in labels]
    ax.bar(xs, means, yerr=errors, capsize=5, color=colors, edgecolor="#263238", linewidth=0.8)
    max_label_height = max(mean + error for mean, error in zip(means, errors, strict=False))
    label_offset = max(means) * 0.025
    ax.set_ylim(0, max_label_height + label_offset * 4.0)
    for x, row, mean in zip(xs, rows, means, strict=False):
        prompt_tokens = row.get("prompt_tokens_mean")
        token_label = f"{prompt_tokens:.0f} tok" if isinstance(prompt_tokens, int | float) else ""
        ax.text(x, mean + label_offset, f"{mean:.1f}\n{token_label}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("TTFT Minus Model Load (s)", fontsize=13)
    ax.set_xticks(xs)
    ax.set_xticklabels(axis_labels, rotation=0, ha="center", fontsize=11)
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        0.96,
        "Error bars: 95% CI from the largest input-token file per model",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        alpha=0.75,
    )

    out_path = input_ttft_dir / "input_ttft_error_bar.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    output_rows, output_skipped = collect_output_tps_rows(args.output_tps_dir)
    output_curves, output_curve_skipped = collect_output_tps_curves(output_rows)
    input_series, input_skipped = collect_input_ttft_series(args.input_ttft_dir)
    input_error_rows = collect_input_ttft_error_rows(input_series)

    output_plot = write_output_tps_plot(args.output_tps_dir, output_rows, output_skipped)
    output_curve_plot = write_output_tps_curve_plot(args.output_tps_dir, output_curves, output_curve_skipped)
    input_plot = write_input_ttft_plot(args.input_ttft_dir, input_series, input_skipped)
    input_error_plot = write_input_ttft_error_bar_plot(args.input_ttft_dir, input_error_rows)

    print(output_plot)
    print(output_curve_plot)
    print(input_plot)
    print(input_error_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
