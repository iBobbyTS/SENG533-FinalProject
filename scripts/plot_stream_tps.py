#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CACHE = PROJECT_ROOT / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "20260404T065739Z_stream_perf_batch(on local)"
COMBINED_EXCLUDE_SUBSTRINGS = ["claude-4.6-opus-reasoning-distilled"]
COMBINED_LABEL_MAP = {
    "qwen3.5-4b": "4B Q4_K_M",
    "qwen/qwen3.5-9b": "9B Q4_K_M",
    "qwen/qwen3.5-9b@q4_k_m": "9B Q4_K_M",
    "qwen/qwen3.5-9b@q6_k": "9B Q6_K",
    "qwen/qwen3.5-9b@q8_0": "9B Q8_0",
    "qwen3.5-27b@q4_k_m": "27B Q4_K_M",
    "qwen3.5-27b@q6_k": "27B Q6_K",
    "qwen/qwen3.5-35b-a3b": "35B-A3B Q4_K_M",
    "qwen/qwen3.5-35b-a3b@q4_k_m": "35B-A3B Q4_K_M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot token-vs-throughput curves from run1_full.json files in a stream_perf_batch result directory."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Run directory to scan. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Generate only the combined plot and skip per-model plots.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    out: list[str] = []
    for char in value:
        if char.isalnum() or char in "._-":
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_curve(run1: dict) -> tuple[list[float], list[float]]:
    delta_events = run1.get("delta_events", [])
    output_event_times = run1.get("output_token_times_seconds") or run1.get("output_event_times_seconds", [])
    usage = run1.get("usage", {}) or {}
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, (int, float)) or completion_tokens <= 0:
        completion_tokens = len(delta_events) or len(output_event_times)
    if not delta_events and not output_event_times:
        return [], []

    first_output_seconds = run1.get("observed_first_output_seconds")
    if not isinstance(first_output_seconds, (int, float)):
        if delta_events:
            first_output_seconds = float(delta_events[0].get("elapsed_seconds") or 0.0)
        else:
            first_output_seconds = float(output_event_times[0] or 0.0)

    event_count = len(delta_events) or len(output_event_times)
    xs: list[float] = []
    ys: list[float] = []
    for index in range(1, event_count + 1):
        if delta_events:
            elapsed = float(delta_events[index - 1].get("elapsed_seconds") or 0.0)
        else:
            elapsed = float(output_event_times[index - 1] or 0.0)
        approx_tokens = float(completion_tokens) * (index / event_count)
        denom = elapsed - float(first_output_seconds)
        tps = math.nan if denom <= 0 else approx_tokens / denom
        xs.append(approx_tokens)
        ys.append(tps)
    return xs, ys


def plot_one(run1_path: Path, out_dir: Path) -> Path:
    data = load_json(run1_path)
    model = str(data.get("model") or run1_path.parents[1].name)
    context = run1_path.parent.name.replace("ctx_", "")
    xs, ys = build_curve(data)
    usage = data.get("usage", {}) or {}
    output_tokens = usage.get("completion_tokens")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, linewidth=1.2, color="#0B7285")

    ax.set_title(f"{model} | ctx_{context}")
    ax.set_xlabel("Output Tokens")
    ax.set_ylabel("Cumulative Tokens/s")
    ax.grid(True, alpha=0.3)

    event_count = len(data.get("delta_events", [])) or len(data.get("output_event_times_seconds", []))
    subtitle = f"completion_tokens={output_tokens}, output_events={event_count}"
    ax.text(0.01, 0.01, subtitle, transform=ax.transAxes, fontsize=9, alpha=0.8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(model)}_ctx_{context}_tps.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def should_exclude_from_combined(model: str) -> bool:
    lowered = model.lower()
    return any(fragment in lowered for fragment in COMBINED_EXCLUDE_SUBSTRINGS)


def format_combined_label(model: str) -> str:
    return COMBINED_LABEL_MAP.get(model, model)


def plot_combined(run1_paths: list[Path], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    included = 0

    for run1_path in run1_paths:
        data = load_json(run1_path)
        model = str(data.get("model") or run1_path.parents[1].name)
        if should_exclude_from_combined(model):
            continue
        xs, ys = build_curve(data)
        if not xs:
            continue
        label = format_combined_label(model)
        ax.plot(xs, ys, linewidth=1.4, label=label)
        included += 1

    if included == 0:
        raise RuntimeError("No eligible models found for the combined plot.")

    ax.set_xlabel("Output Tokens", fontsize=15)
    ax.set_ylabel("Cumulative Tokens/s", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_non_claude_ctx_32768_tps.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    run1_paths = sorted(run_dir.rglob("run1_full.json"))
    if not run1_paths:
        raise SystemExit(f"No run1_full.json files found under: {run_dir}")

    out_dir = run_dir / "plots"
    plot_paths: list[Path] = []
    if not args.combined_only:
        for run1_path in run1_paths:
            plot_paths.append(plot_one(run1_path, out_dir))
    plot_paths.append(plot_combined(run1_paths, out_dir))

    index_lines = [
        "# Stream TPS Plots",
        "",
        f"Source run directory: `{run_dir}`",
        "",
    ]
    for plot_path in plot_paths:
        index_lines.append(f"- `{plot_path.name}`")
    (out_dir / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    for plot_path in plot_paths:
        print(plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
