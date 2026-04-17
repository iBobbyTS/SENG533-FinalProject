#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
PROJECT_CACHE = PROJECT_ROOT / ".cache"
DEFAULT_BASE_URL = "http://127.0.0.1:1234/api/v1"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input_token"
DEFAULT_CONTEXT_LENGTH = 20000
DEFAULT_RUNS = 5
DEFAULT_MAX_OUTPUT_TOKENS = 16
DEFAULT_IDLE_TIMEOUT_SECONDS = 300
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MODELS = [
    "qwen3.5-4b",
    "qwen/qwen3.5-9b",
    "qwen3.5-27b@q4_k_m",
    "qwen3.5-27b@q6_k",
    "qwen/qwen3.5-35b-a3b",
]

PLOT_LABEL_MAP = {
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

T_CRITICAL_975 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636314,
    6: 2.446911848791681,
    7: 2.3646242510102993,
    8: 2.306004135033371,
    9: 2.2621571628540993,
    10: 2.2281388519649385,
    20: 2.0859634472658364,
    30: 2.042272456301238,
    60: 2.0002978210582616,
    120: 1.979930405052777,
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_suite.model_stream_perf import (  # noqa: E402
    _stream_chat,
    load_model_with_timing,
    resolve_explicit_model_targets,
    slugify,
    unload_all_models,
    utc_timestamp,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run input TTFT tests with top-level Swift files and compute a 95% CI "
            "for the longest input file."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--expand-model-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expand model keys such as qwen/qwen3.5-9b into installed quantization variants.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--idle-timeout-seconds", type=int, default=DEFAULT_IDLE_TIMEOUT_SECONDS)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def discover_swift_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix == ".swift")


def select_longest_file(paths: list[Path]) -> Path:
    if not paths:
        raise ValueError("No Swift input files were found.")
    return max(paths, key=lambda path: (path.stat().st_size, path.name))


def build_prompt(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    return (
        "Read the Swift source file below. Reply with exactly one short sentence confirming that you read it.\n\n"
        f"File: {path.name}\n"
        "```swift\n"
        f"{source}\n"
        "```"
    )


def t_critical_975(df: int) -> float:
    if df < 1:
        raise ValueError("degrees of freedom must be at least 1")
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    greater_keys = [key for key in T_CRITICAL_975 if key > df]
    if greater_keys:
        return T_CRITICAL_975[min(greater_keys)]
    return 1.959963984540054


def confidence_interval(samples: list[float], confidence_level: float = DEFAULT_CONFIDENCE_LEVEL) -> dict[str, Any]:
    clean_samples = [float(item) for item in samples if isinstance(item, int | float) and math.isfinite(float(item))]
    count = len(clean_samples)
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "sample_stdev": None,
            "standard_error": None,
            "confidence_level": confidence_level,
            "t_critical": None,
            "ci_half_width": None,
            "ci_lower": None,
            "ci_upper": None,
            "relative_half_width_percent": None,
        }
    mean = statistics.fmean(clean_samples)
    if count == 1:
        return {
            "count": count,
            "mean": mean,
            "sample_stdev": None,
            "standard_error": None,
            "confidence_level": confidence_level,
            "t_critical": None,
            "ci_half_width": None,
            "ci_lower": None,
            "ci_upper": None,
            "relative_half_width_percent": None,
        }
    if confidence_level != 0.95:
        raise ValueError("Only 95% confidence intervals are supported without SciPy.")
    sample_stdev = statistics.stdev(clean_samples)
    standard_error = sample_stdev / math.sqrt(count)
    critical = t_critical_975(count - 1)
    half_width = critical * standard_error
    return {
        "count": count,
        "mean": mean,
        "sample_stdev": sample_stdev,
        "standard_error": standard_error,
        "confidence_level": confidence_level,
        "t_critical": critical,
        "ci_half_width": half_width,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "relative_half_width_percent": (half_width / mean * 100) if mean else None,
    }


def finite_metric_samples(runs: list[dict[str, Any]], key: str) -> list[float]:
    samples: list[float] = []
    for run in runs:
        usage = run.get("usage") if isinstance(run, dict) else None
        value = usage.get(key) if isinstance(usage, dict) else None
        if isinstance(value, int | float) and math.isfinite(float(value)):
            samples.append(float(value))
    return samples


def approximate_input_tps_samples(runs: list[dict[str, Any]]) -> list[float]:
    samples: list[float] = []
    for run in runs:
        usage = run.get("usage") if isinstance(run, dict) else None
        if not isinstance(usage, dict):
            continue
        prompt_tokens = usage.get("prompt_tokens")
        ttft_seconds = usage.get("time_to_first_token_seconds")
        if not isinstance(prompt_tokens, int | float) or not isinstance(ttft_seconds, int | float):
            continue
        if prompt_tokens > 0 and ttft_seconds > 0:
            samples.append(float(prompt_tokens) / float(ttft_seconds))
    return samples


def ttft_minus_model_load_samples(runs: list[dict[str, Any]]) -> list[float]:
    samples: list[float] = []
    for run in runs:
        usage = run.get("usage") if isinstance(run, dict) else None
        if not isinstance(usage, dict):
            continue
        ttft_seconds = usage.get("time_to_first_token_seconds")
        if not isinstance(ttft_seconds, int | float) or not math.isfinite(float(ttft_seconds)):
            continue
        model_load_seconds = usage.get("model_load_time_seconds")
        if not isinstance(model_load_seconds, int | float) or not math.isfinite(float(model_load_seconds)):
            model_load_seconds = 0.0
        samples.append(max(float(ttft_seconds) - float(model_load_seconds), 0.0))
    return samples


def summarize_file_runs(file_path: Path, longest_file: Path, runs: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    ttft_samples = finite_metric_samples(runs, "time_to_first_token_seconds")
    adjusted_ttft_samples = ttft_minus_model_load_samples(runs)
    model_load_samples = finite_metric_samples(runs, "model_load_time_seconds")
    prompt_token_samples = finite_metric_samples(runs, "prompt_tokens")
    completion_token_samples = finite_metric_samples(runs, "completion_tokens")
    input_tps_samples = approximate_input_tps_samples(runs)
    return {
        "file": file_path.name,
        "file_path": str(file_path),
        "is_longest_file": file_path == longest_file,
        "bytes": file_path.stat().st_size,
        "runs": len(runs),
        "result_path": str(out_path),
        "time_to_first_token_seconds_samples": ttft_samples,
        "model_load_time_seconds_samples": model_load_samples,
        "ttft_minus_model_load_seconds_samples": adjusted_ttft_samples,
        "prompt_tokens_samples": prompt_token_samples,
        "completion_tokens_samples": completion_token_samples,
        "approx_input_tokens_per_second_samples": input_tps_samples,
        "time_to_first_token_seconds_ci": confidence_interval(ttft_samples),
        "model_load_time_seconds_ci": confidence_interval(model_load_samples),
        "ttft_minus_model_load_seconds_ci": confidence_interval(adjusted_ttft_samples),
        "prompt_tokens_ci": confidence_interval(prompt_token_samples),
        "completion_tokens_ci": confidence_interval(completion_token_samples),
        "approx_input_tokens_per_second_ci": confidence_interval(input_tps_samples),
    }


def format_float(value: Any, digits: int = 3) -> str:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def format_plot_label(model_key: str) -> str:
    return PLOT_LABEL_MAP.get(model_key, model_key)


def build_plot_series(summary: dict[str, Any]) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for row in summary.get("results", []):
        if not isinstance(row, dict):
            continue
        model_key = str(row.get("model_key") or "")
        points: list[dict[str, Any]] = []
        for file_summary in row.get("file_summaries", []):
            if not isinstance(file_summary, dict):
                continue
            prompt_mean = (file_summary.get("prompt_tokens_ci") or {}).get("mean")
            ttft_mean = (file_summary.get("ttft_minus_model_load_seconds_ci") or {}).get("mean")
            if not isinstance(prompt_mean, int | float) or not isinstance(ttft_mean, int | float):
                continue
            if not math.isfinite(float(prompt_mean)) or not math.isfinite(float(ttft_mean)):
                continue
            points.append(
                {
                    "file": file_summary.get("file"),
                    "bytes": file_summary.get("bytes"),
                    "prompt_tokens_mean": float(prompt_mean),
                    "ttft_seconds_mean": float(ttft_mean),
                }
            )
        if points:
            points.sort(key=lambda point: (point["prompt_tokens_mean"], str(point.get("file") or "")))
            series.append(
                {
                    "model_key": model_key,
                    "label": format_plot_label(model_key),
                    "points": points,
                }
            )
    return series


def write_input_ttft_plot(run_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    plot_dir = run_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    series = build_plot_series(summary)
    data_path = plot_dir / "input_ttft_vs_input_tokens_data.json"
    write_json(data_path, series)
    if not series:
        return {
            "status": "skipped",
            "reason": "no completed model/file summaries",
            "data_path": str(data_path),
        }

    PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(12, 7))
    for item in series:
        xs = [point["prompt_tokens_mean"] for point in item["points"]]
        ys = [point["ttft_seconds_mean"] for point in item["points"]]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=item["label"])

    ax.set_xlabel("Input Tokens", fontsize=15)
    ax.set_ylabel("Mean TTFT (s)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    plot_path = plot_dir / "input_ttft_vs_input_tokens.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return {
        "status": "completed",
        "path": str(plot_path),
        "data_path": str(data_path),
    }


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Input TTFT Confidence Intervals",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Base URL: `{summary['base_url']}`",
        f"- Input directory: `{summary['input_dir']}`",
        f"- Swift files: `{summary['swift_files']}`",
        f"- Longest file: `{summary['longest_file']}`",
        f"- Context length: `{summary['context_length']}`",
        f"- Runs per model/file: `{summary['runs']}`",
        f"- Max output tokens: `{summary['max_output_tokens']}`",
        f"- Confidence level: `{summary['confidence_level']}`",
        f"- TTFT source: `usage.time_to_first_token_seconds - usage.model_load_time_seconds`",
    ]
    input_ttft_plot = summary.get("input_ttft_plot")
    if isinstance(input_ttft_plot, dict):
        if input_ttft_plot.get("status") == "completed":
            lines.append(f"- Input TTFT plot: `{input_ttft_plot.get('path')}`")
        elif input_ttft_plot.get("status"):
            lines.append(f"- Input TTFT plot status: `{input_ttft_plot.get('status')}`")
    lines.extend(
        [
            "",
            "## Longest File 95% CI",
            "",
            "| Model | Quantization | File | Bytes | Samples | Mean TTFT Minus Model Load (s) | 95% CI Lower | 95% CI Upper | CI Half Width | Relative Half Width | Mean Raw TTFT (s) | Mean Model Load (s) | Mean Prompt Tokens | Mean Approx Input Tokens/s | Result |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["results"]:
        longest = row.get("longest_file_summary") or {}
        ttft_ci = longest.get("ttft_minus_model_load_seconds_ci") or {}
        raw_ttft_ci = longest.get("time_to_first_token_seconds_ci") or {}
        model_load_ci = longest.get("model_load_time_seconds_ci") or {}
        prompt_ci = longest.get("prompt_tokens_ci") or {}
        input_tps_ci = longest.get("approx_input_tokens_per_second_ci") or {}
        lines.append(
            f"| {row['model_key']} | {row.get('quantization') or ''} | {longest.get('file') or ''} | "
            f"{longest.get('bytes') or ''} | "
            f"{ttft_ci.get('count') or 0} | "
            f"{format_float(ttft_ci.get('mean'))} | "
            f"{format_float(ttft_ci.get('ci_lower'))} | "
            f"{format_float(ttft_ci.get('ci_upper'))} | "
            f"{format_float(ttft_ci.get('ci_half_width'))} | "
            f"{format_float(ttft_ci.get('relative_half_width_percent'))}% | "
            f"{format_float(raw_ttft_ci.get('mean'))} | "
            f"{format_float(model_load_ci.get('mean'))} | "
            f"{format_float(prompt_ci.get('mean'), 1)} | "
            f"{format_float(input_tps_ci.get('mean'))} | "
            f"`{longest.get('result_path') or ''}` |"
        )

    lines.extend(
        [
            "",
            "## All Files",
            "",
            "| Model | File | Bytes | Status | Samples | Mean TTFT Minus Model Load (s) | Mean Raw TTFT (s) | Mean Model Load (s) | Mean Prompt Tokens | Mean Completion Tokens | Result |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["results"]:
        for file_summary in row.get("file_summaries", []):
            ttft_ci = file_summary.get("ttft_minus_model_load_seconds_ci") or {}
            raw_ttft_ci = file_summary.get("time_to_first_token_seconds_ci") or {}
            model_load_ci = file_summary.get("model_load_time_seconds_ci") or {}
            prompt_ci = file_summary.get("prompt_tokens_ci") or {}
            completion_ci = file_summary.get("completion_tokens_ci") or {}
            lines.append(
                f"| {row['model_key']} | {file_summary.get('file') or ''} | "
                f"{file_summary.get('bytes') or ''} | {row.get('status') or ''} | "
                f"{ttft_ci.get('count') or 0} | "
                f"{format_float(ttft_ci.get('mean'))} | "
                f"{format_float(raw_ttft_ci.get('mean'))} | "
                f"{format_float(model_load_ci.get('mean'))} | "
                f"{format_float(prompt_ci.get('mean'), 1)} | "
                f"{format_float(completion_ci.get('mean'), 1)} | "
                f"`{file_summary.get('result_path') or ''}` |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2 to compute a confidence interval.")

    input_files = discover_swift_files(args.input_dir)
    longest_file = select_longest_file(input_files)
    targets = resolve_explicit_model_targets(
        args.models,
        args.base_url,
        expand_model_variants=args.expand_model_variants,
    )

    run_id = args.run_id or f"{utc_timestamp()}_input_ttft_ci"
    run_root = args.output_root.resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "base_url": args.base_url,
        "input_dir": str(args.input_dir.resolve()),
        "swift_files": [path.name for path in input_files],
        "longest_file": longest_file.name,
        "context_length": args.context_length,
        "runs": args.runs,
        "max_output_tokens": args.max_output_tokens,
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "targets": [
            {
                "source": target.source,
                "model_key": target.model_key,
                "params": target.params,
                "quantization": target.quantization,
                "notes": target.notes,
                "latest_run_label": target.latest_run_label,
                "summary_path": str(target.summary_path) if target.summary_path else None,
                "recorded_model_key": target.recorded_model_key,
            }
            for target in targets
        ],
        "results": [],
    }
    write_json(run_root / "batch_config.json", summary)

    overall_returncode = 0
    for target in summary["targets"]:
        model_key = target["model_key"]
        model_root = run_root / slugify(model_key) / f"ctx_{args.context_length}"
        model_row: dict[str, Any] = {
            **target,
            "context_length": args.context_length,
            "status": "failed",
            "file_summaries": [],
            "longest_file_summary": None,
        }
        print(
            json.dumps(
                {
                    "event": "model_started",
                    "model": model_key,
                    "context_length": args.context_length,
                    "runs": args.runs,
                    "max_output_tokens": args.max_output_tokens,
                }
            ),
            flush=True,
        )

        try:
            model_row["preexisting_loaded_models"] = unload_all_models(args.base_url)
            model_row["load"] = load_model_with_timing(args.base_url, model_key, args.context_length)
            for input_file in input_files:
                file_root = model_root / slugify(input_file.stem)
                prompt = build_prompt(input_file)
                file_runs: list[dict[str, Any]] = []
                for run_index in range(1, args.runs + 1):
                    print(
                        json.dumps(
                            {
                                "event": "input_ttft_run_started",
                                "model": model_key,
                                "file": input_file.name,
                                "run": run_index,
                            }
                        ),
                        flush=True,
                    )
                    run_result = _stream_chat(
                        base_url=args.base_url,
                        model_key=model_key,
                        context_length=args.context_length,
                        prompt=prompt,
                        idle_timeout_seconds=args.idle_timeout_seconds,
                        max_output_tokens=args.max_output_tokens,
                        compact_output_timing=True,
                    )
                    run_result["input_file"] = {
                        "name": input_file.name,
                        "path": str(input_file),
                        "bytes": input_file.stat().st_size,
                    }
                    run_result["run_index"] = run_index
                    write_json(file_root / f"run{run_index}.json", run_result)
                    file_runs.append(run_result)
                    usage = run_result.get("usage") or {}
                    print(
                        json.dumps(
                            {
                                "event": "input_ttft_run_finished",
                                "model": model_key,
                                "file": input_file.name,
                                "run": run_index,
                                "prompt_tokens": usage.get("prompt_tokens"),
                                "ttft_seconds": usage.get("time_to_first_token_seconds"),
                            }
                        ),
                        flush=True,
                    )
                write_json(file_root / "runs.json", file_runs)
                file_summary = summarize_file_runs(input_file, longest_file, file_runs, file_root / "runs.json")
                write_json(file_root / "summary.json", file_summary)
                model_row["file_summaries"].append(file_summary)
                print(
                    json.dumps(
                        {
                            "event": "input_ttft_file_finished",
                            "model": model_key,
                            "file": input_file.name,
                            "prompt_tokens_mean": (file_summary.get("prompt_tokens_ci") or {}).get("mean"),
                            "mean_ttft_minus_model_load_seconds": (
                                file_summary.get("ttft_minus_model_load_seconds_ci") or {}
                            ).get("mean"),
                            "mean_raw_ttft_seconds": (file_summary.get("time_to_first_token_seconds_ci") or {}).get(
                                "mean"
                            ),
                            "mean_model_load_seconds": (file_summary.get("model_load_time_seconds_ci") or {}).get(
                                "mean"
                            ),
                        }
                    ),
                    flush=True,
                )
                if input_file == longest_file:
                    model_row["longest_file_summary"] = file_summary
            model_row["status"] = "completed"
        except Exception as exc:  # noqa: BLE001
            overall_returncode = 1
            model_row["status"] = "failed"
            model_row["error"] = str(exc)
            write_json(model_root / "error.json", model_row)
            if args.stop_on_failure:
                summary["results"].append(model_row)
                write_json(run_root / "input_ttft_summary.json", summary)
                write_summary_markdown(run_root / "input_ttft_summary.md", summary)
                unload_all_models(args.base_url)
                return overall_returncode
        finally:
            model_row["unloaded_after"] = unload_all_models(args.base_url)

        summary["results"].append(model_row)
        summary["input_ttft_plot"] = write_input_ttft_plot(run_root, summary)
        write_json(run_root / "input_ttft_summary.json", summary)
        write_summary_markdown(run_root / "input_ttft_summary.md", summary)
        print(
            json.dumps(
                {
                    "event": "model_finished",
                    "model": model_key,
                    "status": model_row["status"],
                    "longest_file": longest_file.name,
                    "mean_ttft_minus_model_load_seconds": (
                        (
                            (model_row.get("longest_file_summary") or {}).get(
                                "ttft_minus_model_load_seconds_ci"
                            )
                            or {}
                        ).get("mean")
                    ),
                }
            ),
            flush=True,
        )

    summary["input_ttft_plot"] = write_input_ttft_plot(run_root, summary)
    write_json(run_root / "input_ttft_summary.json", summary)
    write_summary_markdown(run_root / "input_ttft_summary.md", summary)
    return overall_returncode


if __name__ == "__main__":
    raise SystemExit(main())
