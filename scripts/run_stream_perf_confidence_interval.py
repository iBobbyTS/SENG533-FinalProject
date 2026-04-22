#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_BASE_URL = "http://127.0.0.1:1234/api/v1"
DEFAULT_CONTEXTS = [32768]
DEFAULT_RUNS = 5
DEFAULT_MAX_OUTPUT_TOKENS = 32768
DEFAULT_TEMPERATURE = 0.0
DEFAULT_IDLE_TIMEOUT_SECONDS = 600
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 60
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MODELS = [
    "qwen3.5-4b",
    "qwen/qwen3.5-9b",
    "qwen3.5-27b@q4_k_m",
    "qwen3.5-27b@q6_k",
    "qwen/qwen3.5-35b-a3b",
]

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
    11: 2.200985160082949,
    12: 2.178812829667229,
    13: 2.1603686564610127,
    14: 2.1447866879169273,
    15: 2.131449545559323,
    16: 2.1199052992210112,
    17: 2.1098155778333156,
    18: 2.10092204024096,
    19: 2.093024054408263,
    20: 2.0859634472658364,
    21: 2.079613844727662,
    22: 2.0738730679040147,
    23: 2.0686576104190406,
    24: 2.0638985616280205,
    25: 2.059538552753294,
    26: 2.055529438642871,
    27: 2.0518305164802833,
    28: 2.048407141795244,
    29: 2.045229642132703,
    30: 2.042272456301238,
    40: 2.021075390306273,
    60: 2.0002978210582616,
    120: 1.979930405052777,
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_suite.model_stream_perf import (  # noqa: E402
    DEFAULT_PROMPT,
    resolve_explicit_model_targets,
    run_generation_perf_pair,
    slugify,
    utc_timestamp,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run compact stream performance tests repeatedly and compute 95% Student t "
            "confidence intervals for tokens/s."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--expand-model-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expand multi-variant model keys such as qwen/qwen3.5-9b into all installed variants.",
    )
    parser.add_argument("--contexts", nargs="+", type=int, default=DEFAULT_CONTEXTS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--idle-timeout-seconds", type=int, default=DEFAULT_IDLE_TIMEOUT_SECONDS)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--plot-combined",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate only the combined stream TPS plot after all runs finish.",
    )
    return parser.parse_args()


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
    sample_stdev = statistics.stdev(clean_samples)
    standard_error = sample_stdev / math.sqrt(count)
    if confidence_level != 0.95:
        raise ValueError("Only 95% confidence intervals are supported without SciPy.")
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


def metric_samples(speed_runs: list[dict[str, Any]], key: str) -> list[float]:
    samples: list[float] = []
    for row in speed_runs:
        usage = row.get("usage") if isinstance(row, dict) else None
        if not isinstance(usage, dict):
            continue
        value = usage.get(key)
        if isinstance(value, int | float):
            samples.append(float(value))
    return samples


def token_samples(speed_runs: list[dict[str, Any]]) -> list[float]:
    return metric_samples(speed_runs, "completion_tokens")


def derived_throughput_samples(speed_runs: list[dict[str, Any]]) -> list[float]:
    samples: list[float] = []
    for row in speed_runs:
        usage = row.get("usage") if isinstance(row, dict) else None
        if not isinstance(usage, dict):
            continue
        completion_tokens = usage.get("completion_tokens")
        token_times = row.get("output_token_times_seconds")
        if not isinstance(completion_tokens, int | float) or not isinstance(token_times, list) or not token_times:
            continue
        final_time = token_times[-1]
        if not isinstance(final_time, int | float):
            continue
        final_time = float(final_time)
        if completion_tokens <= 0 or final_time <= 0:
            continue
        samples.append(float(completion_tokens) / final_time)
    return samples


def derived_seconds_per_token_samples(speed_runs: list[dict[str, Any]]) -> list[float]:
    samples: list[float] = []
    for row in speed_runs:
        usage = row.get("usage") if isinstance(row, dict) else None
        if not isinstance(usage, dict):
            continue
        completion_tokens = usage.get("completion_tokens")
        token_times = row.get("output_token_times_seconds")
        if not isinstance(completion_tokens, int | float) or not isinstance(token_times, list) or not token_times:
            continue
        final_time = token_times[-1]
        if not isinstance(final_time, int | float):
            continue
        final_time = float(final_time)
        if completion_tokens <= 0 or final_time <= 0:
            continue
        samples.append(final_time / float(completion_tokens))
    return samples


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stream Performance Confidence Intervals",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Base URL: `{summary['base_url']}`",
        f"- Contexts: `{summary['contexts']}`",
        f"- Runs per model/context: `{summary['runs']}`",
        f"- Max output tokens: `{summary['max_output_tokens']}`",
        f"- Temperature: `{summary['temperature']}`",
        f"- Confidence level: `{summary['confidence_level']}`",
        f"- Prompt: `{summary['prompt']}`",
        f"- TPS formula: `completion_tokens / output_token_times_seconds[-1]`",
    ]
    combined_plot = summary.get("combined_plot")
    if isinstance(combined_plot, dict):
        plot_paths = combined_plot.get("paths") if isinstance(combined_plot.get("paths"), list) else []
        if combined_plot.get("returncode") == 0 and plot_paths:
            lines.append(f"- Combined plot: `{plot_paths[0]}`")
        elif combined_plot.get("returncode") is not None:
            lines.append(f"- Combined plot status: `failed ({combined_plot.get('returncode')})`")
        stderr = combined_plot.get("stderr")
        if stderr:
            lines.append(f"- Combined plot stderr: `{stderr}`")
    lines.extend(
        [
            "",
            "| Model | Context | Status | Samples | Mean Derived TPS | 95% CI Lower | 95% CI Upper | CI Half Width | Relative Half Width | Mean LM Studio TPS | Mean TTFT (s) | Mean Output Tokens | Result |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["results"]:
        tps_ci = row.get("tokens_per_second_ci") or {}
        lmstudio_tps_ci = row.get("lmstudio_tokens_per_second_ci") or {}
        ttft_ci = row.get("time_to_first_token_seconds_ci") or {}
        output_tokens_ci = row.get("completion_tokens_ci") or {}

        def fmt(value: Any, digits: int = 3) -> str:
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                return ""
            return f"{float(value):.{digits}f}"

        lines.append(
            f"| {row['model_key']} | {row['context_length']} | {row['status']} | "
            f"{tps_ci.get('count') or 0} | "
            f"{fmt(tps_ci.get('mean'))} | "
            f"{fmt(tps_ci.get('ci_lower'))} | "
            f"{fmt(tps_ci.get('ci_upper'))} | "
            f"{fmt(tps_ci.get('ci_half_width'))} | "
            f"{fmt(tps_ci.get('relative_half_width_percent'))}% | "
            f"{fmt(lmstudio_tps_ci.get('mean'))} | "
            f"{fmt(ttft_ci.get('mean'))} | "
            f"{fmt(output_tokens_ci.get('mean'), 1)} | "
            f"`{row.get('result_path', '')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_combined_plot(run_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "plot_stream_tps.py"),
        "--combined-only",
        str(run_root),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "paths": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    args = parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2 to compute a confidence interval.")

    run_id = args.run_id or f"{utc_timestamp()}_stream_perf_ci"
    run_root = args.output_root.resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    targets = resolve_explicit_model_targets(
        args.models,
        args.base_url,
        expand_model_variants=args.expand_model_variants,
    )
    target_rows = [
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
    ]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "base_url": args.base_url,
        "prompt": args.prompt,
        "contexts": args.contexts,
        "runs": args.runs,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "compact_output_timing": True,
        "capture_memory": False,
        "targets": target_rows,
        "results": [],
    }
    write_json(run_root / "batch_config.json", summary)

    overall_returncode = 0
    for target in target_rows:
        model_key = target["model_key"]
        model_root = run_root / slugify(model_key)
        for context_length in args.contexts:
            context_root = model_root / f"ctx_{context_length}"
            print(
                json.dumps(
                    {
                        "event": "model_context_started",
                        "model": model_key,
                        "context_length": context_length,
                        "run_id": run_id,
                        "runs": args.runs,
                        "max_output_tokens": args.max_output_tokens,
                    }
                ),
                flush=True,
            )
            result_row: dict[str, Any] = {
                **target,
                "context_length": context_length,
                "status": "failed",
            }
            try:
                result = run_generation_perf_pair(
                    base_url=args.base_url,
                    model_key=model_key,
                    context_length=context_length,
                    prompt=args.prompt,
                    snapshot_interval_seconds=DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
                    idle_timeout_seconds=args.idle_timeout_seconds,
                    temperature=args.temperature,
                    capture_memory=False,
                    max_output_tokens=args.max_output_tokens,
                    speed_runs=args.runs,
                    compact_output_timing=True,
                )
                write_json(context_root / "result.json", result)
                write_json(context_root / "load.json", result["load"])
                if result["speed_runs"]:
                    write_json(context_root / "run1_full.json", result["speed_runs"][0])
                for index, speed_run in enumerate(result["speed_runs"], start=1):
                    write_json(context_root / f"run{index}_speed.json", speed_run)

                speed_runs = result["speed_runs"]
                tps_samples = derived_throughput_samples(speed_runs)
                seconds_per_token_samples = derived_seconds_per_token_samples(speed_runs)
                lmstudio_tps_samples = metric_samples(speed_runs, "tokens_per_second")
                ttft_samples = metric_samples(speed_runs, "time_to_first_token_seconds")
                completion_token_samples = token_samples(speed_runs)
                result_row.update(
                    {
                        "status": "completed",
                        "result_path": str(context_root / "result.json"),
                        "load_strategy": result["load"].get("strategy"),
                        "load_duration_seconds": result["load"].get("load_duration_seconds"),
                        "tokens_per_second_samples": tps_samples,
                        "seconds_per_token_samples": seconds_per_token_samples,
                        "lmstudio_tokens_per_second_samples": lmstudio_tps_samples,
                        "time_to_first_token_seconds_samples": ttft_samples,
                        "completion_tokens_samples": completion_token_samples,
                        "tokens_per_second_ci": confidence_interval(tps_samples),
                        "seconds_per_token_ci": confidence_interval(seconds_per_token_samples),
                        "lmstudio_tokens_per_second_ci": confidence_interval(lmstudio_tps_samples),
                        "time_to_first_token_seconds_ci": confidence_interval(ttft_samples),
                        "completion_tokens_ci": confidence_interval(completion_token_samples),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                overall_returncode = 1
                result_row.update({"status": "failed", "error": str(exc)})
                write_json(context_root / "error.json", result_row)
                if args.stop_on_failure:
                    summary["results"].append(result_row)
                    write_json(run_root / "confidence_summary.json", summary)
                    write_summary_markdown(run_root / "confidence_summary.md", summary)
                    return overall_returncode

            summary["results"].append(result_row)
            write_json(run_root / "confidence_summary.json", summary)
            write_summary_markdown(run_root / "confidence_summary.md", summary)
            print(
                json.dumps(
                    {
                        "event": "model_context_finished",
                        "model": model_key,
                        "context_length": context_length,
                        "run_id": run_id,
                        "status": result_row["status"],
                        "mean_tps": (result_row.get("tokens_per_second_ci") or {}).get("mean"),
                        "ci_half_width": (result_row.get("tokens_per_second_ci") or {}).get("ci_half_width"),
                    }
                ),
                flush=True,
            )

    if args.plot_combined:
        summary["combined_plot"] = run_combined_plot(run_root)
        if summary["combined_plot"]["returncode"] != 0:
            overall_returncode = 1
        print(
            json.dumps(
                {
                    "event": "combined_plot_finished",
                    "run_id": run_id,
                    "returncode": summary["combined_plot"]["returncode"],
                    "paths": summary["combined_plot"].get("paths", []),
                }
            ),
            flush=True,
        )

    write_json(run_root / "confidence_summary.json", summary)
    write_summary_markdown(run_root / "confidence_summary.md", summary)
    return overall_returncode


if __name__ == "__main__":
    raise SystemExit(main())
