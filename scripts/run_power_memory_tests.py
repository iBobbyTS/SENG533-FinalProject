#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import select
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_BASE_URL = "http://127.0.0.1:1234/api/v1"
DEFAULT_CONTEXT_LENGTH = 32768
DEFAULT_TEMPERATURE = 0.1
DEFAULT_POWER_RUNS = 5
DEFAULT_POWER_INTERVAL_MS = 250
DEFAULT_POWER_WARMUP_SECONDS = 3.0
DEFAULT_POWER_SAMPLE_SECONDS = 10.0
DEFAULT_POWER_STARTUP_TIMEOUT_SECONDS = 15.0
DEFAULT_COOLDOWN_SECONDS = 60.0
DEFAULT_IDLE_TIMEOUT_SECONDS = 600
DEFAULT_MAX_OUTPUT_TOKENS = 32768
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 60
DEFAULT_MEMORY_SPEED_RUNS = 0
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "power_memory_test"

DEFAULT_MODELS = [
    "qwen3.5-4b",
    "qwen/qwen3.5-9b@q4_k_m",
    "qwen/qwen3.5-9b@q6_k",
    "qwen/qwen3.5-9b@q8_0",
    "qwen3.5-27b@q4_k_m",
    "qwen3.5-27b@q6_k",
    "qwen/qwen3.5-35b-a3b@q4_k_m",
]

POWER_METRICS = [
    "all_power",
    "cpu_power",
    "gpu_power",
    "gpu_ram_power",
    "ram_power",
    "sys_power",
    "ane_power",
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
    20: 2.0859634472658364,
    30: 2.042272456301238,
    60: 2.0002978210582616,
    120: 1.979930405052777,
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_suite.model_management import native_api_root_from_base_url  # noqa: E402
from benchmark_suite.model_stream_perf import (  # noqa: E402
    DEFAULT_PROMPT,
    _parse_sse_event,
    load_model_with_timing,
    run_generation_perf_pair,
    slugify,
    unload_all_models,
    utc_timestamp,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LM Studio power and memory tests for the final model set."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--mode",
        choices=["power", "memory", "both"],
        default="both",
        help="Select which test group to run.",
    )
    parser.add_argument("--power-runs", type=int, default=DEFAULT_POWER_RUNS)
    parser.add_argument("--power-interval-ms", type=int, default=DEFAULT_POWER_INTERVAL_MS)
    parser.add_argument("--power-warmup-seconds", type=float, default=DEFAULT_POWER_WARMUP_SECONDS)
    parser.add_argument("--power-sample-seconds", type=float, default=DEFAULT_POWER_SAMPLE_SECONDS)
    parser.add_argument("--power-startup-timeout-seconds", type=float, default=DEFAULT_POWER_STARTUP_TIMEOUT_SECONDS)
    parser.add_argument("--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN_SECONDS)
    parser.add_argument("--idle-timeout-seconds", type=int, default=DEFAULT_IDLE_TIMEOUT_SECONDS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--output-start-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--memory-snapshot-interval-seconds", type=int, default=DEFAULT_SNAPSHOT_INTERVAL_SECONDS)
    parser.add_argument(
        "--memory-speed-runs",
        type=int,
        default=DEFAULT_MEMORY_SPEED_RUNS,
        help="Optional warm speed-only runs before the memory-capture run. Defaults to 0 for memory-only testing.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the run plan and write no benchmark samples.",
    )
    return parser.parse_args()


def metric_summary(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) >= 2 else 0.0,
    }


def t_critical_975(df: int) -> float:
    if df < 1:
        raise ValueError("degrees of freedom must be at least 1")
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    greater_keys = [key for key in T_CRITICAL_975 if key > df]
    if greater_keys:
        return T_CRITICAL_975[min(greater_keys)]
    return 1.959963984540054


def confidence_interval(samples: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in samples if math.isfinite(float(value))]
    count = len(clean)
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "sample_stdev": None,
            "standard_error": None,
            "confidence_level": 0.95,
            "t_critical": None,
            "ci_half_width": None,
            "ci_lower": None,
            "ci_upper": None,
        }
    mean = statistics.fmean(clean)
    if count == 1:
        return {
            "count": count,
            "mean": mean,
            "sample_stdev": None,
            "standard_error": None,
            "confidence_level": 0.95,
            "t_critical": None,
            "ci_half_width": None,
            "ci_lower": None,
            "ci_upper": None,
        }
    sample_stdev = statistics.stdev(clean)
    standard_error = sample_stdev / math.sqrt(count)
    critical = t_critical_975(count - 1)
    half_width = critical * standard_error
    return {
        "count": count,
        "mean": mean,
        "sample_stdev": sample_stdev,
        "standard_error": standard_error,
        "confidence_level": 0.95,
        "t_critical": critical,
        "ci_half_width": half_width,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
    }


def parse_macmon_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def collect_power_samples(
    *,
    duration_seconds: float,
    interval_ms: int,
    startup_timeout_seconds: float = DEFAULT_POWER_STARTUP_TIMEOUT_SECONDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    non_json_lines: list[str] = []
    proc = subprocess.Popen(
        ["macmon", "pipe", "-i", str(interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    first_sample_deadline = time.perf_counter() + startup_timeout_seconds
    sample_deadline: float | None = None
    try:
        assert proc.stdout is not None
        while True:
            now = time.perf_counter()
            if sample_deadline is not None:
                if now >= sample_deadline:
                    break
                next_deadline = sample_deadline
            else:
                if now >= first_sample_deadline:
                    break
                next_deadline = first_sample_deadline
            timeout = max(0.0, min(0.5, next_deadline - now))
            ready, _, _ = select.select([proc.stdout], [], [], timeout)
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            row = parse_macmon_line(line)
            if row is None:
                non_json_lines.append(line.strip())
            else:
                samples.append(row)
                if sample_deadline is None:
                    sample_deadline = time.perf_counter() + duration_seconds
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return samples, non_json_lines


def summarize_power_samples(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for metric in POWER_METRICS:
        values = [float(row[metric]) for row in samples if isinstance(row.get(metric), int | float)]
        summary[metric] = metric_summary(values)
    return summary


def stream_until_stop(
    *,
    base_url: str,
    model_key: str,
    context_length: int,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    idle_timeout_seconds: int,
    first_output_event: threading.Event,
    stop_event: threading.Event,
    result_holder: dict[str, Any],
) -> None:
    native_root = native_api_root_from_base_url(base_url)
    payload = {
        "model": model_key,
        "input": prompt,
        "temperature": temperature,
        "store": False,
        "stream": True,
        "context_length": context_length,
        "max_output_tokens": max_output_tokens,
    }
    session = requests.Session()
    started_perf = time.perf_counter()
    current_event: str | None = None
    current_data: list[bytes] = []
    output_event_count = 0
    first_output_seconds: float | None = None
    final_result: dict[str, Any] | None = None

    def flush_event() -> None:
        nonlocal current_event, current_data, output_event_count, first_output_seconds, final_result
        if current_event is None and not current_data:
            return
        event_name, data = _parse_sse_event(current_event, current_data)
        elapsed = time.perf_counter() - started_perf
        if event_name in {"message.delta", "reasoning.delta"} and str(data.get("content") or ""):
            output_event_count += 1
            if first_output_seconds is None:
                first_output_seconds = elapsed
                first_output_event.set()
        elif event_name == "chat.end":
            final_result = data.get("result", {})
            stop_event.set()
        current_event = None
        current_data = []

    try:
        with session.post(
            f"{native_root}/chat",
            json=payload,
            timeout=(30, idle_timeout_seconds),
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=False):
                if stop_event.is_set():
                    break
                if raw_line is None:
                    continue
                line = raw_line.rstrip(b"\r")
                if not line:
                    flush_event()
                    continue
                if line.startswith(b":"):
                    continue
                if line.startswith(b"event:"):
                    current_event = line[len(b"event:") :].strip().decode("utf-8", errors="replace")
                elif line.startswith(b"data:"):
                    current_data.append(line[len(b"data:") :].lstrip())
            flush_event()
        result_holder.update(
            {
                "status": "completed" if final_result is not None else "stopped",
                "first_output_seconds": first_output_seconds,
                "output_event_count": output_event_count,
                "wall_time_seconds": time.perf_counter() - started_perf,
                "final_stats": (final_result or {}).get("stats", {}) if isinstance(final_result, dict) else {},
            }
        )
    except Exception as exc:  # noqa: BLE001
        result_holder.update(
            {
                "status": "failed",
                "error": str(exc),
                "first_output_seconds": first_output_seconds,
                "output_event_count": output_event_count,
                "wall_time_seconds": time.perf_counter() - started_perf,
            }
        )
        first_output_event.set()


def run_one_power_sample(
    *,
    base_url: str,
    model_key: str,
    context_length: int,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    idle_timeout_seconds: int,
    output_start_timeout_seconds: float,
    warmup_seconds: float,
    sample_seconds: float,
    interval_ms: int,
    startup_timeout_seconds: float,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    first_output_event = threading.Event()
    stop_event = threading.Event()
    generation_result: dict[str, Any] = {}
    power_started_at: str | None = None
    power_ended_at: str | None = None
    samples: list[dict[str, Any]] = []
    non_json_lines: list[str] = []

    preexisting = unload_all_models(base_url)
    load_record: dict[str, Any] | None = None
    unload_after: list[dict[str, Any]] = []
    try:
        load_record = load_model_with_timing(base_url, model_key, context_length)
        generation_thread = threading.Thread(
            target=stream_until_stop,
            kwargs={
                "base_url": base_url,
                "model_key": model_key,
                "context_length": context_length,
                "prompt": prompt,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "idle_timeout_seconds": idle_timeout_seconds,
                "first_output_event": first_output_event,
                "stop_event": stop_event,
                "result_holder": generation_result,
            },
            daemon=True,
        )
        generation_thread.start()
        if not first_output_event.wait(timeout=output_start_timeout_seconds):
            raise TimeoutError(f"{model_key} did not produce output within {output_start_timeout_seconds} seconds")
        if generation_result.get("status") == "failed":
            raise RuntimeError(str(generation_result.get("error") or "generation failed before sampling"))
        time.sleep(warmup_seconds)
        power_started_at = datetime.now(timezone.utc).isoformat()
        samples, non_json_lines = collect_power_samples(
            duration_seconds=sample_seconds,
            interval_ms=interval_ms,
            startup_timeout_seconds=startup_timeout_seconds,
        )
        power_ended_at = datetime.now(timezone.utc).isoformat()
        stop_event.set()
        generation_thread.join(timeout=10)
    finally:
        stop_event.set()
        unload_after = unload_all_models(base_url)

    row = {
        "model_key": model_key,
        "context_length": context_length,
        "prompt": prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "preexisting_loaded_models": preexisting,
        "load": load_record,
        "power_started_at": power_started_at,
        "power_ended_at": power_ended_at,
        "power_interval_ms": interval_ms,
        "power_sample_seconds": sample_seconds,
        "power_sample_count": len(samples),
        "power_summary": summarize_power_samples(samples),
        "generation": generation_result,
        "macmon_non_json_lines": non_json_lines[:20],
        "unloaded_after": unload_after,
    }
    write_json(out_dir / "run.json", row)
    (out_dir / "power_samples.jsonl").write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    return row


def summarize_power_runs(model_key: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_ci: dict[str, Any] = {}
    for metric in POWER_METRICS:
        run_means = [
            run.get("power_summary", {}).get(metric, {}).get("mean")
            for run in runs
        ]
        metric_ci[metric] = confidence_interval(
            [float(value) for value in run_means if isinstance(value, int | float)]
        )
    return {
        "model_key": model_key,
        "run_count": len(runs),
        "status": "completed" if all(run.get("power_sample_count", 0) > 0 for run in runs) else "partial",
        "metric_ci_over_run_means": metric_ci,
        "runs": runs,
    }


def run_power_tests(args: argparse.Namespace, run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    power_root = run_root / "power"
    for model_key in args.models:
        print(json.dumps({"event": "power_model_started", "model": model_key}), flush=True)
        model_runs: list[dict[str, Any]] = []
        model_root = power_root / slugify(model_key)
        for run_index in range(1, args.power_runs + 1):
            print(
                json.dumps({"event": "power_run_started", "model": model_key, "run": run_index}),
                flush=True,
            )
            try:
                run_row = run_one_power_sample(
                    base_url=args.base_url,
                    model_key=model_key,
                    context_length=args.context_length,
                    prompt=args.prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    idle_timeout_seconds=args.idle_timeout_seconds,
                    output_start_timeout_seconds=args.output_start_timeout_seconds,
                    warmup_seconds=args.power_warmup_seconds,
                    sample_seconds=args.power_sample_seconds,
                    interval_ms=args.power_interval_ms,
                    startup_timeout_seconds=args.power_startup_timeout_seconds,
                    out_dir=model_root / f"run_{run_index:02d}",
                )
                run_row["run_index"] = run_index
                model_runs.append(run_row)
                print(
                    json.dumps(
                        {
                            "event": "power_run_finished",
                            "model": model_key,
                            "run": run_index,
                            "sample_count": run_row["power_sample_count"],
                            "all_power_mean": run_row["power_summary"]["all_power"]["mean"],
                        }
                    ),
                    flush=True,
                )
            finally:
                if args.cooldown_seconds > 0:
                    time.sleep(args.cooldown_seconds)
        summary = summarize_power_runs(model_key, model_runs)
        write_json(model_root / "summary.json", summary)
        rows.append(summary)
        write_json(power_root / "power_summary.json", {"models": rows})
    write_power_markdown(power_root / "power_summary.md", rows, args)
    return rows


def memory_row_from_result(model_key: str, context_length: int, result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    snapshots = result.get("second_run_memory_snapshots") or []
    first_snapshot = snapshots[0] if snapshots else {}
    final_snapshot = snapshots[-1] if snapshots else {}
    final_memory_analysis = final_snapshot.get("memory_analysis") or {}
    final_footprint = final_snapshot.get("footprint") or {}
    return {
        "model_key": model_key,
        "context_length": context_length,
        "status": "completed" if snapshots else "completed_no_memory_snapshots",
        "result_path": str(result_path),
        "load": result.get("load"),
        "first_run_usage": (result.get("first_run_full_stats") or {}).get("usage"),
        "second_run_usage": (result.get("second_run_control") or {}).get("usage"),
        "snapshot_count": len(snapshots),
        "first_snapshot": first_snapshot,
        "final_snapshot": final_snapshot,
        "final_rss_bytes": final_snapshot.get("rss_bytes"),
        "final_vms_bytes": final_snapshot.get("vms_bytes"),
        "final_phys_footprint_bytes": final_footprint.get("phys_footprint_bytes"),
        "final_memory_analysis_resident_sum_bytes": final_memory_analysis.get("resident_sum_bytes"),
        "final_memory_analysis_swapped_sum_bytes": final_memory_analysis.get("swapped_sum_bytes"),
    }


def run_memory_tests(args: argparse.Namespace, run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    memory_root = run_root / "memory"
    for model_key in args.models:
        print(json.dumps({"event": "memory_model_started", "model": model_key}), flush=True)
        model_root = memory_root / slugify(model_key) / f"ctx_{args.context_length}"
        try:
            result = run_generation_perf_pair(
                base_url=args.base_url,
                model_key=model_key,
                context_length=args.context_length,
                prompt=args.prompt,
                temperature=args.temperature,
                snapshot_interval_seconds=args.memory_snapshot_interval_seconds,
                idle_timeout_seconds=args.idle_timeout_seconds,
                capture_memory=True,
                max_output_tokens=args.max_output_tokens,
                speed_runs=args.memory_speed_runs,
                compact_output_timing=True,
            )
            write_json(model_root / "result.json", result)
            write_json(model_root / "load.json", result["load"])
            write_json(model_root / "run1_speed.json", result["first_run_full_stats"])
            write_json(model_root / "run2_control.json", result["second_run_control"])
            write_json(model_root / "run2_memory_snapshots.json", result["second_run_memory_snapshots"])
            row = memory_row_from_result(model_key, args.context_length, result, model_root / "result.json")
        except Exception as exc:  # noqa: BLE001
            unload_all_models(args.base_url)
            row = {
                "model_key": model_key,
                "context_length": args.context_length,
                "status": "failed",
                "error": str(exc),
            }
            write_json(model_root / "error.json", row)
        rows.append(row)
        write_json(memory_root / "memory_summary.json", {"models": rows})
        print(
            json.dumps(
                {
                    "event": "memory_model_finished",
                    "model": model_key,
                    "status": row["status"],
                    "snapshot_count": row.get("snapshot_count"),
                }
            ),
            flush=True,
        )
    write_memory_markdown(memory_root / "memory_summary.md", rows, args)
    return rows


def bytes_to_gib(value: Any) -> str:
    if not isinstance(value, int | float):
        return ""
    return f"{float(value) / (1024**3):.2f}"


def write_power_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Power Test Summary",
        "",
        f"- Context length: `{args.context_length}`",
        f"- Temperature: `{args.temperature}`",
        f"- Runs per model: `{args.power_runs}`",
        f"- macmon interval: `{args.power_interval_ms} ms`",
        f"- Warmup after first output: `{args.power_warmup_seconds} s`",
        f"- Sampling duration: `{args.power_sample_seconds} s`",
        f"- macmon startup timeout: `{args.power_startup_timeout_seconds} s`",
        f"- Cooldown after unload: `{args.cooldown_seconds} s`",
        "",
        "| Model | Runs | Status | all_power mean | all_power 95% CI | gpu_power mean | gpu_power 95% CI | ram_power mean | ram_power 95% CI | sys_power mean | sys_power 95% CI |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        ci = row.get("metric_ci_over_run_means", {})

        def mean(metric: str) -> str:
            value = (ci.get(metric) or {}).get("mean")
            return f"{value:.3f}" if isinstance(value, int | float) else ""

        def half(metric: str) -> str:
            value = (ci.get(metric) or {}).get("ci_half_width")
            return f"±{value:.3f}" if isinstance(value, int | float) else ""

        lines.append(
            f"| {row['model_key']} | {row['run_count']} | {row['status']} | "
            f"{mean('all_power')} | {half('all_power')} | "
            f"{mean('gpu_power')} | {half('gpu_power')} | "
            f"{mean('ram_power')} | {half('ram_power')} | "
            f"{mean('sys_power')} | {half('sys_power')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_memory_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Memory Test Summary",
        "",
        f"- Context length: `{args.context_length}`",
        f"- Temperature: `{args.temperature}`",
        f"- Prompt: `{args.prompt}`",
        f"- Max output tokens: `{args.max_output_tokens}`",
        "",
        "| Model | Status | Snapshots | Final RSS (GiB) | Final VMS (GiB) | Final Footprint (GiB) | MemoryAnalysis Resident Sum (GiB) | MemoryAnalysis Swapped Sum (GiB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_key']} | {row['status']} | {row.get('snapshot_count') or 0} | "
            f"{bytes_to_gib(row.get('final_rss_bytes'))} | "
            f"{bytes_to_gib(row.get('final_vms_bytes'))} | "
            f"{bytes_to_gib(row.get('final_phys_footprint_bytes'))} | "
            f"{bytes_to_gib(row.get('final_memory_analysis_resident_sum_bytes'))} | "
            f"{bytes_to_gib(row.get('final_memory_analysis_swapped_sum_bytes'))} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_plan(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan = {
        "run_id": args.run_id,
        "mode": args.mode,
        "base_url": args.base_url,
        "models": args.models,
        "context_length": args.context_length,
        "temperature": args.temperature,
        "prompt": args.prompt,
        "power_runs": args.power_runs,
        "power_interval_ms": args.power_interval_ms,
        "power_warmup_seconds": args.power_warmup_seconds,
        "power_sample_seconds": args.power_sample_seconds,
        "power_startup_timeout_seconds": args.power_startup_timeout_seconds,
        "cooldown_seconds": args.cooldown_seconds,
        "memory_snapshot_interval_seconds": args.memory_snapshot_interval_seconds,
        "memory_speed_runs": args.memory_speed_runs,
        "max_output_tokens": args.max_output_tokens,
        "dry_run": args.dry_run,
    }
    write_json(path, plan)
    return plan


def main() -> int:
    args = parse_args()
    args.run_id = args.run_id or f"{utc_timestamp()}_power_memory_test"
    run_root = args.output_root.resolve() / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    plan = write_run_plan(run_root / "run_plan.json", args)
    if args.dry_run:
        print(json.dumps({"event": "dry_run", "run_root": str(run_root), "plan": plan}, ensure_ascii=False))
        return 0

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "base_url": args.base_url,
        "models": args.models,
        "context_length": args.context_length,
        "mode": args.mode,
        "power": None,
        "memory": None,
    }
    if args.mode in {"power", "both"}:
        summary["power"] = run_power_tests(args, run_root)
        write_json(run_root / "summary.json", summary)
    if args.mode in {"memory", "both"}:
        summary["memory"] = run_memory_tests(args, run_root)
        write_json(run_root / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
