from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .config import (
    BENCHMARK_ORDER,
    CACHE_ROOT,
    DATA_ROOT,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    RESULTS_ROOT,
)
from .model_management import managed_model
from .reporting import enrich_benchmark_summary, write_run_markdown
from .utils import ensure_dir, utc_timestamp, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all benchmarks with memory and power monitoring.")
    parser.add_argument("--profile", choices=["probe", "smoke", "initial", "mixed"], default="initial")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--external-root", type=Path, default=Path(__file__).resolve().parents[1] / "external")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--interval-ms", type=int, default=500, help="Memory sampling interval in milliseconds.")
    parser.add_argument("--power-interval-ms", type=int, default=1000, help="macmon power sampling interval in milliseconds.")
    parser.add_argument("--skip-model-management", action="store_true")
    parser.add_argument("--disable-power-monitoring", action="store_true")
    parser.add_argument(
        "--per-benchmark-contexts",
        default=None,
        help="JSON object mapping benchmark names to context lengths, e.g. {\"mmlu_pro\":8192}.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_context_map(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--per-benchmark-contexts must be a JSON object.")
    context_map: dict[str, int] = {}
    for key, value in parsed.items():
        if key not in BENCHMARK_ORDER:
            raise ValueError(f"Unsupported benchmark in context map: {key}")
        if not isinstance(value, int):
            raise ValueError(f"Context length for {key} must be an integer.")
        context_map[key] = value
    return context_map


def wait_for_power_stream(power_path: Path, power_proc: subprocess.Popen[str], timeout_seconds: float = 5.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if power_path.exists():
            lines = [line.strip() for line in power_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                try:
                    json.loads(lines[0])
                    return
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"macmon failed to start: {lines[0]}") from exc
        if power_proc.poll() is not None:
            break
        time.sleep(0.2)

    if power_path.exists():
        lines = [line.strip() for line in power_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            try:
                json.loads(lines[0])
                return
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"macmon failed to start: {lines[0]}") from exc
    raise RuntimeError("macmon did not produce a valid power sample during startup.")


def main() -> int:
    args = parse_args()
    context_map = parse_context_map(args.per_benchmark_contexts)
    if context_map and args.skip_model_management:
        raise ValueError("Cannot combine --per-benchmark-contexts with --skip-model-management.")
    run_id = args.run_id or utc_timestamp()
    run_root = ensure_dir(args.output_root / run_id)
    monitored_root = ensure_dir(run_root / "monitored")
    power_path = monitored_root / "power.jsonl"
    run_started = time.perf_counter()
    power_handle = None
    power_proc = None
    if not args.disable_power_monitoring:
        power_handle = power_path.open("w", encoding="utf-8")
        power_proc = subprocess.Popen(
            ["macmon", "pipe", "-i", str(args.power_interval_ms)],
            stdout=power_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    row_paths: list[Path] = []

    try:
        if power_proc is not None:
            wait_for_power_stream(power_path, power_proc)
        use_outer_model_management = not args.skip_model_management and not context_map
        with managed_model(
            args.base_url,
            args.model,
            context_length=args.context_length,
            enabled=use_outer_model_management,
        ):
            for benchmark in BENCHMARK_ORDER:
                meta_path = monitored_root / f"{benchmark}.meta.json"
                memory_path = monitored_root / f"{benchmark}.memory.json"
                row_path = monitored_root / f"{benchmark}.row.json"
                benchmark_context = context_map.get(benchmark, args.context_length)
                benchmark_cmd = [
                    sys.executable,
                    "-m",
                    "benchmark_suite.run_monitored_benchmark",
                    "--benchmark",
                    benchmark,
                    "--profile",
                    args.profile,
                    "--model",
                    args.model,
                    "--base-url",
                    args.base_url,
                    "--seed",
                    str(args.seed),
                    "--results-root",
                    str(args.output_root),
                    "--run-id",
                    run_id,
                    "--meta-out",
                    str(meta_path),
                    "--memory-out",
                    str(memory_path),
                    "--interval-ms",
                    str(args.interval_ms),
                ]
                if benchmark_context is not None:
                    benchmark_cmd.extend(["--context-length", str(benchmark_context)])
                if use_outer_model_management:
                    benchmark_cmd.append("--skip-model-management")
                if args.disable_power_monitoring:
                    benchmark_cmd.append("--disable-power-monitoring")
                subprocess.run(benchmark_cmd, check=False)
                if power_handle is not None:
                    power_handle.flush()
                finalize_cmd = [
                    sys.executable,
                    "-m",
                    "benchmark_suite.finalize_monitored_run",
                    "--meta",
                    str(meta_path),
                    "--memory",
                    str(memory_path),
                    "--power",
                    str(power_path),
                    "--out",
                    str(row_path),
                ]
                subprocess.run(finalize_cmd, check=True)
                row_paths.append(row_path)
    finally:
        if power_proc is not None:
            power_proc.terminate()
            try:
                power_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                power_proc.kill()
        if power_handle is not None:
            power_handle.close()

    benchmark_summaries = []
    for row_path in row_paths:
        row = load_json(row_path)
        benchmark_summary_path = Path(row["benchmark_summary"])
        benchmark_summary = load_json(benchmark_summary_path)
        enriched_summary, _ = enrich_benchmark_summary(
            benchmark_summary,
            benchmark_summary_path.parent,
            row.get("duration_seconds"),
        )
        enriched_summary["memory_max_bytes"] = row.get("memory_max_bytes")
        enriched_summary["memory_mean_bytes"] = row.get("memory_mean_bytes")
        enriched_summary["context_length"] = row.get("context_length")
        enriched_summary["virtual_max_bytes"] = row.get("virtual_max_bytes")
        enriched_summary["virtual_mean_bytes"] = row.get("virtual_mean_bytes")
        enriched_summary["swap_used_max_bytes"] = row.get("swap_used_max_bytes")
        enriched_summary["swap_used_mean_bytes"] = row.get("swap_used_mean_bytes")
        enriched_summary["power_max_watts"] = row.get("power_max_watts")
        enriched_summary["power_mean_watts"] = row.get("power_mean_watts")
        enriched_summary["gpu_power_max_watts"] = row.get("gpu_power_max_watts")
        enriched_summary["gpu_power_mean_watts"] = row.get("gpu_power_mean_watts")
        enriched_summary["ram_power_max_watts"] = row.get("ram_power_max_watts")
        enriched_summary["ram_power_mean_watts"] = row.get("ram_power_mean_watts")
        enriched_summary["sys_power_max_watts"] = row.get("sys_power_max_watts")
        enriched_summary["sys_power_mean_watts"] = row.get("sys_power_mean_watts")
        benchmark_summaries.append(enriched_summary)

    run_summary = {
        "run_id": run_id,
        "profile": args.profile,
        "model": args.model,
        "base_url": args.base_url,
        "time_total_seconds": time.perf_counter() - run_started,
        "benchmarks": benchmark_summaries,
    }
    write_json(run_root / "summary.json", run_summary)
    write_run_markdown(run_root / "summary.md", run_summary)
    write_run_markdown(monitored_root / "summary.md", run_summary)
    return 0 if all(item["status"] != "failed" for item in benchmark_summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
