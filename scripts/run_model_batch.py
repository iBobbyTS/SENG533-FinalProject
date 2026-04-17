#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_BASE_URL = "http://127.0.0.1:1234/api/v1"
DEFAULT_PROFILE = "initial"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_suite.config import RECOMMENDED_BENCHMARK_CONTEXTS

# Edit this list to add or remove model/context combinations.
# The integer value is the fallback context length if a benchmark is not listed
# in BENCHMARK_CONTEXTS.
MODEL_RUNS: list[tuple[str, int]] = [
    ("qwen3.5-27b-claude-4.6-opus-reasoning-distilled", 262144),  # Q6_K
]

BENCHMARK_CONTEXTS: dict[str, int] = dict(RECOMMENDED_BENCHMARK_CONTEXTS)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def slugify_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def select_runner_python() -> str:
    candidate = PROJECT_ROOT / ".venv-bench" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark_suite.run_all_monitored for a predefined list of model/context pairs."
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=["probe", "smoke", "initial", "mixed"])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--batch-run-id", default=None)
    parser.add_argument("--disable-power-monitoring", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def summarize_run(summary: dict[str, Any]) -> dict[str, Any]:
    benchmarks = summary.get("benchmarks", [])
    benchmark_statuses = {item.get("benchmark", "unknown"): item.get("status", "unknown") for item in benchmarks}
    completed = sum(1 for item in benchmarks if item.get("status") == "completed")
    failed = sum(1 for item in benchmarks if item.get("status") == "failed")
    return {
        "benchmark_count": len(benchmarks),
        "completed_count": completed,
        "failed_count": failed,
        "benchmark_statuses": benchmark_statuses,
    }


def write_batch_markdown(batch_root: Path, batch_summary: dict[str, Any]) -> None:
    lines = [
        "# Batch Benchmark Summary",
        "",
        f"- Batch run ID: `{batch_summary['batch_run_id']}`",
        f"- Profile: `{batch_summary['profile']}`",
        f"- Base URL: `{batch_summary['base_url']}`",
        f"- Power monitoring: `{'disabled' if batch_summary['disable_power_monitoring'] else 'enabled'}`",
        f"- Benchmark contexts: `{json.dumps(batch_summary['benchmark_contexts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "| Model | Fallback Context | Return Code | Run Status | Completed | Failed | Run ID | Result Directory |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in batch_summary["runs"]:
        lines.append(
            f"| {row['model']} | {row['fallback_context_length']} | {row['returncode']} | {row['run_status']} | "
            f"{row['completed_count']} | {row['failed_count']} | {row['run_id']} | `{row['result_dir']}` |"
        )
    (batch_root / "batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_model(
    *,
    runner_python: str,
    profile: str,
    base_url: str,
    results_root: Path,
    batch_root: Path,
    model: str,
    fallback_context_length: int,
    benchmark_contexts: dict[str, int],
    disable_power_monitoring: bool,
) -> dict[str, Any]:
    run_id = f"{utc_timestamp()}_{slugify_model_name(model)}_ctxmulti"
    result_dir = results_root / run_id
    command = [
        runner_python,
        "-m",
        "benchmark_suite.run_all_monitored",
        "--profile",
        profile,
        "--base-url",
        base_url,
        "--model",
        model,
        "--run-id",
        run_id,
        "--output-root",
        str(results_root),
        "--per-benchmark-contexts",
        json.dumps(benchmark_contexts, separators=(",", ":")),
    ]
    if fallback_context_length is not None:
        command.extend(["--context-length", str(fallback_context_length)])
    if disable_power_monitoring:
        command.append("--disable-power-monitoring")

    started_at = datetime.now(timezone.utc).isoformat()
    started_perf = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "run_started",
                "model": model,
                "fallback_context_length": fallback_context_length,
                "benchmark_contexts": benchmark_contexts,
                "run_id": run_id,
            }
        )
    )
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    duration_seconds = time.perf_counter() - started_perf
    ended_at = datetime.now(timezone.utc).isoformat()

    summary_path = result_dir / "summary.json"
    run_status = "missing_summary"
    completed_count = 0
    failed_count = 0
    benchmark_statuses: dict[str, Any] = {}
    if summary_path.exists():
        summary = load_json(summary_path)
        run_stats = summarize_run(summary)
        benchmark_statuses = run_stats["benchmark_statuses"]
        completed_count = run_stats["completed_count"]
        failed_count = run_stats["failed_count"]
        if failed_count == 0 and completed_count == run_stats["benchmark_count"]:
            run_status = "completed"
        else:
            run_status = "failed"
    elif completed.returncode != 0:
        run_status = "failed"

    row = {
        "model": model,
        "fallback_context_length": fallback_context_length,
        "benchmark_contexts": benchmark_contexts,
        "run_id": run_id,
        "result_dir": str(result_dir),
        "summary_path": str(summary_path),
        "returncode": completed.returncode,
        "run_status": run_status,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "benchmark_statuses": benchmark_statuses,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "command": command,
    }
    write_json(batch_root / f"{run_id}.json", row)
    print(
        json.dumps(
            {
                "event": "run_finished",
                "model": model,
                "fallback_context_length": fallback_context_length,
                "benchmark_contexts": benchmark_contexts,
                "run_id": run_id,
                "returncode": completed.returncode,
                "run_status": run_status,
            }
        )
    )
    return row


def main() -> int:
    args = parse_args()
    batch_run_id = args.batch_run_id or f"{utc_timestamp()}_model_batch"
    batch_root = args.results_root / batch_run_id
    batch_root.mkdir(parents=True, exist_ok=True)

    batch_summary: dict[str, Any] = {
        "batch_run_id": batch_run_id,
        "profile": args.profile,
        "base_url": args.base_url,
        "disable_power_monitoring": args.disable_power_monitoring,
        "models": MODEL_RUNS,
        "benchmark_contexts": BENCHMARK_CONTEXTS,
        "runs": [],
    }
    write_json(batch_root / "batch_config.json", batch_summary)

    runner_python = select_runner_python()
    overall_returncode = 0

    for model, fallback_context_length in MODEL_RUNS:
        row = run_one_model(
            runner_python=runner_python,
            profile=args.profile,
            base_url=args.base_url,
            results_root=args.results_root,
            batch_root=batch_root,
            model=model,
            fallback_context_length=fallback_context_length,
            benchmark_contexts=BENCHMARK_CONTEXTS,
            disable_power_monitoring=args.disable_power_monitoring,
        )
        batch_summary["runs"].append(row)
        write_json(batch_root / "batch_summary.json", batch_summary)
        write_batch_markdown(batch_root, batch_summary)
        if row["returncode"] != 0:
            overall_returncode = 1
            if args.stop_on_failure:
                break

    return overall_returncode


if __name__ == "__main__":
    raise SystemExit(main())
