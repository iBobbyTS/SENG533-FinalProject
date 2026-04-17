#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_MODEL_PROGRESS = RESULTS_ROOT / "20260330_model_progress.md"
DEFAULT_BASE_URL = "http://192.168.31.76:1234/api/v1"
DEFAULT_CONTEXTS = [32768]
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 60
DEFAULT_IDLE_TIMEOUT_SECONDS = 600

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_suite.model_stream_perf import (  # noqa: E402
    DEFAULT_PROMPT,
    render_batch_markdown,
    resolve_explicit_model_targets,
    resolve_recorded_model_targets,
    run_generation_perf_pair,
    slugify,
    utc_timestamp,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run long-generation performance sampling for recorded LM Studio models. "
            "For each target model and context, the script loads the model, always runs "
            "one streaming chat, and optionally runs a second chat with minute-level "
            "memory snapshots."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-progress", type=Path, default=DEFAULT_MODEL_PROGRESS)
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=DEFAULT_CONTEXTS,
        help="Context lengths to test. Default: 4096 65536",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--snapshot-interval-seconds", type=int, default=DEFAULT_SNAPSHOT_INTERVAL_SECONDS)
    parser.add_argument("--idle-timeout-seconds", type=int, default=DEFAULT_IDLE_TIMEOUT_SECONDS)
    parser.add_argument(
        "--capture-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture LM Studio backend memory snapshots during the second run. "
            "Enabled by default; use --no-capture-memory when the backend is on a remote host."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional safety cap for generated output. Default: no explicit cap; LM Studio GUI limits apply.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional explicit LM Studio model keys to run. If omitted, use the models recorded in "
            "the model progress table, with the 9B family reduced to the currently selected GUI variant."
        ),
    )
    parser.add_argument(
        "--expand-model-variants",
        action="store_true",
        help=(
            "When --models includes a multi-variant LM Studio model key, run every listed variant. "
            "For qwen/qwen3.5-9b this expands to @q4_k_m, @q6_k, and @q8_0."
        ),
    )
    parser.add_argument(
        "--speed-runs",
        type=int,
        default=1,
        help="Number of streaming speed runs to execute before the optional memory run. Default: 1.",
    )
    parser.add_argument(
        "--compact-output-timing",
        action="store_true",
        help=(
            "Store only output event timestamps instead of full delta event rows and raw final output. "
            "This reduces JSON size and CPU overhead during long generations."
        ),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def select_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.models:
        targets = resolve_explicit_model_targets(
            args.models,
            args.base_url,
            expand_model_variants=args.expand_model_variants,
        )
    else:
        targets = resolve_recorded_model_targets(args.model_progress.resolve(), args.base_url)
    return [
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


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"{utc_timestamp()}_stream_perf_batch"
    output_root = args.output_root.resolve()
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    selected_targets = select_targets(args)
    batch_summary: dict[str, Any] = {
        "run_id": run_id,
        "base_url": args.base_url,
        "prompt": args.prompt,
        "contexts": args.contexts,
        "snapshot_interval_seconds": args.snapshot_interval_seconds,
        "idle_timeout_seconds": args.idle_timeout_seconds,
        "capture_memory": args.capture_memory,
        "max_output_tokens": args.max_output_tokens,
        "expand_model_variants": args.expand_model_variants,
        "speed_runs": args.speed_runs,
        "compact_output_timing": args.compact_output_timing,
        "targets": selected_targets,
        "results": [],
    }
    write_json(run_root / "batch_config.json", batch_summary)

    overall_returncode = 0
    for target in selected_targets:
        model_key = target["model_key"]
        model_root = run_root / slugify(model_key)
        for context_length in args.contexts:
            context_root = model_root / f"ctx_{context_length}"
            result_row = {
                **target,
                "context_length": context_length,
                "status": "failed",
            }
            print(
                json.dumps(
                    {
                        "event": "model_context_started",
                        "model": model_key,
                        "context_length": context_length,
                        "run_id": run_id,
                        "speed_runs": args.speed_runs,
                        "capture_memory": args.capture_memory,
                        "compact_output_timing": args.compact_output_timing,
                    }
                ),
                flush=True,
            )
            try:
                result = run_generation_perf_pair(
                    base_url=args.base_url,
                    model_key=model_key,
                    context_length=context_length,
                    prompt=args.prompt,
                    snapshot_interval_seconds=args.snapshot_interval_seconds,
                    idle_timeout_seconds=args.idle_timeout_seconds,
                    capture_memory=args.capture_memory,
                    max_output_tokens=args.max_output_tokens,
                    speed_runs=args.speed_runs,
                    compact_output_timing=args.compact_output_timing,
                )
                write_json(context_root / "result.json", result)
                write_json(context_root / "load.json", result["load"])
                write_json(context_root / "run1_full.json", result["first_run_full_stats"])
                for index, speed_run in enumerate(result["speed_runs"], start=1):
                    write_json(context_root / f"run{index}_speed.json", speed_run)
                write_json(context_root / "run2_control.json", result["second_run_control"])
                write_json(context_root / "run2_memory_snapshots.json", result["second_run_memory_snapshots"])
                write_json(context_root / "meta.json", {
                    "model": model_key,
                    "context_length": context_length,
                    "prompt": args.prompt,
                    "source": target["source"],
                    "recorded_model_key": target["recorded_model_key"],
                    "summary_path": target["summary_path"],
                })

                result_row.update(
                    {
                        "status": "completed",
                        "result_path": str(context_root / "result.json"),
                        "load_duration_seconds": result["load"]["load_duration_seconds"],
                        "load_strategy": result["load"].get("strategy"),
                        "speed_run_count": len(result["speed_runs"]),
                        "speed_run_usages": [item.get("usage", {}) for item in result["speed_runs"]],
                        "first_run_usage": result["first_run_full_stats"]["usage"],
                        "second_run_usage": result["second_run_control"]["usage"],
                        "memory_snapshot_count": len(result["second_run_memory_snapshots"]),
                        "memory_snapshot_final": result["second_run_memory_snapshots"][-1] if result["second_run_memory_snapshots"] else {},
                    }
                )
                output_tokens = result["first_run_full_stats"]["usage"].get("completion_tokens")
                speed_run_output_tokens = [
                    item.get("usage", {}).get("completion_tokens") for item in result["speed_runs"]
                ]
            except Exception as exc:  # noqa: BLE001
                overall_returncode = 1
                error_payload = {
                    "model": model_key,
                    "context_length": context_length,
                    "error": str(exc),
                }
                write_json(context_root / "error.json", error_payload)
                result_row.update(
                    {
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                output_tokens = None
                speed_run_output_tokens = []
                if args.stop_on_failure:
                    batch_summary["results"].append(result_row)
                    write_json(run_root / "batch_summary.json", batch_summary)
                    (run_root / "batch_summary.md").write_text(render_batch_markdown(batch_summary), encoding="utf-8")
                    return overall_returncode
            batch_summary["results"].append(result_row)
            write_json(run_root / "batch_summary.json", batch_summary)
            (run_root / "batch_summary.md").write_text(render_batch_markdown(batch_summary), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "event": "model_context_finished",
                        "model": model_key,
                        "context_length": context_length,
                        "run_id": run_id,
                        "status": result_row["status"],
                        "output_tokens": output_tokens,
                        "speed_run_output_tokens": speed_run_output_tokens,
                    }
                ),
                flush=True,
            )

    return overall_returncode


if __name__ == "__main__":
    raise SystemExit(main())
