from __future__ import annotations

import argparse
import time
from pathlib import Path

from .benchmarks.registry import BENCHMARK_REGISTRY
from .client import LMStudioClient
from .config import BENCHMARK_ORDER, CACHE_ROOT, DATA_ROOT, DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_SEED, RESULTS_ROOT
from .reporting import enrich_benchmark_summary, write_benchmark_markdown, write_benchmark_summary, write_run_markdown
from .types import RunConfig
from .utils import ensure_dir, utc_timestamp, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all benchmarks against LM Studio.")
    parser.add_argument("--profile", choices=["probe", "smoke", "initial", "mixed"], default="initial")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--external-root", type=Path, default=Path(__file__).resolve().parents[1] / "external")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or utc_timestamp()
    run_root = ensure_dir(args.output_root / run_id)
    client = LMStudioClient(base_url=args.base_url, model=args.model)
    summaries = []
    run_started = time.perf_counter()
    for benchmark in BENCHMARK_ORDER:
        benchmark_started = time.perf_counter()
        config = RunConfig(
            benchmark=benchmark,
            profile=args.profile,
            base_url=args.base_url,
            model=args.model,
            seed=args.seed,
            run_id=run_id,
            output_root=run_root,
            data_root=args.data_root,
            cache_root=args.cache_root,
            external_root=args.external_root,
        )
        try:
            summary = BENCHMARK_REGISTRY[benchmark](config, client)
        except Exception as exc:  # noqa: BLE001
            summary = {
                "benchmark": benchmark,
                "profile": args.profile,
                "status": "failed",
                "model": args.model,
                "base_url": args.base_url,
                "sample_count": 0,
                "metric": "N/A",
                "score": None,
                "correct_count": None,
                "notes": [f"Runner failed: {exc}"],
                "repo": "",
                "dataset": "",
            }
        duration_seconds = time.perf_counter() - benchmark_started
        benchmark_dir = run_root / benchmark
        summary, predictions = enrich_benchmark_summary(summary, benchmark_dir, duration_seconds)
        summaries.append(summary)
        write_benchmark_summary(benchmark_dir / "summary.json", summary)
        write_benchmark_markdown(benchmark_dir / "summary.md", summary, predictions)
        print(f"[done] {benchmark}: status={summary['status']} score={summary.get('score')}")

    run_summary = {
        "run_id": run_id,
        "profile": args.profile,
        "model": args.model,
        "base_url": args.base_url,
        "time_total_seconds": time.perf_counter() - run_started,
        "benchmarks": summaries,
    }
    write_json(run_root / "summary.json", run_summary)
    write_run_markdown(run_root / "summary.md", run_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
