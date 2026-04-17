from __future__ import annotations

import argparse
import time
from pathlib import Path

from .benchmarks.registry import BENCHMARK_REGISTRY
from .client import LMStudioClient
from .config import CACHE_ROOT, DATA_ROOT, DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_SEED, RESULTS_ROOT
from .model_management import managed_model
from .reporting import enrich_benchmark_summary, write_benchmark_markdown, write_benchmark_summary
from .types import RunConfig
from .utils import ensure_dir, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one benchmark against LM Studio.")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_REGISTRY), required=True)
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
    parser.add_argument("--skip-model-management", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or utc_timestamp()
    run_root = ensure_dir(args.output_root / run_id)
    client = LMStudioClient(base_url=args.base_url, model=args.model)
    config = RunConfig(
        benchmark=args.benchmark,
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
    started = time.perf_counter()
    try:
        with managed_model(
            args.base_url,
            args.model,
            context_length=args.context_length,
            enabled=not args.skip_model_management,
        ):
            summary = BENCHMARK_REGISTRY[args.benchmark](config, client)
    except Exception as exc:  # noqa: BLE001
        summary = {
            "benchmark": args.benchmark,
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
    benchmark_dir = run_root / args.benchmark
    summary, predictions = enrich_benchmark_summary(summary, benchmark_dir, time.perf_counter() - started)
    write_benchmark_summary(benchmark_dir / "summary.json", summary)
    write_benchmark_markdown(benchmark_dir / "summary.md", summary, predictions)
    print(f"[done] {args.benchmark}: status={summary['status']} score={summary.get('score')}")
    return 0 if summary["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
