from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from .benchmarks.registry import BENCHMARK_REGISTRY
from .client import LMStudioClient
from .config import CACHE_ROOT, DATA_ROOT, DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_SEED, RESULTS_ROOT
from .reporting import enrich_benchmark_summary, write_benchmark_markdown, write_benchmark_summary, write_run_markdown
from .types import RunConfig
from .utils import ensure_dir, utc_timestamp, write_json

BENCHMARK_WEIGHTS = {
    "mmlu_pro": 0.4,
    "gsm8k": 0.6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 10-seed reliability trials for knowledge and reasoning.")
    parser.add_argument("--profile", choices=["probe", "smoke", "initial", "mixed"], default="smoke")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trial-count", type=int, default=10)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--external-root", type=Path, default=Path(__file__).resolve().parents[1] / "external")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    return parser.parse_args()


def _mse(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"{utc_timestamp()}_reliability"
    run_root = ensure_dir(args.output_root / run_id)
    rng = random.Random(args.seed)
    trial_seeds = rng.sample(range(1, 10**9), args.trial_count)
    benchmark_scores: dict[str, list[float]] = {name: [] for name in BENCHMARK_WEIGHTS}
    trial_records = []
    run_started = time.perf_counter()

    for trial_index, trial_seed in enumerate(trial_seeds, start=1):
        trial_dir = ensure_dir(run_root / f"trial_{trial_index:02d}")
        client = LMStudioClient(base_url=args.base_url, model=args.model)
        trial_summary = {
            "trial_index": trial_index,
            "seed": trial_seed,
            "benchmarks": [],
        }
        for benchmark in BENCHMARK_WEIGHTS:
            config = RunConfig(
                benchmark=benchmark,
                profile=args.profile,
                base_url=args.base_url,
                model=args.model,
                seed=trial_seed,
                run_id=f"{run_id}_trial_{trial_index:02d}",
                output_root=trial_dir,
                data_root=args.data_root,
                cache_root=args.cache_root,
                external_root=args.external_root,
            )
            benchmark_started = time.perf_counter()
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
            benchmark_dir = trial_dir / benchmark
            summary, predictions = enrich_benchmark_summary(summary, benchmark_dir, time.perf_counter() - benchmark_started)
            write_benchmark_summary(benchmark_dir / "summary.json", summary)
            write_benchmark_markdown(benchmark_dir / "summary.md", summary, predictions)
            if isinstance(summary.get("score"), (int, float)):
                benchmark_scores[benchmark].append(float(summary["score"]))
            trial_summary["benchmarks"].append(summary)
            print(
                f"[reliability] trial={trial_index:02d} seed={trial_seed} "
                f"{benchmark}: status={summary['status']} score={summary.get('score')}"
            )
        trial_records.append(trial_summary)

    mse_by_benchmark = {benchmark: _mse(scores) for benchmark, scores in benchmark_scores.items()}
    weighted_mse = None
    if all(mse_by_benchmark[benchmark] is not None for benchmark in BENCHMARK_WEIGHTS):
        weighted_mse = sum(mse_by_benchmark[benchmark] * weight for benchmark, weight in BENCHMARK_WEIGHTS.items())

    summary = {
        "run_id": run_id,
        "profile": args.profile,
        "model": args.model,
        "base_url": args.base_url,
        "trial_count": args.trial_count,
        "trial_seeds": trial_seeds,
        "weights": BENCHMARK_WEIGHTS,
        "mse_by_benchmark": mse_by_benchmark,
        "weighted_mse": weighted_mse,
        "trial_scores": benchmark_scores,
        "trials": trial_records,
        "time_total_seconds": time.perf_counter() - run_started,
        "notes": [
            "Reliability only; Robustness is intentionally excluded.",
            "MSE is computed against the mean of the 10 trial scores for each benchmark.",
            "Knowledge uses MMLU-Pro with weight 0.4; Reasoning uses GSM8K with weight 0.6.",
        ],
    }
    write_json(run_root / "summary.json", summary)
    write_run_markdown(
        run_root / "summary.md",
        {
            "run_id": run_id,
            "profile": f"reliability:{args.profile}",
            "model": args.model,
            "base_url": args.base_url,
            "time_total_seconds": time.perf_counter() - run_started,
            "benchmarks": [
                {
                    "benchmark": benchmark,
                    "status": "completed" if mse_by_benchmark[benchmark] is not None else "partial",
                    "score": mse_by_benchmark[benchmark],
                    "sample_count": len(benchmark_scores[benchmark]),
                    "metric": "mean squared error across trial scores",
                    "mean_decode_tps": None,
                    "mean_approx_prefill_tps": None,
                    "time_total_seconds": None,
                }
                for benchmark in BENCHMARK_WEIGHTS
            ]
            + [
                {
                    "benchmark": "weighted_reliability",
                    "status": "completed" if weighted_mse is not None else "partial",
                    "score": weighted_mse,
                    "sample_count": args.trial_count,
                    "metric": "0.4*MMLU-Pro MSE + 0.6*GSM8K MSE",
                    "mean_decode_tps": None,
                    "mean_approx_prefill_tps": None,
                    "time_total_seconds": None,
                }
            ],
        },
    )
    print(f"[done] reliability: weighted_mse={weighted_mse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
