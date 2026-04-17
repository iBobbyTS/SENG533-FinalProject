from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reporting import enrich_benchmark_summary, write_run_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate monitored benchmark rows into Markdown.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    rows = [load_json(path) for path in args.inputs]
    rows.sort(key=lambda row: row["benchmark"])
    benchmark_summaries = []
    for row in rows:
        benchmark_summary_path = Path(row["benchmark_summary"])
        benchmark_summary = load_json(benchmark_summary_path)
        enriched_summary, _ = enrich_benchmark_summary(
            benchmark_summary,
            benchmark_summary_path.parent,
            row.get("duration_seconds"),
        )
        enriched_summary["memory_max_bytes"] = row.get("memory_max_bytes")
        enriched_summary["memory_mean_bytes"] = row.get("memory_mean_bytes")
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

    run_root = Path(rows[0]["benchmark_summary"]).resolve().parents[1] if rows else None
    run_summary = {
        "run_id": run_root.name if run_root else "monitored_run",
        "profile": rows[0]["profile"] if rows else "N/A",
        "model": rows[0]["model"] if rows else "N/A",
        "base_url": benchmark_summaries[0].get("base_url", "N/A") if benchmark_summaries else "N/A",
        "time_total_seconds": sum(
            float(row["duration_seconds"])
            for row in rows
            if isinstance(row.get("duration_seconds"), (int, float))
        ),
        "benchmarks": benchmark_summaries,
    }
    write_run_markdown(args.out, run_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
