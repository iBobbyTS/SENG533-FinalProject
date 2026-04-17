from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine benchmark, memory, and power metrics into one summary row.")
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--power", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def iso_to_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metric_values(power_rows: list[dict], started_at: datetime, ended_at: datetime, key: str) -> list[float]:
    values: list[float] = []
    for row in power_rows:
        timestamp = iso_to_dt(row["timestamp"])
        if not (started_at <= timestamp <= ended_at):
            continue
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def main() -> int:
    args = parse_args()
    meta = load_json(args.meta)
    memory_rows = load_json(args.memory)
    power_rows = parse_jsonl(args.power)
    benchmark_summary = load_json(Path(meta["benchmark_summary"]))

    memory_values = [row["rss_bytes"] for row in memory_rows if row.get("rss_bytes") is not None]
    virtual_values = [row["vms_bytes"] for row in memory_rows if row.get("vms_bytes") is not None]
    swap_used_values = [row["swap_used_bytes"] for row in memory_rows if row.get("swap_used_bytes") is not None]
    pagein_values = [row["pageins"] for row in memory_rows if row.get("pageins") is not None]
    memory_source = "largest_lm_studio_node_rss" if memory_values else None
    virtual_source = "largest_lm_studio_node_vms" if virtual_values else None
    swap_source = "sysctl_vm.swapusage" if swap_used_values else None
    if not memory_values:
        fallback_memory_values = [
            row.get("memory", {}).get("ram_usage")
            for row in power_rows
            if row.get("memory", {}).get("ram_usage") is not None
        ]
        if fallback_memory_values:
            memory_values = fallback_memory_values
            memory_source = "macmon_system_ram_usage"
    if not swap_used_values:
        fallback_swap_values = [
            row.get("memory", {}).get("swap_usage")
            for row in power_rows
            if row.get("memory", {}).get("swap_usage") is not None
        ]
        if fallback_swap_values:
            swap_used_values = fallback_swap_values
            swap_source = "macmon_memory_swap_usage"
    started_at = iso_to_dt(meta["started_at"])
    ended_at = iso_to_dt(meta["ended_at"])
    all_power_values = metric_values(power_rows, started_at, ended_at, "all_power")
    gpu_power_values = metric_values(power_rows, started_at, ended_at, "gpu_power")
    ram_power_values = metric_values(power_rows, started_at, ended_at, "ram_power")
    sys_power_values = metric_values(power_rows, started_at, ended_at, "sys_power")

    summary = {
        "benchmark": meta["benchmark"],
        "profile": meta["profile"],
        "model": meta["model"],
        "context_length": meta.get("context_length"),
        "duration_seconds": meta["duration_seconds"],
        "memory_source": memory_source,
        "virtual_source": virtual_source if virtual_values else None,
        "swap_source": swap_source if swap_used_values else None,
        "memory_max_bytes": max(memory_values) if memory_values else None,
        "memory_mean_bytes": mean(memory_values),
        "virtual_max_bytes": max(virtual_values) if virtual_values else None,
        "virtual_mean_bytes": mean(virtual_values),
        "swap_used_max_bytes": max(swap_used_values) if swap_used_values else None,
        "swap_used_mean_bytes": mean(swap_used_values),
        "pageins_start": pagein_values[0] if pagein_values else None,
        "pageins_end": pagein_values[-1] if pagein_values else None,
        "pageins_delta": (
            max(0, pagein_values[-1] - pagein_values[0]) if len(pagein_values) >= 2 else None
        ),
        "power_max_watts": max(all_power_values) if all_power_values else None,
        "power_mean_watts": mean(all_power_values),
        "gpu_power_max_watts": max(gpu_power_values) if gpu_power_values else None,
        "gpu_power_mean_watts": mean(gpu_power_values),
        "ram_power_max_watts": max(ram_power_values) if ram_power_values else None,
        "ram_power_mean_watts": mean(ram_power_values),
        "sys_power_max_watts": max(sys_power_values) if sys_power_values else None,
        "sys_power_mean_watts": mean(sys_power_values),
        "benchmark_score": benchmark_summary.get("score"),
        "benchmark_status": benchmark_summary.get("status"),
        "benchmark_summary": meta["benchmark_summary"],
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
