from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from .config import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_SEED, RESULTS_ROOT
from .utils import ensure_dir


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_size_to_bytes(size_str: str | None) -> int | None:
    if not size_str:
        return None
    cleaned = size_str.strip().upper()
    if cleaned.endswith("B") and len(cleaned) > 1 and cleaned[-2] in {"K", "M", "G", "T", "P"}:
        cleaned = cleaned[:-1]
    multiplier = 1
    if cleaned.endswith("K"):
        multiplier = 1024
        cleaned = cleaned[:-1]
    elif cleaned.endswith("M"):
        multiplier = 1024**2
        cleaned = cleaned[:-1]
    elif cleaned.endswith("G"):
        multiplier = 1024**3
        cleaned = cleaned[:-1]
    elif cleaned.endswith("T"):
        multiplier = 1024**4
        cleaned = cleaned[:-1]
    elif cleaned.endswith("P"):
        multiplier = 1024**5
        cleaned = cleaned[:-1]
    try:
        return int(float(cleaned) * multiplier)
    except ValueError:
        return None


def read_swap_usage() -> dict[str, int | None]:
    try:
        result = subprocess.run(
            ["sysctl", "vm.swapusage"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return {
            "swap_total_bytes": None,
            "swap_used_bytes": None,
            "swap_free_bytes": None,
        }

    if result.returncode != 0:
        return {
            "swap_total_bytes": None,
            "swap_used_bytes": None,
            "swap_free_bytes": None,
        }

    match = re.search(
        r"total\s*=\s*([0-9.]+[KMGTP]?)\s+used\s*=\s*([0-9.]+[KMGTP]?)\s+free\s*=\s*([0-9.]+[KMGTP]?)",
        result.stdout,
        flags=re.IGNORECASE,
    )
    if not match:
        return {
            "swap_total_bytes": None,
            "swap_used_bytes": None,
            "swap_free_bytes": None,
        }

    return {
        "swap_total_bytes": parse_size_to_bytes(match.group(1)),
        "swap_used_bytes": parse_size_to_bytes(match.group(2)),
        "swap_free_bytes": parse_size_to_bytes(match.group(3)),
    }


def find_largest_lm_studio_node() -> dict | None:
    max_rss = 0
    target: dict | None = None
    try:
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = proc.info["name"] or ""
                if "node" not in name.lower():
                    continue
                cmdline = " ".join(proc.cmdline() or [])
                if "lm studio" in cmdline.lower():
                    memory_info = proc.memory_info()
                    rss = memory_info.rss
                    if rss > max_rss:
                        max_rss = rss
                        target = {
                            "pid": proc.info["pid"],
                            "process_name": name,
                            "rss_bytes": rss,
                            "vms_bytes": getattr(memory_info, "vms", None),
                            "pageins": getattr(memory_info, "pageins", None),
                            "pfaults": getattr(memory_info, "pfaults", None),
                            "selection_rule": "largest_lm_studio_node_rss",
                        }
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, SystemError, OSError):
                continue
    except PermissionError:
        return None
    return target


def sample_memory(stop_event: threading.Event, interval_seconds: float, samples: list[dict]) -> None:
    while not stop_event.is_set():
        timestamp = iso_now()
        target = find_largest_lm_studio_node()
        sample = {
            "timestamp": timestamp,
            "pid": None,
            "process_name": None,
            "rss_bytes": None,
            "vms_bytes": None,
            "pageins": None,
            "pfaults": None,
            "swap_total_bytes": None,
            "swap_used_bytes": None,
            "swap_free_bytes": None,
        }
        if target is not None:
            sample.update(target)
        sample.update(read_swap_usage())
        samples.append(sample)
        stop_event.wait(interval_seconds)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one benchmark and sample LM Studio memory usage.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--profile", choices=["probe", "smoke", "initial", "mixed"], default="probe")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--meta-out", type=Path, required=True)
    parser.add_argument("--memory-out", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument("--power-out", type=Path, default=None)
    parser.add_argument("--row-out", type=Path, default=None)
    parser.add_argument("--power-interval-ms", type=int, default=1000)
    parser.add_argument("--skip-model-management", action="store_true")
    parser.add_argument("--disable-power-monitoring", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.meta_out.parent)
    ensure_dir(args.memory_out.parent)
    if args.power_out is not None:
        ensure_dir(args.power_out.parent)
    if args.row_out is not None:
        ensure_dir(args.row_out.parent)
    stop_event = threading.Event()
    samples: list[dict] = []
    thread = threading.Thread(
        target=sample_memory,
        args=(stop_event, args.interval_ms / 1000.0, samples),
        daemon=True,
    )
    command = [
        sys.executable,
        "-m",
        "benchmark_suite.run_benchmark",
        "--benchmark",
        args.benchmark,
        "--profile",
        args.profile,
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--seed",
        str(args.seed),
        "--output-root",
        str(args.results_root),
        "--run-id",
        args.run_id,
    ]
    if args.context_length is not None:
        command.extend(["--context-length", str(args.context_length)])
    if args.skip_model_management:
        command.append("--skip-model-management")
    started_at = iso_now()
    start_perf = time.perf_counter()
    power_handle = None
    power_proc = None
    thread.start()
    try:
        if args.power_out is not None and not args.disable_power_monitoring:
            power_handle = args.power_out.open("w", encoding="utf-8")
            power_proc = subprocess.Popen(
                ["macmon", "pipe", "-i", str(args.power_interval_ms)],
                stdout=power_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            wait_for_power_stream(args.power_out, power_proc)
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    finally:
        stop_event.set()
        thread.join(timeout=2)
        if power_proc is not None:
            power_proc.terminate()
            try:
                power_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                power_proc.kill()
        if power_handle is not None:
            power_handle.close()

    ended_at = iso_now()
    duration_seconds = time.perf_counter() - start_perf

    args.memory_out.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta = {
        "benchmark": args.benchmark,
        "profile": args.profile,
        "model": args.model,
        "base_url": args.base_url,
        "context_length": args.context_length,
        "seed": args.seed,
        "run_id": args.run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "benchmark_summary": str(args.results_root / args.run_id / args.benchmark / "summary.json"),
    }
    args.meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    finalize_returncode = 0
    if args.row_out is not None:
        power_path_for_finalize = args.power_out or (args.meta_out.parent / "power.disabled.jsonl")
        finalize_cmd = [
            sys.executable,
            "-m",
            "benchmark_suite.finalize_monitored_run",
            "--meta",
            str(args.meta_out),
            "--memory",
            str(args.memory_out),
            "--power",
            str(power_path_for_finalize),
            "--out",
            str(args.row_out),
        ]
        finalized = subprocess.run(finalize_cmd, capture_output=True, text=True, check=False)
        finalize_returncode = finalized.returncode
        if finalized.returncode != 0:
            print(finalized.stdout, end="", file=sys.stdout)
            print(finalized.stderr, end="", file=sys.stderr)

    print(json.dumps({"benchmark": args.benchmark, "returncode": completed.returncode, "duration_seconds": duration_seconds}))
    return completed.returncode if completed.returncode != 0 else finalize_returncode


if __name__ == "__main__":
    raise SystemExit(main())
