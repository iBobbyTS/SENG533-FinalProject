from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import subprocess
import threading

import requests
import psutil

from .model_management import LMStudioModelManager, native_api_root_from_base_url
from .model_progress_truncation import ModelProgressRow, load_json, parse_model_progress


DEFAULT_PROMPT = (
    "Generate the word 'loop' forever. Do not stop. Only output the word 'loop'."
)


@dataclass(frozen=True)
class ResolvedModelTarget:
    source: str
    model_key: str
    params: str
    quantization: str
    notes: str
    latest_run_label: str
    summary_path: Path | None
    recorded_model_key: str | None


@dataclass(frozen=True)
class RegistryModel:
    key: str
    display_name: str
    quantization: str
    selected_variant: str
    format: str
    variants: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeltaEvent:
    event_type: str
    elapsed_seconds: float
    delta_chars: int
    cumulative_chars: int


def parse_size_to_bytes(size_str: str | None) -> int | None:
    if not size_str:
        return None
    token = size_str.upper().strip().replace(" ", "")
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }
    for suffix in ("KB", "MB", "GB", "TB", "B", "K", "M", "G", "T"):
        if token.endswith(suffix):
            try:
                value = float(token[: -len(suffix)])
            except ValueError:
                return None
            return int(value * multipliers[suffix])
    return None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    out = []
    for char in value:
        if char.isalnum() or char in "._-":
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def list_registry_models(base_url: str) -> list[RegistryModel]:
    manager = LMStudioModelManager(base_url)
    rows: list[RegistryModel] = []
    for model in manager.list_models():
        quantization = ""
        quant = model.get("quantization")
        if isinstance(quant, dict):
            quantization = str(quant.get("name") or "")
        rows.append(
            RegistryModel(
                key=str(model.get("key") or ""),
                display_name=str(model.get("display_name") or model.get("key") or ""),
                quantization=quantization,
                selected_variant=str(model.get("selected_variant") or ""),
                format=str(model.get("format") or ""),
                variants=tuple(str(item) for item in model.get("variants", []) if item),
            )
        )
    return rows


def find_largest_lm_studio_node() -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_rss = -1
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if "node" not in name:
                continue
            cmdline = " ".join(proc.cmdline()).lower()
            if "lm studio" not in cmdline:
                continue
            memory_info = proc.memory_info()
            rss = getattr(memory_info, "rss", 0)
            if rss > best_rss:
                best_rss = rss
                best = {
                    "pid": proc.info["pid"],
                    "name": proc.info.get("name") or "",
                    "cmdline": proc.cmdline(),
                    "rss_bytes": rss,
                    "vms_bytes": getattr(memory_info, "vms", None),
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, SystemError, OSError):
            continue
    return best


def parse_footprint_output(output: str) -> dict[str, Any]:
    size_pattern = r"([0-9]+(?:\.[0-9]+)?\s*(?:[KMGTP]?B|[KMGTP]))"
    overall_match = re.search(rf"Footprint:\s+{size_pattern}", output)
    phys_match = re.search(rf"phys_footprint:\s+{size_pattern}", output)
    peak_match = re.search(rf"phys_footprint_peak:\s+{size_pattern}", output)
    total_match = re.search(
        rf"^\s*{size_pattern}\s+{size_pattern}\s+{size_pattern}\s+[0-9]+\s+TOTAL$",
        output,
        re.MULTILINE,
    )
    return {
        "footprint_bytes": parse_size_to_bytes(overall_match.group(1)) if overall_match else None,
        "phys_footprint_bytes": parse_size_to_bytes(phys_match.group(1)) if phys_match else None,
        "phys_footprint_peak_bytes": parse_size_to_bytes(peak_match.group(1)) if peak_match else None,
        "total_dirty_bytes": parse_size_to_bytes(total_match.group(1)) if total_match else None,
        "total_clean_bytes": parse_size_to_bytes(total_match.group(2)) if total_match else None,
        "total_reclaimable_bytes": parse_size_to_bytes(total_match.group(3)) if total_match else None,
    }


def sample_footprint(pid: int) -> dict[str, Any]:
    output = subprocess.check_output(["footprint", str(pid)], stderr=subprocess.STDOUT).decode("utf-8", errors="replace")
    parsed = parse_footprint_output(output)
    parsed["raw_excerpt"] = output[:1000]
    return parsed


def parse_memory_analysis_output(output: str) -> dict[str, Any]:
    footprint_match = re.search(r"Physical footprint:\s+([0-9\.]+[KMGTP])", output)
    row_pattern = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_\s\.\(\)-]+?)\s+([0-9\.]+[KMGTP])\s+([0-9\.]+[KMGTP])\s+([0-9\.]+[KMGTP])\s+([0-9\.]+[KMGTP])",
        re.MULTILINE,
    )
    categories = {
        "model": {"virtual_bytes": 0, "resident_bytes": 0, "swapped_bytes": 0},
        "context_gpu": {"virtual_bytes": 0, "resident_bytes": 0, "swapped_bytes": 0},
        "software": {"virtual_bytes": 0, "resident_bytes": 0, "swapped_bytes": 0},
    }

    for match in row_pattern.finditer(output):
        region_name = match.group(1).strip()
        if region_name.startswith("TOTAL"):
            continue
        virt = parse_size_to_bytes(match.group(2)) or 0
        res = parse_size_to_bytes(match.group(3)) or 0
        swap = parse_size_to_bytes(match.group(5)) or 0
        if region_name.startswith("mapped file"):
            bucket = "model"
        elif (
            region_name.startswith("VM_ALLOCATE")
            or region_name.startswith("shared memory")
            or region_name.startswith("IOAccelerator")
        ):
            bucket = "context_gpu"
        else:
            bucket = "software"
        categories[bucket]["virtual_bytes"] += virt
        categories[bucket]["resident_bytes"] += res
        categories[bucket]["swapped_bytes"] += swap

    return {
        "reported_physical_footprint_bytes": parse_size_to_bytes(footprint_match.group(1)) if footprint_match else None,
        "categories": categories,
        "resident_sum_bytes": sum(bucket["resident_bytes"] for bucket in categories.values()),
        "swapped_sum_bytes": sum(bucket["swapped_bytes"] for bucket in categories.values()),
        "raw_excerpt": output[:1000],
    }


def sample_memory_analysis(pid: int) -> dict[str, Any]:
    output = subprocess.check_output(["vmmap", "-summary", str(pid)], stderr=subprocess.STDOUT).decode("utf-8", errors="replace")
    return parse_memory_analysis_output(output)


def collect_memory_snapshot(pid: int, elapsed_seconds: float, is_final_snapshot: bool) -> dict[str, Any]:
    started = time.perf_counter()
    rss_bytes: int | None = None
    vms_bytes: int | None = None
    try:
        proc = psutil.Process(pid)
        info = proc.memory_info()
        rss_bytes = getattr(info, "rss", None)
        vms_bytes = getattr(info, "vms", None)
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, SystemError, OSError):
        pass

    footprint_payload: dict[str, Any] = {}
    memory_analysis_payload: dict[str, Any] = {}
    footprint_error: str | None = None
    memory_analysis_error: str | None = None

    try:
        footprint_payload = sample_footprint(pid)
    except Exception as exc:  # noqa: BLE001
        footprint_error = str(exc)

    try:
        memory_analysis_payload = sample_memory_analysis(pid)
    except Exception as exc:  # noqa: BLE001
        memory_analysis_error = str(exc)

    return {
        "elapsed_seconds": elapsed_seconds,
        "is_final_snapshot": is_final_snapshot,
        "rss_bytes": rss_bytes,
        "vms_bytes": vms_bytes,
        "footprint": footprint_payload,
        "footprint_error": footprint_error,
        "memory_analysis": memory_analysis_payload,
        "memory_analysis_error": memory_analysis_error,
        "snapshot_duration_seconds": time.perf_counter() - started,
    }


def collect_memory_snapshots_during_run(
    *,
    pid: int,
    sampler_thread_done: threading.Event,
    snapshots: list[dict[str, Any]],
    started_perf: float,
    snapshot_interval_seconds: int,
) -> threading.Thread:
    def worker() -> None:
        next_mark = float(snapshot_interval_seconds)
        while not sampler_thread_done.is_set():
            now_elapsed = time.perf_counter() - started_perf
            wait_seconds = next_mark - now_elapsed
            if wait_seconds > 0:
                if sampler_thread_done.wait(wait_seconds):
                    break
            elapsed = time.perf_counter() - started_perf
            snapshots.append(collect_memory_snapshot(pid, elapsed, is_final_snapshot=False))
            next_mark += float(snapshot_interval_seconds)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def _summary_model_key(row: ModelProgressRow) -> str:
    summary = load_json(row.summary_path)
    model_key = summary.get("model")
    if not isinstance(model_key, str) or not model_key:
        raise RuntimeError(f"Missing model key in {row.summary_path}")
    return model_key


def _resolve_model_key(
    row: ModelProgressRow,
    summary_model_key: str,
    registry_by_key: dict[str, RegistryModel],
) -> str | None:
    if summary_model_key in registry_by_key:
        return summary_model_key

    if row.current_selected_variant and row.current_selected_variant in registry_by_key:
        return row.current_selected_variant

    direct_suffix = f"{summary_model_key}@{row.quantization.lower()}"
    if direct_suffix in registry_by_key:
        return direct_suffix

    candidates = [
        model.key
        for model in registry_by_key.values()
        if model.key.startswith(f"{summary_model_key}@") and model.quantization == row.quantization
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def resolve_recorded_model_targets(model_progress_path: Path, base_url: str) -> list[ResolvedModelTarget]:
    progress_rows = parse_model_progress(model_progress_path)
    registry_models = list_registry_models(base_url)
    registry_by_key = {model.key: model for model in registry_models}

    rows_with_keys: list[tuple[ModelProgressRow, str]] = [
        (row, _summary_model_key(row)) for row in progress_rows
    ]

    selected_summary_keys: set[str] = set()
    selected_rows: list[tuple[ModelProgressRow, str]] = []

    groups: dict[str, list[tuple[ModelProgressRow, str]]] = {}
    for row, summary_model_key in rows_with_keys:
        groups.setdefault(summary_model_key, []).append((row, summary_model_key))

    for summary_model_key, group_rows in groups.items():
        registry_entry = registry_by_key.get(summary_model_key)
        if registry_entry and registry_entry.selected_variant and len(group_rows) > 1:
            matching = [row_pair for row_pair in group_rows if row_pair[0].quantization == registry_entry.quantization]
            if matching:
                selected_rows.extend(matching[:1])
                selected_summary_keys.add(summary_model_key)

    for row, summary_model_key in rows_with_keys:
        if summary_model_key in selected_summary_keys:
            continue
        selected_rows.append((row, summary_model_key))

    targets: list[ResolvedModelTarget] = []
    seen_model_keys: set[str] = set()
    for row, summary_model_key in selected_rows:
        resolved_key = _resolve_model_key(row, summary_model_key, registry_by_key)
        if resolved_key is None or resolved_key in seen_model_keys:
            continue
        seen_model_keys.add(resolved_key)
        targets.append(
            ResolvedModelTarget(
                source="model_progress",
                model_key=resolved_key,
                params=row.params,
                quantization=row.quantization,
                notes=row.notes,
                latest_run_label=row.latest_run_label,
                summary_path=row.summary_path,
                recorded_model_key=summary_model_key,
            )
        )

    return targets


def _quantization_from_model_key(model_key: str) -> str:
    suffix = model_key.rsplit("@", maxsplit=1)[-1] if "@" in model_key else model_key
    match = re.search(r"q[0-9]+(?:_[a-z0-9]+)+|[0-9]+bit", suffix, flags=re.IGNORECASE)
    return match.group(0).upper() if match else ""


def resolve_explicit_model_targets(
    model_keys: list[str],
    base_url: str,
    *,
    expand_model_variants: bool = False,
) -> list[ResolvedModelTarget]:
    registry_models = list_registry_models(base_url)
    registry_by_key = {model.key: model for model in registry_models}
    targets: list[ResolvedModelTarget] = []
    seen: set[str] = set()
    expanded_model_keys: list[str] = []
    for model_key in model_keys:
        registry_entry = registry_by_key.get(model_key)
        if expand_model_variants and registry_entry and registry_entry.variants:
            expanded_model_keys.extend(registry_entry.variants)
        else:
            expanded_model_keys.append(model_key)

    for model_key in expanded_model_keys:
        if model_key in seen:
            continue
        seen.add(model_key)
        registry_entry = registry_by_key.get(model_key)
        quantization = registry_entry.quantization if registry_entry else ""
        if not quantization:
            quantization = _quantization_from_model_key(model_key)
        targets.append(
            ResolvedModelTarget(
                source="explicit",
                model_key=model_key,
                params="",
                quantization=quantization,
                notes="",
                latest_run_label="",
                summary_path=None,
                recorded_model_key=model_key if registry_entry else None,
            )
        )
    return targets


def _parse_sse_event(event_type: str | None, raw_parts: list[bytes]) -> tuple[str, dict[str, Any]]:
    if not raw_parts:
        return event_type or "unknown", {}

    payload_variants: list[str] = []
    for joiner in (b"\n", b"", None):
        if joiner is None:
            text = "\n".join(part.decode("utf-8", errors="replace") for part in raw_parts)
        else:
            text = joiner.join(raw_parts).decode("utf-8", errors="strict")
        if text not in payload_variants:
            payload_variants.append(text)

    last_error: Exception | None = None
    for variant in payload_variants:
        try:
            data = json.loads(variant)
            return event_type or str(data.get("type") or "unknown"), data
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise last_error if last_error is not None else RuntimeError("Failed to parse SSE event")


def _stream_chat(
    *,
    base_url: str,
    model_key: str,
    context_length: int,
    prompt: str,
    temperature: float = 0.0,
    idle_timeout_seconds: int,
    max_output_tokens: int | None = None,
    compact_output_timing: bool = False,
) -> dict[str, Any]:
    native_root = native_api_root_from_base_url(base_url)
    session = requests.Session()
    payload = {
        "model": model_key,
        "input": prompt,
        "temperature": temperature,
        "store": False,
        "stream": True,
        "context_length": context_length,
    }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens

    started_at = datetime.now(timezone.utc).isoformat()
    started_perf = time.perf_counter()
    current_event: str | None = None
    current_data: list[bytes] = []
    final_result: dict[str, Any] | None = None
    output_chars = 0
    message_chars = 0
    reasoning_chars = 0
    first_output_at: float | None = None
    delta_events: list[DeltaEvent] = []
    stream_delta_times_seconds: list[float] = []
    chat_end_elapsed_seconds: float | None = None
    stream_rows: list[dict[str, Any]] = []
    event_counts: dict[str, int] = {}
    model_load_time_seconds: float | None = None

    def flush_event() -> None:
        nonlocal current_event, current_data, final_result, output_chars, message_chars, reasoning_chars, first_output_at, model_load_time_seconds, chat_end_elapsed_seconds
        if current_event is None and not current_data:
            return
        event_name, data = _parse_sse_event(current_event, current_data)
        elapsed_seconds = time.perf_counter() - started_perf
        event_counts[event_name] = event_counts.get(event_name, 0) + 1

        row: dict[str, Any] | None = None
        if not compact_output_timing:
            row = {
                "event_type": event_name,
                "elapsed_seconds": elapsed_seconds,
            }
            if "progress" in data:
                row["progress"] = data.get("progress")

        if event_name in {"message.delta", "reasoning.delta"}:
            content = str(data.get("content") or "")
            delta_chars = len(content)
            output_chars += delta_chars
            if event_name == "message.delta":
                message_chars += delta_chars
            else:
                reasoning_chars += delta_chars
            if first_output_at is None:
                first_output_at = elapsed_seconds
            stream_delta_times_seconds.append(elapsed_seconds)
            if not compact_output_timing:
                delta_events.append(
                    DeltaEvent(
                        event_type=event_name,
                        elapsed_seconds=elapsed_seconds,
                        delta_chars=delta_chars,
                        cumulative_chars=output_chars,
                    )
                )
                assert row is not None
                row["delta_chars"] = delta_chars
                row["cumulative_output_chars"] = output_chars
        elif event_name == "model_load.end":
            model_load_time_seconds = data.get("model_load_time_seconds")
            if row is not None:
                row["model_load_time_seconds"] = model_load_time_seconds
        elif event_name == "chat.end":
            chat_end_elapsed_seconds = elapsed_seconds
            final_result = data.get("result", {})

        if row is not None:
            stream_rows.append(row)
        current_event = None
        current_data = []

    with session.post(
        f"{native_root}/chat",
        json=payload,
        timeout=(30, idle_timeout_seconds),
        stream=True,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=False):
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

    ended_perf = time.perf_counter()
    ended_at = datetime.now(timezone.utc).isoformat()
    if final_result is None:
        raise RuntimeError("LM Studio native stream ended without chat.end")

    stats = final_result.get("stats", {}) if isinstance(final_result, dict) else {}
    output_items = final_result.get("output", []) if isinstance(final_result, dict) and not compact_output_timing else []
    message_text = "".join(
        item.get("content", "")
        for item in output_items
        if isinstance(item, dict) and item.get("type") == "message"
    )
    reasoning_text = "".join(
        item.get("content", "")
        for item in output_items
        if isinstance(item, dict) and item.get("type") == "reasoning"
    )
    usage = {
        "prompt_tokens": stats.get("input_tokens"),
        "completion_tokens": stats.get("total_output_tokens"),
        "total_tokens": (
            (stats.get("input_tokens") or 0) + (stats.get("total_output_tokens") or 0)
            if stats.get("input_tokens") is not None and stats.get("total_output_tokens") is not None
            else None
        ),
        "reasoning_output_tokens": stats.get("reasoning_output_tokens"),
        "tokens_per_second": stats.get("tokens_per_second"),
        "time_to_first_token_seconds": stats.get("time_to_first_token_seconds"),
        "model_load_time_seconds": stats.get("model_load_time_seconds") or model_load_time_seconds,
    }
    output_token_times_seconds = list(stream_delta_times_seconds)
    token_time_adjustment: dict[str, Any] = {
        "raw_stream_delta_count": len(stream_delta_times_seconds),
        "usage_completion_tokens": usage["completion_tokens"],
        "padded_count": 0,
        "truncated_count": 0,
    }
    completion_tokens = usage["completion_tokens"]
    if isinstance(completion_tokens, int) and completion_tokens >= 0:
        if len(output_token_times_seconds) < completion_tokens:
            padded_count = completion_tokens - len(output_token_times_seconds)
            pad_time = chat_end_elapsed_seconds if chat_end_elapsed_seconds is not None else ended_perf - started_perf
            output_token_times_seconds.extend([pad_time] * padded_count)
            token_time_adjustment["padded_count"] = padded_count
            token_time_adjustment["pad_time_seconds"] = pad_time
        elif len(output_token_times_seconds) > completion_tokens:
            truncated_count = len(output_token_times_seconds) - completion_tokens
            output_token_times_seconds = output_token_times_seconds[:completion_tokens]
            token_time_adjustment["truncated_count"] = truncated_count

    return {
        "status": "completed",
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_time_seconds": ended_perf - started_perf,
        "model": model_key,
        "prompt": prompt,
        "temperature": temperature,
        "message_preview": message_text[:500],
        "reasoning_preview": reasoning_text[:500],
        "output_chars": output_chars,
        "message_chars": message_chars,
        "reasoning_chars": reasoning_chars,
        "observed_first_output_seconds": first_output_at,
        "usage": usage,
        "event_counts": event_counts,
        "compact_output_timing": compact_output_timing,
        "output_token_times_seconds": output_token_times_seconds,
        "stream_delta_times_seconds": stream_delta_times_seconds,
        "output_event_times_seconds": output_token_times_seconds,
        "output_token_time_adjustment": token_time_adjustment,
        "delta_events": [
            {
                "event_type": item.event_type,
                "elapsed_seconds": item.elapsed_seconds,
                "delta_chars": item.delta_chars,
                "cumulative_chars": item.cumulative_chars,
            }
            for item in delta_events
        ],
        "stream_rows": stream_rows,
        "raw_result": None if compact_output_timing else final_result,
    }


def _chat_auto_load_model_with_timing(base_url: str, model_key: str, context_length: int) -> dict[str, Any]:
    native_root = native_api_root_from_base_url(base_url)
    payload = {
        "model": model_key,
        "input": "load",
        "temperature": 0.0,
        "store": False,
        "max_output_tokens": 1,
        "context_length": context_length,
    }
    started = time.perf_counter()
    response = requests.post(f"{native_root}/chat", json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    load_duration_seconds = time.perf_counter() - started
    return {
        "model": model_key,
        "context_length": context_length,
        "load_duration_seconds": load_duration_seconds,
        "strategy": "chat_auto_load",
        "response": {
            "model_instance_id": data.get("model_instance_id"),
            "stats": data.get("stats", {}),
        },
    }


def load_model_with_timing(base_url: str, model_key: str, context_length: int) -> dict[str, Any]:
    manager = LMStudioModelManager(base_url)
    started = time.perf_counter()
    strategy = "models_load"
    try:
        response = manager.load_model(model_key, context_length=context_length)
    except RuntimeError as exc:
        if "@" not in model_key:
            raise
        fallback = _chat_auto_load_model_with_timing(base_url, model_key, context_length)
        fallback["models_load_error"] = str(exc)
        return fallback
    load_duration_seconds = time.perf_counter() - started
    return {
        "model": model_key,
        "context_length": context_length,
        "load_duration_seconds": load_duration_seconds,
        "strategy": strategy,
        "response": response,
    }


def unload_all_models(base_url: str) -> list[dict[str, Any]]:
    manager = LMStudioModelManager(base_url)
    unloaded = manager.unload_all()
    return [
        {
            "model_key": item.model_key,
            "display_name": item.display_name,
            "instance_id": item.instance_id,
            "config": item.config,
        }
        for item in unloaded
    ]


def run_generation_perf_pair(
    *,
    base_url: str,
    model_key: str,
    context_length: int,
    prompt: str,
    snapshot_interval_seconds: int,
    idle_timeout_seconds: int,
    temperature: float = 0.0,
    capture_memory: bool = True,
    max_output_tokens: int | None = None,
    speed_runs: int = 1,
    compact_output_timing: bool = False,
) -> dict[str, Any]:
    if speed_runs < 0:
        raise ValueError("speed_runs must be non-negative")
    if speed_runs == 0 and not capture_memory:
        raise ValueError("speed_runs must be at least 1 when memory capture is disabled")

    unload_before = unload_all_models(base_url)
    result: dict[str, Any] | None = None
    try:
        load_record = load_model_with_timing(base_url, model_key, context_length)
        speed_run_rows: list[dict[str, Any]] = []
        for _ in range(speed_runs):
            speed_run_rows.append(
                _stream_chat(
                    base_url=base_url,
                    model_key=model_key,
                    context_length=context_length,
                    prompt=prompt,
                    temperature=temperature,
                    idle_timeout_seconds=idle_timeout_seconds,
                    max_output_tokens=max_output_tokens,
                    compact_output_timing=compact_output_timing,
                )
            )
        run_one = speed_run_rows[0] if speed_run_rows else {
            "status": "skipped",
            "reason": "speed_runs_disabled",
            "started_at": None,
            "ended_at": None,
            "wall_time_seconds": None,
            "usage": {},
            "event_counts": {},
            "output_chars": 0,
            "message_chars": 0,
            "reasoning_chars": 0,
            "observed_first_output_seconds": None,
        }
        pid: int | None = None
        memory_snapshots: list[dict[str, Any]] = []

        if capture_memory:
            target = find_largest_lm_studio_node()
            pid = int(target["pid"]) if target and target.get("pid") is not None else None
            if pid is None:
                raise RuntimeError(f"Could not locate the LM Studio backend node for {model_key}")

            memory_started_perf = time.perf_counter()
            sampler_done = threading.Event()
            sampler_thread = collect_memory_snapshots_during_run(
                pid=pid,
                sampler_thread_done=sampler_done,
                snapshots=memory_snapshots,
                started_perf=memory_started_perf,
                snapshot_interval_seconds=snapshot_interval_seconds,
            )
            try:
                run_two = _stream_chat(
                    base_url=base_url,
                    model_key=model_key,
                    context_length=context_length,
                    prompt=prompt,
                    temperature=temperature,
                    idle_timeout_seconds=idle_timeout_seconds,
                    max_output_tokens=max_output_tokens,
                    compact_output_timing=compact_output_timing,
                )
            finally:
                sampler_done.set()
                sampler_thread.join(timeout=snapshot_interval_seconds + 30)
            memory_snapshots.append(
                collect_memory_snapshot(
                    pid,
                    time.perf_counter() - memory_started_perf,
                    is_final_snapshot=True,
                )
            )
        else:
            run_two = {
                "status": "skipped",
                "reason": "capture_memory_disabled",
                "started_at": None,
                "ended_at": None,
                "wall_time_seconds": None,
                "usage": {},
                "event_counts": {},
                "output_chars": 0,
                "message_chars": 0,
                "reasoning_chars": 0,
                "observed_first_output_seconds": None,
            }
        result = {
            "model": model_key,
            "context_length": context_length,
            "prompt": prompt,
            "temperature": temperature,
            "snapshot_interval_seconds": snapshot_interval_seconds,
            "capture_memory": capture_memory,
            "speed_runs_requested": speed_runs,
            "compact_output_timing": compact_output_timing,
            "preexisting_loaded_models": unload_before,
            "load": load_record,
            "pid": pid,
            "speed_runs": speed_run_rows,
            "first_run_full_stats": run_one,
            "second_run_control": {
                "status": run_two["status"],
                "reason": run_two.get("reason"),
                "started_at": run_two["started_at"],
                "ended_at": run_two["ended_at"],
                "wall_time_seconds": run_two["wall_time_seconds"],
                "usage": run_two["usage"],
                "event_counts": run_two["event_counts"],
                "output_chars": run_two["output_chars"],
                "message_chars": run_two["message_chars"],
                "reasoning_chars": run_two["reasoning_chars"],
                "observed_first_output_seconds": run_two["observed_first_output_seconds"],
            },
            "second_run_memory_snapshots": memory_snapshots,
        }
    finally:
        unload_after = unload_all_models(base_url)
        if result is not None:
            result["unloaded_after"] = unload_after
    if result is None:
        raise RuntimeError(f"Failed to collect generation performance for {model_key} @ ctx {context_length}")
    return result


def render_batch_markdown(batch_summary: dict[str, Any]) -> str:
    lines = [
        "# Generation Performance Batch Summary",
        "",
        f"- Run ID: `{batch_summary['run_id']}`",
        f"- Base URL: `{batch_summary['base_url']}`",
        f"- Prompt: `{batch_summary['prompt']}`",
        f"- Contexts: `{batch_summary['contexts']}`",
        f"- Snapshot interval: `{batch_summary['snapshot_interval_seconds']}` seconds",
        f"- Speed runs: `{batch_summary.get('speed_runs', 1)}`",
        f"- Compact output timing: `{batch_summary.get('compact_output_timing', False)}`",
        "",
        "| Model | Source | Params | Quantization | Notes | Context | Status | Load Strategy | Load (s) | Speed Runs | Run 1 TPS | Run 1 TTFT (s) | Run 1 Output Tokens | Run 2 Output Tokens | Memory Snapshots | Final RSS (GiB) | Final Footprint (GiB) | Final MemoryAnalysis Sum (GiB) |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in batch_summary["results"]:
        first_usage = row.get("first_run_usage", {})
        second_usage = row.get("second_run_usage", {})
        final_memory = row.get("memory_snapshot_final", {})
        final_rss_bytes = final_memory.get("rss_bytes")
        final_footprint_bytes = ((final_memory.get("footprint") or {}).get("phys_footprint_bytes"))
        final_memory_sum_bytes = ((final_memory.get("memory_analysis") or {}).get("resident_sum_bytes"))

        def gib(value: Any) -> str:
            if not isinstance(value, (int, float)):
                return ""
            return f"{value / (1024**3):.2f}"

        lines.append(
            f"| {row['model_key']} | {row['source']} | {row['params']} | {row['quantization']} | {row['notes']} | "
            f"{row['context_length']} | {row['status']} | "
            f"{row.get('load_strategy') or ''} | "
            f"{row.get('load_duration_seconds') or ''} | "
            f"{row.get('speed_run_count') or 0} | "
            f"{first_usage.get('tokens_per_second') or ''} | "
            f"{first_usage.get('time_to_first_token_seconds') or ''} | "
            f"{first_usage.get('completion_tokens') or ''} | "
            f"{second_usage.get('completion_tokens') or ''} | "
            f"{row.get('memory_snapshot_count') or 0} | "
            f"{gib(final_rss_bytes)} | "
            f"{gib(final_footprint_bytes)} | "
            f"{gib(final_memory_sum_bytes)} |"
        )

    return "\n".join(lines) + "\n"
