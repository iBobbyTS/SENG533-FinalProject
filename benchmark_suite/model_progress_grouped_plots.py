from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


BENCHMARK_ORDER = [
    "mmlu_pro",
    "gsm8k",
    "mbpp",
    "vqa",
    "longbench",
    "truthfulqa",
]

BENCHMARK_LABELS = {
    "mmlu_pro": "MMLU-Pro",
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "vqa": "VQA",
    "longbench": "LongBench",
    "truthfulqa": "TruthfulQA",
}

RUN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


@dataclass(frozen=True)
class ModelProgressRow:
    model_name: str
    params: str
    quantization: str
    size_gb: float
    notes: str
    run_label: str
    summary_path: Path
    status: str
    completed: int
    failed: int
    failure_notes: str


@dataclass(frozen=True)
class BenchmarkPoint:
    benchmark: str
    score_percent: float | None
    status: str
    truncated_rows: int


@dataclass(frozen=True)
class BenchmarkOverride:
    params: str
    quantization: str
    notes: str
    benchmark: str
    summary_path: Path


def parse_model_progress_markdown(path: Path) -> list[ModelProgressRow]:
    rows: list[ModelProgressRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("| Model Name") or stripped.startswith("|------------"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 12:
            continue
        link_match = RUN_LINK_RE.fullmatch(cells[5])
        if not link_match:
            continue
        run_label, summary_path = link_match.groups()
        rows.append(
            ModelProgressRow(
                model_name=cells[0],
                params=cells[1],
                quantization=cells[2],
                size_gb=float(cells[3]),
                notes=cells[4],
                run_label=run_label,
                summary_path=Path(unquote(summary_path)),
                status=cells[6],
                completed=int(cells[7]),
                failed=int(cells[8]),
                failure_notes=cells[9],
            )
        )
    return rows


def load_truncation_index(path: Path) -> dict[tuple[str, str], int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    index: dict[tuple[str, str], int] = {}
    for row in rows:
        summary_path = str(Path(row["summary_path"]).resolve())
        index[(summary_path, row["benchmark"])] = int(row["truncated_rows"])
    return index


def load_benchmark_overrides(path: Path) -> dict[tuple[str, str, str, str], BenchmarkOverride]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    overrides: dict[tuple[str, str, str, str], BenchmarkOverride] = {}
    for row in rows:
        summary_path = Path(row["summary_path"])
        if not summary_path.is_absolute():
            summary_path = (path.parent / summary_path).resolve()
        override = BenchmarkOverride(
            params=str(row["params"]),
            quantization=str(row["quantization"]),
            notes=str(row["notes"]),
            benchmark=str(row["benchmark"]),
            summary_path=summary_path,
        )
        overrides[(override.params, override.quantization, override.notes, override.benchmark)] = override
    return overrides


def resolve_summary_path(summary_path: Path, search_roots: list[Path]) -> Path:
    if summary_path.exists():
        return summary_path
    if not summary_path.is_absolute():
        return summary_path
    parts = summary_path.parts
    try:
        results_index = parts.index("results")
    except ValueError:
        return summary_path
    if results_index + 1 >= len(parts):
        return summary_path

    run_relative = Path(*parts[results_index + 1 :])
    for root in search_roots:
        candidate = root / run_relative
        if candidate.exists():
            return candidate
    return summary_path


def _summary_benchmarks(summary: dict) -> list[dict]:
    benchmarks = summary.get("benchmarks")
    if isinstance(benchmarks, list):
        return benchmarks
    if "benchmark" in summary:
        return [summary]
    return []


def load_benchmark_points(
    summary_path: Path,
    truncation_index: dict[tuple[str, str], int],
    truncation_summary_path: Path | None = None,
) -> dict[str, BenchmarkPoint]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_key = str((truncation_summary_path or summary_path).resolve())
    points: dict[str, BenchmarkPoint] = {}
    for benchmark in _summary_benchmarks(summary):
        name = benchmark["benchmark"]
        raw_score = benchmark.get("score")
        score_percent = None if raw_score is None else float(raw_score) * 100.0
        points[name] = BenchmarkPoint(
            benchmark=name,
            score_percent=score_percent,
            status=str(benchmark.get("status") or ""),
            truncated_rows=truncation_index.get((summary_key, name), 0),
        )
    return points


def apply_benchmark_overrides(
    row: ModelProgressRow,
    points: dict[str, BenchmarkPoint],
    overrides: dict[tuple[str, str, str, str], BenchmarkOverride],
    truncation_index: dict[tuple[str, str], int],
) -> dict[str, BenchmarkPoint]:
    updated = dict(points)
    for benchmark in BENCHMARK_ORDER:
        override = overrides.get((row.params, row.quantization, row.notes, benchmark))
        if override is None:
            continue
        override_points = load_benchmark_points(override.summary_path, truncation_index)
        updated[benchmark] = override_points[benchmark]
    return updated


def find_row(rows: list[ModelProgressRow], params: str, quantization: str, notes: str) -> ModelProgressRow:
    for row in rows:
        if row.params == params and row.quantization == quantization and row.notes == notes:
            return row
    raise KeyError(f"Model row not found: params={params}, quantization={quantization}, notes={notes}")


def transpose_group_series(
    series: list[tuple[str, dict[str, BenchmarkPoint]]],
) -> tuple[list[str], dict[str, list[tuple[str, BenchmarkPoint]]]]:
    model_labels = [label for label, _ in series]
    benchmark_series: dict[str, list[tuple[str, BenchmarkPoint]]] = {name: [] for name in BENCHMARK_ORDER}
    for label, points in series:
        for benchmark in BENCHMARK_ORDER:
            benchmark_series[benchmark].append((label, points[benchmark]))
    return model_labels, benchmark_series
