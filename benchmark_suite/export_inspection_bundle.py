from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .utils import ensure_dir, load_jsonl


DEFAULT_FIELDS = [
    "index",
    "task",
    "task_index",
    "row_index",
    "question_id",
    "prompt",
    "task_id",
    "image_id",
    "category",
    "id",
    "gold",
    "answers",
    "prediction",
    "correct",
    "score",
    "finish_reason",
    "loop_risk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark runs into a human-readable inspection bundle.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="append", dest="runs", required=True, help="Run spec in the form label=/abs/or/rel/path")
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid run spec: {spec}")
    label, raw_path = spec.split("=", 1)
    return label.strip(), Path(raw_path.strip())


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        ensure_dir(target.parent)
        shutil.copy2(source, target)


def format_usage(row: dict) -> str:
    usage = row.get("usage")
    if not isinstance(usage, dict):
        return "N/A"
    return (
        f"prompt={usage.get('prompt_tokens', 'N/A')}, "
        f"completion={usage.get('completion_tokens', 'N/A')}, "
        f"total={usage.get('total_tokens', 'N/A')}"
    )


def write_benchmark_markdown(predictions_path: Path, output_path: Path) -> int:
    rows = load_jsonl(predictions_path)
    lines = [
        f"# {predictions_path.parent.name}",
        "",
        f"- Records: `{len(rows)}`",
        f"- Source: `{predictions_path}`",
        "",
    ]
    for row in rows:
        key_lines = []
        for field in DEFAULT_FIELDS:
            if field == "prompt":
                continue
            if field in row:
                key_lines.append(f"- {field}: `{row[field]}`")
        key_lines.append(f"- usage: `{format_usage(row)}`")
        lines.append(f"## Record {row.get('index', row.get('row_index', '?'))}")
        lines.extend(key_lines)
        prompt = row.get("prompt")
        if prompt:
            lines.append("- prompt:")
            lines.append("```text")
            if isinstance(prompt, str):
                lines.append(prompt)
            else:
                lines.append(str(prompt))
            lines.append("```")
        response_text = row.get("response_text", "")
        if response_text:
            lines.append("- response_text:")
            lines.append("```text")
            lines.append(response_text)
            lines.append("```")
        code = row.get("code")
        if code:
            lines.append("- code:")
            lines.append("```text")
            lines.append(code)
            lines.append("```")
        execution_log = row.get("execution_log")
        if execution_log:
            lines.append("- execution_log:")
            lines.append("```text")
            lines.append(execution_log)
            lines.append("```")
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def export_run(label: str, run_root: Path, bundle_root: Path) -> tuple[int, list[str]]:
    target_root = ensure_dir(bundle_root / label)
    benchmarks: list[str] = []
    total_records = 0

    copy_if_exists(run_root / "summary.json", target_root / "summary.json")
    copy_if_exists(run_root / "summary.md", target_root / "summary.md")

    for benchmark_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        predictions_path = benchmark_dir / "predictions.jsonl"
        if not predictions_path.exists():
            continue
        benchmarks.append(benchmark_dir.name)
        target_benchmark = ensure_dir(target_root / benchmark_dir.name)
        copy_if_exists(predictions_path, target_benchmark / "predictions.jsonl")
        copy_if_exists(benchmark_dir / "summary.json", target_benchmark / "summary.json")
        copy_if_exists(benchmark_dir / "summary.md", target_benchmark / "summary.md")
        copy_if_exists(benchmark_dir / "details.json", target_benchmark / "details.json")
        copy_if_exists(benchmark_dir / "breakdown.json", target_benchmark / "breakdown.json")
        copy_if_exists(benchmark_dir / "per_task.json", target_benchmark / "per_task.json")
        total_records += write_benchmark_markdown(predictions_path, target_benchmark / "inspection.md")

    return total_records, benchmarks


def main() -> int:
    args = parse_args()
    bundle_root = ensure_dir(args.out)
    index_lines = [
        "# Inspection Bundle",
        "",
        "This directory is for manual inspection of raw benchmark outputs, with emphasis on detecting unproductive reasoning, overly long lead-ins, repeated output, or looping behavior.",
        "",
        "Notes:",
        "- `prompt` is the full prompt actually sent to the model; multimodal samples also include the image path.",
        "- `response_text` is the raw model response.",
        "- `finish_reason` and `loop_risk` help identify truncation or repetition.",
        "",
    ]

    for spec in args.runs:
        label, run_root = parse_run_spec(spec)
        total_records, benchmarks = export_run(label, run_root, bundle_root)
        index_lines.append(f"## {label}")
        index_lines.append(f"- run_root: `{run_root}`")
        index_lines.append(f"- exported_benchmarks: `{', '.join(benchmarks) if benchmarks else 'none'}`")
        index_lines.append(f"- exported_records: `{total_records}`")
        index_lines.append("")

    (bundle_root / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
