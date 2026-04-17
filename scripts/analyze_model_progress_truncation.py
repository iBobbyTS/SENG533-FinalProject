#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_suite.model_progress_truncation import analyze_model_progress, render_markdown


DEFAULT_MODEL_PROGRESS = (
    PROJECT_ROOT / "results" / "20260330_model_progress.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token truncation counts for runs listed in the model progress markdown."
    )
    parser.add_argument(
        "--model-progress",
        type=Path,
        default=DEFAULT_MODEL_PROGRESS,
        help="Path to the model progress markdown file.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Markdown output path. Defaults to <model-progress-stem>_truncation.md beside the input file.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="JSON output path. Defaults to <model-progress-stem>_truncation.json beside the input file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_progress_path = args.model_progress.resolve()
    out_md = args.out_md or model_progress_path.with_name(f"{model_progress_path.stem}_truncation.md")
    out_json = args.out_json or model_progress_path.with_name(f"{model_progress_path.stem}_truncation.json")

    rows = analyze_model_progress(model_progress_path)
    markdown = render_markdown(model_progress_path, rows)

    out_md.write_text(markdown, encoding="utf-8")
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(out_md)
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
