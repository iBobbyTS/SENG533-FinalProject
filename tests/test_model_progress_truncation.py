from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_suite.model_progress_truncation import (
    analyze_model_progress,
    analyze_predictions,
    parse_model_progress,
)


class ModelProgressTruncationTests(unittest.TestCase):
    def test_parse_model_progress_extracts_summary_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "results" / "run1" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text("{}", encoding="utf-8")
            markdown_path = root / "model_progress.md"
            markdown_path.write_text(
                "\n".join(
                    [
                        "| Model Name | Params | Quantization | Notes | Latest Run | Status | Completed | Failed | Failure Notes | Current Registry Quantization | Current Selected Variant |",
                        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
                        f"| Qwen3.5 | 9B | Q8_0 | Dense model | [run1]({summary_path}) | Partial | 5 | 1 | note |  |  |",
                    ]
                ),
                encoding="utf-8",
            )

            rows = parse_model_progress(markdown_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].summary_path, summary_path)
            self.assertEqual(rows[0].quantization, "Q8_0")

    def test_analyze_predictions_counts_explicit_and_context_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictions_path = Path(tmp) / "predictions.jsonl"
            records = [
                {
                    "finish_reason": "length",
                    "usage": {"total_tokens": 3000},
                },
                {
                    "finish_reason": None,
                    "usage": {"total_tokens": 4096},
                },
                {
                    "finish_reason": None,
                    "usage": {"total_tokens": 2048},
                },
            ]
            predictions_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            result = analyze_predictions(predictions_path, context_length=4096)
            self.assertEqual(result["prediction_rows"], 3)
            self.assertEqual(result["rows_with_usage"], 3)
            self.assertEqual(result["explicit_length_rows"], 1)
            self.assertEqual(result["context_cap_hit_rows"], 1)
            self.assertEqual(result["truncated_rows"], 2)

    def test_analyze_model_progress_reads_benchmark_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "results" / "run1"
            run_root.mkdir(parents=True)
            summary_path = run_root / "summary.json"
            benchmark_root = run_root / "gsm8k"
            benchmark_root.mkdir()
            summary_path.write_text(
                json.dumps(
                    {
                        "run_id": "run1",
                        "profile": "initial",
                        "model": "qwen/qwen3.5-9b",
                        "benchmarks": [
                            {
                                "benchmark": "gsm8k",
                                "status": "completed",
                                "sample_count": 2,
                                "context_length": 4096,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (benchmark_root / "predictions.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"finish_reason": None, "usage": {"total_tokens": 4096}}),
                        json.dumps({"finish_reason": None, "usage": {"total_tokens": 1000}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            markdown_path = root / "model_progress.md"
            markdown_path.write_text(
                "\n".join(
                    [
                        "| Model Name | Params | Quantization | Notes | Latest Run | Status | Completed | Failed | Failure Notes | Current Registry Quantization | Current Selected Variant |",
                        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
                        f"| Qwen3.5 | 9B | Q8_0 | Dense model | [run1]({summary_path}) | Partial | 5 | 1 | note |  |  |",
                    ]
                ),
                encoding="utf-8",
            )

            rows = analyze_model_progress(markdown_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["benchmark"], "gsm8k")
            self.assertEqual(rows[0]["truncated_rows"], 1)
            self.assertEqual(rows[0]["context_cap_hit_rows"], 1)


if __name__ == "__main__":
    unittest.main()
