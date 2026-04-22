from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_suite.model_progress_grouped_plots import (
    BenchmarkPoint,
    apply_benchmark_overrides,
    find_row,
    load_benchmark_overrides,
    load_benchmark_points,
    load_truncation_index,
    parse_model_progress_markdown,
    resolve_summary_path,
    transpose_group_series,
)


class ModelProgressGroupedPlotsTests(unittest.TestCase):
    def test_parse_model_progress_markdown_extracts_summary_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "results" / "run" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text("{}", encoding="utf-8")
            markdown = root / "model_progress.md"
            markdown.write_text(
                "\n".join(
                    [
                        "# Model Progress Summary",
                        "",
                        "| Model Name | Params | Quantization | Size (GB) | Notes | Latest Run | Status | Completed | Failed | Failure Notes | Current Registry Quantization | Current Selected Variant |",
                        "|------------|---------|--------------|-----------|-------|------------|--------|----------:|-------:|---------------|-------------------------------|--------------------------|",
                        f"| Qwen3.5 | 27B | Q4_K_M | 16.54 | Dense model | [run_1]({summary_path.as_posix().replace(' ', '%20')}) | Completed | 6 | 0 | None |  |  |",
                    ]
                ),
                encoding="utf-8",
            )
            rows = parse_model_progress_markdown(markdown)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].run_label, "run_1")
        self.assertEqual(rows[0].summary_path, summary_path)
        self.assertEqual(rows[0].size_gb, 16.54)

    def test_load_truncation_index_and_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {"benchmark": "mmlu_pro", "status": "completed", "score": 0.75},
                            {"benchmark": "vqa", "status": "failed", "score": None},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            trunc_path = root / "trunc.json"
            trunc_path.write_text(
                json.dumps(
                    [
                        {"summary_path": str(summary_path.resolve()), "benchmark": "mmlu_pro", "truncated_rows": 3},
                        {"summary_path": str(summary_path.resolve()), "benchmark": "vqa", "truncated_rows": 0},
                    ]
                ),
                encoding="utf-8",
            )

            trunc_index = load_truncation_index(trunc_path)
            points = load_benchmark_points(summary_path, trunc_index)

        self.assertEqual(points["mmlu_pro"].score_percent, 75.0)
        self.assertEqual(points["mmlu_pro"].truncated_rows, 3)
        self.assertIsNone(points["vqa"].score_percent)
        self.assertEqual(points["vqa"].status, "failed")

    def test_load_benchmark_points_accepts_single_benchmark_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "mbpp" / "summary.json"
            summary_path.parent.mkdir()
            summary_path.write_text(
                json.dumps({"benchmark": "mbpp", "status": "completed", "score": 0.76}),
                encoding="utf-8",
            )

            points = load_benchmark_points(summary_path, {})

        self.assertEqual(points["mbpp"].score_percent, 76.0)
        self.assertEqual(points["mbpp"].status, "completed")

    def test_apply_benchmark_overrides_replaces_only_matching_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            aggregate_summary = root / "aggregate_summary.json"
            aggregate_summary.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {"benchmark": "mbpp", "status": "completed", "score": 0.0},
                            {"benchmark": "vqa", "status": "completed", "score": 0.7},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            override_summary = root / "rerun" / "summary.json"
            override_summary.parent.mkdir()
            override_summary.write_text(
                json.dumps({"benchmark": "mbpp", "status": "completed", "score": 0.76}),
                encoding="utf-8",
            )
            override_path = root / "overrides.json"
            override_path.write_text(
                json.dumps(
                    [
                        {
                            "params": "27B",
                            "quantization": "Q4_K_M",
                            "notes": "Dense model",
                            "benchmark": "mbpp",
                            "summary_path": "rerun/summary.json",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            row = parse_model_progress_markdown(
                self._write_markdown(root / "progress.md", [("27B", "Q4_K_M", "Dense model", aggregate_summary)])
            )[0]

            points = load_benchmark_points(aggregate_summary, {})
            overrides = load_benchmark_overrides(override_path)
            updated = apply_benchmark_overrides(row, points, overrides, {})

        self.assertEqual(updated["mbpp"].score_percent, 76.0)
        self.assertEqual(updated["vqa"].score_percent, 70.0)

    def test_resolve_summary_path_finds_moved_result_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stale_path = root / "results" / "run_1" / "summary.json"
            moved_path = root / "results" / "inteligence_benchmark" / "run_1" / "summary.json"
            moved_path.parent.mkdir(parents=True)
            moved_path.write_text("{}", encoding="utf-8")

            resolved = resolve_summary_path(stale_path, [root / "results" / "inteligence_benchmark"])

        self.assertEqual(resolved, moved_path)

    def test_find_row_distinguishes_same_params_and_quant_by_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            summary_path.write_text("{}", encoding="utf-8")
            rows = [
                parse_model_progress_markdown(
                    self._write_markdown(
                        root / "a.md",
                        [
                            ("27B", "Q4_K_M", "Dense model", summary_path),
                            ("27B", "Q4_K_M", "Claude 4.6 Opus reasoning distilled", summary_path),
                        ],
                    )
                )
            ][0]

        row = find_row(rows, "27B", "Q4_K_M", "Claude 4.6 Opus reasoning distilled")
        self.assertEqual(row.notes, "Claude 4.6 Opus reasoning distilled")

    def test_transpose_group_series_returns_model_clusters(self) -> None:
        series = [
            (
                "4B Q4_K_M",
                {
                    "mmlu_pro": BenchmarkPoint("mmlu_pro", 30.0, "completed", 0),
                    "gsm8k": BenchmarkPoint("gsm8k", 93.0, "completed", 1),
                    "mbpp": BenchmarkPoint("mbpp", 60.0, "completed", 2),
                    "vqa": BenchmarkPoint("vqa", 81.0, "completed", 3),
                    "longbench": BenchmarkPoint("longbench", 36.0, "completed", 4),
                    "truthfulqa": BenchmarkPoint("truthfulqa", 73.0, "completed", 5),
                },
            ),
            (
                "9B Q4_K_M",
                {
                    "mmlu_pro": BenchmarkPoint("mmlu_pro", 74.0, "completed", 6),
                    "gsm8k": BenchmarkPoint("gsm8k", 92.0, "completed", 7),
                    "mbpp": BenchmarkPoint("mbpp", 52.0, "completed", 8),
                    "vqa": BenchmarkPoint("vqa", 72.0, "completed", 9),
                    "longbench": BenchmarkPoint("longbench", 45.0, "completed", 10),
                    "truthfulqa": BenchmarkPoint("truthfulqa", 82.0, "completed", 11),
                },
            ),
        ]

        labels, benchmark_series = transpose_group_series(series)

        self.assertEqual(labels, ["4B Q4_K_M", "9B Q4_K_M"])
        self.assertEqual([label for label, _ in benchmark_series["mmlu_pro"]], labels)
        self.assertEqual(benchmark_series["mmlu_pro"][0][1].score_percent, 30.0)
        self.assertEqual(benchmark_series["mmlu_pro"][1][1].truncated_rows, 6)

    @staticmethod
    def _write_markdown(path: Path, entries: list[tuple[str, str, str, Path]]) -> Path:
        lines = [
            "| Model Name | Params | Quantization | Size (GB) | Notes | Latest Run | Status | Completed | Failed | Failure Notes | Current Registry Quantization | Current Selected Variant |",
            "|------------|---------|--------------|-----------|-------|------------|--------|----------:|-------:|---------------|-------------------------------|--------------------------|",
        ]
        for i, (params, quant, notes, summary_path) in enumerate(entries, start=1):
            lines.append(
                f"| Qwen3.5 | {params} | {quant} | 1.0 | {notes} | [run_{i}]({summary_path}) | Completed | 6 | 0 | None |  |  |"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
