from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_grouped_performance_results.py"


def load_module():
    spec = importlib.util.spec_from_file_location("plot_grouped_performance_results", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlotGroupedPerformanceResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_collect_output_tps_rows_reads_completed_results_and_skips_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "confidence_summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "results": [
                            {
                                "model_key": "qwen/qwen3.5-9b@q6_k",
                                "status": "completed",
                                "context_length": 32768,
                                "tokens_per_second_ci": {"mean": 18.0, "ci_half_width": 0.5, "count": 5},
                                "completion_tokens_ci": {"mean": 32739.0},
                            },
                            {
                                "model_key": "qwen3.5-27b@q6_k",
                                "status": "failed",
                                "tokens_per_second_ci": {},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows, skipped = self.module.collect_output_tps_rows(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "9B Q6_K")
        self.assertEqual(rows[0]["mean_tps"], 18.0)
        self.assertEqual(rows[0]["ci_half_width"], 0.5)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["model_key"], "qwen3.5-27b@q6_k")

    def test_collect_output_tps_rows_keeps_latest_completed_result_for_same_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_dir = root / "20260417_old"
            new_dir = root / "20260420_new"
            old_dir.mkdir()
            new_dir.mkdir()
            base_result = {
                "model_key": "qwen/qwen3.5-9b@q8_0",
                "status": "completed",
                "context_length": 32768,
                "tokens_per_second_ci": {"mean": 17.0, "ci_half_width": 0.5, "count": 5},
                "completion_tokens_ci": {"mean": 2855.0},
            }
            (old_dir / "confidence_summary.json").write_text(
                json.dumps({"run_id": "20260417_old", "results": [base_result]}),
                encoding="utf-8",
            )
            newer_result = {
                **base_result,
                "tokens_per_second_ci": {"mean": 14.9, "ci_half_width": 0.1, "count": 4},
                "completion_tokens_ci": {"mean": 32739.0},
            }
            (new_dir / "confidence_summary.json").write_text(
                json.dumps({"run_id": "20260420_new", "results": [newer_result]}),
                encoding="utf-8",
            )

            rows, skipped = self.module.collect_output_tps_rows(root)

        self.assertEqual(skipped, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "9B Q8_0")
        self.assertEqual(rows[0]["run_id"], "20260420_new")
        self.assertEqual(rows[0]["mean_tps"], 14.9)
        self.assertEqual(rows[0]["completion_tokens_mean"], 32739.0)

    def test_output_tps_curve_from_speed_runs_averages_token_progression(self) -> None:
        curve = self.module.output_tps_curve_from_speed_runs(
            [
                {
                    "observed_first_output_seconds": 0.0,
                    "output_token_times_seconds": [0.0, 1.0, 2.0, 4.0],
                },
                {
                    "observed_first_output_seconds": 0.0,
                    "output_token_times_seconds": [0.0, 2.0, 4.0, 8.0],
                },
            ],
            max_points=10,
        )

        self.assertEqual([point["output_tokens"] for point in curve], [2.0, 3.0, 4.0])
        self.assertEqual([point["sample_count"] for point in curve], [2.0, 2.0, 2.0])
        self.assertEqual([point["mean_tps"] for point in curve], [1.5, 1.125, 0.75])

    def test_collect_input_ttft_series_includes_retry_completed_series(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            main_dir = root / "main"
            retry_dir = root / "retry"
            main_dir.mkdir()
            retry_dir.mkdir()
            (main_dir / "input_ttft_summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "main",
                        "results": [
                            {
                                "model_key": "qwen3.5-27b@q4_k_m",
                                "status": "failed",
                                "file_summaries": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (retry_dir / "input_ttft_summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "retry",
                        "results": [
                            {
                                "model_key": "qwen3.5-27b@q4_k_m",
                                "status": "completed",
                                "context_length": 20000,
                                "file_summaries": [
                                    {
                                        "file": "Small.swift",
                                        "prompt_tokens_ci": {"mean": 100.0},
                                        "ttft_minus_model_load_seconds_ci": {
                                            "mean": 2.0,
                                            "ci_half_width": 0.1,
                                        },
                                    },
                                    {
                                        "file": "Large.swift",
                                        "prompt_tokens_ci": {"mean": 500.0},
                                        "ttft_minus_model_load_seconds_ci": {
                                            "mean": 8.0,
                                            "ci_half_width": 0.3,
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            series, skipped = self.module.collect_input_ttft_series(root)

        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["label"], "27B Q4_K_M")
        self.assertEqual([point["file"] for point in series[0]["points"]], ["Small.swift", "Large.swift"])
        self.assertEqual([point["prompt_tokens_mean"] for point in series[0]["points"]], [100.0, 500.0])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["run_id"], "main")

    def test_collect_input_ttft_error_rows_uses_largest_input_token_file(self) -> None:
        rows = self.module.collect_input_ttft_error_rows(
            [
                {
                    "run_id": "retry",
                    "model_key": "qwen3.5-27b@q4_k_m",
                    "label": "27B Q4_K_M",
                    "context_length": 20000,
                    "points": [
                        {
                            "file": "Small.swift",
                            "prompt_tokens_mean": 100.0,
                            "ttft_seconds_mean": 2.0,
                            "ttft_ci_half_width": 0.1,
                        },
                        {
                            "file": "Large.swift",
                            "prompt_tokens_mean": 500.0,
                            "ttft_seconds_mean": 8.0,
                            "ttft_ci_half_width": 0.3,
                        },
                    ],
                }
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["file"], "Large.swift")
        self.assertEqual(rows[0]["prompt_tokens_mean"], 500.0)
        self.assertEqual(rows[0]["mean_ttft_seconds"], 8.0)
        self.assertEqual(rows[0]["ci_half_width"], 0.3)

    def test_model_sort_key_orders_known_labels(self) -> None:
        labels = ["27B Q6_K", "4B Q4_K_M", "9B Q8_0", "35B-A3B Q4_K_M"]

        self.assertEqual(
            sorted(labels, key=self.module.model_sort_key),
            ["4B Q4_K_M", "9B Q8_0", "27B Q6_K", "35B-A3B Q4_K_M"],
        )

    def test_format_axis_model_label_puts_quantization_on_second_line(self) -> None:
        self.assertEqual(self.module.format_axis_model_label("9B Q8_0"), "9B\nQ8_0")
        self.assertEqual(self.module.format_axis_model_label("35B-A3B Q4_K_M"), "35B-A3B\nQ4_K_M")
        self.assertEqual(self.module.format_axis_model_label("27B Opus Q4_K_M"), "27B Opus\nQ4_K_M")

    def test_model_color_uses_stable_known_and_fallback_colors(self) -> None:
        self.assertEqual(self.module.model_color("9B Q6_K"), self.module.MODEL_COLOR_MAP["9B Q6_K"])
        self.assertEqual(self.module.model_color("9B Q6_K"), self.module.model_color("9B Q6_K"))
        self.assertEqual(self.module.model_color("unknown-model"), self.module.model_color("unknown-model"))


if __name__ == "__main__":
    unittest.main()
