from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_input_ttft_confidence_interval.py"
SPEC = importlib.util.spec_from_file_location("run_input_ttft_confidence_interval", SCRIPT_PATH)
assert SPEC is not None
input_ttft = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(input_ttft)


class InputTtftConfidenceIntervalTests(unittest.TestCase):
    def test_discover_swift_files_ignores_non_swift_and_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            keep_a = root / "A.swift"
            keep_b = root / "B.swift"
            keep_a.write_text("a", encoding="utf-8")
            keep_b.write_text("bb", encoding="utf-8")
            (root / ".DS_Store").write_text("ignored", encoding="utf-8")
            nested = root / "Nested"
            nested.mkdir()
            (nested / "C.swift").write_text("ignored", encoding="utf-8")

            discovered = input_ttft.discover_swift_files(root)

        self.assertEqual([path.name for path in discovered], ["A.swift", "B.swift"])

    def test_select_longest_file_uses_top_level_file_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            short = root / "Short.swift"
            long = root / "Long.swift"
            short.write_text("a", encoding="utf-8")
            long.write_text("abc", encoding="utf-8")

            result = input_ttft.select_longest_file([short, long])

        self.assertEqual(result.name, "Long.swift")

    def test_build_prompt_prepends_prefix_before_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Source.swift"
            path.write_text("let value = 1", encoding="utf-8")

            prompt = input_ttft.build_prompt(path, prompt_prefix="Z\n")

        self.assertTrue(prompt.startswith("Z\nRead the Swift source file below."))
        self.assertIn("let value = 1", prompt)

    def test_prompt_prefix_for_run_uses_upper_then_lower_sequence(self) -> None:
        self.assertEqual(input_ttft.prompt_prefix_for_run(1), "A\n")
        self.assertEqual(input_ttft.prompt_prefix_for_run(26), "Z\n")
        self.assertEqual(input_ttft.prompt_prefix_for_run(27), "a\n")
        self.assertEqual(input_ttft.prompt_prefix_for_run(52), "z\n")
        self.assertEqual(input_ttft.prompt_prefix_for_run(53), "A\n")
        self.assertEqual(input_ttft.prompt_prefix_for_run(1, enabled=False), "")

    def test_prefix_sequence_warning_when_runs_exceed_sequence(self) -> None:
        warning = input_ttft.prefix_sequence_warning(53)

        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertEqual(warning["warning"], "prefix_sequence_reuse")
        self.assertEqual(warning["unique_prefixes"], 52)
        self.assertIsNone(input_ttft.prefix_sequence_warning(52))
        self.assertIsNone(input_ttft.prefix_sequence_warning(53, enabled=False))

    def test_approximate_input_tps_samples_uses_prompt_tokens_over_ttft(self) -> None:
        rows = [
            {"usage": {"prompt_tokens": 100, "time_to_first_token_seconds": 2.0}},
            {"usage": {"prompt_tokens": 150, "time_to_first_token_seconds": 3.0}},
            {"usage": {"prompt_tokens": 150, "time_to_first_token_seconds": 0.0}},
        ]

        self.assertEqual(input_ttft.approximate_input_tps_samples(rows), [50.0, 50.0])

    def test_ttft_minus_model_load_samples_subtracts_per_run_load_time(self) -> None:
        rows = [
            {"usage": {"time_to_first_token_seconds": 10.0, "model_load_time_seconds": 3.0}},
            {"usage": {"time_to_first_token_seconds": 1.0, "model_load_time_seconds": 2.0}},
            {"usage": {"time_to_first_token_seconds": 4.0, "model_load_time_seconds": None}},
        ]

        self.assertEqual(input_ttft.ttft_minus_model_load_samples(rows), [7.0, 0.0, 4.0])

    def test_confidence_interval_for_five_ttft_samples(self) -> None:
        result = input_ttft.confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(result["count"], 5)
        self.assertEqual(result["mean"], 3.0)
        self.assertTrue(math.isclose(result["t_critical"], 2.7764451051977987))
        self.assertTrue(math.isclose(result["ci_half_width"], 1.9632431614775606))

    def test_build_plot_series_uses_mean_prompt_tokens_and_ttft(self) -> None:
        summary = {
            "results": [
                {
                    "model_key": "qwen/qwen3.5-9b@q6_k",
                    "file_summaries": [
                        {
                            "file": "Long.swift",
                            "bytes": 100,
                            "prompt_tokens_ci": {"mean": 2000.0},
                            "ttft_minus_model_load_seconds_ci": {"mean": 1.5},
                        },
                        {
                            "file": "Short.swift",
                            "bytes": 10,
                            "prompt_tokens_ci": {"mean": 500.0},
                            "ttft_minus_model_load_seconds_ci": {"mean": 0.5},
                        },
                    ],
                }
            ]
        }

        series = input_ttft.build_plot_series(summary)

        self.assertEqual(series[0]["label"], "9B Q6_K")
        self.assertEqual([point["file"] for point in series[0]["points"]], ["Short.swift", "Long.swift"])
        self.assertEqual([point["prompt_tokens_mean"] for point in series[0]["points"]], [500.0, 2000.0])
        self.assertEqual([point["ttft_seconds_mean"] for point in series[0]["points"]], [0.5, 1.5])


if __name__ == "__main__":
    unittest.main()
