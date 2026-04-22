from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_stream_perf_confidence_interval.py"
SPEC = importlib.util.spec_from_file_location("run_stream_perf_confidence_interval", SCRIPT_PATH)
assert SPEC is not None
stream_perf_ci = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stream_perf_ci)


class StreamPerfConfidenceIntervalTests(unittest.TestCase):
    def test_default_run_configuration_matches_current_batch_plan(self) -> None:
        self.assertEqual(stream_perf_ci.DEFAULT_MAX_OUTPUT_TOKENS, 32768)
        self.assertEqual(stream_perf_ci.DEFAULT_TEMPERATURE, 0.0)
        self.assertNotIn("qwen3.5-27b@q8_0", stream_perf_ci.DEFAULT_MODELS)

    def test_confidence_interval_uses_student_t_for_five_samples(self) -> None:
        result = stream_perf_ci.confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(result["count"], 5)
        self.assertEqual(result["mean"], 3.0)
        self.assertTrue(math.isclose(result["t_critical"], 2.7764451051977987))
        self.assertTrue(math.isclose(result["sample_stdev"], math.sqrt(2.5)))
        self.assertTrue(math.isclose(result["ci_half_width"], 1.9632431614775606))
        self.assertTrue(math.isclose(result["ci_lower"], 1.0367568385224394))
        self.assertTrue(math.isclose(result["ci_upper"], 4.963243161477561))

    def test_confidence_interval_ignores_missing_and_non_finite_values(self) -> None:
        result = stream_perf_ci.confidence_interval([10.0, float("nan"), float("inf"), 12.0])

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["mean"], 11.0)
        self.assertIsNotNone(result["ci_half_width"])

    def test_metric_samples_reads_usage_values(self) -> None:
        rows = [
            {"usage": {"tokens_per_second": 10.0}},
            {"usage": {"tokens_per_second": 12}},
            {"usage": {"tokens_per_second": None}},
            {"usage": {}},
        ]

        self.assertEqual(stream_perf_ci.metric_samples(rows, "tokens_per_second"), [10.0, 12.0])

    def test_derived_throughput_uses_last_token_time(self) -> None:
        rows = [
            {
                "usage": {"completion_tokens": 256, "tokens_per_second": 100.0},
                "output_token_times_seconds": [0.1, 4.0],
            },
            {
                "usage": {"completion_tokens": 128, "tokens_per_second": 100.0},
                "output_token_times_seconds": [0.2, 2.0],
            },
        ]

        self.assertEqual(stream_perf_ci.derived_throughput_samples(rows), [64.0, 64.0])
        self.assertEqual(stream_perf_ci.derived_seconds_per_token_samples(rows), [4.0 / 256, 2.0 / 128])


if __name__ == "__main__":
    unittest.main()
