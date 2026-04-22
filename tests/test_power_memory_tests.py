from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_power_memory_tests.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_power_memory_tests", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PowerMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_metric_summary_records_distribution(self) -> None:
        summary = self.module.metric_summary([1.0, 2.0, 4.0])

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)
        self.assertAlmostEqual(summary["mean"], statistics.fmean([1.0, 2.0, 4.0]))
        self.assertAlmostEqual(summary["std"], statistics.stdev([1.0, 2.0, 4.0]))

    def test_confidence_interval_uses_t_distribution_for_five_runs(self) -> None:
        interval = self.module.confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])

        expected_half_width = 2.7764451051977987 * statistics.stdev([1, 2, 3, 4, 5]) / math.sqrt(5)
        self.assertEqual(interval["count"], 5)
        self.assertAlmostEqual(interval["mean"], 3.0)
        self.assertAlmostEqual(interval["ci_half_width"], expected_half_width)
        self.assertAlmostEqual(interval["ci_lower"], 3.0 - expected_half_width)
        self.assertAlmostEqual(interval["ci_upper"], 3.0 + expected_half_width)

    def test_parse_macmon_line_accepts_json_only(self) -> None:
        self.assertEqual(
            self.module.parse_macmon_line('{"all_power": 12.5, "gpu_power": 3.1}'),
            {"all_power": 12.5, "gpu_power": 3.1},
        )
        self.assertIsNone(self.module.parse_macmon_line("Error: Failed to create subscription"))
        self.assertIsNone(self.module.parse_macmon_line("[1, 2, 3]"))

    def test_summarize_power_runs_computes_ci_over_run_means(self) -> None:
        runs = [
            {"power_sample_count": 40, "power_summary": {"all_power": {"mean": value}}}
            for value in [10.0, 12.0, 14.0, 16.0, 18.0]
        ]

        summary = self.module.summarize_power_runs("model", runs)

        self.assertEqual(summary["status"], "completed")
        all_power_ci = summary["metric_ci_over_run_means"]["all_power"]
        self.assertEqual(all_power_ci["count"], 5)
        self.assertAlmostEqual(all_power_ci["mean"], 14.0)

    def test_memory_row_from_result_extracts_final_snapshot(self) -> None:
        result = {
            "load": {"strategy": "models_load"},
            "first_run_full_stats": {"usage": {"completion_tokens": 256}},
            "second_run_control": {"usage": {"completion_tokens": 256}},
            "second_run_memory_snapshots": [
                {"rss_bytes": 100, "vms_bytes": 1000},
                {
                    "rss_bytes": 200,
                    "vms_bytes": 2000,
                    "footprint": {"phys_footprint_bytes": 300},
                    "memory_analysis": {
                        "resident_sum_bytes": 400,
                        "swapped_sum_bytes": 500,
                    },
                },
            ],
        }

        row = self.module.memory_row_from_result("model", 32768, result, Path("result.json"))

        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["snapshot_count"], 2)
        self.assertEqual(row["final_rss_bytes"], 200)
        self.assertEqual(row["final_vms_bytes"], 2000)
        self.assertEqual(row["final_phys_footprint_bytes"], 300)
        self.assertEqual(row["final_memory_analysis_resident_sum_bytes"], 400)
        self.assertEqual(row["final_memory_analysis_swapped_sum_bytes"], 500)

    def test_write_run_plan_records_power_and_memory_settings(self) -> None:
        args = argparse.Namespace(
            run_id="unit",
            mode="both",
            base_url="http://127.0.0.1:1234/api/v1",
            models=["a", "b"],
            context_length=32768,
            temperature=0.1,
            prompt="prompt",
            power_runs=5,
            power_interval_ms=250,
            power_warmup_seconds=3.0,
            power_sample_seconds=10.0,
            power_startup_timeout_seconds=15.0,
            cooldown_seconds=60.0,
            memory_snapshot_interval_seconds=60,
            memory_speed_runs=0,
            max_output_tokens=32768,
            dry_run=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_plan.json"
            plan = self.module.write_run_plan(path, args)

            self.assertTrue(path.exists())
            self.assertEqual(plan["models"], ["a", "b"])
            self.assertEqual(plan["temperature"], 0.1)
            self.assertEqual(plan["power_interval_ms"], 250)
            self.assertEqual(plan["power_startup_timeout_seconds"], 15.0)
            self.assertEqual(plan["memory_snapshot_interval_seconds"], 60)
            self.assertEqual(plan["memory_speed_runs"], 0)


if __name__ == "__main__":
    unittest.main()
