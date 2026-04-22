from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_suite.plot_memory_clustered import bytes_to_gib, load_memory_series, load_power_memory_series


class PlotMemoryClusteredTests(unittest.TestCase):
    def test_bytes_to_gib(self) -> None:
        self.assertAlmostEqual(bytes_to_gib(1024**3), 1.0)
        self.assertAlmostEqual(bytes_to_gib(0), 0.0)

    def test_load_memory_series_extracts_first_and_last_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            model_dirs = [
                "qwen3.5-4b",
                "qwen_qwen3.5-9b",
                "qwen3.5-27b_q4_k_m",
                "qwen3.5-27b_q6_k",
                "qwen_qwen3.5-35b-a3b",
            ]
            for index, model_dir in enumerate(model_dirs, start=1):
                target = run_dir / model_dir / "ctx_32768"
                target.mkdir(parents=True)
                base = index * 100
                snapshots = [
                    {
                        "rss_bytes": base + 1,
                        "footprint": {"phys_footprint_bytes": base + 2},
                        "memory_analysis": {
                            "swapped_sum_bytes": base + 20,
                            "categories": {
                                "model": {"resident_bytes": base + 3},
                                "context_gpu": {"resident_bytes": base + 4},
                            }
                        },
                    },
                    {
                        "rss_bytes": base + 11,
                        "footprint": {"phys_footprint_bytes": base + 12},
                        "memory_analysis": {
                            "swapped_sum_bytes": base + 30,
                            "categories": {
                                "model": {"resident_bytes": base + 13},
                                "context_gpu": {"resident_bytes": base + 14},
                            }
                        },
                    },
                ]
                (target / "run2_memory_snapshots.json").write_text(
                    json.dumps(snapshots),
                    encoding="utf-8",
                )

            series = load_memory_series(run_dir)

        self.assertEqual(
            [item.label for item in series],
            ["4B\nQ4_K_M", "9B\nQ4_K_M", "27B\nQ4_K_M", "27B\nQ6_K", "35B-A3B\nQ4_K_M"],
        )
        self.assertEqual(series[0].rss_initial_bytes, 101)
        self.assertEqual(series[0].rss_final_bytes, 111)
        self.assertEqual(series[0].footprint_initial_bytes, 102)
        self.assertEqual(series[0].footprint_final_bytes, 112)
        self.assertEqual(series[0].swap_initial_bytes, 120)
        self.assertEqual(series[0].swap_final_bytes, 130)
        self.assertEqual(series[0].model_bucket_bytes, 103)
        self.assertEqual(series[0].context_initial_bytes, 104)
        self.assertEqual(series[0].context_final_bytes, 114)

    def test_load_power_memory_series_uses_latest_successful_row(self) -> None:
        def row(model_key: str, status: str, base: int, completion_tokens: int | None = None) -> dict:
            payload = {
                "model_key": model_key,
                "status": status,
                "first_snapshot": {
                    "rss_bytes": base + 1,
                    "footprint": {"phys_footprint_bytes": base + 2},
                    "memory_analysis": {
                        "swapped_sum_bytes": base + 20,
                        "categories": {
                            "model": {"resident_bytes": base + 3},
                            "context_gpu": {"resident_bytes": base + 4},
                        },
                    },
                },
                "final_snapshot": {
                    "rss_bytes": base + 11,
                    "footprint": {"phys_footprint_bytes": base + 12},
                    "memory_analysis": {
                        "swapped_sum_bytes": base + 30,
                        "categories": {
                            "model": {"resident_bytes": base + 13},
                            "context_gpu": {"resident_bytes": base + 14},
                        },
                    },
                },
            }
            if completion_tokens is not None:
                payload["first_run_usage"] = {"completion_tokens": completion_tokens}
            if status != "completed":
                payload["error"] = "timed out"
            return payload

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run1 = root / "20260420T000000Z_power_memory_test"
            run1.mkdir()
            (run1 / "summary.json").write_text(
                json.dumps(
                    {
                        "memory": [
                            row("model-a", "completed", 100, completion_tokens=32739),
                            row("model-b", "failed", 200),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            run2 = root / "20260420T010000Z_power_memory_test"
            run2.mkdir()
            (run2 / "summary.json").write_text(
                json.dumps(
                    {
                        "memory": [
                            row("model-a", "failed", 300),
                            row("model-b", "completed", 400, completion_tokens=2855),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            series = load_power_memory_series(
                root,
                model_specs=(("A", "model-a"), ("B", "model-b"), ("C", "model-c")),
            )

        self.assertEqual(series[0].status, "completed")
        self.assertEqual(series[0].source_run, "20260420T000000Z_power_memory_test")
        self.assertIsNotNone(series[0].memory)
        self.assertEqual(series[0].memory.rss_initial_bytes, 101)
        self.assertEqual(series[0].completion_tokens, 32739)
        self.assertEqual(series[1].status, "completed")
        self.assertEqual(series[1].source_run, "20260420T010000Z_power_memory_test")
        self.assertEqual(series[1].completion_tokens, 2855)
        self.assertEqual(series[2].status, "missing")
        self.assertIsNone(series[2].memory)


if __name__ == "__main__":
    unittest.main()
