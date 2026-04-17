from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_suite.plot_memory_clustered import bytes_to_gib, load_memory_series


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


if __name__ == "__main__":
    unittest.main()
