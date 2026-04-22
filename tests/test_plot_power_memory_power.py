from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_power_memory_power.py"


def load_module():
    spec = importlib.util.spec_from_file_location("plot_power_memory_power", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlotPowerMemoryPowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_collect_power_rows_uses_latest_completed_run(self) -> None:
        def power_row(model_key: str, status: str, sys_mean: float) -> dict:
            return {
                "model_key": model_key,
                "status": status,
                "metric_ci_over_run_means": {
                    "sys_power": {"mean": sys_mean, "ci_half_width": 1.0},
                    "gpu_power": {"mean": 2.0, "ci_half_width": 0.2},
                    "cpu_power": {"mean": 3.0, "ci_half_width": 0.3},
                    "ram_power": {"mean": 4.0, "ci_half_width": 0.4},
                },
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "20260420T000000Z_power_memory_test"
            first.mkdir()
            (first / "summary.json").write_text(
                json.dumps({"power": [power_row("qwen3.5-4b", "completed", 10.0)]}),
                encoding="utf-8",
            )
            second = root / "20260420T010000Z_power_memory_test"
            second.mkdir()
            (second / "summary.json").write_text(
                json.dumps(
                    {
                        "power": [
                            power_row("qwen3.5-4b", "completed", 20.0),
                            power_row("qwen/qwen3.5-9b@q4_k_m", "failed", 30.0),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows = self.module.collect_power_rows(root)

        self.assertEqual(rows[0]["model_key"], "qwen3.5-4b")
        self.assertEqual(rows[0]["source_run"], "20260420T010000Z_power_memory_test")
        self.assertEqual(rows[0]["metrics"]["sys_power"]["mean"], 20.0)
        self.assertEqual(rows[0]["metrics"]["gpu_power"]["mean"], 2.0)
        self.assertEqual(rows[1]["model_key"], "qwen/qwen3.5-9b@q4_k_m")
        self.assertEqual(rows[1]["status"], "missing")

    def test_bar_and_legend_labels_match_report_language(self) -> None:
        self.assertEqual(
            [item[1] for item in self.module.BAR_ORDER],
            ["System Power", "GPU Power", "RAM Power", "CPU Power"],
        )
        self.assertEqual(
            [item[1] for item in self.module.LEGEND_ORDER],
            ["System Power", "GPU Power", "RAM Power", "CPU Power"],
        )


if __name__ == "__main__":
    unittest.main()
