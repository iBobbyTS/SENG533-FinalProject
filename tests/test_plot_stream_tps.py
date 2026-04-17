from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_stream_tps.py"


def load_module():
    spec = importlib.util.spec_from_file_location("plot_stream_tps_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlotStreamTpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_format_combined_label_maps_known_models(self) -> None:
        self.assertEqual(self.module.format_combined_label("qwen3.5-4b"), "4B Q4_K_M")
        self.assertEqual(self.module.format_combined_label("qwen/qwen3.5-9b"), "9B Q4_K_M")
        self.assertEqual(self.module.format_combined_label("qwen/qwen3.5-9b@q4_k_m"), "9B Q4_K_M")
        self.assertEqual(self.module.format_combined_label("qwen/qwen3.5-9b@q6_k"), "9B Q6_K")
        self.assertEqual(self.module.format_combined_label("qwen/qwen3.5-9b@q8_0"), "9B Q8_0")
        self.assertEqual(self.module.format_combined_label("qwen3.5-27b@q4_k_m"), "27B Q4_K_M")
        self.assertEqual(self.module.format_combined_label("qwen3.5-27b@q6_k"), "27B Q6_K")
        self.assertEqual(self.module.format_combined_label("qwen/qwen3.5-35b-a3b"), "35B-A3B Q4_K_M")
        self.assertEqual(self.module.format_combined_label("qwen/qwen3.5-35b-a3b@q4_k_m"), "35B-A3B Q4_K_M")

    def test_format_combined_label_falls_back_for_unknown_models(self) -> None:
        self.assertEqual(self.module.format_combined_label("unknown-model"), "unknown-model")

    def test_build_curve_uses_compact_output_token_times(self) -> None:
        xs, ys = self.module.build_curve(
            {
                "usage": {"completion_tokens": 4},
                "observed_first_output_seconds": 1.0,
                "output_token_times_seconds": [1.0, 2.0, 3.0, 5.0],
            }
        )

        self.assertEqual(xs, [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(math.isnan(ys[0]))
        self.assertEqual(ys[1:], [2.0, 1.5, 1.0])


if __name__ == "__main__":
    unittest.main()
