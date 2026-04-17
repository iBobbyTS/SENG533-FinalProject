from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_model_progress_groups.py"


def load_module():
    spec = importlib.util.spec_from_file_location("plot_model_progress_groups_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlotModelProgressGroupsScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_groups_cover_five_requested_comparisons(self) -> None:
        groups = self.module.GROUPS
        self.assertEqual(len(groups), 5)
        self.assertEqual(groups[0][0], "01_q4_model_size")
        self.assertEqual(groups[-1][0], "05_ultra_low_quantization")

    def test_default_output_directory_name(self) -> None:
        self.assertEqual(self.module.DEFAULT_OUT_DIR.name, "20260330_model_progress_grouped_plots")

    def test_group_labels_use_short_model_codes(self) -> None:
        self.assertEqual([item[0] for item in self.module.GROUPS[0][2]], ["4B", "9B", "27B"])
        self.assertEqual([item[0] for item in self.module.GROUPS[1][2]], ["Q4_K_M", "Q6_K", "Q8_0"])
        self.assertEqual([item[0] for item in self.module.GROUPS[3][2]], ["Official", "Opus"])
        self.assertEqual([item[0] for item in self.module.GROUPS[4][2]], ["Q2_K", "Q4_K_M"])


if __name__ == "__main__":
    unittest.main()
