from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_suite.model_stream_perf import (
    RegistryModel,
    parse_footprint_output,
    resolve_explicit_model_targets,
    resolve_recorded_model_targets,
    run_generation_perf_pair,
)


class ModelStreamPerfTests(unittest.TestCase):
    def test_resolve_recorded_model_targets_prefers_current_9b_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results = root / "results"
            results.mkdir()

            def write_summary(name: str, model: str) -> Path:
                run_dir = results / name
                run_dir.mkdir(parents=True)
                path = run_dir / "summary.json"
                path.write_text(json.dumps({"model": model, "benchmarks": []}), encoding="utf-8")
                return path

            summary_9b_q4 = write_summary("run_9b_q4", "qwen/qwen3.5-9b")
            summary_9b_q6 = write_summary("run_9b_q6", "qwen/qwen3.5-9b")
            summary_9b_q8 = write_summary("run_9b_q8", "qwen/qwen3.5-9b")
            summary_27b_distilled = write_summary(
                "run_27b_distilled_q4",
                "qwen3.5-27b-claude-4.6-opus-reasoning-distilled",
            )

            markdown = root / "model_progress.md"
            markdown.write_text(
                "\n".join(
                    [
                        "# Model Progress Summary",
                        "",
                        "| Model Name | Params | Quantization | Size (GB) | Notes | Latest Run | Status | Completed | Failed | Failure Notes | Current Registry Quantization | Current Selected Variant |",
                        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                        f"| Qwen3.5 | 9B | Q4_K_M | 5.63 | Dense model | [run_9b_q4]({summary_9b_q4}) | Completed | 6 | 0 | None |  |  |",
                        f"| Qwen3.5 | 9B | Q6_K | 7.36 | Dense model | [run_9b_q6]({summary_9b_q6}) | Completed | 6 | 0 | None |  |  |",
                        f"| Qwen3.5 | 9B | Q8_0 | 9.53 | Dense model | [run_9b_q8]({summary_9b_q8}) | Partial | 5 | 1 | None |  |  |",
                        f"| Qwen3.5 | 27B | Q4_K_M | 16.53 | Claude 4.6 Opus reasoning distilled | [run_27b_distilled_q4]({summary_27b_distilled}) | Completed | 6 | 0 | None |  |  |",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            registry_rows = [
                RegistryModel(
                    key="qwen/qwen3.5-9b",
                    display_name="Qwen3.5 9B",
                    quantization="Q4_K_M",
                    selected_variant="qwen/qwen3.5-9b@q4_k_m",
                    format="gguf",
                ),
                RegistryModel(
                    key="qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q4_k_m",
                    display_name="Qwen3.5 27B",
                    quantization="Q4_K_M",
                    selected_variant="",
                    format="gguf",
                ),
            ]

            with patch("benchmark_suite.model_stream_perf.list_registry_models", return_value=registry_rows):
                targets = resolve_recorded_model_targets(markdown, "http://127.0.0.1:1234/api/v1")

        self.assertEqual(
            [target.model_key for target in targets],
            [
                "qwen/qwen3.5-9b",
                    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q4_k_m",
                ],
            )

    def test_resolve_explicit_model_targets_can_expand_variants(self) -> None:
        registry_rows = [
            RegistryModel(
                key="qwen/qwen3.5-9b",
                display_name="Qwen3.5 9B",
                quantization="Q4_K_M",
                selected_variant="qwen/qwen3.5-9b@q4_k_m",
                format="gguf",
                variants=(
                    "qwen/qwen3.5-9b@q4_k_m",
                    "qwen/qwen3.5-9b@q6_k",
                    "qwen/qwen3.5-9b@q8_0",
                ),
            )
        ]

        with patch("benchmark_suite.model_stream_perf.list_registry_models", return_value=registry_rows):
            targets = resolve_explicit_model_targets(
                ["qwen/qwen3.5-9b"],
                "http://127.0.0.1:1234/api/v1",
                expand_model_variants=True,
            )

        self.assertEqual(
            [target.model_key for target in targets],
            [
                "qwen/qwen3.5-9b@q4_k_m",
                "qwen/qwen3.5-9b@q6_k",
                "qwen/qwen3.5-9b@q8_0",
            ],
        )
        self.assertEqual([target.quantization for target in targets], ["Q4_K_M", "Q6_K", "Q8_0"])

    def test_parse_footprint_output_extracts_core_fields(self) -> None:
        parsed = parse_footprint_output(
            """
======================================================================
node [334]: 64-bit    Footprint: 3742726664 B (16384 bytes per page)
======================================================================
      Dirty         Clean   Reclaimable    Regions    Category
        ---           ---           ---        ---    ---
3288039424 B           0 B           0 B         57    untagged (VM_ALLOCATE)
        0 B  5628198912 B           0 B          9    mapped file
        ---           ---           ---        ---    ---
3742726664 B  5678694400 B       98304 B       4558    TOTAL

Auxiliary data:
    phys_footprint: 3742759432 B
    phys_footprint_peak: 3742775816 B
"""
        )
        self.assertEqual(parsed["footprint_bytes"], 3742726664)
        self.assertEqual(parsed["phys_footprint_bytes"], 3742759432)
        self.assertEqual(parsed["phys_footprint_peak_bytes"], 3742775816)
        self.assertEqual(parsed["total_clean_bytes"], 5678694400)

    def test_parse_footprint_output_accepts_mb_units(self) -> None:
        parsed = parse_footprint_output(
            """
======================================================================
node [18918]: 64-bit    Footprint: 493 MB (16384 bytes per page)
======================================================================
  Dirty      Clean  Reclaimable    Regions    Category
    ---        ---          ---        ---    ---
 144 MB        0 B          0 B         18    Owned physical footprint (unmapped) (graphics)
    ---        ---          ---        ---    ---
 493 MB     100 MB       288 KB       1000    TOTAL

Auxiliary data:
    phys_footprint: 494 MB
    phys_footprint_peak: 500 MB
"""
        )
        self.assertEqual(parsed["footprint_bytes"], 493 * 1024 * 1024)
        self.assertEqual(parsed["phys_footprint_bytes"], 494 * 1024 * 1024)
        self.assertEqual(parsed["total_reclaimable_bytes"], 288 * 1024)

    def test_run_generation_perf_pair_skips_memory_capture_when_disabled(self) -> None:
        load_record = {
            "model": "remote-model",
            "context_length": 4096,
            "load_duration_seconds": 1.23,
            "response": {"instance_id": "remote-model"},
        }
        run_one = {
            "status": "completed",
            "started_at": "2026-04-03T00:00:00+00:00",
            "ended_at": "2026-04-03T00:00:05+00:00",
            "wall_time_seconds": 5.0,
            "usage": {"completion_tokens": 128},
            "event_counts": {"chat.end": 1},
            "output_chars": 512,
            "message_chars": 512,
            "reasoning_chars": 0,
            "observed_first_output_seconds": 0.5,
        }

        with (
            patch("benchmark_suite.model_stream_perf.unload_all_models", side_effect=[[], []]) as unload_all_mock,
            patch("benchmark_suite.model_stream_perf.load_model_with_timing", return_value=load_record),
            patch("benchmark_suite.model_stream_perf._stream_chat", return_value=run_one) as stream_mock,
            patch("benchmark_suite.model_stream_perf.find_largest_lm_studio_node") as find_node_mock,
            patch("benchmark_suite.model_stream_perf.collect_memory_snapshots_during_run") as collect_snapshots_mock,
            patch("benchmark_suite.model_stream_perf.collect_memory_snapshot") as collect_snapshot_mock,
        ):
            result = run_generation_perf_pair(
                base_url="http://192.168.31.76:1234/api/v1",
                model_key="remote-model",
                context_length=4096,
                prompt="loop",
                temperature=0.1,
                snapshot_interval_seconds=60,
                idle_timeout_seconds=600,
                capture_memory=False,
            )

        self.assertEqual(unload_all_mock.call_count, 2)
        self.assertEqual(stream_mock.call_count, 1)
        find_node_mock.assert_not_called()
        collect_snapshots_mock.assert_not_called()
        collect_snapshot_mock.assert_not_called()
        self.assertFalse(result["capture_memory"])
        self.assertIsNone(result["pid"])
        self.assertEqual(result["first_run_full_stats"], run_one)
        self.assertEqual(result["second_run_control"]["status"], "skipped")
        self.assertEqual(result["second_run_control"]["reason"], "capture_memory_disabled")
        self.assertEqual(result["second_run_control"]["usage"], {})
        self.assertEqual(result["second_run_memory_snapshots"], [])

    def test_run_generation_perf_pair_supports_multiple_speed_runs(self) -> None:
        load_record = {
            "model": "remote-model",
            "context_length": 4096,
            "load_duration_seconds": 1.23,
            "strategy": "models_load",
            "response": {"instance_id": "remote-model"},
        }

        def fake_stream(**kwargs: object) -> dict[str, object]:
            return {
                "status": "completed",
                "started_at": "2026-04-03T00:00:00+00:00",
                "ended_at": "2026-04-03T00:00:05+00:00",
                "wall_time_seconds": 5.0,
                "usage": {"completion_tokens": 128},
                "event_counts": {"chat.end": 1},
                "output_chars": 512,
                "message_chars": 512,
                "reasoning_chars": 0,
                "observed_first_output_seconds": 0.5,
                "compact_output_timing": kwargs["compact_output_timing"],
                "output_token_times_seconds": [0.5, 0.6],
            }

        with (
            patch("benchmark_suite.model_stream_perf.unload_all_models", side_effect=[[], []]),
            patch("benchmark_suite.model_stream_perf.load_model_with_timing", return_value=load_record),
            patch("benchmark_suite.model_stream_perf._stream_chat", side_effect=fake_stream) as stream_mock,
        ):
            result = run_generation_perf_pair(
                base_url="http://192.168.31.76:1234/api/v1",
                model_key="remote-model",
                context_length=4096,
                prompt="loop",
                temperature=0.1,
                snapshot_interval_seconds=60,
                idle_timeout_seconds=600,
                capture_memory=False,
                speed_runs=3,
                compact_output_timing=True,
            )

        self.assertEqual(stream_mock.call_count, 3)
        for call in stream_mock.call_args_list:
            self.assertEqual(call.kwargs["temperature"], 0.1)
        self.assertEqual(len(result["speed_runs"]), 3)
        self.assertTrue(result["compact_output_timing"])
        self.assertEqual(result["speed_runs_requested"], 3)
        self.assertEqual(result["first_run_full_stats"], result["speed_runs"][0])
        self.assertEqual(result["temperature"], 0.1)

    def test_run_generation_perf_pair_supports_memory_only_run(self) -> None:
        load_record = {
            "model": "remote-model",
            "context_length": 4096,
            "load_duration_seconds": 1.23,
            "strategy": "models_load",
            "response": {"instance_id": "remote-model"},
        }
        memory_run = {
            "status": "completed",
            "started_at": "2026-04-03T00:00:00+00:00",
            "ended_at": "2026-04-03T00:00:05+00:00",
            "wall_time_seconds": 5.0,
            "usage": {"completion_tokens": 128},
            "event_counts": {"chat.end": 1},
            "output_chars": 512,
            "message_chars": 512,
            "reasoning_chars": 0,
            "observed_first_output_seconds": 0.5,
        }

        class FinishedThread:
            def join(self, timeout: float | None = None) -> None:
                return None

        with (
            patch("benchmark_suite.model_stream_perf.unload_all_models", side_effect=[[], []]),
            patch("benchmark_suite.model_stream_perf.load_model_with_timing", return_value=load_record),
            patch("benchmark_suite.model_stream_perf._stream_chat", return_value=memory_run) as stream_mock,
            patch("benchmark_suite.model_stream_perf.find_largest_lm_studio_node", return_value={"pid": 123}),
            patch("benchmark_suite.model_stream_perf.collect_memory_snapshots_during_run", return_value=FinishedThread()),
            patch(
                "benchmark_suite.model_stream_perf.collect_memory_snapshot",
                return_value={"rss_bytes": 1024, "is_final_snapshot": True},
            ),
        ):
            result = run_generation_perf_pair(
                base_url="http://192.168.31.76:1234/api/v1",
                model_key="remote-model",
                context_length=4096,
                prompt="loop",
                temperature=0.1,
                snapshot_interval_seconds=60,
                idle_timeout_seconds=600,
                capture_memory=True,
                speed_runs=0,
                compact_output_timing=True,
            )

        self.assertEqual(stream_mock.call_count, 1)
        self.assertEqual(result["speed_runs"], [])
        self.assertEqual(result["speed_runs_requested"], 0)
        self.assertEqual(result["first_run_full_stats"]["status"], "skipped")
        self.assertEqual(result["first_run_full_stats"]["reason"], "speed_runs_disabled")
        self.assertEqual(result["second_run_control"]["usage"], {"completion_tokens": 128})
        self.assertEqual(result["second_run_memory_snapshots"], [{"rss_bytes": 1024, "is_final_snapshot": True}])


if __name__ == "__main__":
    unittest.main()
