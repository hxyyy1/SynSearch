from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from MABSyn import baseline_mab, linucb, result_utils, summarize_results


class MABSynTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = Path.cwd() / ".test_artifacts" / self._testMethodName
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.original_workdir = result_utils.get_workdir()
        result_utils.set_workdir(str(self.test_root / ".mabsyn_work"))

    def tearDown(self) -> None:
        result_utils.set_workdir(self.original_workdir)
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def test_baseline_parse_optional_kv_args(self) -> None:
        parsed = baseline_mab.parse_optional_kv_args(
            ["--steps", "12", "--episodes", "8", "--workdir", ".mabsyn_work"]
        )
        self.assertEqual(parsed["steps"], "12")
        self.assertEqual(parsed["episodes"], "8")
        self.assertEqual(parsed["workdir"], ".mabsyn_work")
        parsed_flag = baseline_mab.parse_optional_kv_args(["--debug-search", "--steps", "4"])
        self.assertEqual(parsed_flag["debug-search"], "true")
        self.assertEqual(parsed_flag["steps"], "4")

    def test_baseline_parse_stats_and_reward(self) -> None:
        nodes, level = baseline_mab.parse_abc_stats("i/o = 3/1 nd = 77 lev = 9")
        self.assertEqual(nodes, 77)
        self.assertEqual(level, 9)
        self.assertAlmostEqual(
            baseline_mab.compute_reward(100, 80, 20, 16, True),
            0.2,
        )
        self.assertAlmostEqual(
            baseline_mab.compute_reward(100, 80, 20, 10, True, and_reward_weight=0.5, lev_reward_weight=0.5),
            0.35,
        )
        self.assertEqual(
            baseline_mab.compute_reward(100, None, 20, None, False),
            baseline_mab.SEVERE_NEGATIVE_REWARD,
        )

    def test_debug_trace_csv_and_json_are_written_for_synthetic_result(self) -> None:
        result = {
            "benchmark": "tc_public_1/input.blif",
            "status": "ok",
            "best": {"nodes": 10, "level": 3, "recipe_str": "balance"},
            "config": {
                "episodes": 1,
                "steps_per_episode": 1,
                "ucb_c": 2.0,
                "and_reward_weight": 0.5,
                "lev_reward_weight": 0.2,
                "seed": 2,
            },
            "debug_trace": [
                {
                    "episode": 1,
                    "step": 1,
                    "action_index": 0,
                    "action": "rewrite",
                    "selected": 1,
                    "count_before": 0,
                    "q_before": 0.0,
                    "bonus_before": None,
                    "score_before": None,
                    "cold_start": 1,
                    "count_after_select": 1,
                    "sequence_prefix": "rewrite",
                    "count_after_update": 1,
                    "q_after_update": 0.1,
                    "episode_reward": 0.1,
                    "episode_success": 1,
                    "episode_nodes": 9,
                    "episode_level": 3,
                    "episode_peak_memory_kb": 1024,
                    "episode_sequence": "rewrite",
                    "best_nodes_after_episode": 9,
                    "best_level_after_episode": 3,
                }
            ],
        }
        baseline_mab.write_result_artifacts(result)
        debug_csv_path = Path(result_utils.get_results_root()) / "baseline_mab" / "tc_public_1__input.debug.csv"
        debug_json_path = Path(result_utils.get_results_root()) / "baseline_mab" / "tc_public_1__input.debug.json"
        self.assertTrue(debug_csv_path.exists())
        self.assertTrue(debug_json_path.exists())
        with debug_csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["action"], "rewrite")
        self.assertEqual(rows[0]["episode_sequence"], "rewrite")
        with debug_json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["benchmark"], "tc_public_1/input.blif")
        self.assertEqual(payload["episodes"][0]["episode"], 1)
        self.assertEqual(payload["episodes"][0]["steps"][0]["selected_action"], "rewrite")
        self.assertEqual(payload["episodes"][0]["steps"][0]["actions"][0]["action"], "rewrite")

    def test_linucb_parse_helpers(self) -> None:
        kv = linucb.parse_optional_kv_args(["--alpha", "1.2", "--workdir", ".mabsyn_work"])
        self.assertEqual(kv["alpha"], "1.2")
        self.assertEqual(kv["workdir"], ".mabsyn_work")
        nodes, level = linucb.parse_abc_stats("and = 45 lev = 7")
        self.assertEqual((nodes, level), (45, 7))
        ave_fanout, max_fanout = linucb.parse_fanout_stats("fanouts: ave = 1.50 max = 7")
        self.assertEqual((ave_fanout, max_fanout), (1.5, 7.0))

    def test_result_utils_writes_under_workdir(self) -> None:
        workdir = result_utils.get_workdir()
        self.assertTrue(workdir.endswith(".mabsyn_work"))
        path = result_utils.write_benchmark_result(
            "baseline_mab",
            {
                "benchmark": "tc_public_1/input.blif",
                "best": {"nodes": 10, "level": 3},
                "runtime_sec": 0.25,
                "peak_memory_kb": 2048,
            },
        )
        self.assertTrue(path.startswith(workdir))
        self.assertTrue(path.endswith("results/baseline_mab/tc_public_1__input.json"))
        self.assertTrue(Path(path).exists())

    def test_summarize_results_generates_csv_for_synthetic_rows(self) -> None:
        result_utils.write_benchmark_result(
            "baseline_mab",
            {
                "benchmark": "tc_public_1/input.blif",
                "best": {"nodes": 12, "level": 4},
                "runtime_sec": 1.5,
                "peak_memory_kb": 2048,
            },
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            summarize_results.main(["baseline_mab", "--workdir", result_utils.get_workdir()])
        csv_path = Path(result_utils.get_results_root()) / "baseline_mab" / "summary.csv"
        self.assertTrue(csv_path.exists())
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(
            rows[0],
            [
                "file",
                "method",
                "variant_label",
                "steps",
                "iterations",
                "seed",
                "ucb_c",
                "alpha",
                "lambda",
                "and_reward_weight",
                "lev_reward_weight",
                "and",
                "lev",
                "runtime_sec",
                "peak_memory_kb",
            ],
        )
        self.assertEqual(rows[1][0], "tc_public_1/input.blif")
        self.assertEqual(rows[1][1], "baseline_mab")
        self.assertIn("method=baseline_mab", rows[1][2])
        self.assertEqual(rows[1][11], "12")
        self.assertEqual(rows[1][14], "2048")
        self.assertIn("CSV written to:", stdout.getvalue())

    def test_default_output_path_and_parent_creation(self) -> None:
        nested = result_utils.ensure_parent_dir(str(self.test_root / "nested" / "baseline.json"))
        Path(nested).write_text(json.dumps({"ok": True}), encoding="utf-8")
        self.assertTrue(Path(nested).exists())

    def test_result_json_outputs_are_opt_in(self) -> None:
        self.assertIsNone(result_utils.optional_output_path({}, "result-json"))
        baseline_explicit = result_utils.optional_output_path(
            {"result-json": str(self.test_root / "explicit" / "baseline.json")},
            "result-json",
        )
        linucb_explicit = result_utils.optional_output_path(
            {"result-json": str(self.test_root / "explicit" / "linucb.json")},
            "result-json",
        )
        self.assertIsNotNone(baseline_explicit)
        self.assertIsNotNone(linucb_explicit)
        self.assertTrue(str(baseline_explicit).endswith("explicit/baseline.json"))
        self.assertTrue(str(linucb_explicit).endswith("explicit/linucb.json"))

    def test_run_search_subcommand_handles_batch_without_aggregate_json(self) -> None:
        baseline_result = {
            "benchmark": "tc_public_1/input.blif",
            "status": "ok",
            "initial": {"nodes": 10, "level": 2},
            "best": {"nodes": 9, "level": 2, "recipe_str": "balance"},
            "improvement": {"nodes_reduced": 1, "ratio": 0.1},
            "runtime_sec": 0.2,
            "peak_memory_kb": 1024,
        }
        linucb_result = {
            "benchmark": "tc_public_1/input.blif",
            "status": "ok",
            "initial": {"nodes": 10, "level": 2},
            "best": {"nodes": 8, "level": 2, "recipe_str": "rewrite", "recipe_best": "rewrite"},
            "improvement": {"nodes_reduced": 2, "ratio": 0.2},
            "runtime_sec": 0.3,
            "peak_memory_kb": 2048,
        }

        with patch.object(
            baseline_mab,
            "_discover_designs_or_exit",
            return_value={"tc_public_1/input.blif": "/tmp/mock.blif"},
        ):
            with patch.object(baseline_mab, "optimize_one_benchmark", return_value=baseline_result):
                baseline_mab.main(["run-search", "--workdir", result_utils.get_workdir()])

        with patch.object(
            linucb,
            "_discover_designs_or_exit",
            return_value={"tc_public_1/input.blif": "/tmp/mock.blif"},
        ):
            with patch.object(linucb, "optimize_one_benchmark", return_value=linucb_result):
                linucb.main(["run-search", "--workdir", result_utils.get_workdir()])

        baseline_summary = Path(result_utils.get_results_root()) / "baseline_mab" / "_summary.json"
        linucb_summary = Path(result_utils.get_results_root()) / "linucb" / "_summary.json"
        self.assertTrue(baseline_summary.exists())
        self.assertTrue(linucb_summary.exists())
