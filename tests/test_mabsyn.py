from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from MABSyn import baseline_mab, baseline_mab_prefix, linucb, result_utils, summarize_results


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
        parsed_monitor = baseline_mab.parse_optional_kv_args(["--external-monitor", "--steps", "4"])
        self.assertEqual(parsed_monitor["external-monitor"], "true")
        prefix_args = baseline_mab_prefix._build_parser().parse_args(["run-search"])
        self.assertEqual(prefix_args.ucb_c, baseline_mab_prefix.DEFAULT_BANDIT_C)
        self.assertFalse(prefix_args.external_monitor)
        self.assertTrue(
            baseline_mab_prefix._build_parser().parse_args(["run-search", "--external-monitor"]).external_monitor
        )
        self.assertFalse(baseline_mab._build_parser().parse_args(["run-search"]).external_monitor)
        self.assertTrue(baseline_mab._build_parser().parse_args(["run-search", "--external-monitor"]).external_monitor)
        self.assertFalse(linucb._build_parser().parse_args(["run-search"]).external_monitor)
        self.assertTrue(linucb._build_parser().parse_args(["run-search", "--external-monitor"]).external_monitor)

    def test_baseline_parse_stats_and_reward(self) -> None:
        nodes, level = baseline_mab.parse_abc_stats("i/o = 3/1 nd = 77 lev = 9")
        self.assertEqual(nodes, 77)
        self.assertEqual(level, 9)
        self.assertAlmostEqual(
            baseline_mab.compute_reward(100, 80, 20, 16, True),
            20 ** 0.5 / 10,
        )
        self.assertAlmostEqual(
            baseline_mab.compute_reward(100, 80, 20, 10, True, and_reward_weight=0.5, lev_reward_weight=0.5),
            ((20 / 100) ** 0.5 + (10 / 20) ** 0.5) / 2,
        )
        self.assertEqual(
            baseline_mab.compute_reward(100, None, 20, None, False),
            baseline_mab.SEVERE_NEGATIVE_REWARD,
        )
        self.assertTrue(
            baseline_mab.is_better_result(
                reference_nodes=100,
                reference_level=20,
                candidate_nodes=82,
                candidate_level=10,
                incumbent_nodes=80,
                incumbent_level=16,
                and_reward_weight=0.5,
                lev_reward_weight=0.5,
            )
        )
        self.assertFalse(
            baseline_mab.is_better_result(
                reference_nodes=100,
                reference_level=20,
                candidate_nodes=80,
                candidate_level=16,
                incumbent_nodes=82,
                incumbent_level=10,
                and_reward_weight=0.5,
                lev_reward_weight=0.5,
            )
        )

    def test_debug_trace_csv_and_json_are_written_for_synthetic_result(self) -> None:
        result = {
            "benchmark": "design_1/input.blif",
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
        debug_csv_path = Path(result_utils.get_results_root()) / "baseline_mab" / "design_1__input.debug.csv"
        debug_json_path = Path(result_utils.get_results_root()) / "baseline_mab" / "design_1__input.debug.json"
        self.assertTrue(debug_csv_path.exists())
        self.assertTrue(debug_json_path.exists())
        with debug_csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["action"], "rewrite")
        self.assertEqual(rows[0]["episode_sequence"], "rewrite")
        with debug_json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["benchmark"], "design_1/input.blif")
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

    def test_baseline_prefix_cache_reuses_snapshots(self) -> None:
        calls: list[str] = []

        def fake_subprocess_run(args, capture_output, text, timeout, check):
            self.assertTrue(capture_output)
            self.assertTrue(text)
            self.assertFalse(check)
            self.assertEqual(timeout, 30)
            command_string = args[2]
            calls.append(command_string)
            match = re.search(r"write_aiger\s+('([^']+)'|([^;]+))", command_string)
            self.assertIsNotNone(match)
            raw_path = match.group(2) or match.group(3)
            snapshot_path = Path(str(raw_path).strip())
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text("mock aig", encoding="utf-8")
            if "read_blif" in command_string:
                stdout = "and = 100 lev = 20"
            else:
                stdout = "and = 90 lev = 18"
                self.assertIn("read_aiger", command_string)
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")

        with patch.object(
            baseline_mab_prefix.subprocess,
            "run",
            side_effect=fake_subprocess_run,
        ):
            cache = baseline_mab_prefix.PrefixCache("/tmp/mock.blif", abc_bin="abc", timeout_sec=30)
            try:
                root = cache.get_or_build(())
                child = cache.get_or_build(("balance",))
                child_cached = cache.get_or_build(("balance",))
            finally:
                cache.close()

        self.assertTrue(root.success)
        self.assertTrue(child.success)
        self.assertTrue(child_cached.success)
        self.assertEqual(len(calls), 2)
        self.assertEqual(child.snapshot_path, child_cached.snapshot_path)
        self.assertEqual(child.nodes, 90)
        self.assertEqual(child.level, 18)

    def test_baseline_prefix_best_recipe_uses_full_length_sequence(self) -> None:
        class FakeBandit:
            next_instance = 0

            def __init__(self, action_list, c=0.0, rng=None):
                del c, rng
                self.actions = list(action_list)
                self.total_steps = 0
                self.counts = [0 for _ in self.actions]
                self.q_values = [0.0 for _ in self.actions]
                self.selected_index = FakeBandit.next_instance
                FakeBandit.next_instance += 1

            def select_action(self) -> int:
                return self.selected_index

            def update(self, trial_index, reward) -> None:
                del trial_index, reward

        class FakePrefixCache:
            def __init__(self, blif_path, abc_bin, timeout_sec):
                del blif_path, abc_bin, timeout_sec
                self.peak_memory_kb = None
                self.entries = {
                    (): baseline_mab_prefix.PrefixCacheEntry(
                        prefix=(),
                        nodes=100,
                        level=20,
                        snapshot_path="/tmp/root.aig",
                        success=True,
                        runtime_sec=0.01,
                        log="root",
                        cache_hit=False,
                        peak_memory_kb=None,
                    ),
                    ("balance",): baseline_mab_prefix.PrefixCacheEntry(
                        prefix=("balance",),
                        nodes=50,
                        level=10,
                        snapshot_path="/tmp/balance.aig",
                        success=True,
                        runtime_sec=0.01,
                        log="balance",
                        cache_hit=False,
                        peak_memory_kb=None,
                    ),
                    ("balance", "rewrite"): baseline_mab_prefix.PrefixCacheEntry(
                        prefix=("balance", "rewrite"),
                        nodes=90,
                        level=18,
                        snapshot_path="/tmp/balance_rewrite.aig",
                        success=True,
                        runtime_sec=0.01,
                        log="balance rewrite",
                        cache_hit=False,
                        peak_memory_kb=None,
                    ),
                }

            def get_or_build(self, prefix):
                return self.entries[prefix]

            def close(self) -> None:
                return None

        FakeBandit.next_instance = 0
        with patch.object(baseline_mab_prefix, "PrefixCache", FakePrefixCache):
            with patch.object(baseline_mab_prefix, "UCBBandit", FakeBandit):
                with patch.object(baseline_mab_prefix, "actions", ["balance", "rewrite"]):
                    with patch.object(baseline_mab_prefix, "K_STEPS", 2):
                        with patch.object(baseline_mab_prefix, "N_EPISODES", 1):
                            result = baseline_mab_prefix.optimize_one_benchmark(
                                "/tmp/mock.blif",
                                benchmark_name="toy.blif",
                            )

        self.assertEqual(result["best"]["recipe_str"], "balance; rewrite")
        self.assertEqual(result["best"]["nodes"], 90)
        self.assertEqual(result["best"]["level"], 18)

    def test_result_utils_writes_under_workdir(self) -> None:
        workdir = result_utils.get_workdir()
        self.assertTrue(workdir.endswith(".mabsyn_work"))
        path = result_utils.write_benchmark_result(
            "baseline_mab",
            {
                "benchmark": "design_1/input.blif",
                "best": {"nodes": 10, "level": 3},
                "runtime_sec": 0.25,
                "peak_memory_kb": 2048,
            },
        )
        self.assertTrue(path.startswith(workdir))
        self.assertTrue(path.endswith("results/baseline_mab/design_1__input.json"))
        self.assertTrue(Path(path).exists())

    def test_summarize_results_generates_csv_for_synthetic_rows(self) -> None:
        result_utils.write_benchmark_result(
            "baseline_mab",
            {
                "benchmark": "design_1/input.blif",
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
                "variant_label",
                "and",
                "lev",
                "runtime_sec",
                "peak_memory_kb",
            ],
        )
        self.assertEqual(rows[1][0], "design_1/input.blif")
        self.assertIn("steps=", rows[1][1])
        self.assertEqual(rows[1][2], "12")
        self.assertEqual(rows[1][5], "2048")
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
            "benchmark": "design_1/input.blif",
            "status": "ok",
            "initial": {"nodes": 10, "level": 2},
            "best": {"nodes": 9, "level": 2, "recipe_str": "balance"},
            "improvement": {"nodes_reduced": 1, "ratio": 0.1},
            "runtime_sec": 0.2,
            "peak_memory_kb": 1024,
        }
        linucb_result = {
            "benchmark": "design_1/input.blif",
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
            return_value={"design_1/input.blif": "/tmp/mock.blif"},
        ):
            with patch.object(baseline_mab, "optimize_one_benchmark", return_value=baseline_result):
                baseline_mab.main(["run-search", "--workdir", result_utils.get_workdir()])

        with patch.object(
            linucb,
            "_discover_designs_or_exit",
            return_value={"design_1/input.blif": "/tmp/mock.blif"},
        ):
            with patch.object(linucb, "optimize_one_benchmark", return_value=linucb_result):
                linucb.main(["run-search", "--workdir", result_utils.get_workdir()])

        baseline_summary = Path(result_utils.get_results_root()) / "baseline_mab" / "_summary.json"
        linucb_summary = Path(result_utils.get_results_root()) / "linucb" / "_summary.json"
        self.assertTrue(baseline_summary.exists())
        self.assertTrue(linucb_summary.exists())
