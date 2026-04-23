from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from alphasyn.backend import ABCBackend, immediate_reward, parse_abc_and_count, parse_abc_stats
from alphasyn.mcts import run_search
from alphasyn.types import BackendResult, BaselineInfo, SearchConfig, format_sequence_for_abc


class FakeBackend:
    def __init__(self, and_count_map: dict[tuple[str, ...], int], workdir: Path) -> None:
        self.and_count_map = and_count_map
        self.workdir = workdir

    def version(self) -> str:
        return "fake-abc"

    def compute_baseline(self, design_path: Path) -> BaselineInfo:
        return BaselineInfo(
            initial_and_count=self.and_count_map[()],
            initial_lev_count=max(0, self.and_count_map[()] // 10),
            and_baseline=10.0,
            lev_baseline=1.0,
            heuristic_and_count=self.and_count_map[()],
            heuristic_lev_count=max(0, self.and_count_map[()] // 10),
            heuristic_steps=12,
        )

    def evaluate_prefix(self, design_path: Path, prefix: tuple[str, ...]) -> BackendResult:
        and_count = self.and_count_map[prefix]
        snapshot_path = self.workdir / f"{'_'.join(prefix) or 'root'}.aig"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text("fake", encoding="ascii")
        return BackendResult(
            and_count=and_count,
            lev_count=max(0, and_count // 10),
            runtime_sec=0.01,
            snapshot_path=snapshot_path,
            log="fake",
            cache_hit=False,
        )


class CoreTests(unittest.TestCase):
    def test_parse_abc_and_count_supports_multiple_formats(self) -> None:
        self.assertEqual(parse_abc_and_count("and = 42 lev = 9"), 42)
        self.assertEqual(parse_abc_and_count("i/o = 3/1 nd = 77 edge = 110"), 77)
        self.assertEqual(parse_abc_stats("and = 42 lev = 9"), (42, 9))

    def test_immediate_reward_blends_and_and_lev_without_rescaling(self) -> None:
        self.assertAlmostEqual(
            immediate_reward(100, 90, 20, 16, 5.0, 2.0),
            2**0.5,
        )
        self.assertAlmostEqual(
            immediate_reward(90, 100, 16, 20, 5.0, 2.0),
            -(2**0.5),
        )

    def test_search_prefers_best_q_plus_r_and_reuses_tree(self) -> None:
        and_counts = {
            (): 100,
            ("balance",): 95,
            ("rewrite",): 80,
            ("rewrite-z",): 99,
            ("balance", "balance"): 94,
            ("balance", "rewrite"): 70,
            ("balance", "rewrite-z"): 98,
            ("rewrite", "balance"): 60,
            ("rewrite", "rewrite"): 65,
            ("rewrite", "rewrite-z"): 78,
        }
        test_workdir = Path.cwd() / ".test_artifacts" / "test_search_prefers_best_q_plus_r_and_reuses_tree"
        if test_workdir.exists():
            shutil.rmtree(test_workdir)
        test_workdir.mkdir(parents=True, exist_ok=True)
        backend = FakeBackend(and_counts, test_workdir)
        design_path = test_workdir / "toy.blif"
        design_path.write_text(".model toy\n.end\n", encoding="ascii")
        config = SearchConfig(
            design_name="toy",
            design_path=design_path,
            action_space=("balance", "rewrite", "rewrite-z"),
            sequence_length=2,
            search_iterations=4,
            cpuct=1.0,
            mu_discount=0.9,
            seed=0,
            debug_search=True,
            workdir=test_workdir,
        )
        try:
            result = run_search(config, backend)
        finally:
            shutil.rmtree(test_workdir)

        self.assertEqual(result.sequence, ("rewrite", "balance"))
        self.assertEqual(result.final_and_count, 60)
        self.assertEqual(result.final_lev_count, 6)
        self.assertEqual(result.steps[0].selected_action, "rewrite")
        self.assertEqual(result.steps[1].selected_action, "balance")
        self.assertEqual(result.steps[0].lev_count, 8)
        self.assertGreater(result.steps[0].action_scores["rewrite"], result.steps[0].action_scores["balance"])
        payload = result.to_json_dict()
        self.assertEqual(payload["final_and"], 60)
        self.assertEqual(payload["final_lev"], 6)
        self.assertEqual(payload["baseline"]["initial_and"], 100)
        self.assertEqual(payload["steps"][0]["and"], 80)
        self.assertEqual(payload["steps"][0]["lev"], 8)
        self.assertEqual(payload["steps"][0]["root_visits"], 3)
        self.assertIn("q", payload["steps"][0]["action_debug"]["rewrite"])
        self.assertIn("r", payload["steps"][0]["action_debug"]["rewrite"])
        self.assertIn("u", payload["steps"][0]["action_debug"]["rewrite"])
        self.assertIn("selection_score", payload["steps"][0]["action_debug"]["rewrite"])
        self.assertTrue(payload["steps"][0]["iteration_traces"])
        self.assertIn("nodes", payload["steps"][0]["iteration_traces"][0])
        self.assertEqual(payload["sequence"], "rewrite; balance")
        self.assertEqual(format_sequence_for_abc(result.sequence), "rewrite; balance")

    def test_backend_skips_peak_memory_collection_by_default(self) -> None:
        test_workdir = Path.cwd() / ".test_artifacts" / "test_backend_skips_peak_memory_collection_by_default"
        if test_workdir.exists():
            shutil.rmtree(test_workdir)
        test_workdir.mkdir(parents=True, exist_ok=True)
        backend = ABCBackend("abc", test_workdir)
        completed = subprocess.CompletedProcess(
            ["abc", "-c", "print_stats"],
            0,
            stdout="and = 42 lev = 9\n",
            stderr="",
        )
        try:
            with patch("alphasyn.backend.subprocess.run", return_value=completed) as run_mock:
                result = backend._run_and_cache(
                    key="demo",
                    read_command="read_blif /tmp/demo.blif",
                    commands=("strash",),
                )
            self.assertEqual(result.and_count, 42)
            self.assertEqual(result.lev_count, 9)
            self.assertIsNone(result.peak_memory_kb)
            run_mock.assert_called_once()
        finally:
            backend.cleanup_cache()
            shutil.rmtree(test_workdir, ignore_errors=True)

    def test_backend_ignores_legacy_peak_memory_flag(self) -> None:
        test_workdir = Path.cwd() / ".test_artifacts" / "test_backend_ignores_legacy_peak_memory_flag"
        if test_workdir.exists():
            shutil.rmtree(test_workdir)
        test_workdir.mkdir(parents=True, exist_ok=True)
        backend = ABCBackend("abc", test_workdir)
        completed = subprocess.CompletedProcess(
            ["abc", "-c", "print_stats"],
            0,
            stdout="and = 30 lev = 7\n",
            stderr="",
        )
        try:
            with patch("alphasyn.backend.subprocess.run", return_value=completed) as run_mock:
                result = backend._run_and_cache(
                    key="demo",
                    read_command="read_blif /tmp/demo.blif",
                    commands=("strash",),
                )
            self.assertEqual(result.and_count, 30)
            self.assertEqual(result.lev_count, 7)
            self.assertIsNone(result.peak_memory_kb)
            self.assertIsNone(backend.get_peak_memory_kb())
            run_mock.assert_called_once()
        finally:
            backend.cleanup_cache()
            shutil.rmtree(test_workdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
