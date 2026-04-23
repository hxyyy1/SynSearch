from __future__ import annotations

import csv
import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from alphasyn.mcts import run_search as run_mcts
from alphasyn.types import BackendResult, BaselineInfo, SearchConfig
from hybridsyn import cli as hybridsyn_cli
from hybridsyn.search import HybridSearchConfig, run_search


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

    def get_peak_memory_kb(self) -> float | None:
        return 2048.0

    def cleanup_cache(self) -> None:
        return None


class FakeCLIBackend(FakeBackend):
    COUNTS = {
        (): 100,
        ("balance",): 95,
        ("rewrite",): 80,
        ("rewrite-z",): 94,
        ("refactor",): 93,
        ("refactor-z",): 92,
        ("resub",): 91,
        ("resub-z",): 90,
        ("rewrite", "balance"): 60,
        ("rewrite", "rewrite"): 70,
        ("rewrite", "rewrite-z"): 72,
        ("rewrite", "refactor"): 68,
        ("rewrite", "refactor-z"): 66,
        ("rewrite", "resub"): 64,
        ("rewrite", "resub-z"): 62,
    }

    def __init__(
        self,
        abc_bin: str | None,
        workdir: Path,
    ) -> None:
        del abc_bin
        super().__init__(self.COUNTS, workdir)


class HybridSynTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = Path.cwd() / ".test_artifacts" / self._testMethodName
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.test_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def _write_design(self, relative_path: str = "toy.blif") -> Path:
        design_path = self.test_root / relative_path
        design_path.parent.mkdir(parents=True, exist_ok=True)
        design_path.write_text(".model toy\n.end\n", encoding="ascii")
        return design_path

    def test_hybrid_handoff_and_mcts_continuation(self) -> None:
        and_counts = {
            (): 100,
            ("balance",): 95,
            ("rewrite",): 80,
            ("balance", "balance"): 70,
            ("balance", "rewrite"): 75,
            ("rewrite", "balance"): 60,
            ("rewrite", "rewrite"): 65,
        }
        design_path = self._write_design()
        backend = FakeBackend(and_counts, self.test_root)
        result = run_search(
            HybridSearchConfig(
                design_name="toy",
                design_path=design_path,
                action_space=("balance", "rewrite"),
                sequence_length=2,
                warmup_steps=1,
                warmup_episodes=2,
                search_iterations=4,
                cpuct=1.0,
                seed=0,
                debug_search=True,
                workdir=self.test_root,
            ),
            backend,
        )

        self.assertEqual(result.metadata["warmup"]["selected_prefix"], ["rewrite"])
        self.assertEqual(len(result.metadata["warmup"]["top_candidates"]), 1)
        self.assertEqual(result.sequence, ("rewrite", "balance"))
        self.assertEqual(result.final_and_count, 60)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].step_index, 2)
        self.assertEqual(result.steps[0].selected_action, "balance")

    def test_top_k_warmup_candidates_allow_mcts_to_recover_better_branch(self) -> None:
        and_counts = {
            (): 100,
            ("balance",): 75,
            ("rewrite",): 80,
            ("balance", "balance"): 70,
            ("balance", "rewrite"): 72,
            ("rewrite", "balance"): 40,
            ("rewrite", "rewrite"): 75,
        }
        design_path = self._write_design()
        backend = FakeBackend(and_counts, self.test_root)
        result = run_search(
            HybridSearchConfig(
                design_name="toy",
                design_path=design_path,
                action_space=("balance", "rewrite"),
                sequence_length=2,
                warmup_steps=1,
                warmup_episodes=2,
                warmup_top_k=2,
                search_iterations=4,
                cpuct=1.0,
                seed=0,
                debug_search=True,
                workdir=self.test_root,
            ),
            backend,
        )

        self.assertEqual(
            [candidate["prefix"] for candidate in result.metadata["warmup"]["top_candidates"]],
            [["balance"], ["rewrite"]],
        )
        self.assertEqual(result.metadata["selected_warmup_candidate"]["prefix"], ["rewrite"])
        self.assertEqual(result.sequence, ("rewrite", "balance"))
        self.assertEqual(result.final_and_count, 40)

    def test_warmup_zero_matches_pure_mcts(self) -> None:
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
        design_path = self._write_design()
        backend = FakeBackend(and_counts, self.test_root)
        hybrid_result = run_search(
            HybridSearchConfig(
                design_name="toy",
                design_path=design_path,
                action_space=("balance", "rewrite", "rewrite-z"),
                sequence_length=2,
                warmup_steps=0,
                warmup_episodes=4,
                search_iterations=4,
                cpuct=1.0,
                seed=0,
                debug_search=True,
                workdir=self.test_root,
            ),
            backend,
        )
        pure_result = run_mcts(
            SearchConfig(
                design_name="toy",
                design_path=design_path,
                action_space=("balance", "rewrite", "rewrite-z"),
                sequence_length=2,
                search_iterations=4,
                cpuct=1.0,
                seed=0,
                debug_search=True,
                workdir=self.test_root,
            ),
            backend,
        )

        self.assertEqual(hybrid_result.sequence, pure_result.sequence)
        self.assertEqual(hybrid_result.final_and_count, pure_result.final_and_count)
        self.assertEqual(hybrid_result.final_lev_count, pure_result.final_lev_count)
        self.assertEqual(
            [step.selected_action for step in hybrid_result.steps],
            [step.selected_action for step in pure_result.steps],
        )

    def test_warmup_clamps_to_sequence_length_for_bandit_only_search(self) -> None:
        and_counts = {
            (): 100,
            ("balance",): 90,
            ("rewrite",): 80,
        }
        design_path = self._write_design()
        backend = FakeBackend(and_counts, self.test_root)
        result = run_search(
            HybridSearchConfig(
                design_name="toy",
                design_path=design_path,
                action_space=("balance", "rewrite"),
                sequence_length=1,
                warmup_steps=4,
                warmup_episodes=2,
                search_iterations=3,
                seed=0,
                workdir=self.test_root,
            ),
            backend,
        )

        self.assertEqual(result.metadata["warmup"]["effective_warmup_steps"], 1)
        self.assertEqual(result.sequence, ("rewrite",))
        self.assertEqual(result.steps, ())

    def test_handoff_tie_break_is_deterministic(self) -> None:
        and_counts = {
            (): 100,
            ("balance",): 80,
            ("rewrite",): 80,
        }
        design_path = self._write_design()
        backend = FakeBackend(and_counts, self.test_root)
        result = run_search(
            HybridSearchConfig(
                design_name="toy",
                design_path=design_path,
                action_space=("balance", "rewrite"),
                sequence_length=1,
                warmup_steps=1,
                warmup_episodes=2,
                search_iterations=2,
                seed=0,
                workdir=self.test_root,
            ),
            backend,
        )

        self.assertEqual(result.metadata["warmup"]["selected_prefix"], ["balance"])
        self.assertEqual(result.sequence, ("balance",))

    def test_cli_run_search_and_summarize_write_expected_artifacts(self) -> None:
        dataset_root = self.test_root / "dataset"
        design_path = dataset_root / "input.blif"
        design_path.parent.mkdir(parents=True, exist_ok=True)
        design_path.write_text(".model toy\n.end\n", encoding="ascii")
        workdir = self.test_root / ".hybridsyn_work"

        stdout = io.StringIO()
        with patch.object(hybridsyn_cli, "ABCBackend", FakeCLIBackend):
            with redirect_stdout(stdout):
                exit_code = hybridsyn_cli.main(
                    [
                        "run-search",
                        f"--workdir={workdir}",
                        f"--dataset-root={dataset_root}",
                        "--sequence-length=2",
                        "--warmup-steps=1",
                        "--warmup-episodes=7",
                        "--search-iterations=2",
                        "--debug-search",
                    ]
                )
        self.assertEqual(exit_code, 0)
        progress_output = stdout.getvalue()
        self.assertIn("[input.blif] start and=100 lev=10 warmup=1x7 topk=1 mcts=2", progress_output)
        self.assertIn("[input.blif] warmup 7/7 best=80/8 prefix=rewrite", progress_output)
        self.assertIn("[input.blif] branch 1/1 handoff and=80 lev=8 prefix=rewrite", progress_output)
        self.assertIn("[input.blif] branch 1/1 step 2/2 balance -> 60/6", progress_output)
        self.assertIn("[input.blif] done and=60 lev=6 selected=rewrite", progress_output)

        results_dir = workdir / "results"
        json_paths = sorted(results_dir.glob("*.json"))
        warmup_paths = sorted(results_dir.glob("*.warmup.csv"))
        debug_paths = sorted(results_dir.glob("*.debug.csv"))
        trace_paths = sorted(results_dir.glob("*.trace.csv"))
        self.assertEqual(len(json_paths), 1)
        self.assertEqual(len(warmup_paths), 1)
        self.assertEqual(len(debug_paths), 1)
        self.assertEqual(len(trace_paths), 1)

        result_payload = json.loads(json_paths[0].read_text(encoding="ascii"))
        self.assertEqual(result_payload["sequence"], "rewrite; balance")
        self.assertEqual(result_payload["metadata"]["warmup"]["selected_prefix"], ["rewrite"])
        self.assertEqual(result_payload["metadata"]["selected_warmup_candidate"]["prefix"], ["rewrite"])

        summary_stdout = io.StringIO()
        summary_path = workdir / "summary.csv"
        with patch.object(hybridsyn_cli, "ABCBackend", FakeCLIBackend):
            with redirect_stdout(summary_stdout):
                summarize_exit = hybridsyn_cli.main(
                    [
                        "summarize",
                        f"--workdir={workdir}",
                        f"--output={summary_path}",
                    ]
                )
        self.assertEqual(summarize_exit, 0)
        self.assertTrue(summary_path.exists())
        with summary_path.open("r", encoding="ascii", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["design_name"], "input.blif")
        self.assertIn("warmup=1", rows[0]["variant_label"])
        self.assertIn("topk=1", rows[0]["variant_label"])
        self.assertIn("ucb_c=0.4", rows[0]["variant_label"])

    def test_summarize_sorts_designs_in_natural_order(self) -> None:
        workdir = self.test_root / ".hybridsyn_work"
        results_dir = workdir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        summary_path = workdir / "summary.csv"

        def write_result(design_name: str) -> None:
            payload = {
                "design_name": design_name,
                "seed": 0,
                "final_and": 10,
                "final_lev": 2,
                "total_runtime_sec": 1.0,
                "peak_memory_kb": None,
                "baseline": {
                    "initial_and": 12,
                    "initial_lev": 3,
                    "heuristic_and": 11,
                    "heuristic_lev": 2,
                },
                "metadata": {
                    "config": {
                        "sequence_length": 2,
                        "warmup_steps": 1,
                        "warmup_episodes": 2,
                        "warmup_top_k": 1,
                        "search_iterations": 2,
                        "ucb_c": 0.4,
                        "cpuct": 1.0,
                        "mu_discount": 0.9,
                    }
                },
            }
            result_path = results_dir / f"{design_name.replace('/', '__')}.json"
            result_path.write_text(json.dumps(payload, indent=2), encoding="ascii")

        write_result("tc_public_10/input.blif")
        write_result("tc_public_2/input.blif")
        write_result("tc_public_1/input.blif")

        summarize_exit = hybridsyn_cli.main(
            [
                "summarize",
                f"--workdir={workdir}",
                f"--output={summary_path}",
            ]
        )

        self.assertEqual(summarize_exit, 0)
        with summary_path.open("r", encoding="ascii", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            [row["design_name"] for row in rows],
            ["tc_public_1/input.blif", "tc_public_2/input.blif", "tc_public_10/input.blif"],
        )

    def test_cli_parser_exposes_external_monitor_flag(self) -> None:
        parser = hybridsyn_cli._build_parser()
        default_args = parser.parse_args(["run-search"])
        enabled_args = parser.parse_args(["run-search", "--external-monitor"])

        self.assertFalse(default_args.external_monitor)
        self.assertEqual(default_args.warmup_top_k, 1)
        self.assertTrue(enabled_args.external_monitor)


if __name__ == "__main__":
    unittest.main()
