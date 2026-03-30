from __future__ import annotations

import random
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from SASyn.core_types import BaselineInfo, BackendResult, SearchConfig
from SASyn.sa import (
    _estimate_initial_temperature,
    _temperature_at,
    energy_from_counts,
    initial_sequence,
    run_search,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

ACTION_DELTAS = {
    "balance": (1, 0),
    "rewrite": (3, 1),
    "rewrite-z": (2, 2),
    "refactor": (2, 1),
    "refactor-z": (1, 2),
    "resub": (4, 1),
    "resub-z": (3, 2),
}


class FakeSABackend:
    def version(self) -> str:
        return "fake-abc"

    def compute_baseline(self, design_path: Path) -> BaselineInfo:
        return BaselineInfo(
            initial_and_count=100,
            initial_lev_count=20,
            and_baseline=10.0,
            lev_baseline=2.0,
            heuristic_and_count=92,
            heuristic_lev_count=18,
            heuristic_steps=10,
        )

    def evaluate_sequence(self, design_path: Path, sequence: tuple[str, ...]) -> BackendResult:
        and_count = 100
        lev_count = 20
        for index, action in enumerate(sequence, start=1):
            and_delta, lev_delta = ACTION_DELTAS[action]
            and_count -= and_delta * index
            lev_count -= lev_delta
        return BackendResult(
            and_count=max(and_count, 1),
            lev_count=max(lev_count, 1),
            runtime_sec=0.01,
            log="fake",
        )


class SASynCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = REPO_ROOT / ".test_artifacts" / self._testMethodName
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.design_path = self.workdir / "toy.blif"
        self.design_path.write_text(".model toy\n.end\n", encoding="ascii")
        self.backend = FakeSABackend()

    def tearDown(self) -> None:
        if self.workdir.exists():
            shutil.rmtree(self.workdir)

    def test_energy_from_counts_uses_normalized_cost(self) -> None:
        baseline = self.backend.compute_baseline(self.design_path)
        self.assertAlmostEqual(
            energy_from_counts(80, 10, baseline, 0.7, 0.3),
            (0.7 * 0.8) + (0.3 * 0.5),
        )

    def test_initial_sequence_uses_seed_actions_and_truncates(self) -> None:
        config = SearchConfig(
            design_name="toy",
            design_path=self.design_path,
            action_space=("balance", "rewrite", "refactor"),
            sequence_length=6,
            workdir=self.workdir,
        )
        self.assertEqual(
            initial_sequence(config),
            ("balance", "rewrite", "refactor", "balance", "rewrite", "balance"),
        )

    def test_initial_temperature_falls_back_without_uphill_moves(self) -> None:
        baseline = self.backend.compute_baseline(self.design_path)
        config = SearchConfig(
            design_name="toy",
            design_path=self.design_path,
            action_space=("balance",),
            sequence_length=1,
            min_temperature=0.2,
            workdir=self.workdir,
        )
        sequence = ("balance",)
        current = self.backend.evaluate_sequence(self.design_path, sequence)
        current_energy = energy_from_counts(
            current.and_count,
            current.lev_count,
            baseline,
            config.and_weight,
            config.lev_weight,
        )
        temperature = _estimate_initial_temperature(
            config,
            baseline,
            self.backend,
            sequence,
            current_energy,
            random.Random(0),
        )
        self.assertEqual(temperature, 0.2)

    def test_temperature_schedule_handles_edge_cases(self) -> None:
        self.assertEqual(_temperature_at(0, 1, 2.0, 0.1), 2.0)
        self.assertAlmostEqual(_temperature_at(3, 4, 2.0, 0.1), 0.1)

    def test_run_search_is_reproducible_and_json_shape_is_stable(self) -> None:
        config = SearchConfig(
            design_name="toy",
            design_path=self.design_path,
            action_space=("balance", "rewrite", "resub"),
            sequence_length=4,
            search_iterations=12,
            seed=7,
            debug_search=True,
            workdir=self.workdir,
        )
        result_one = run_search(config, self.backend)
        result_two = run_search(config, self.backend)

        self.assertEqual(result_one.sequence, result_two.sequence)
        self.assertEqual(result_one.final_and_count, result_two.final_and_count)
        self.assertEqual(result_one.final_lev_count, result_two.final_lev_count)
        self.assertEqual(len(result_one.iterations), 12)

        payload = result_one.to_json_dict()
        self.assertEqual(payload["design_name"], "toy")
        self.assertEqual(payload["abc_version"], "fake-abc")
        self.assertEqual(payload["seed"], 7)
        self.assertIsInstance(payload["sequence"], str)
        self.assertIn("iterations", payload)
        self.assertEqual(len(payload["iterations"]), 12)
        self.assertIn("initial_temperature_used", payload["metadata"])


class SASynCLITests(unittest.TestCase):
    def test_help_commands(self) -> None:
        commands = [
            [],
            ["prepare-data"],
            ["run-search"],
            ["summarize"],
        ]
        for args in commands:
            with self.subTest(args=args):
                completed = subprocess.run(
                    [sys.executable, "-m", "SASyn", *args, "--help"],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout.lower())
                self.assertIn("sasyn", completed.stdout.lower())
