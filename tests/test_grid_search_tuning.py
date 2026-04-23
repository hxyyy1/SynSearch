from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from scripts import grid_search_tuning


class GridSearchTuningTests(unittest.TestCase):
    def test_iter_grid_respects_param_order(self) -> None:
        combinations = grid_search_tuning.iter_grid(
            {
                "search-iterations": [10, 20],
                "sequence-length": [8],
                "cpuct": [0.5, 1.0],
            },
            ("sequence-length", "search-iterations", "cpuct"),
        )
        self.assertEqual(
            combinations,
            [
                {"sequence-length": 8, "search-iterations": 10, "cpuct": 0.5},
                {"sequence-length": 8, "search-iterations": 10, "cpuct": 1.0},
                {"sequence-length": 8, "search-iterations": 20, "cpuct": 0.5},
                {"sequence-length": 8, "search-iterations": 20, "cpuct": 1.0},
            ],
        )

    def test_format_trial_name_is_stable(self) -> None:
        name = grid_search_tuning._format_trial_name(
            3,
            {
                "steps": 10,
                "episodes": 20,
                "ucb-c": 0.4,
            },
        )
        self.assertEqual(name, "trial_003__episodes=20__steps=10__ucb_c=0.4")

    def test_rescore_all_trials_uses_global_compare_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            summary_a = output_root / "a.csv"
            summary_b = output_root / "b.csv"
            summary_a.write_text("design_name,final_and,final_lev,total_runtime_sec\nx,1,1,1\n", encoding="utf-8")
            summary_b.write_text("design_name,final_and,final_lev,total_runtime_sec\nx,1,1,1\n", encoding="utf-8")
            rows = [
                {
                    "algorithm": "alphasyn",
                    "trial_index": 1,
                    "trial_name": "trial_a",
                    "variant_label": "sequence-length=10",
                    "summary_path": str(summary_a),
                },
                {
                    "algorithm": "baseline_mab",
                    "trial_index": 2,
                    "trial_name": "trial_b",
                    "variant_label": "steps=10",
                    "summary_path": str(summary_b),
                },
            ]

            fake_loaded_a = object()
            fake_loaded_b = object()
            fake_aggregate = [
                {
                    "algorithm": "AlphaSyn",
                    "variant_label": None,
                    "design_count": 3,
                    "final_score": 25.0,
                    "summary_path": str(summary_a.resolve()),
                },
                {
                    "algorithm": "MABSyn-UCB1",
                    "variant_label": None,
                    "design_count": 3,
                    "final_score": 20.0,
                    "summary_path": str(summary_b.resolve()),
                },
            ]
            fake_details = [{"design": "x", "rank": 1}]

            with (
                mock.patch.object(
                    grid_search_tuning.cas,
                    "load_summaries",
                    side_effect=[[fake_loaded_a], [fake_loaded_b]],
                ) as load_summaries,
                mock.patch.object(
                    grid_search_tuning.cas,
                    "compare_summaries",
                    return_value=(fake_aggregate, fake_details, ["x"], ["and"], 1.0),
                ) as compare_summaries,
            ):
                rescored = grid_search_tuning._rescore_all_trials(rows, output_root=output_root)

            self.assertEqual(len(rescored), 2)
            self.assertEqual(rescored[0]["global_rank"], 1)
            self.assertEqual(rescored[0]["algorithm"], "alphasyn")
            self.assertEqual(rescored[0]["final_score"], 25.0)
            self.assertEqual(rescored[1]["global_rank"], 2)
            self.assertEqual(rescored[1]["algorithm"], "baseline_mab")
            self.assertEqual(rescored[1]["design_count"], 3)
            self.assertTrue((output_root / "global_compare" / "aggregate.csv").exists())
            self.assertTrue((output_root / "global_compare" / "details.csv").exists())
            self.assertEqual(load_summaries.call_count, 2)
            compare_summaries.assert_called_once()


if __name__ == "__main__":
    unittest.main()
