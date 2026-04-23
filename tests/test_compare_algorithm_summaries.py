from __future__ import annotations

import csv
import io
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import compare_algorithm_summaries


REPO_ROOT = Path(__file__).resolve().parents[1]


class CompareAlgorithmSummariesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = REPO_ROOT / ".test_artifacts" / self._testMethodName
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.workdir.exists():
            shutil.rmtree(self.workdir)

    def _write_csv(self, path: Path, headers: list[str], rows: list[list[object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)

    def test_default_summary_specs_include_hybridsyn(self) -> None:
        specs = compare_algorithm_summaries._parse_summary_specs([])
        self.assertEqual(
            [spec.name for spec in specs],
            ["MCTSyn", "HybridSyn", "SASyn", "MABSyn-UCB1", "MABSyn-UCB1Prefix"],
        )

    def test_score_for_rank_uses_minimum_threshold_after_sixth_place(self) -> None:
        self.assertEqual(compare_algorithm_summaries._score_for_rank(1), 10)
        self.assertEqual(compare_algorithm_summaries._score_for_rank(2), 9)
        self.assertEqual(compare_algorithm_summaries._score_for_rank(3), 8)
        self.assertEqual(compare_algorithm_summaries._score_for_rank(4), 7)
        self.assertEqual(compare_algorithm_summaries._score_for_rank(5), 6)
        self.assertEqual(compare_algorithm_summaries._score_for_rank(6), 5)
        self.assertEqual(compare_algorithm_summaries._score_for_rank(9), 5)

    def test_compare_summaries_uses_common_designs_and_writes_outputs(self) -> None:
        mct_path = self.workdir / "mct.csv"
        hybrid_path = self.workdir / "hybrid.csv"
        sa_path = self.workdir / "sa.csv"
        mab_path = self.workdir / "mab.csv"
        aggregate_path = self.workdir / "aggregate.csv"
        detail_path = self.workdir / "detail.csv"

        self._write_csv(
            mct_path,
            ["design_name", "final_and", "final_lev", "total_runtime_sec", "peak_memory_kb"],
            [
                ["d1", 100, 10, 5.0, 1000],
                ["d2", 200, 20, 10.0, 2000],
            ],
        )
        self._write_csv(
            hybrid_path,
            ["design_name", "final_and", "final_lev", "total_runtime_sec", "peak_memory_kb"],
            [
                ["d1", 95, 11, 4.5, 950],
                ["d2", 198, 19, 9.5, 1800],
            ],
        )
        self._write_csv(
            sa_path,
            ["design_name", "final_and", "final_lev", "total_runtime_sec", "peak_memory_kb"],
            [
                ["d1", 90, 12, 6.0, 900],
                ["d2", 210, 18, 9.0, 2500],
            ],
        )
        self._write_csv(
            mab_path,
            ["file", "variant_label", "and", "lev", "runtime_sec", "peak_memory_kb"],
            [
                ["d1", "method=baseline_mab,steps=10,iters=20", 120, 9, 4.0, 1100],
                ["d2", "method=baseline_mab,steps=10,iters=20", 205, 21, 8.0, 1500],
                ["d3", "method=baseline_mab,steps=10,iters=20", 999, 999, 999.0, 9999],
            ],
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = compare_algorithm_summaries.main(
                [
                    f"--summary=MCTSyn={mct_path}",
                    f"--summary=HybridSyn={hybrid_path}",
                    f"--summary=SASyn={sa_path}",
                    f"--summary=MABSyn={mab_path}",
                    f"--output={aggregate_path}",
                    f"--details-output={detail_path}",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertTrue(aggregate_path.exists())
        self.assertTrue(detail_path.exists())

        with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["algorithm"] for row in rows], ["HybridSyn", "SASyn", "MABSyn", "MCTSyn"])
        self.assertEqual(rows[0]["design_count"], "2")
        self.assertAlmostEqual(float(rows[0]["final_score"]), 18.1)
        self.assertAlmostEqual(float(rows[1]["final_score"]), 16.8)
        self.assertAlmostEqual(float(rows[2]["final_score"]), 16.6)
        self.assertAlmostEqual(float(rows[3]["final_score"]), 16.5)

        with detail_path.open("r", encoding="utf-8", newline="") as handle:
            detail_rows = list(csv.DictReader(handle))
        self.assertEqual(len(detail_rows), 8)
        d1_rows = [row for row in detail_rows if row["design"] == "d1"]
        self.assertEqual([row["algorithm"] for row in d1_rows], ["HybridSyn", "SASyn", "MABSyn", "MCTSyn"])
        self.assertEqual([row["rank"] for row in d1_rows], ["1", "1", "3", "3"])
        self.assertEqual(d1_rows[0]["and_rank"], "2")
        self.assertEqual(d1_rows[0]["and_score"], "9")
        mab_row = next(row for row in d1_rows if row["algorithm"] == "MABSyn")
        self.assertEqual(mab_row["variant_label"], "method=baseline_mab,steps=10,iters=20")

    def test_compare_summaries_treats_variant_labels_as_separate_entries(self) -> None:
        mct_path = self.workdir / "mct.csv"
        sa_path = self.workdir / "sa.csv"
        aggregate_path = self.workdir / "aggregate.csv"
        detail_path = self.workdir / "detail.csv"

        self._write_csv(
            mct_path,
            ["design_name", "variant_label", "final_and", "final_lev", "total_runtime_sec"],
            [
                ["d1", "cpuct=1.0", 100, 10, 5.0],
                ["d2", "cpuct=1.0", 110, 11, 5.5],
                ["d1", "cpuct=2.0", 95, 10, 4.5],
                ["d2", "cpuct=2.0", 105, 11, 4.8],
            ],
        )
        self._write_csv(
            sa_path,
            ["design_name", "variant_label", "final_and", "final_lev", "total_runtime_sec"],
            [
                ["d1", "baseline", 98, 10, 5.2],
                ["d2", "baseline", 108, 11, 5.0],
            ],
        )

        exit_code = compare_algorithm_summaries.main(
            [
                f"--summary=MCTSyn={mct_path}",
                f"--summary=SASyn={sa_path}",
                f"--output={aggregate_path}",
                f"--details-output={detail_path}",
            ]
        )

        self.assertEqual(exit_code, 0)
        with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [(row["algorithm"], row["variant_label"]) for row in rows],
            [("MCTSyn", "cpuct=2.0"), ("SASyn", "baseline"), ("MCTSyn", "cpuct=1.0")],
        )
        self.assertEqual([row["design_count"] for row in rows], ["2", "2", "2"])
        self.assertAlmostEqual(float(rows[0]["final_score"]), 18.0)
        self.assertAlmostEqual(float(rows[1]["final_score"]), 16.4)
        self.assertAlmostEqual(float(rows[2]["final_score"]), 15.4)

        with detail_path.open("r", encoding="utf-8", newline="") as handle:
            detail_rows = list(csv.DictReader(handle))
        self.assertEqual(len(detail_rows), 6)
        self.assertEqual(
            [(row["algorithm"], row["variant_label"]) for row in detail_rows if row["design"] == "d1"],
            [("MCTSyn", "cpuct=2.0"), ("SASyn", "baseline"), ("MCTSyn", "cpuct=1.0")],
        )
        self.assertEqual(
            [row["rank"] for row in detail_rows if row["design"] == "d1"],
            ["1", "2", "3"],
        )
        d1_rows = [row for row in detail_rows if row["design"] == "d1"]
        self.assertEqual([row["final_score"] for row in d1_rows], ["9.0", "8.1", "7.8"])

    def test_compare_summaries_treats_dash_metrics_as_missing(self) -> None:
        mct_path = self.workdir / "mct.csv"
        mab_path = self.workdir / "mab.csv"
        aggregate_path = self.workdir / "aggregate.csv"
        detail_path = self.workdir / "detail.csv"

        self._write_csv(
            mct_path,
            ["design_name", "final_and", "final_lev", "total_runtime_sec"],
            [
                ["d1", 100, 10, 5.0],
                ["d2", 110, 11, 6.0],
            ],
        )
        self._write_csv(
            mab_path,
            ["file", "variant_label", "and", "lev", "runtime_sec"],
            [
                ["d1", "trial-a", "-", "-", "-"],
                ["d2", "trial-a", 105, 10, 5.5],
            ],
        )

        exit_code = compare_algorithm_summaries.main(
            [
                f"--summary=MCTSyn={mct_path}",
                f"--summary=MABSyn={mab_path}",
                f"--output={aggregate_path}",
                f"--details-output={detail_path}",
            ]
        )

        self.assertEqual(exit_code, 0)
        with detail_path.open("r", encoding="utf-8", newline="") as handle:
            detail_rows = list(csv.DictReader(handle))
        self.assertEqual([row["design"] for row in detail_rows], ["d2", "d2"])

    def test_compare_summaries_keeps_design_when_only_one_variant_row_is_missing(self) -> None:
        mct_path = self.workdir / "mct.csv"
        prefix_path = self.workdir / "prefix.csv"
        aggregate_path = self.workdir / "aggregate.csv"
        detail_path = self.workdir / "detail.csv"

        self._write_csv(
            mct_path,
            ["design_name", "variant_label", "final_and", "final_lev", "total_runtime_sec"],
            [
                ["tc_public_1/input.blif", "cpuct=1.0", 100, 10, 5.0],
                ["tc_public_2/input.blif", "cpuct=1.0", 110, 11, 6.0],
            ],
        )
        self._write_csv(
            prefix_path,
            ["file", "variant_label", "and", "lev", "runtime_sec"],
            [
                ["tc_public_1/input.blif", "method=baseline_mab_prefix,steps=10,iters=100", "-", "-", "-"],
                ["tc_public_1/input.blif", "method=baseline_mab_prefix,steps=10,iters=20", 95, 9, 4.5],
                ["tc_public_2/input.blif", "method=baseline_mab_prefix,steps=10,iters=20", 108, 10, 5.5],
            ],
        )

        exit_code = compare_algorithm_summaries.main(
            [
                f"--summary=MCTSyn={mct_path}",
                f"--summary=MABSyn-UCB1Prefix={prefix_path}",
                f"--output={aggregate_path}",
                f"--details-output={detail_path}",
            ]
        )

        self.assertEqual(exit_code, 0)
        with detail_path.open("r", encoding="utf-8", newline="") as handle:
            detail_rows = list(csv.DictReader(handle))
        tc1_rows = [row for row in detail_rows if row["design"] == "tc_public_1/input.blif"]
        self.assertEqual(len(tc1_rows), 2)
        self.assertEqual([row["algorithm"] for row in tc1_rows], ["MABSyn-UCB1Prefix", "MCTSyn"])

    def test_compare_summaries_skips_non_overlapping_variants(self) -> None:
        mct_path = self.workdir / "mct.csv"
        sa_path = self.workdir / "sa.csv"
        aggregate_path = self.workdir / "aggregate.csv"
        detail_path = self.workdir / "detail.csv"

        self._write_csv(
            mct_path,
            ["design_name", "variant_label", "final_and", "final_lev", "total_runtime_sec"],
            [
                ["d1", "shared", 100, 10, 5.0],
                ["d2", "shared", 110, 11, 5.5],
                ["x1", "private", 90, 9, 4.0],
            ],
        )
        self._write_csv(
            sa_path,
            ["design_name", "variant_label", "final_and", "final_lev", "total_runtime_sec"],
            [
                ["d1", "baseline", 98, 10, 5.2],
                ["d2", "baseline", 108, 11, 5.0],
            ],
        )

        exit_code = compare_algorithm_summaries.main(
            [
                f"--summary=MCTSyn={mct_path}",
                f"--summary=SASyn={sa_path}",
                f"--output={aggregate_path}",
                f"--details-output={detail_path}",
            ]
        )

        self.assertEqual(exit_code, 0)
        with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            [(row["algorithm"], row["variant_label"], row["design_count"]) for row in rows],
            [("SASyn", "baseline", "2"), ("MCTSyn", "shared", "2")],
        )


if __name__ == "__main__":
    unittest.main()
