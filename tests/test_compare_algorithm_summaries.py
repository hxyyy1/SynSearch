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

    def test_compare_summaries_uses_common_designs_and_writes_outputs(self) -> None:
        mct_path = self.workdir / "mct.csv"
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
            sa_path,
            ["design_name", "final_and", "final_lev", "total_runtime_sec", "peak_memory_kb"],
            [
                ["d1", 90, 12, 6.0, 900],
                ["d2", 210, 18, 9.0, 2500],
            ],
        )
        self._write_csv(
            mab_path,
            ["file", "and", "lev", "runtime_sec", "peak_memory_kb"],
            [
                ["d1", 120, 9, 4.0, 1100],
                ["d2", 205, 21, 8.0, 1500],
                ["d3", 999, 999, 999.0, 9999],
            ],
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = compare_algorithm_summaries.main(
                [
                    f"--summary=MCTSyn={mct_path}",
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
        self.assertEqual([row["algorithm"] for row in rows], ["MCTSyn", "MABSyn", "SASyn"])
        self.assertEqual(rows[0]["design_count"], "2")
        self.assertAlmostEqual(float(rows[0]["final_score"]), 10.621212121212123)
        self.assertAlmostEqual(float(rows[1]["final_score"]), 10.428571428571429)
        self.assertAlmostEqual(float(rows[2]["final_score"]), 9.587662337662333)

        with detail_path.open("r", encoding="utf-8", newline="") as handle:
            detail_rows = list(csv.DictReader(handle))
        self.assertEqual(len(detail_rows), 6)
        d1_rows = [row for row in detail_rows if row["design"] == "d1"]
        self.assertEqual([row["algorithm"] for row in d1_rows], ["MABSyn", "MCTSyn", "SASyn"])


if __name__ == "__main__":
    unittest.main()
