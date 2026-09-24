from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from alphasyn import cli as mcts_cli
from hybridsyn import cli as hybrid_cli
from SASyn import cli as sa_cli
from MABSyn import baseline_mab, baseline_mab_prefix, linucb


SEARCH_MODULES = (mcts_cli, hybrid_cli, sa_cli, baseline_mab, baseline_mab_prefix, linucb)
BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "benchmarks"


class BenchmarkTests(unittest.TestCase):
    def test_default_searches_select_the_same_bundled_circuits(self) -> None:
        expected = {path.name: path.resolve() for path in BENCHMARK_ROOT.rglob("*.blif")}
        self.assertEqual(len(expected), 26)
        self.assertIn("adder.blif", expected)
        self.assertIn("bfly.abc.blif", expected)
        for module in SEARCH_MODULES:
            with self.subTest(module=module.__name__):
                args = module._build_parser().parse_args(["run-search"])
                self.assertEqual(args.dataset_root, Path("benchmarks"))
                actual = module._discover_designs_or_exit(BENCHMARK_ROOT)
                self.assertEqual({name: Path(path) for name, path in actual.items()}, expected)

    def test_prepare_data_defaults_to_benchmarks(self) -> None:
        for module in (mcts_cli, hybrid_cli, sa_cli):
            with self.subTest(module=module.__name__):
                args = module._build_parser().parse_args(["prepare-data"])
                self.assertEqual(args.dataset_root, Path("benchmarks"))

    def test_duplicate_filenames_keep_distinct_names_across_algorithms(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {}
            for name in ("suite_a/shared.blif", "suite_b/shared.blif", "unique.blif"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(".model toy\n.end\n", encoding="ascii")
                expected[name] = path.resolve()
            for module in SEARCH_MODULES:
                with self.subTest(module=module.__name__):
                    actual = module._discover_designs_or_exit(root)
                    self.assertEqual({name: Path(path) for name, path in actual.items()}, expected)
