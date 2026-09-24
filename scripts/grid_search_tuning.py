from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".grid_search"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import compare_algorithm_summaries as cas


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    compare_name: str
    run_module: str
    summarize_module: str
    summary_relpath: Path
    param_order: tuple[str, ...]


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "alphasyn": AlgorithmSpec(
        key="alphasyn",
        compare_name="AlphaSyn",
        run_module="alphasyn",
        summarize_module="alphasyn",
        summary_relpath=Path("summary.csv"),
        param_order=("sequence-length", "search-iterations", "cpuct", "mu-discount", "seed"),
    ),
    "baseline_mab": AlgorithmSpec(
        key="baseline_mab",
        compare_name="MABSyn-UCB1",
        run_module="MABSyn.baseline_mab",
        summarize_module="MABSyn.baseline_mab",
        summary_relpath=Path("results/baseline_mab/summary.csv"),
        param_order=("steps", "episodes", "ucb-c", "seed"),
    ),
    "baseline_mab_prefix": AlgorithmSpec(
        key="baseline_mab_prefix",
        compare_name="MABSyn-UCB1Prefix",
        run_module="MABSyn.baseline_mab_prefix",
        summarize_module="MABSyn.baseline_mab_prefix",
        summary_relpath=Path("results/baseline_mab_prefix/summary.csv"),
        param_order=("steps", "episodes", "ucb-c", "seed"),
    ),
}


DEFAULT_GRIDS: dict[str, dict[str, list[Any]]] = {
    "alphasyn": {
        "sequence-length": [10],
        "search-iterations": [20],
        "cpuct": [0.5, 1.0, 1.5],
        "mu-discount": [0.9],
        "seed": [0],
    },
    "baseline_mab": {
        "steps": [10],
        "episodes": [20],
        "ucb-c": [0.4, 1.0, 1.5],
        "seed": [0,1,2,3,4,5,6,7,8,9],
    },
    "baseline_mab_prefix": {
        "steps": [10],
        "episodes": [20],
        "ucb-c": [0.2, 0.4, 0.8],
        "seed": [0,1,2,3,4,5,6,7,8,9],
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid search tuner for AlphaSyn, baseline_mab, and baseline_mab_prefix."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(ALGORITHMS),
        default=list(ALGORITHMS),
        help="Algorithms to tune.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("benchmarks"))
    parser.add_argument(
        "--design",
        action="append",
        default=[],
        help="Specific design under dataset-root. Can be repeated. Omit to run the full dataset.",
    )
    parser.add_argument("--abc-bin", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=None,
        help="Optional JSON file overriding the default parameter grids.",
    )
    parser.add_argument(
        "--max-trials-per-algorithm",
        type=int,
        default=None,
        help="Optional cap on grid combinations per algorithm after enumeration order.",
    )
    return parser


def _load_grid_config(path: Path | None) -> dict[str, dict[str, list[Any]]]:
    if path is None:
        return {key: dict(value) for key, value in DEFAULT_GRIDS.items()}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Grid config must be a JSON object.")
    merged = {key: dict(value) for key, value in DEFAULT_GRIDS.items()}
    for algo, algo_grid in payload.items():
        if algo not in ALGORITHMS:
            raise SystemExit(f"Unknown algorithm in grid config: {algo}")
        if not isinstance(algo_grid, dict):
            raise SystemExit(f"Grid config for {algo} must be a JSON object.")
        normalized: dict[str, list[Any]] = {}
        for name, values in algo_grid.items():
            if not isinstance(values, list) or not values:
                raise SystemExit(f"Grid values for {algo}.{name} must be a non-empty JSON list.")
            normalized[name] = values
        merged[algo] = normalized
    return merged
def iter_grid(grid: dict[str, list[Any]], param_order: tuple[str, ...]) -> list[dict[str, Any]]:
    keys = [key for key in param_order if key in grid]
    missing = [key for key in grid if key not in keys]
    keys.extend(sorted(missing))
    combinations: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[key] for key in keys)):
        combinations.append(dict(zip(keys, values, strict=True)))
    return combinations


def _format_trial_name(index: int, params: dict[str, Any]) -> str:
    fragments = [f"{key.replace('-', '_')}={params[key]}" for key in sorted(params)]
    return f"trial_{index:03d}__" + "__".join(fragments)


def _format_variant_label(spec: AlgorithmSpec, params: dict[str, Any]) -> str:
    ordered_keys = [key for key in spec.param_order if key in params]
    trailing_keys = [key for key in sorted(params) if key not in ordered_keys]
    keys = [*ordered_keys, *trailing_keys]
    return ",".join(f"{key}={params[key]}" for key in keys)


def _run_command(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def _run_trial(
    *,
    spec: AlgorithmSpec,
    workdir: Path,
    dataset_root: Path,
    designs: list[str],
    abc_bin: str | None,
    params: dict[str, Any],
) -> Path:
    run_base = [
        sys.executable,
        "-m",
        spec.run_module,
        "run-search",
        "--external-monitor",
        f"--workdir={workdir}",
        f"--dataset-root={dataset_root}",
    ]
    if abc_bin:
        run_base.append(f"--abc-bin={abc_bin}")
    run_base.extend(f"--{key}={value}" for key, value in params.items())

    if designs:
        for design in designs:
            _run_command([*run_base, f"--design={design}"], cwd=REPO_ROOT)
    else:
        _run_command(run_base, cwd=REPO_ROOT)

    _run_command(
        [
            sys.executable,
            "-m",
            spec.summarize_module,
            "summarize",
            f"--workdir={workdir}",
        ],
        cwd=REPO_ROOT,
    )
    summary_path = (workdir / spec.summary_relpath).resolve()
    if not summary_path.exists():
        raise SystemExit(f"Summary not found after trial: {summary_path}")
    return summary_path


def _rescore_all_trials(
    rows: list[dict[str, Any]],
    *,
    output_root: Path,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    specs: list[cas.SummarySpec] = []
    metadata_by_summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        summary_path = str(Path(row["summary_path"]).resolve())
        metadata_by_summary[summary_path] = row
        specs.append(cas.SummarySpec(ALGORITHMS[str(row["algorithm"])].compare_name, Path(summary_path)))

    loaded_summaries = [
        summary
        for spec in specs
        for summary in cas.load_summaries(spec)
    ]
    aggregate_rows, detail_rows, _, _, _ = cas.compare_summaries(
        loaded_summaries,
        weights={"and": 0.5, "lev": 0.2, "runtime": 0.2, "memory": 0.1},
        missing_policy="common",
    )

    compare_root = (output_root / "global_compare").resolve()
    compare_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = compare_root / "aggregate.csv"
    detail_path = compare_root / "details.csv"
    cas._write_csv(aggregate_path, aggregate_rows)
    cas._write_csv(detail_path, detail_rows)

    rescored_rows: list[dict[str, Any]] = []
    for rank, aggregate_row in enumerate(aggregate_rows, start=1):
        summary_path = str(Path(str(aggregate_row["summary_path"])).resolve())
        metadata = metadata_by_summary.get(summary_path)
        if metadata is None:
            raise SystemExit(f"Summary missing from trial metadata during global compare: {summary_path}")
        rescored_rows.append(
            {
                "global_rank": rank,
                **metadata,
                "design_count": aggregate_row["design_count"],
                "final_score": aggregate_row["final_score"],
                "global_compare_aggregate_csv": str(aggregate_path),
                "global_compare_details_csv": str(detail_path),
            }
        )
    return rescored_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root
    grid_config = _load_grid_config(args.grid_config)

    for algo in args.algorithms:
        if algo not in grid_config:
            raise SystemExit(f"Missing grid config for algorithm: {algo}")

    raw_rows: list[dict[str, Any]] = []
    for algo in args.algorithms:
        spec = ALGORITHMS[algo]
        algo_output_root = output_root / algo
        trial_root = algo_output_root / "trials"
        trial_root.mkdir(parents=True, exist_ok=True)

        combinations = iter_grid(grid_config[algo], spec.param_order)
        if args.max_trials_per_algorithm is not None:
            combinations = combinations[: args.max_trials_per_algorithm]

        result_rows: list[dict[str, Any]] = []
        for index, params in enumerate(combinations, start=1):
            trial_name = _format_trial_name(index, params)
            workdir = trial_root / trial_name / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)

            summary_path = _run_trial(
                spec=spec,
                workdir=workdir,
                dataset_root=dataset_root,
                designs=args.design,
                abc_bin=args.abc_bin,
                params=params,
            )
            row: dict[str, Any] = {
                "algorithm": algo,
                "trial_index": index,
                "trial_name": trial_name,
                "variant_label": _format_variant_label(spec, params),
                "summary_path": str(summary_path),
            }
            result_rows.append(row)

        _write_csv(algo_output_root / "results.raw.csv", result_rows)
        raw_rows.extend(result_rows)

    rescored_rows = _rescore_all_trials(raw_rows, output_root=output_root)
    by_algorithm: dict[str, list[dict[str, Any]]] = {algo: [] for algo in args.algorithms}
    for row in rescored_rows:
        by_algorithm[str(row["algorithm"])].append(row)

    for algo in args.algorithms:
        algo_rows = sorted(
            by_algorithm[algo],
            key=lambda item: (int(item["global_rank"]), int(item["trial_index"])),
        )
        for rank, row in enumerate(algo_rows, start=1):
            row["algorithm_rank"] = rank
        ordered_rows = [
            {"algorithm_rank": row["algorithm_rank"], **{k: v for k, v in row.items() if k != "algorithm_rank"}}
            for row in algo_rows
        ]
        _write_csv(output_root / algo / "results.csv", ordered_rows)

    _write_csv(output_root / "all_results.csv", rescored_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
