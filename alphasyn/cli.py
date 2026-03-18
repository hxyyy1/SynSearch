from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

from .backend import ABCBackend, BackendError
from .dataset import discover_blif_designs, write_manifest
from .mcts import run_search
from .types import SearchConfig, format_sequence_for_abc


def _resolve_local_path(path: Path) -> Path:
    cwd = Path.cwd().resolve()
    candidate = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
    if not candidate.is_relative_to(cwd):
        raise SystemExit(f"Path '{path}' must stay under the current working directory: {cwd}")
    return candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphasyn")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Scan tc_public and emit a manifest.")
    prepare.add_argument(
        "--workdir",
        type=Path,
        default=Path(".alphasyn_work"),
        help="Directory for manifests, cache, and results.",
    )
    prepare.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("tc_public"),
        help="Directory containing the target .blif files.",
    )

    run = subparsers.add_parser("run-search", help="Run core AlphaSyn w/o nn search.")
    run.add_argument(
        "--workdir",
        type=Path,
        default=Path(".alphasyn_work"),
        help="Directory for manifests, cache, and results.",
    )
    run.add_argument(
        "--abc-bin",
        default=os.environ.get("ABC_BIN"),
        help="Path to ABC executable.",
    )
    run.add_argument("--dataset-root", type=Path, default=Path("tc_public"))
    run.add_argument("--design", help="Single .blif filename under dataset-root. Default: run all.")
    run.add_argument("--sequence-length", type=int, default=24)
    run.add_argument("--search-iterations", type=int, default=64)
    run.add_argument("--cpuct", type=float, default=1.0)
    run.add_argument("--mu-discount", type=float, default=0.9)
    run.add_argument("--seed", type=int, default=0)

    summarize = subparsers.add_parser("summarize", help="Aggregate JSON search results into CSV.")
    summarize.add_argument(
        "--workdir",
        type=Path,
        default=Path(".alphasyn_work"),
        help="Directory for manifests, cache, and results.",
    )
    summarize.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing JSON outputs from run-search.",
    )
    summarize.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary CSV output path.",
    )
    return parser


def _load_results(results_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="ascii"))
        for path in sorted(results_dir.glob("*.json"))
    ]


def _discover_designs_or_exit(dataset_root: Path) -> dict[str, Path]:
    try:
        return discover_blif_designs(dataset_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _group_results_by_design(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in results:
        design_name = str(row["design_name"])
        current_seed = int(row["seed"])
        existing = grouped.get(design_name)
        if existing is None or current_seed < int(existing["seed"]):
            grouped[design_name] = row
    return [grouped[name] for name in sorted(grouped, key=_design_sort_key)]


def _design_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def _command_prepare_data(args: argparse.Namespace) -> int:
    args.workdir = _resolve_local_path(args.workdir)
    output_path = args.workdir / "manifest.json"
    try:
        manifest = write_manifest(args.dataset_root, output_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, indent=2))
    return 0


def _command_run_search(args: argparse.Namespace) -> int:
    args.workdir = _resolve_local_path(args.workdir)
    designs = _discover_designs_or_exit(args.dataset_root)
    if args.design:
        if args.design not in designs:
            available = ", ".join(sorted(designs))
            raise SystemExit(
                f"Unknown design '{args.design}'. Available designs: {available}"
            )
        selected = {args.design: designs[args.design]}
    else:
        selected = designs

    backend = ABCBackend(args.abc_bin, args.workdir)
    results_dir = args.workdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for design_name, design_path in selected.items():
        config = SearchConfig(
            design_name=design_name,
            design_path=design_path,
            sequence_length=args.sequence_length,
            search_iterations=args.search_iterations,
            cpuct=args.cpuct,
            mu_discount=args.mu_discount,
            seed=args.seed,
            workdir=args.workdir,
        )
        result = run_search(config, backend)
        output_path = results_dir / f"{design_name}.seed{args.seed}.json"
        output_path.write_text(json.dumps(result.to_json_dict(), indent=2), encoding="ascii")
        sequence_text = format_sequence_for_abc(result.sequence)
        print(f"design: {design_name}")
        print(f"final_and: {result.final_and_count}")
        print(f"final_lev: {result.final_lev_count}")
        print(f"sequence: {sequence_text}")
        print(f"output_json: {output_path}")
    return 0


def _command_summarize(args: argparse.Namespace) -> int:
    args.workdir = _resolve_local_path(args.workdir)
    results_dir = _resolve_local_path(args.results_dir) if args.results_dir else (args.workdir / "results")
    output_path = _resolve_local_path(args.output) if args.output else (args.workdir / "summary.csv")
    results = _group_results_by_design(_load_results(results_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "design_name",
                "initial_and",
                "initial_lev",
                "heuristic_and",
                "heuristic_lev",
                "final_and",
                "final_lev",
            ],
        )
        writer.writeheader()
        for row in results:
            baseline = row["baseline"]
            assert isinstance(baseline, dict)
            writer.writerow(
                {
                    "design_name": row["design_name"],
                    "initial_and": baseline["initial_and"],
                    "initial_lev": baseline["initial_lev"],
                    "heuristic_and": baseline["heuristic_and"],
                    "heuristic_lev": baseline["heuristic_lev"],
                    "final_and": row["final_and"],
                    "final_lev": row["final_lev"],
                }
            )
    print(str(output_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "prepare-data":
            return _command_prepare_data(args)
        if args.command == "run-search":
            return _command_run_search(args)
        if args.command == "summarize":
            return _command_summarize(args)
    except BackendError as exc:
        parser.exit(status=2, message=f"{exc}\n")

    parser.exit(status=2, message="Unknown command.\n")
    return 2
