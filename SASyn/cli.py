from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

from .backend import ABCBackend, BackendError
from .dataset import discover_blif_designs, write_manifest
from .sa import run_search
from .core_types import SearchConfig, SearchResult, format_sequence_for_abc


def _resolve_local_path(path: Path) -> Path:
    cwd = Path.cwd().resolve()
    candidate = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
    if not candidate.is_relative_to(cwd):
        raise SystemExit(f"Path '{path}' must stay under the current working directory: {cwd}")
    return candidate


def _result_file_stem(design_name: str) -> str:
    return design_name.replace("/", "__")


def _discover_designs_or_exit(dataset_root: Path) -> dict[str, Path]:
    try:
        return discover_blif_designs(dataset_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sasyn")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Scan BLIF data and emit a manifest.")
    prepare.add_argument("--workdir", type=Path, default=Path(".sasyn_work"))
    prepare.add_argument("--dataset-root", type=Path, default=Path("tc_public"))

    run = subparsers.add_parser("run-search", help="Run simulated annealing sequence search.")
    run.add_argument("--workdir", type=Path, default=Path(".sasyn_work"))
    run.add_argument("--abc-bin", default=os.environ.get("ABC_BIN"), help="Path to ABC executable.")
    run.add_argument("--dataset-root", type=Path, default=Path("tc_public"))
    run.add_argument("--design", help="Single .blif filename under dataset-root. Default: run all.")
    run.add_argument("--sequence-length", type=int, default=24)
    run.add_argument("--search-iterations", type=int, default=1000)
    run.add_argument("--and-weight", type=float, default=0.7)
    run.add_argument("--lev-weight", type=float, default=0.3)
    run.add_argument(
        "--initial-temperature",
        type=float,
        default=None,
        help="Override the initial temperature. Omit to auto-calibrate.",
    )
    run.add_argument("--min-temperature", type=float, default=1e-3)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--debug-search", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Aggregate JSON search results into CSV.")
    summarize.add_argument("--workdir", type=Path, default=Path(".sasyn_work"))
    summarize.add_argument("--results-dir", type=Path, default=None)
    summarize.add_argument("--output", type=Path, default=None)
    return parser


def _load_results(results_dir: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(path.read_text(encoding="ascii"))
        for path in sorted(results_dir.glob("*.json"))
    ]
    rows.sort(key=_result_sort_key)
    return rows


def _result_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    design_name = str(row.get("design_name", ""))
    parts = re.split(r"(\d+)", design_name)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


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
            raise SystemExit(f"Unknown design '{args.design}'. Available designs: {available}")
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
            and_weight=args.and_weight,
            lev_weight=args.lev_weight,
            initial_temperature=args.initial_temperature,
            min_temperature=args.min_temperature,
            seed=args.seed,
            debug_search=args.debug_search,
            workdir=args.workdir,
        )
        result = run_search(config, backend)
        result_stem = _result_file_stem(design_name)
        output_path = results_dir / f"{result_stem}.json"
        output_path.write_text(json.dumps(result.to_json_dict(), indent=2), encoding="ascii")
        debug_csv_path = None
        if args.debug_search:
            debug_csv_path = results_dir / f"{result_stem}.debug.csv"
            _write_debug_csv(result, debug_csv_path)
        print(f"design: {design_name}")
        print(f"final_and: {result.final_and_count}")
        print(f"final_lev: {result.final_lev_count}")
        print(f"sequence: {format_sequence_for_abc(result.sequence)}")
        print(f"output_json: {output_path}")
        if args.debug_search:
            print(f"output_debug_csv: {debug_csv_path}")
    return 0


def _write_debug_csv(result: SearchResult, output_path: Path) -> None:
    with output_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "design_name",
                "iteration",
                "temperature",
                "move_type",
                "accepted",
                "delta_energy",
                "current_and",
                "current_lev",
                "current_energy",
                "best_and",
                "best_lev",
                "best_energy",
                "current_sequence",
                "best_sequence",
            ],
        )
        writer.writeheader()
        for iteration in result.iterations:
            writer.writerow(
                {
                    "design_name": result.design_name,
                    "iteration": iteration.iteration,
                    "temperature": iteration.temperature,
                    "move_type": iteration.move_type,
                    "accepted": iteration.accepted,
                    "delta_energy": iteration.delta_energy,
                    "current_and": iteration.current_and_count,
                    "current_lev": iteration.current_lev_count,
                    "current_energy": iteration.current_energy,
                    "best_and": iteration.best_and_count,
                    "best_lev": iteration.best_lev_count,
                    "best_energy": iteration.best_energy,
                    "current_sequence": format_sequence_for_abc(iteration.current_sequence),
                    "best_sequence": format_sequence_for_abc(iteration.best_sequence),
                }
            )


def _command_summarize(args: argparse.Namespace) -> int:
    args.workdir = _resolve_local_path(args.workdir)
    results_dir = _resolve_local_path(args.results_dir) if args.results_dir else (args.workdir / "results")
    output_path = _resolve_local_path(args.output) if args.output else (args.workdir / "summary.csv")
    results = _load_results(results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "design_name",
                "search_iterations",
                "initial_and",
                "initial_lev",
                "heuristic_and",
                "heuristic_lev",
                "final_and",
                "final_lev",
                "total_runtime_sec",
                "peak_memory_kb",
            ],
        )
        writer.writeheader()
        for row in results:
            baseline = row["baseline"]
            assert isinstance(baseline, dict)
            metadata = row["metadata"]
            assert isinstance(metadata, dict)
            config = metadata["config"]
            assert isinstance(config, dict)
            writer.writerow(
                {
                    "design_name": row["design_name"],
                    "search_iterations": config["search_iterations"],
                    "initial_and": baseline["initial_and"],
                    "initial_lev": baseline["initial_lev"],
                    "heuristic_and": baseline["heuristic_and"],
                    "heuristic_lev": baseline["heuristic_lev"],
                    "final_and": row["final_and"],
                    "final_lev": row["final_lev"],
                    "total_runtime_sec": row["total_runtime_sec"],
                    "peak_memory_kb": row.get("peak_memory_kb", ""),
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
