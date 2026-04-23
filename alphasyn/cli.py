from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from external_monitoring import (
    is_external_monitor_active,
    remove_flag,
    run_under_external_monitor,
    upsert_option,
)

from .backend import ABCBackend, BackendError
from .dataset import discover_blif_designs, write_manifest
from .mcts import run_search
from .types import (
    ACTION_TO_ABC_COMMAND,
    DEFAULT_ACTION_SPACE,
    SearchConfig,
    SearchResult,
    format_sequence_for_abc,
)


def _resolve_local_path(path: Path) -> Path:
    cwd = Path.cwd().resolve()
    candidate = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
    if not candidate.is_relative_to(cwd):
        raise SystemExit(f"Path '{path}' must stay under the current working directory: {cwd}")
    return candidate


def _parse_action_space(raw_actions: str | None) -> tuple[str, ...]:
    if raw_actions is None:
        return DEFAULT_ACTION_SPACE
    parsed = tuple(item.strip() for item in raw_actions.split(",") if item.strip())
    if not parsed:
        raise SystemExit("--actions must contain at least one action.")
    unknown = [item for item in parsed if item not in ACTION_TO_ABC_COMMAND]
    if unknown:
        available = ", ".join(sorted(ACTION_TO_ABC_COMMAND))
        raise SystemExit(
            f"Unknown action(s) in --actions: {', '.join(unknown)}. Available actions: {available}"
        )
    return parsed


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
    run.add_argument("--actions", default=None, help="Comma-separated action labels to search.")
    run.add_argument(
        "--external-monitor",
        action="store_true",
        help="Run each selected design under the external monitor and patch result JSON metrics.",
    )
    run.add_argument(
        "--debug-search",
        action="store_true",
        help="Print per-step root action Q/R/U/visit statistics and store them in the JSON output.",
    )

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


def _result_file_stem(design_name: str) -> str:
    return design_name.replace("/", "__")


def _cpuct_tag(cpuct: float) -> str:
    return f"cpuct{cpuct:g}"


def _discover_designs_or_exit(dataset_root: Path) -> dict[str, Path]:
    try:
        return discover_blif_designs(dataset_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _group_results_by_design(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], dict[str, object]] = {}
    for row in results:
        design_name = str(row["design_name"])
        metadata = row["metadata"]
        assert isinstance(metadata, dict)
        config = metadata["config"]
        assert isinstance(config, dict)
        cpuct = float(config["cpuct"])
        current_seed = int(row["seed"])
        key = (design_name, cpuct)
        existing = grouped.get(key)
        if existing is None or current_seed < int(existing["seed"]):
            grouped[key] = row
    return [
        grouped[key]
        for key in sorted(grouped, key=lambda item: (_design_sort_key(item[0]), item[1]))
    ]


def _design_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(?:^|/|_)(\d+)(?:/|$)", name)
    if match:
        return int(match.group(1)), name
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

    try:
        for design_name, design_path in selected.items():
            config = SearchConfig(
                design_name=design_name,
                design_path=design_path,
                action_space=_parse_action_space(args.actions),
                sequence_length=args.sequence_length,
                search_iterations=args.search_iterations,
                cpuct=args.cpuct,
                mu_discount=args.mu_discount,
                seed=args.seed,
                debug_search=args.debug_search,
                workdir=args.workdir,
            )
            result = run_search(config, backend)
            result_stem = _result_file_stem(design_name)
            cpuct_tag = _cpuct_tag(args.cpuct)
            output_path = results_dir / f"{result_stem}.{cpuct_tag}.json"
            output_path.write_text(json.dumps(result.to_json_dict(), indent=2), encoding="ascii")
            sequence_text = format_sequence_for_abc(result.sequence)
            debug_csv_path = None
            trace_csv_path = None
            if args.debug_search:
                debug_csv_path = results_dir / f"{result_stem}.{cpuct_tag}.debug.csv"
                _write_debug_csv(result, debug_csv_path)
                trace_csv_path = results_dir / f"{result_stem}.{cpuct_tag}.trace.csv"
                _write_iteration_trace_csv(result, trace_csv_path)
            print(f"design: {design_name}")
            print(f"final_and: {result.final_and_count}")
            print(f"final_lev: {result.final_lev_count}")
            print(f"sequence: {sequence_text}")
            print(f"output_json: {output_path}")
            if args.debug_search:
                print(f"output_debug_csv: {debug_csv_path}")
                print(f"output_trace_csv: {trace_csv_path}")
    finally:
        backend.cleanup_cache()
    return 0


def _run_with_external_monitor(args: argparse.Namespace, raw_argv: list[str]) -> int | None:
    if is_external_monitor_active() or not args.external_monitor:
        return None
    args.workdir = _resolve_local_path(args.workdir)
    base_argv = remove_flag(raw_argv, "--external-monitor")
    designs = _discover_designs_or_exit(args.dataset_root)
    selected_designs = [args.design] if args.design else sorted(designs, key=_design_sort_key)
    for design_name in selected_designs:
        patch_json = args.workdir / "results" / f"{_result_file_stem(design_name)}.{_cpuct_tag(args.cpuct)}.json"
        design_argv = upsert_option(base_argv, "--design", design_name)
        exit_code = run_under_external_monitor(
            raw_argv=design_argv,
            patch_json=patch_json,
            module_name="alphasyn",
        )
        if exit_code != 0:
            return exit_code
    return 0


def _write_debug_csv(result: SearchResult, output_path: Path) -> None:
    with output_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "design_name",
                "cpuct",
                "step_index",
                "selected_action",
                "root_visits",
                "root_value",
                "action",
                "q",
                "r",
                "u",
                "q_plus_r",
                "selection_score",
                "visits",
                "prior",
                "child_and",
                "child_lev",
            ],
        )
        writer.writeheader()
        for step in result.steps:
            for action, info in step.action_debug.items():
                writer.writerow(
                    {
                        "design_name": result.design_name,
                        "cpuct": result.metadata["config"]["cpuct"],
                        "step_index": step.step_index,
                        "selected_action": step.selected_action,
                        "root_visits": step.root_visits,
                        "root_value": step.root_value,
                        "action": action,
                        "q": info["q"],
                        "r": info["r"],
                        "u": info["u"],
                        "q_plus_r": info["q_plus_r"],
                        "selection_score": info["selection_score"],
                        "visits": info["visits"],
                        "prior": info["prior"],
                        "child_and": info["child_and"],
                        "child_lev": info["child_lev"],
                    }
                )


def _write_iteration_trace_csv(result: SearchResult, output_path: Path) -> None:
    with output_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "design_name",
                "cpuct",
                "step_index",
                "iteration_index",
                "selected_actions",
                "expanded_prefix",
                "root_visits",
                "root_value",
                "node_prefix",
                "node_and",
                "node_lev",
                "node_visits",
                "node_value",
                "action",
                "q",
                "r",
                "u",
                "q_plus_r",
                "selection_score",
                "edge_visits",
                "prior",
                "child_and",
                "child_lev",
            ],
        )
        writer.writeheader()
        for step in result.steps:
            for trace in step.iteration_traces:
                nodes = trace["nodes"]
                assert isinstance(nodes, list)
                for node in nodes:
                    assert isinstance(node, dict)
                    children = node["children"]
                    assert isinstance(children, dict)
                    for action, info in children.items():
                        assert isinstance(info, dict)
                        writer.writerow(
                            {
                                "design_name": result.design_name,
                                "cpuct": result.metadata["config"]["cpuct"],
                                "step_index": step.step_index,
                                "iteration_index": trace["iteration_index"],
                                "selected_actions": " -> ".join(trace["selected_actions"]),
                                "expanded_prefix": (
                                    " -> ".join(trace["expanded_prefix"])
                                    if trace["expanded_prefix"] is not None
                                    else ""
                                ),
                                "root_visits": trace["root_visits"],
                                "root_value": trace["root_value"],
                                "node_prefix": " -> ".join(node["prefix"]),
                                "node_and": node["and"],
                                "node_lev": node["lev"],
                                "node_visits": node["visits"],
                                "node_value": node["value"],
                                "action": action,
                                "q": info["q"],
                                "r": info["r"],
                                "u": info["u"],
                                "q_plus_r": info["q_plus_r"],
                                "selection_score": info["selection_score"],
                                "edge_visits": info["visits"],
                                "prior": info["prior"],
                                "child_and": info["child_and"],
                                "child_lev": info["child_lev"],
                            }
                        )



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
                "variant_label",
                # "sequence_length",
                # "cpuct",
                # "search_iterations",
                # "mu_discount",
                # "seed",
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
            variant_label = (
                f"seq={config['sequence_length']},"
                f"iters={config['search_iterations']},"
                f"cpuct={config['cpuct']},"
                f"mu={config['mu_discount']},"
                # f"seed={config['seed']}"
            )
            writer.writerow(
                {
                    "design_name": row["design_name"],
                    "variant_label": variant_label,
                    # "sequence_length": config["sequence_length"],
                    # "cpuct": config["cpuct"],
                    # "search_iterations": config["search_iterations"],
                    # "mu_discount": config["mu_discount"],
                    # "seed": config["seed"],
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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(raw_argv)

    try:
        if args.command == "prepare-data":
            return _command_prepare_data(args)
        if args.command == "run-search":
            monitored = _run_with_external_monitor(args, raw_argv)
            if monitored is not None:
                return monitored
            return _command_run_search(args)
        if args.command == "summarize":
            return _command_summarize(args)
    except BackendError as exc:
        parser.exit(status=2, message=f"{exc}\n")

    parser.exit(status=2, message="Unknown command.\n")
    return 2
