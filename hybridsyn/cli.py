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

from alphasyn.backend import ABCBackend, BackendError
from alphasyn.dataset import discover_blif_designs, write_manifest
from alphasyn.types import (
    ACTION_TO_ABC_COMMAND,
    DEFAULT_ACTION_SPACE,
    SearchResult,
    format_sequence_for_abc,
)

from .search import HybridSearchConfig, run_search


def _resolve_local_path(path: Path) -> Path:
    cwd = Path.cwd().resolve()
    candidate = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
    if not candidate.is_relative_to(cwd):
        raise SystemExit(f"Path '{path}' must stay under the current working directory: {cwd}")
    return candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hybridsyn")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Scan datasets and emit a manifest.")
    prepare.add_argument("--workdir", type=Path, default=Path(".hybridsyn_work"))
    prepare.add_argument("--dataset-root", type=Path, default=Path("benchmarks"))

    run = subparsers.add_parser("run-search", help="Run HybridSyn search.")
    run.add_argument("--workdir", type=Path, default=Path(".hybridsyn_work"))
    run.add_argument("--abc-bin", default=os.environ.get("ABC_BIN"))
    run.add_argument("--dataset-root", type=Path, default=Path("benchmarks"))
    run.add_argument("--design", help="Single .blif filename under dataset-root. Default: run all.")
    run.add_argument("--sequence-length", type=int, default=24)
    run.add_argument("--warmup-steps", type=int, default=4)
    run.add_argument("--warmup-episodes", type=int, default=None)
    run.add_argument("--warmup-top-k", type=int, default=1)
    run.add_argument("--ucb-c", type=float, default=0.4)
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
    run.add_argument("--debug-search", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Aggregate JSON search results into CSV.")
    summarize.add_argument("--workdir", type=Path, default=Path(".hybridsyn_work"))
    summarize.add_argument("--results-dir", type=Path, default=None)
    summarize.add_argument("--output", type=Path, default=None)
    return parser


def _load_results(results_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="ascii"))
        for path in sorted(results_dir.glob("*.json"))
    ]


def _natural_sort_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def _result_file_stem(design_name: str) -> str:
    return design_name.replace("/", "__")


def _variant_label(config: dict[str, object]) -> str:
    return (
        f"seq={config['sequence_length']},"
        f"warmup={config['warmup_steps']},"
        f"warmup_episodes={config['warmup_episodes']},"
        f"topk={config['warmup_top_k']},"
        f"iters={config['search_iterations']},"
        f"ucb_c={config['ucb_c']},"
        f"cpuct={config['cpuct']},"
        f"mu={config['mu_discount']}"
    )


def _variant_tag(config: HybridSearchConfig) -> str:
    return (
        f"seq{config.sequence_length}"
        f".warm{min(max(config.warmup_steps, 0), config.sequence_length)}"
        f".we{config.warmup_episodes}"
        f".topk{max(1, config.warmup_top_k)}"
        f".ucb{config.ucb_c:g}"
        f".iters{config.search_iterations}"
        f".cpuct{config.cpuct:g}"
        f".mu{config.mu_discount:g}"
    )


def _discover_designs_or_exit(dataset_root: Path) -> dict[str, Path]:
    try:
        return discover_blif_designs(dataset_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


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


def _format_prefix(prefix: list[str]) -> str:
    if not prefix:
        return "-"
    return " ".join(prefix)


def _make_progress_reporter(design_name: str):
    def reporter(event: str, payload: dict[str, object]) -> None:
        if event == "design_start":
            print(
                f"[{design_name}] start and={payload['initial_and']} lev={payload['initial_lev']} "
                f"warmup={payload['warmup_steps']}x{payload['warmup_episodes']} topk={payload['warmup_top_k']} "
                f"mcts={payload['search_iterations']}"
            )
            return
        if event == "warmup_skipped":
            print(f"[{design_name}] warmup skipped")
            return
        if event == "warmup_progress":
            print(
                f"[{design_name}] warmup {payload['episode']}/{payload['warmup_episodes']} "
                f"best={payload['best_and']}/{payload['best_lev']} "
                f"prefix={_format_prefix(payload['best_prefix'])}"
            )
            return
        if event == "handoff":
            print(
                f"[{design_name}] branch {payload['branch_index']}/{payload['branch_count']} handoff "
                f"and={payload['and']} lev={payload['lev']} "
                f"prefix={_format_prefix(payload['prefix'])}"
            )
            return
        if event == "mcts_step":
            print(
                f"[{design_name}] branch {payload['branch_index']}/{payload['branch_count']} "
                f"step {payload['step_index']}/{payload['sequence_length']} "
                f"{payload['selected_action']} -> {payload['and']}/{payload['lev']}"
            )
            return
        if event == "design_done":
            print(
                f"[{design_name}] done and={payload['final_and']} lev={payload['final_lev']} "
                f"selected={_format_prefix(payload['selected_branch_prefix'])} "
                f"time={float(payload['runtime_sec']):.2f}s"
            )
    return reporter


def _group_results_by_design(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in results:
        design_name = str(row["design_name"])
        metadata = row["metadata"]
        assert isinstance(metadata, dict)
        config = metadata["config"]
        assert isinstance(config, dict)
        variant_label = _variant_label(config)
        current_seed = int(row["seed"])
        key = (design_name, variant_label)
        existing = grouped.get(key)
        if existing is None or current_seed < int(existing["seed"]):
            grouped[key] = row
    return [
        grouped[key]
        for key in sorted(grouped, key=lambda item: (_natural_sort_key(item[0]), item[1]))
    ]


def _command_prepare_data(args: argparse.Namespace) -> int:
    args.workdir = _resolve_local_path(args.workdir)
    output_path = args.workdir / "manifest.json"
    try:
        manifest = write_manifest(args.dataset_root, output_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, indent=2))
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


def _write_warmup_debug_csv(result: SearchResult, output_path: Path) -> None:
    metadata = result.metadata["warmup"]
    assert isinstance(metadata, dict)
    debug_trace = metadata.get("debug_trace", [])
    assert isinstance(debug_trace, list)
    with output_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "design_name",
                "warmup_steps",
                "warmup_episodes",
                "ucb_c",
                "episode",
                "step",
                "action_index",
                "action",
                "selected",
                "count_before",
                "q_before",
                "bonus_before",
                "score_before",
                "cold_start",
                "count_after_select",
                "count_after_update",
                "q_after_update",
                "sequence_prefix",
                "step_reward",
                "step_and",
                "step_lev",
                "step_cache_hit",
                "evaluation_error",
            ],
        )
        writer.writeheader()
        for row in debug_trace:
            payload = dict(row)
            payload["design_name"] = result.design_name
            payload["warmup_steps"] = result.metadata["config"]["warmup_steps"]
            payload["warmup_episodes"] = result.metadata["config"]["warmup_episodes"]
            payload["ucb_c"] = result.metadata["config"]["ucb_c"]
            writer.writerow(payload)


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

    try:
        for design_name, design_path in selected.items():
            warmup_episodes = args.warmup_episodes or args.search_iterations
            config = HybridSearchConfig(
                design_name=design_name,
                design_path=design_path,
                action_space=_parse_action_space(args.actions),
                sequence_length=args.sequence_length,
                warmup_steps=args.warmup_steps,
                warmup_episodes=warmup_episodes,
                warmup_top_k=args.warmup_top_k,
                ucb_c=args.ucb_c,
                search_iterations=args.search_iterations,
                cpuct=args.cpuct,
                mu_discount=args.mu_discount,
                seed=args.seed,
                debug_search=args.debug_search,
                workdir=args.workdir,
            )
            result = run_search(
                config,
                backend,
                progress_callback=_make_progress_reporter(design_name),
            )
            result_stem = _result_file_stem(design_name)
            variant_tag = _variant_tag(config)
            output_path = results_dir / f"{result_stem}.{variant_tag}.json"
            output_path.write_text(json.dumps(result.to_json_dict(), indent=2), encoding="ascii")

            warmup_debug_csv_path = None
            debug_csv_path = None
            trace_csv_path = None
            if args.debug_search:
                warmup_debug_csv_path = results_dir / f"{result_stem}.{variant_tag}.warmup.csv"
                _write_warmup_debug_csv(result, warmup_debug_csv_path)
                debug_csv_path = results_dir / f"{result_stem}.{variant_tag}.debug.csv"
                _write_debug_csv(result, debug_csv_path)
                trace_csv_path = results_dir / f"{result_stem}.{variant_tag}.trace.csv"
                _write_iteration_trace_csv(result, trace_csv_path)

            print(f"design: {design_name}")
            print(f"final_and: {result.final_and_count}")
            print(f"final_lev: {result.final_lev_count}")
            print(f"sequence: {format_sequence_for_abc(result.sequence)}")
            print(f"output_json: {output_path}")
            if args.debug_search:
                print(f"output_warmup_csv: {warmup_debug_csv_path}")
                print(f"output_debug_csv: {debug_csv_path}")
                print(f"output_trace_csv: {trace_csv_path}")
    finally:
        backend.cleanup_cache()
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
                "variant_label",
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
                    "variant_label": _variant_label(config),
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


def _run_with_external_monitor(args: argparse.Namespace, raw_argv: list[str]) -> int | None:
    if is_external_monitor_active() or not args.external_monitor:
        return None
    args.workdir = _resolve_local_path(args.workdir)
    base_argv = remove_flag(raw_argv, "--external-monitor")
    designs = _discover_designs_or_exit(args.dataset_root)
    selected_designs = [args.design] if args.design else sorted(designs, key=_natural_sort_key)
    warmup_episodes = args.warmup_episodes if args.warmup_episodes is not None else args.search_iterations
    variant_tag = (
        f"seq{args.sequence_length}"
        f".warm{min(max(args.warmup_steps, 0), args.sequence_length)}"
        f".we{warmup_episodes}"
        f".topk{max(1, args.warmup_top_k)}"
        f".ucb{args.ucb_c:g}"
        f".iters{args.search_iterations}"
        f".cpuct{args.cpuct:g}"
        f".mu{args.mu_discount:g}"
    )
    for design_name in selected_designs:
        patch_json = args.workdir / "results" / f"{_result_file_stem(design_name)}.{variant_tag}.json"
        design_argv = upsert_option(base_argv, "--design", design_name)
        exit_code = run_under_external_monitor(
            raw_argv=design_argv,
            patch_json=patch_json,
            module_name="hybridsyn",
        )
        if exit_code != 0:
            return exit_code
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
