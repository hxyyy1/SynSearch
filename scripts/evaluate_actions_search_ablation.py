from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import action_eval_common as common
import compare_algorithm_summaries as cas


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    compare_name: str
    run_module: str
    summarize_module: str
    summary_relpath: Path


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "alphasyn": AlgorithmSpec(
        key="alphasyn",
        compare_name="AlphaSyn",
        run_module="alphasyn",
        summarize_module="alphasyn",
        summary_relpath=Path("summary.csv"),
    ),
    # "hybridsyn": AlgorithmSpec(
    #     key="hybridsyn",
    #     compare_name="HybridSyn",
    #     run_module="hybridsyn",
    #     summarize_module="hybridsyn",
    #     summary_relpath=Path("summary.csv"),
    # ),
    # "sasyn": AlgorithmSpec(
    #     key="sasyn",
    #     compare_name="SASyn",
    #     run_module="SASyn",
    #     summarize_module="SASyn",
    #     summary_relpath=Path("summary.csv"),
    # ),
    "baseline_mab": AlgorithmSpec(
        key="baseline_mab",
        compare_name="MABSyn-UCB1",
        run_module="MABSyn.baseline_mab",
        summarize_module="MABSyn.baseline_mab",
        summary_relpath=Path("results/baseline_mab/summary.csv"),
    ),
    "baseline_mab_prefix": AlgorithmSpec(
        key="baseline_mab_prefix",
        compare_name="MABSyn-UCB1Prefix",
        run_module="MABSyn.baseline_mab_prefix",
        summarize_module="MABSyn.baseline_mab_prefix",
        summary_relpath=Path("results/baseline_mab_prefix/summary.csv"),
    ),
    # "linucb": AlgorithmSpec(
    #     key="linucb",
    #     compare_name="MABSyn-LinUCB",
    #     run_module="MABSyn.linucb",
    #     summarize_module="MABSyn.linucb",
    #     summary_relpath=Path("results/linucb/summary.csv"),
    # ),
}


def _resolve_local_output_root(path: Path) -> Path:
    candidate = (Path.cwd().resolve() / path).resolve() if not path.is_absolute() else path.resolve()
    cwd = Path.cwd().resolve()
    if not candidate.is_relative_to(cwd):
        raise SystemExit(
            f"Path '{path}' must stay under the current working directory because run-search workdirs "
            f"are restricted to: {cwd}"
        )
    return candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 3: run full-search action-space ablations and globally compare all trials."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(ALGORITHMS),
        default=["alphasyn", "baseline_mab", "baseline_mab_prefix"],
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("tc_public"))
    parser.add_argument(
        "--design",
        action="append",
        default=[],
        help="Specific design under dataset-root. Can be repeated.",
    )
    parser.add_argument("--abc-bin", default=None, help="Path to ABC executable.")
    parser.add_argument(
        "--action-label",
        action="append",
        default=[],
        help="Candidate action label to include. Repeatable. Default: all built-in candidates.",
    )
    parser.add_argument(
        "--action-spec",
        action="append",
        default=[],
        metavar="LABEL=COMMAND",
        help="Override or add a candidate action command.",
    )
    parser.add_argument(
        "--preset-json",
        type=Path,
        default=None,
        help="Optional JSON file mapping preset name to action-label list.",
    )
    parser.add_argument(
        "--include-combined-preset",
        action="store_true",
        help="Also add one preset containing all selected candidate actions.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON file with per-algorithm run-search options.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".action_eval/search_ablation"),
    )
    return parser


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=common.REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def _run_command_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=common.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _summary_path_for_trial(spec: AlgorithmSpec, workdir: Path) -> Path:
    return (workdir / spec.summary_relpath).resolve()


def _trial_meta_path(preset_root: Path) -> Path:
    return (preset_root / "trial_meta.json").resolve()


def _infer_action_labels(preset_label: str) -> list[str] | None:
    if preset_label == "baseline":
        return list(common.DEFAULT_ACTION_SPACE)
    if preset_label.startswith("plus_"):
        return [*common.DEFAULT_ACTION_SPACE, preset_label[len("plus_"):]]
    return None


def _build_trial_row(
    *,
    algorithm_key: str,
    spec: AlgorithmSpec,
    preset_label: str,
    action_labels: list[str],
    summary_path: Path,
) -> dict[str, Any]:
    return {
        "algorithm": algorithm_key,
        "compare_name": spec.compare_name,
        "preset_label": preset_label,
        "action_labels": ",".join(action_labels),
        "actions_arg": ",".join(common.resolve_algorithm_actions(algorithm_key, action_labels)),
        "summary_path": str(summary_path.resolve()),
    }


def _write_trial_meta(preset_root: Path, row: dict[str, Any]) -> None:
    _trial_meta_path(preset_root).write_text(
        json.dumps(row, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _discover_existing_trial_rows(
    output_root: Path,
    *,
    algorithms: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm_key in algorithms:
        spec = ALGORITHMS[algorithm_key]
        algo_root = output_root / algorithm_key
        if not algo_root.exists():
            continue
        for preset_root in sorted(path for path in algo_root.iterdir() if path.is_dir()):
            summary_path = _summary_path_for_trial(spec, preset_root / "workdir")
            if not summary_path.exists():
                continue
            meta_path = _trial_meta_path(preset_root)
            if meta_path.exists():
                row = json.loads(meta_path.read_text(encoding="utf-8"))
                row["summary_path"] = str(summary_path.resolve())
                rows.append(row)
                continue
            action_labels = _infer_action_labels(preset_root.name)
            if action_labels is None:
                print(
                    f"[warn] skipping preset without metadata: {preset_root}",
                    file=sys.stderr,
                )
                continue
            row = _build_trial_row(
                algorithm_key=algorithm_key,
                spec=spec,
                preset_label=preset_root.name,
                action_labels=action_labels,
                summary_path=summary_path,
            )
            _write_trial_meta(preset_root, row)
            rows.append(row)
    return rows


def _rescore_rows(rows: list[dict[str, Any]], *, output_root: Path) -> list[dict[str, Any]]:
    specs = [
        cas.SummarySpec(row["compare_name"], Path(str(row["summary_path"])))
        for row in rows
    ]
    aggregate_rows, detail_rows = common.compare_summary_specs(specs, output_root=output_root / "global_compare")
    metadata_by_path = {
        str(Path(str(row["summary_path"])).resolve()): row
        for row in rows
    }
    rescored: list[dict[str, Any]] = []
    for rank, aggregate_row in enumerate(aggregate_rows, start=1):
        summary_path = str(Path(str(aggregate_row["summary_path"])).resolve())
        metadata = metadata_by_path[summary_path]
        rescored.append(
            {
                "global_rank": rank,
                "algorithm": metadata["algorithm"],
                "compare_name": metadata["compare_name"],
                "preset_label": metadata["preset_label"],
                "action_labels": metadata["action_labels"],
                "actions_arg": metadata["actions_arg"],
                "summary_path": summary_path,
                "design_count": aggregate_row["design_count"],
                "final_score": aggregate_row["final_score"],
            }
        )
    common.write_csv(output_root / "all_results.csv", rescored)
    common.write_csv(output_root / "global_compare" / "aggregate_merged.csv", rescored)
    common.write_csv(output_root / "global_compare" / "details_merged.csv", detail_rows)
    return rescored


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    abc_bin = common.resolve_abc_bin(args.abc_bin)
    output_root = _resolve_local_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    action_specs = common.load_action_specs(labels=args.action_label, raw_specs=args.action_spec)
    presets = common.load_presets(
        preset_json=args.preset_json,
        action_labels=list(action_specs),
        include_combined=args.include_combined_preset,
    )
    algorithm_options = common.load_algorithm_options(args.config_json)
    selected_designs = common.discover_selected_designs(args.dataset_root, args.design)

    failure_rows: list[dict[str, Any]] = []
    for algorithm_key in args.algorithms:
        spec = ALGORITHMS[algorithm_key]
        algo_root = output_root / algorithm_key
        algo_root.mkdir(parents=True, exist_ok=True)
        per_algo_failures: list[dict[str, Any]] = []

        for preset_label, action_labels in presets.items():
            resolved_actions = common.resolve_algorithm_actions(algorithm_key, action_labels)
            preset_root = algo_root / preset_label
            workdir = preset_root / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            summary_path = _summary_path_for_trial(spec, workdir)

            if summary_path.exists():
                print(f"[reuse] {algorithm_key}/{preset_label}: {summary_path}")
            else:
                successful_designs = 0
                for design_name in selected_designs:
                    command = [
                        sys.executable,
                        "-m",
                        spec.run_module,
                        "run-search",
                        "--external-monitor",
                        f"--workdir={workdir}",
                        f"--dataset-root={args.dataset_root}",
                        f"--actions={','.join(resolved_actions)}",
                        f"--abc-bin={abc_bin}",
                        *common.option_list_from_mapping(algorithm_options.get(algorithm_key, {})),
                        f"--design={design_name}",
                    ]
                    print(f"[run] {algorithm_key}/{preset_label}/{design_name}")
                    completed = _run_command_capture(command)
                    if completed.returncode == 0:
                        successful_designs += 1
                        continue
                    excerpt = "\n".join(
                        part.strip()
                        for part in (completed.stdout or "", completed.stderr or "")
                        if part and part.strip()
                    )
                    if len(excerpt) > 4000:
                        excerpt = excerpt[-4000:]
                    failure = {
                        "algorithm": algorithm_key,
                        "preset_label": preset_label,
                        "design_name": design_name,
                        "returncode": completed.returncode,
                        "command": " ".join(command),
                        "output_excerpt": excerpt,
                    }
                    per_algo_failures.append(failure)
                    failure_rows.append(failure)
                    print(f"[fail] {algorithm_key}/{preset_label}/{design_name} rc={completed.returncode}")

                if successful_designs > 0:
                    _run_command(
                        [
                            sys.executable,
                            "-m",
                            spec.summarize_module,
                            "summarize",
                            f"--workdir={workdir}",
                        ]
                    )
                if not summary_path.exists():
                    print(f"[skip] {algorithm_key}/{preset_label}: no successful designs, summary not created")
                    continue
            row = _build_trial_row(
                algorithm_key=algorithm_key,
                spec=spec,
                preset_label=preset_label,
                action_labels=action_labels,
                summary_path=summary_path,
            )
            _write_trial_meta(preset_root, row)

        common.write_csv(algo_root / "failures.csv", per_algo_failures)

    discovered_rows = _discover_existing_trial_rows(output_root, algorithms=args.algorithms)
    by_algorithm_trials: dict[str, list[dict[str, Any]]] = {algorithm_key: [] for algorithm_key in args.algorithms}
    for row in discovered_rows:
        by_algorithm_trials[str(row["algorithm"])].append(row)
    for algorithm_key, rows in by_algorithm_trials.items():
        rows.sort(key=lambda item: str(item["preset_label"]))
        common.write_csv(output_root / algorithm_key / "trials.csv", rows)

    if not discovered_rows:
        common.write_csv(output_root / "failures.csv", failure_rows)
        raise SystemExit("No successful trial summaries were produced.")

    rescored_rows = _rescore_rows(discovered_rows, output_root=output_root)
    by_algorithm: dict[str, list[dict[str, Any]]] = {algorithm_key: [] for algorithm_key in args.algorithms}
    for row in rescored_rows:
        by_algorithm[str(row["algorithm"])].append(row)
    for algorithm_key, rows in by_algorithm.items():
        rows.sort(key=lambda item: int(item["global_rank"]))
        for rank, row in enumerate(rows, start=1):
            row["algorithm_rank"] = rank
        common.write_csv(output_root / algorithm_key / "results.csv", rows)
    common.write_csv(output_root / "failures.csv", failure_rows)

    print(f"Global compare aggregate CSV written to: {output_root / 'global_compare' / 'aggregate.csv'}")
    print(f"Global compare detail CSV written to: {output_root / 'global_compare' / 'details.csv'}")
    print(f"Merged ranking CSV written to: {output_root / 'all_results.csv'}")
    if failure_rows:
        print(f"Failure log CSV written to: {output_root / 'failures.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
