from __future__ import annotations

import argparse
import shlex
import tempfile
from pathlib import Path

import action_eval_common as common
import compare_algorithm_summaries as cas


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 1: evaluate candidate ABC commands directly on strashed BLIF designs."
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
        "--output-root",
        type=Path,
        default=Path(".action_eval/static"),
    )
    return parser


def _baseline_script(design_path: Path, snapshot_path: Path) -> str:
    return (
        f"read_blif {shlex.quote(str(design_path.resolve()))}; "
        "strash; "
        "print_stats; "
        f"write_aiger {shlex.quote(str(snapshot_path))}"
    )


def _action_script(design_path: Path, action_command: str, snapshot_path: Path) -> str:
    return (
        f"read_blif {shlex.quote(str(design_path.resolve()))}; "
        "strash; "
        f"{action_command}; "
        "print_stats; "
        f"write_aiger {shlex.quote(str(snapshot_path))}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    abc_bin = common.resolve_abc_bin(args.abc_bin)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    action_specs = common.load_action_specs(labels=args.action_label, raw_specs=args.action_spec)
    designs = common.discover_selected_designs(args.dataset_root, args.design)

    raw_rows: list[dict[str, object]] = []
    summary_rows: dict[str, list[dict[str, object]]] = {common.NOOP_LABEL: []}
    for label in action_specs:
        summary_rows[label] = []

    with tempfile.TemporaryDirectory(prefix="action_static_", dir=str(output_root)) as temp_dir:
        temp_root = Path(temp_dir)
        for design_name, design_path in designs.items():
            baseline_snapshot = temp_root / f"{design_name.replace('/', '__')}.baseline.aig"
            baseline_measurement = common.evaluate_abc_script(
                abc_bin,
                _baseline_script(design_path, baseline_snapshot),
            )
            if not baseline_measurement.success or baseline_measurement.and_count is None or baseline_measurement.lev_count is None:
                raise SystemExit(f"Baseline strash failed for {design_name}: {baseline_measurement.error or baseline_measurement.stderr}")

            baseline_runtime = baseline_measurement.runtime_sec or 0.0
            baseline_memory = baseline_measurement.peak_memory_kb or 0.0
            raw_rows.append(
                {
                    "design_name": design_name,
                    "action_label": common.NOOP_LABEL,
                    "abc_command": "strash",
                    "status": "ok",
                    "final_and": baseline_measurement.and_count,
                    "final_lev": baseline_measurement.lev_count,
                    "runtime_sec": baseline_runtime,
                    "peak_memory_kb": baseline_memory,
                    "delta_and": 0,
                    "delta_lev": 0,
                    "returncode": baseline_measurement.returncode,
                }
            )
            summary_rows[common.NOOP_LABEL].append(
                {
                    "design_name": design_name,
                    "variant_label": common.NOOP_LABEL,
                    "final_and": baseline_measurement.and_count,
                    "final_lev": baseline_measurement.lev_count,
                    "total_runtime_sec": baseline_runtime,
                    "peak_memory_kb": baseline_memory,
                }
            )

            for label, action_command in action_specs.items():
                snapshot_path = temp_root / f"{design_name.replace('/', '__')}.{label}.aig"
                measurement = common.evaluate_abc_script(
                    abc_bin,
                    _action_script(design_path, action_command, snapshot_path),
                )
                final_and, final_lev, runtime_sec, peak_memory_kb = common.penalized_metrics(
                    baseline_and=baseline_measurement.and_count,
                    baseline_lev=baseline_measurement.lev_count,
                    baseline_runtime_sec=baseline_measurement.runtime_sec,
                    baseline_peak_memory_kb=baseline_measurement.peak_memory_kb,
                    measurement=measurement,
                )
                raw_rows.append(
                    {
                        "design_name": design_name,
                        "action_label": label,
                        "abc_command": action_command,
                        "status": "ok" if measurement.success else "failed",
                        "final_and": final_and,
                        "final_lev": final_lev,
                        "runtime_sec": runtime_sec,
                        "peak_memory_kb": peak_memory_kb,
                        "delta_and": final_and - baseline_measurement.and_count,
                        "delta_lev": final_lev - baseline_measurement.lev_count,
                        "returncode": measurement.returncode,
                        "error": measurement.error or "",
                    }
                )
                summary_rows[label].append(
                    {
                        "design_name": design_name,
                        "variant_label": label,
                        "final_and": final_and,
                        "final_lev": final_lev,
                        "total_runtime_sec": runtime_sec,
                        "peak_memory_kb": peak_memory_kb,
                    }
                )

    raw_path = output_root / "raw_details.csv"
    common.write_csv(raw_path, raw_rows)

    summary_specs: list[cas.SummarySpec] = []
    summaries_root = output_root / "summaries"
    for label, rows in summary_rows.items():
        summary_path = summaries_root / f"{label}.csv"
        common.write_summary_csv(summary_path, rows)
        summary_specs.append(cas.SummarySpec(label, summary_path))

    common.compare_summary_specs(summary_specs, output_root=output_root / "compare")
    print(f"Raw detail CSV written to: {raw_path}")
    print(f"Compare aggregate CSV written to: {output_root / 'compare' / 'aggregate.csv'}")
    print(f"Compare detail CSV written to: {output_root / 'compare' / 'details.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
