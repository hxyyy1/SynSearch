from __future__ import annotations

import argparse
import json
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path

import action_eval_common as common
import compare_algorithm_summaries as cas
from alphasyn.backend import ABCBackend


@dataclass(frozen=True)
class PrefixSample:
    sample_name: str
    design_name: str
    design_path: Path
    source_result: Path
    prefix: tuple[str, ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 2: evaluate candidate ABC commands on sampled intermediate search states."
    )
    parser.add_argument(
        "--result-path",
        action="append",
        default=[],
        help="Result JSON file or directory containing search-result JSON files. Repeatable.",
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
        "--max-prefixes-per-design",
        type=int,
        default=5,
        help="Maximum sampled prefixes per design across all provided result files.",
    )
    parser.add_argument(
        "--include-root",
        action="store_true",
        help="Include the empty prefix for each sampled design.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".action_eval/prefixes"),
    )
    return parser


def _iter_result_files(paths: list[str]) -> list[Path]:
    result_files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path.is_dir():
            result_files.extend(sorted(candidate for candidate in path.glob("*.json") if candidate.is_file()))
            continue
        if path.is_file():
            result_files.append(path)
            continue
        raise SystemExit(f"Result path not found: {path}")
    if not result_files:
        raise SystemExit("No result JSON files found. Pass --result-path with a file or directory.")
    return sorted(set(result_files))


def _select_evenly(prefixes: list[tuple[str, ...]], limit: int) -> list[tuple[str, ...]]:
    if limit <= 0 or len(prefixes) <= limit:
        return prefixes
    if limit == 1:
        return [prefixes[-1]]
    selected: list[tuple[str, ...]] = []
    last_index = len(prefixes) - 1
    for slot in range(limit):
        index = round(slot * last_index / (limit - 1))
        candidate = prefixes[index]
        if not selected or candidate != selected[-1]:
            selected.append(candidate)
    return selected


def _load_samples(
    result_files: list[Path],
    *,
    include_root: bool,
    max_prefixes_per_design: int,
) -> list[PrefixSample]:
    grouped: dict[str, list[PrefixSample]] = {}
    seen_keys: set[tuple[str, tuple[str, ...]]] = set()
    for result_path in result_files:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        design_name = str(payload.get("design_name") or "")
        raw_design_path = payload.get("design_path")
        steps = payload.get("steps")
        if not design_name or not isinstance(raw_design_path, str) or not isinstance(steps, list):
            continue
        design_path = Path(raw_design_path).resolve()
        prefixes: list[tuple[str, ...]] = []
        if include_root:
            prefixes.append(tuple())
        for step in steps:
            if not isinstance(step, dict):
                continue
            raw_prefix = step.get("prefix")
            if not isinstance(raw_prefix, list) or not all(isinstance(item, str) for item in raw_prefix):
                continue
            prefixes.append(tuple(raw_prefix))
        prefixes = _select_evenly(prefixes, max_prefixes_per_design)
        for index, prefix in enumerate(prefixes, start=1):
            dedupe_key = (str(design_path), prefix)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            sample_name = f"{design_name}::p{len(prefix):02d}::{index:02d}"
            grouped.setdefault(design_name, []).append(
                PrefixSample(
                    sample_name=sample_name,
                    design_name=design_name,
                    design_path=design_path,
                    source_result=result_path,
                    prefix=prefix,
                )
            )
    samples: list[PrefixSample] = []
    for design_name in sorted(grouped):
        samples.extend(grouped[design_name][:max_prefixes_per_design])
    if not samples:
        raise SystemExit("No usable sampled prefixes found in the provided result JSON files.")
    return samples


def _noop_script(snapshot_path: Path, output_path: Path) -> str:
    return (
        f"read_aiger {shlex.quote(str(snapshot_path))}; "
        "print_stats; "
        f"write_aiger {shlex.quote(str(output_path))}"
    )


def _action_script(snapshot_path: Path, action_command: str, output_path: Path) -> str:
    return (
        f"read_aiger {shlex.quote(str(snapshot_path))}; "
        f"{action_command}; "
        "print_stats; "
        f"write_aiger {shlex.quote(str(output_path))}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    abc_bin = common.resolve_abc_bin(args.abc_bin)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    action_specs = common.load_action_specs(labels=args.action_label, raw_specs=args.action_spec)
    result_files = _iter_result_files(args.result_path)
    samples = _load_samples(
        result_files,
        include_root=args.include_root,
        max_prefixes_per_design=args.max_prefixes_per_design,
    )

    raw_rows: list[dict[str, object]] = []
    summary_rows: dict[str, list[dict[str, object]]] = {common.NOOP_LABEL: []}
    for label in action_specs:
        summary_rows[label] = []

    backend = ABCBackend(abc_bin, output_root / "backend_cache")
    try:
        with tempfile.TemporaryDirectory(prefix="action_prefix_", dir=str(output_root)) as temp_dir:
            temp_root = Path(temp_dir)
            for sample in samples:
                state = backend.evaluate_prefix(sample.design_path, sample.prefix)
                noop_output = temp_root / f"{sample.sample_name.replace('/', '__')}.noop.aig"
                noop_measurement = common.evaluate_abc_script(
                    abc_bin,
                    _noop_script(state.snapshot_path, noop_output),
                )
                if not noop_measurement.success or noop_measurement.and_count is None or noop_measurement.lev_count is None:
                    raise SystemExit(f"Failed to evaluate sampled state {sample.sample_name}: {noop_measurement.error or noop_measurement.stderr}")

                noop_runtime = noop_measurement.runtime_sec or 0.0
                noop_memory = noop_measurement.peak_memory_kb or 0.0
                raw_rows.append(
                    {
                        "sample_name": sample.sample_name,
                        "design_name": sample.design_name,
                        "source_result": str(sample.source_result),
                        "prefix": "; ".join(sample.prefix) if sample.prefix else "-",
                        "prefix_length": len(sample.prefix),
                        "action_label": common.NOOP_LABEL,
                        "abc_command": "noop",
                        "status": "ok",
                        "state_and": state.and_count,
                        "state_lev": state.lev_count,
                        "final_and": noop_measurement.and_count,
                        "final_lev": noop_measurement.lev_count,
                        "runtime_sec": noop_runtime,
                        "peak_memory_kb": noop_memory,
                        "delta_and": noop_measurement.and_count - state.and_count,
                        "delta_lev": noop_measurement.lev_count - state.lev_count,
                    }
                )
                summary_rows[common.NOOP_LABEL].append(
                    {
                        "design_name": sample.sample_name,
                        "variant_label": common.NOOP_LABEL,
                        "final_and": noop_measurement.and_count,
                        "final_lev": noop_measurement.lev_count,
                        "total_runtime_sec": noop_runtime,
                        "peak_memory_kb": noop_memory,
                    }
                )

                for label, action_command in action_specs.items():
                    output_path = temp_root / f"{sample.sample_name.replace('/', '__')}.{label}.aig"
                    measurement = common.evaluate_abc_script(
                        abc_bin,
                        _action_script(state.snapshot_path, action_command, output_path),
                    )
                    final_and, final_lev, runtime_sec, peak_memory_kb = common.penalized_metrics(
                        baseline_and=state.and_count,
                        baseline_lev=state.lev_count,
                        baseline_runtime_sec=noop_measurement.runtime_sec,
                        baseline_peak_memory_kb=noop_measurement.peak_memory_kb,
                        measurement=measurement,
                    )
                    raw_rows.append(
                        {
                            "sample_name": sample.sample_name,
                            "design_name": sample.design_name,
                            "source_result": str(sample.source_result),
                            "prefix": "; ".join(sample.prefix) if sample.prefix else "-",
                            "prefix_length": len(sample.prefix),
                            "action_label": label,
                            "abc_command": action_command,
                            "status": "ok" if measurement.success else "failed",
                            "state_and": state.and_count,
                            "state_lev": state.lev_count,
                            "final_and": final_and,
                            "final_lev": final_lev,
                            "runtime_sec": runtime_sec,
                            "peak_memory_kb": peak_memory_kb,
                            "delta_and": final_and - state.and_count,
                            "delta_lev": final_lev - state.lev_count,
                            "returncode": measurement.returncode,
                            "error": measurement.error or "",
                        }
                    )
                    summary_rows[label].append(
                        {
                            "design_name": sample.sample_name,
                            "variant_label": label,
                            "final_and": final_and,
                            "final_lev": final_lev,
                            "total_runtime_sec": runtime_sec,
                            "peak_memory_kb": peak_memory_kb,
                        }
                    )
    finally:
        backend.cleanup_cache()

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
