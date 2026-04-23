from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import compare_algorithm_summaries as cas
from alphasyn.backend import parse_abc_stats
from alphasyn.dataset import discover_blif_designs
from alphasyn.types import ACTION_TO_ABC_COMMAND, DEFAULT_ACTION_SPACE


DEFAULT_CANDIDATE_ACTIONS: dict[str, str] = {
    "fraig": "fraig",
    "fx": "renode; sop; fx; strash",
    "mfs": "renode; mfs; strash",
    "dsd": "dsd; strash",
    "dch": "dch; strash",
    "extract": "extract",
    "extract-a": "extract -a",
    "collapse": "collapse; strash",
}
NOOP_LABEL = "baseline"
TIME_PREFIX = "__ACTION_EVAL_TIME__"


@dataclass(frozen=True)
class ActionMeasurement:
    returncode: int
    runtime_sec: float | None
    peak_memory_kb: float | None
    stdout: str
    stderr: str
    and_count: int | None
    lev_count: int | None
    success: bool
    error: str | None


def _find_time_binary() -> str | None:
    for candidate in ("/usr/bin/time", shutil.which("gtime"), shutil.which("time")):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _parse_metrics_file(path: Path) -> tuple[float | None, float | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, None
    for line in text.splitlines():
        if not line.startswith(f"{TIME_PREFIX} "):
            continue
        _, runtime_text, memory_text = line.split()
        return float(runtime_text), float(memory_text)
    return None, None


def run_command_with_metrics(command: list[str]) -> tuple[int, float | None, float | None, str, str]:
    time_bin = _find_time_binary()
    if time_bin is None:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        return completed.returncode, None, None, completed.stdout or "", completed.stderr or ""

    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as handle:
        metrics_path = Path(handle.name)
    wrapped = [
        time_bin,
        "-f",
        f"{TIME_PREFIX} %e %M",
        "-o",
        str(metrics_path),
        *command,
    ]
    try:
        completed = subprocess.run(wrapped, check=False, capture_output=True, text=True)
        runtime_sec, peak_memory_kb = _parse_metrics_file(metrics_path)
        return (
            completed.returncode,
            runtime_sec,
            peak_memory_kb,
            completed.stdout or "",
            completed.stderr or "",
        )
    finally:
        metrics_path.unlink(missing_ok=True)


def evaluate_abc_script(abc_bin: str, script: str) -> ActionMeasurement:
    returncode, runtime_sec, peak_memory_kb, stdout, stderr = run_command_with_metrics(
        [abc_bin, "-c", script]
    )
    payload = "\n".join(part for part in (stdout, stderr) if part).strip()
    and_count: int | None = None
    lev_count: int | None = None
    error: str | None = None
    try:
        and_count, lev_count = parse_abc_stats(payload)
    except Exception as exc:  # pragma: no cover - defensive, parse helper already tested elsewhere
        error = str(exc)
    success = returncode == 0 and and_count is not None and lev_count is not None
    if returncode != 0 and error is None:
        error = f"ABC exited with return code {returncode}"
    return ActionMeasurement(
        returncode=returncode,
        runtime_sec=runtime_sec,
        peak_memory_kb=peak_memory_kb,
        stdout=stdout,
        stderr=stderr,
        and_count=and_count,
        lev_count=lev_count,
        success=success,
        error=error,
    )


def resolve_abc_bin(raw_abc_bin: str | None) -> str:
    if raw_abc_bin:
        return raw_abc_bin
    for candidate in ("abc", "yosys-abc", "berkeley-abc"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    fallback = REPO_ROOT.parent / "abc" / "abc"
    if fallback.exists():
        return str(fallback)
    raise SystemExit("ABC executable not found. Pass --abc-bin explicitly.")


def load_action_specs(
    *,
    labels: list[str],
    raw_specs: list[str],
) -> dict[str, str]:
    specs = dict(DEFAULT_CANDIDATE_ACTIONS)
    for item in raw_specs:
        if "=" not in item:
            raise SystemExit(f"Invalid --action-spec: {item}")
        label, command = item.split("=", 1)
        label = label.strip()
        command = command.strip()
        if not label or not command:
            raise SystemExit(f"Invalid --action-spec: {item}")
        specs[label] = command
    selected_labels = labels or list(DEFAULT_CANDIDATE_ACTIONS)
    unknown = [label for label in selected_labels if label not in specs]
    if unknown:
        raise SystemExit(f"Unknown action labels: {', '.join(sorted(unknown))}")
    return {label: specs[label] for label in selected_labels}


def discover_selected_designs(dataset_root: Path, raw_designs: list[str]) -> dict[str, Path]:
    designs = discover_blif_designs(dataset_root)
    if not raw_designs:
        return designs
    missing = [design for design in raw_designs if design not in designs]
    if missing:
        available = ", ".join(sorted(designs))
        raise SystemExit(
            f"Unknown design(s): {', '.join(missing)}. Available designs: {available}"
        )
    return {design: designs[design] for design in raw_designs}


def _penalized_metric(
    candidate_value: float | None,
    fallback_value: float | None,
    *,
    multiplier: float,
    minimum: float,
) -> float:
    if isinstance(candidate_value, (int, float)):
        return float(candidate_value)
    if isinstance(fallback_value, (int, float)):
        return max(float(fallback_value) * multiplier, minimum)
    return minimum


def penalized_metrics(
    *,
    baseline_and: int,
    baseline_lev: int,
    baseline_runtime_sec: float | None,
    baseline_peak_memory_kb: float | None,
    measurement: ActionMeasurement,
) -> tuple[float, float, float, float]:
    return (
        _penalized_metric(measurement.and_count, baseline_and, multiplier=10.0, minimum=1.0),
        _penalized_metric(measurement.lev_count, baseline_lev, multiplier=10.0, minimum=1.0),
        _penalized_metric(
            measurement.runtime_sec,
            baseline_runtime_sec,
            multiplier=10.0,
            minimum=1e6,
        ),
        _penalized_metric(
            measurement.peak_memory_kb,
            baseline_peak_memory_kb,
            multiplier=10.0,
            minimum=1e6,
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "design_name",
                "variant_label",
                "final_and",
                "final_lev",
                "total_runtime_sec",
                "peak_memory_kb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def compare_summary_specs(
    specs: list[cas.SummarySpec],
    *,
    output_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    loaded = [summary for spec in specs for summary in cas.load_summaries(spec)]
    aggregate_rows, detail_rows, _, _, _ = cas.compare_summaries(
        loaded,
        weights={"and": 0.5, "lev": 0.2, "runtime": 0.2, "memory": 0.1},
        missing_policy="common",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    cas._write_csv(output_root / "aggregate.csv", aggregate_rows)
    cas._write_csv(output_root / "details.csv", detail_rows)
    return aggregate_rows, detail_rows


def build_default_presets(action_labels: list[str], *, include_combined: bool) -> dict[str, list[str]]:
    presets: dict[str, list[str]] = {"baseline": list(DEFAULT_ACTION_SPACE)}
    for label in action_labels:
        preset_name = f"plus_{label}"
        presets[preset_name] = [*DEFAULT_ACTION_SPACE, label]
    if include_combined and action_labels:
        presets["all_selected"] = [*DEFAULT_ACTION_SPACE, *action_labels]
    return presets


def load_presets(
    *,
    preset_json: Path | None,
    action_labels: list[str],
    include_combined: bool,
) -> dict[str, list[str]]:
    if preset_json is None:
        return build_default_presets(action_labels, include_combined=include_combined)
    payload = json.loads(preset_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Preset JSON must be an object mapping preset name to action-label list.")
    presets: dict[str, list[str]] = {}
    for preset_name, values in payload.items():
        if not isinstance(preset_name, str) or not preset_name:
            raise SystemExit("Preset names must be non-empty strings.")
        if not isinstance(values, list) or not values or not all(isinstance(item, str) for item in values):
            raise SystemExit(f"Preset '{preset_name}' must be a non-empty list of action labels.")
        presets[preset_name] = values
    return presets


def load_algorithm_options(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Algorithm config JSON must be an object.")
    normalized: dict[str, dict[str, Any]] = {}
    for algorithm, options in payload.items():
        if not isinstance(options, dict):
            raise SystemExit(f"Algorithm config for '{algorithm}' must be an object.")
        normalized[algorithm] = options
    return normalized


def option_list_from_mapping(options: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in options.items():
        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(option)
            continue
        argv.append(f"{option}={value}")
    return argv


def resolve_algorithm_actions(
    algorithm_key: str,
    action_labels: list[str],
) -> list[str]:
    if algorithm_key in {"alphasyn", "hybridsyn", "sasyn"}:
        return action_labels
    return [ACTION_TO_ABC_COMMAND[label] for label in action_labels]

