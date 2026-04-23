from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


METRIC_PREFIX = "__EXTERNAL_METRICS__"


def _find_time_binary() -> str | None:
    for candidate in ("/usr/bin/time", shutil.which("gtime"), shutil.which("time")):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a command under an external monitor and optionally patch result JSON files "
            "with the measured wall time and peak RSS."
        )
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional output JSON file containing the measured metrics.",
    )
    parser.add_argument(
        "--patch-json",
        action="append",
        type=Path,
        default=[],
        help="Result JSON file to patch after the command exits. Can be provided multiple times.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute. Use '--' before the command.",
    )
    return parser


def _strip_command_prefix(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def _measure_with_time(command: list[str]) -> tuple[int, float | None, float | None]:
    time_bin = _find_time_binary()
    if time_bin is None:
        started = subprocess.Popen(command)
        return_code = started.wait()
        return return_code, None, None

    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as handle:
        metrics_path = Path(handle.name)

    wrapped = [
        time_bin,
        "-f",
        f"{METRIC_PREFIX} %e %M",
        "-o",
        str(metrics_path),
        *command,
    ]
    try:
        completed = subprocess.run(wrapped, check=False)
        runtime_sec, peak_memory_kb = _parse_metrics_file(metrics_path)
        return completed.returncode, runtime_sec, peak_memory_kb
    finally:
        metrics_path.unlink(missing_ok=True)


def _parse_metrics_file(path: Path) -> tuple[float | None, float | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, None
    for line in text.splitlines():
        if not line.startswith(f"{METRIC_PREFIX} "):
            continue
        _, runtime_text, memory_text = line.split()
        return float(runtime_text), float(memory_text)
    return None, None


def _patch_result_json(path: Path, runtime_sec: float | None, peak_memory_kb: float | None) -> None:
    if not path.exists():
        print(
            f"[external_monitor] patch target does not exist, skipping: {path}",
            file=sys.stderr,
        )
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    updated = _patch_payload(payload, runtime_sec=runtime_sec, peak_memory_kb=peak_memory_kb)
    path.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")


def _patch_payload(
    payload: dict[str, Any],
    *,
    runtime_sec: float | None,
    peak_memory_kb: float | None,
) -> dict[str, Any]:
    updated = dict(payload)
    if "total_runtime_sec" in updated and runtime_sec is not None:
        updated["total_runtime_sec"] = runtime_sec
    if "runtime_sec" in updated and runtime_sec is not None:
        updated["runtime_sec"] = runtime_sec
    if "peak_memory_kb" in updated and peak_memory_kb is not None:
        updated["peak_memory_kb"] = peak_memory_kb

    results = updated.get("results")
    if isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict):
        updated_results = [dict(results[0])]
        if "total_runtime_sec" in updated_results[0] and runtime_sec is not None:
            updated_results[0]["total_runtime_sec"] = runtime_sec
        if "runtime_sec" in updated_results[0] and runtime_sec is not None:
            updated_results[0]["runtime_sec"] = runtime_sec
        if "peak_memory_kb" in updated_results[0] and peak_memory_kb is not None:
            updated_results[0]["peak_memory_kb"] = peak_memory_kb
        updated["results"] = updated_results
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = _strip_command_prefix(list(args.command))
    if not command:
        raise SystemExit("No command provided. Use '-- <command> ...'.")

    return_code, runtime_sec, peak_memory_kb = _measure_with_time(command)
    metrics_payload = {
        "command": command,
        "returncode": return_code,
        "runtime_sec": runtime_sec,
        "peak_memory_kb": peak_memory_kb,
    }

    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    for patch_path in args.patch_json:
        _patch_result_json(patch_path, runtime_sec=runtime_sec, peak_memory_kb=peak_memory_kb)

    print(json.dumps(metrics_payload, ensure_ascii=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
