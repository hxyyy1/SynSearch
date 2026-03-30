#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from typing import Dict, List, Sequence


DEFAULT_WORKDIR_NAME = ".mabsyn_work"
WORKDIR = os.path.abspath(DEFAULT_WORKDIR_NAME)
RESULTS_ROOT = os.path.join(WORKDIR, "results")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_workdir(path: str) -> str:
    global WORKDIR, RESULTS_ROOT
    WORKDIR = os.path.abspath(path)
    RESULTS_ROOT = os.path.join(WORKDIR, "results")
    ensure_dir(WORKDIR)
    ensure_dir(RESULTS_ROOT)
    return WORKDIR


def get_workdir() -> str:
    ensure_dir(WORKDIR)
    return WORKDIR


def get_results_root() -> str:
    ensure_dir(RESULTS_ROOT)
    return RESULTS_ROOT


def default_output_path(filename: str) -> str:
    return os.path.join(get_workdir(), filename)


def ensure_parent_dir(path: str) -> str:
    resolved = os.path.abspath(path)
    parent = os.path.dirname(resolved)
    if parent:
        ensure_dir(parent)
    return resolved


def optional_output_path(
    kv: Dict[str, str],
    key: str,
    *,
    default_filename: str | None = None,
) -> str | None:
    if key in kv:
        return ensure_parent_dir(kv[key])
    if default_filename is None:
        return None
    return ensure_parent_dir(default_output_path(default_filename))


def get_cache_root(cache_name: str) -> str:
    path = os.path.join(get_workdir(), "cache", cache_name)
    ensure_dir(path)
    return path


def natural_sort_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def get_method_results_dir(method_name: str) -> str:
    path = os.path.join(get_results_root(), method_name)
    ensure_dir(path)
    return path


def benchmark_to_result_filename(benchmark_name: str) -> str:
    normalized = benchmark_name.replace("\\", "/").strip("/")
    stem, _ = os.path.splitext(normalized)
    safe = stem.replace("/", "__")
    return f"{safe}.json"


def write_benchmark_result(method_name: str, result: Dict) -> str:
    benchmark_name = str(result.get("benchmark", "unknown"))
    result_dir = get_method_results_dir(method_name)
    result_path = os.path.join(result_dir, benchmark_to_result_filename(benchmark_name))
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result_path


def write_method_summary(method_name: str, payload: Dict, filename: str = "_summary.json") -> str:
    result_dir = get_method_results_dir(method_name)
    summary_path = os.path.join(result_dir, filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return summary_path


def load_all_result_rows(method_names: Sequence[str] | None = None) -> List[Dict]:
    results_root = get_results_root()
    if method_names is None:
        if not os.path.isdir(results_root):
            return []
        method_names = sorted(
            name for name in os.listdir(results_root)
            if os.path.isdir(os.path.join(results_root, name))
        )

    rows: List[Dict] = []
    for method_name in method_names:
        method_dir = os.path.join(results_root, method_name)
        if not os.path.isdir(method_dir):
            continue
        for name in sorted(os.listdir(method_dir)):
            if not name.endswith(".json") or name.startswith("_"):
                continue
            path = os.path.join(method_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            rows.append(
                {
                    "method": method_name,
                    "result_file": name,
                    "path": path,
                    "payload": payload,
                }
            )
    rows.sort(
        key=lambda row: (
            natural_sort_key(str(row["payload"].get("benchmark", ""))),
            natural_sort_key(str(row["method"])),
            natural_sort_key(str(row["result_file"])),
        )
    )
    return rows


def extract_table_rows(rows: Sequence[Dict]) -> List[List[str]]:
    table_rows: List[List[str]] = []
    for row in rows:
        payload = row["payload"]
        best = payload.get("best", {})
        runtime_sec = payload.get("runtime_sec")
        peak_memory_kb = payload.get("peak_memory_kb")
        runtime_text = "-"
        peak_memory_text = "-"
        if isinstance(runtime_sec, (int, float)):
            runtime_text = f"{float(runtime_sec):.3f}"
        if isinstance(peak_memory_kb, (int, float)):
            peak_memory_text = f"{float(peak_memory_kb):.0f}"
        table_rows.append(
            [
                str(payload.get("benchmark", row["result_file"])),
                str(best.get("nodes", "-")),
                str(best.get("level", "-")),
                runtime_text,
                peak_memory_text,
            ]
        )
    return table_rows


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def format_row(cells: Sequence[str]) -> str:
        return " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(cells))

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


set_workdir(DEFAULT_WORKDIR_NAME)
