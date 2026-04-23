#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import sys

from . import result_utils


def _parse_args(argv: list[str]) -> tuple[list[str] | None, str]:
    method_names: list[str] = []
    workdir = result_utils.DEFAULT_WORKDIR_NAME
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--workdir":
            if index + 1 >= len(argv):
                raise SystemExit("--workdir requires a value")
            workdir = argv[index + 1]
            index += 2
            continue
        method_names.append(token)
        index += 1
    return (method_names or None), workdir


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    method_names, workdir = _parse_args(argv)
    result_utils.set_workdir(workdir)
    rows = result_utils.load_all_result_rows(method_names=method_names)
    if not rows:
        print("No result files found.")
        return

    table_rows = result_utils.extract_table_rows(rows)
    results_root = result_utils.get_results_root()
    if method_names and len(method_names) == 1:
        result_path = os.path.join(results_root, method_names[0])
    else:
        result_path = results_root
    csv_path = os.path.join(result_path, "summary.csv")
    result_utils.ensure_dir(result_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "file",
                # "method",
                "variant_label",
                # "steps",
                # "iterations",
                # "seed",
                # "ucb_c",
                # "alpha",
                # "lambda",
                # "and_reward_weight",
                # "lev_reward_weight",
                "and",
                "lev",
                "runtime_sec",
                "peak_memory_kb",
            )
        )
        writer.writerows(table_rows)
    print(f"CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
