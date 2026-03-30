#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import sys

from . import baseline_mab as mab
from . import result_utils


def main() -> None:
    kv = mab.parse_optional_kv_args(sys.argv[1:])

    abc_bin = kv.get("abc-bin", mab.ABC_BIN)
    workdir = kv.get("workdir", result_utils.DEFAULT_WORKDIR_NAME)
    result_utils.set_workdir(workdir)
    result_json = result_utils.optional_output_path(
        kv,
        "result-json",
    )
    ucb_c = float(kv.get("ucb-c", "2.0"))
    log_level = kv.get("log-level", "INFO").upper()

    mab.K_STEPS = int(kv.get("steps", str(mab.K_STEPS)))
    mab.N_EPISODES = int(kv.get("episodes", str(mab.N_EPISODES)))
    mab.ABC_TIMEOUT_SEC = int(kv.get("timeout", str(mab.ABC_TIMEOUT_SEC)))
    mab.SPIKE_FACTOR = float(kv.get("spike-factor", str(mab.SPIKE_FACTOR)))
    mab.SEVERE_NEGATIVE_REWARD = float(
        kv.get("severe-negative-reward", str(mab.SEVERE_NEGATIVE_REWARD))
    )
    mab.RANDOM_SEED = int(kv.get("seed", str(mab.RANDOM_SEED)))

    if "actions" in kv:
        parsed_actions = [x.strip() for x in kv["actions"].split(",") if x.strip()]
        if parsed_actions:
            mab.actions[:] = parsed_actions

    mab.setup_logging()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

    logging.info("ABC二进制: %s", abc_bin)
    logging.info("Benchmark目录: %s", os.path.abspath(mab.BENCHMARK_DIR))

    blif_files = mab.find_blif_files(mab.BENCHMARK_DIR)
    if not blif_files:
        logging.error("未找到 .blif 文件，请检查目录: %s", mab.BENCHMARK_DIR)
        return

    all_results = {
        "meta": {
            "abc_bin": abc_bin,
            "benchmark_dir": os.path.abspath(mab.BENCHMARK_DIR),
            "num_benchmarks": len(blif_files),
        },
        "results": [],
    }

    for blif_path in blif_files:
        rel_name = os.path.relpath(blif_path, mab.BENCHMARK_DIR)
        result = mab.optimize_one_benchmark(
            blif_path,
            benchmark_name=rel_name,
            bandit_c=ucb_c,
            abc_bin=abc_bin,
        )
        all_results["results"].append(result)
        mab.write_result_artifacts(result)

    all_results["summary"] = mab.build_summary(all_results["results"])
    summary_path = result_utils.write_method_summary("baseline_mab", all_results)

    if result_json:
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    summary = all_results["summary"]
    logging.info(
        "汇总: total=%d success=%d failed=%d avg_ratio=%.4f global_ratio=%.4f",
        summary["total_cases"],
        summary["success_cases"],
        summary["failed_cases"],
        summary["avg_improvement_ratio"],
        summary["global_improvement_ratio"],
    )
    logging.info("全部完成，汇总结果已写入: %s", os.path.abspath(summary_path))
    if result_json:
        logging.info("聚合结果 JSON 已写入: %s", os.path.abspath(result_json))


if __name__ == "__main__":
    main()
