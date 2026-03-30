#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import random
import sys

import numpy as np

from . import linucb as linucb
from . import result_utils


def main() -> None:
    _, kv = linucb.parse_runtime_args(sys.argv)

    abc_bin = kv.get("abc-bin", linucb.ABC_BIN)
    workdir = kv.get("workdir", result_utils.DEFAULT_WORKDIR_NAME)
    result_utils.set_workdir(workdir)
    result_json = result_utils.optional_output_path(
        kv,
        "result-json",
    )
    log_level = kv.get("log-level", "INFO").upper()

    linucb.K_STEPS = int(kv.get("steps", str(linucb.K_STEPS)))
    linucb.N_EPISODES = int(kv.get("episodes", str(linucb.N_EPISODES)))
    linucb.ABC_TIMEOUT_SEC = int(kv.get("timeout", str(linucb.ABC_TIMEOUT_SEC)))
    linucb.SPIKE_FACTOR = float(kv.get("spike-factor", str(linucb.SPIKE_FACTOR)))
    linucb.SEVERE_NEGATIVE_REWARD = float(
        kv.get("severe-negative-reward", str(linucb.SEVERE_NEGATIVE_REWARD))
    )
    linucb.TIMEOUT_NEGATIVE_REWARD = float(
        kv.get("timeout-negative-reward", str(linucb.TIMEOUT_NEGATIVE_REWARD))
    )
    linucb.RANDOM_SEED = int(kv.get("seed", str(linucb.RANDOM_SEED)))

    alpha_str = kv.get(
        "alpha",
        kv.get("linucb-alpha", kv.get("ucb-c", str(linucb.ALPHA))),
    )
    lambda_str = kv.get("lambda", kv.get("linucb-lambda", str(linucb.REG_LAMBDA)))
    linucb.ALPHA = float(alpha_str)
    linucb.REG_LAMBDA = float(lambda_str)
    linucb.LONG_TERM_ROLLOUTS = int(
        kv.get("long-term-rollouts", str(linucb.LONG_TERM_ROLLOUTS))
    )
    linucb.LONG_TERM_HORIZON = int(
        kv.get("long-term-horizon", str(linucb.LONG_TERM_HORIZON))
    )
    linucb.RETURN_BACK_THRESHOLD = float(
        kv.get("return-back-threshold", str(linucb.RETURN_BACK_THRESHOLD))
    )

    if "actions" in kv:
        parsed_actions = [x.strip() for x in kv["actions"].split(",") if x.strip()]
        if parsed_actions:
            linucb.actions[:] = parsed_actions

    linucb.setup_logging()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    random.seed(linucb.RANDOM_SEED)
    np.random.seed(linucb.RANDOM_SEED)

    logging.info("ABC二进制: %s", abc_bin)
    logging.info("Benchmark目录: %s", os.path.abspath(linucb.BENCHMARK_DIR))
    logging.info(
        "参数: steps=%d iters_per_step=%d timeout=%d alpha=%.3f lambda=%.3f "
        "rollouts=%d horizon=%d return_back_threshold=%.4f seed=%d",
        linucb.K_STEPS,
        linucb.N_EPISODES,
        linucb.ABC_TIMEOUT_SEC,
        linucb.ALPHA,
        linucb.REG_LAMBDA,
        linucb.LONG_TERM_ROLLOUTS,
        linucb.LONG_TERM_HORIZON,
        linucb.RETURN_BACK_THRESHOLD,
        linucb.RANDOM_SEED,
    )

    blif_files = linucb.find_blif_files(linucb.BENCHMARK_DIR)
    if not blif_files:
        logging.error("未找到 .blif 文件，请检查目录: %s", linucb.BENCHMARK_DIR)
        return

    all_results = {
        "meta": {
            "abc_bin": abc_bin,
            "benchmark_dir": os.path.abspath(linucb.BENCHMARK_DIR),
            "num_benchmarks": len(blif_files),
            "alpha": linucb.ALPHA,
            "lambda": linucb.REG_LAMBDA,
            "long_term_rollouts": linucb.LONG_TERM_ROLLOUTS,
            "long_term_horizon": linucb.LONG_TERM_HORIZON,
            "return_back_threshold": linucb.RETURN_BACK_THRESHOLD,
            "seed": linucb.RANDOM_SEED,
        },
        "results": [],
    }

    for blif_path in blif_files:
        rel_name = os.path.relpath(blif_path, linucb.BENCHMARK_DIR)
        result = linucb.optimize_one_benchmark(
            blif_path,
            benchmark_name=rel_name,
            alpha=linucb.ALPHA,
            reg_lambda=linucb.REG_LAMBDA,
            abc_bin=abc_bin,
            seed=linucb.RANDOM_SEED,
        )
        all_results["results"].append(result)
        linucb.write_result_artifacts(result)

    all_results["summary"] = linucb.build_summary(all_results["results"])
    summary_path = result_utils.write_method_summary("linucb", all_results)

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
