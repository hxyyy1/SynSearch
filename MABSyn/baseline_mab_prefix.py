#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UCB1 baseline with step-wise prefix evaluation and AIG prefix caching.

- 每个 episode 从初始电路开始
- 每个 step 选择一个动作后，立即评估当前 prefix
- reward 使用当前 prefix 相对上一个 prefix 的边际改进
- 通过前缀 AIG 快照缓存复用 ABC 评估结果，减少重复开销
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple

from external_monitoring import (
    is_external_monitor_active,
    remove_flag,
    run_under_external_monitor,
    upsert_option,
)

from . import result_utils, summarize_results
from .baseline_mab import (
    ABC_BIN as DEFAULT_ABC_BIN,
    ABC_TIMEOUT_SEC as DEFAULT_ABC_TIMEOUT_SEC,
    AND_REWARD_WEIGHT as DEFAULT_AND_REWARD_WEIGHT,
    K_STEPS as DEFAULT_K_STEPS,
    LEV_REWARD_WEIGHT as DEFAULT_LEV_REWARD_WEIGHT,
    N_EPISODES as DEFAULT_N_EPISODES,
    RANDOM_SEED as DEFAULT_RANDOM_SEED,
    SEVERE_NEGATIVE_REWARD as DEFAULT_SEVERE_NEGATIVE_REWARD,
    SPIKE_FACTOR as DEFAULT_SPIKE_FACTOR,
    UCBBandit,
    actions as BASELINE_ACTIONS,
    build_summary,
    compute_reward,
    derive_benchmark_name,
    find_blif_files,
    natural_sort_key,
    parse_abc_stats,
    setup_logging,
    snapshot_action_state,
    write_result_artifacts,
)


METHOD_NAME = "baseline_mab_prefix"
RESULT_JSON = "baseline_mab_prefix_results.json"
DEFAULT_BANDIT_C = 0.4

actions = list(BASELINE_ACTIONS)

N_EPISODES = DEFAULT_N_EPISODES
K_STEPS = DEFAULT_K_STEPS
ABC_TIMEOUT_SEC = DEFAULT_ABC_TIMEOUT_SEC
SPIKE_FACTOR = DEFAULT_SPIKE_FACTOR
SEVERE_NEGATIVE_REWARD = DEFAULT_SEVERE_NEGATIVE_REWARD
AND_REWARD_WEIGHT = DEFAULT_AND_REWARD_WEIGHT
LEV_REWARD_WEIGHT = DEFAULT_LEV_REWARD_WEIGHT
RANDOM_SEED = DEFAULT_RANDOM_SEED
def compute_weighted_result_cost(
    reference_nodes: int,
    candidate_nodes: int,
    reference_level: Optional[int],
    candidate_level: Optional[int],
    and_reward_weight: float = AND_REWARD_WEIGHT,
    lev_reward_weight: float = LEV_REWARD_WEIGHT,
) -> Optional[float]:
    if reference_level is None or candidate_level is None:
        return None

    weight_sum = and_reward_weight + lev_reward_weight
    if weight_sum <= 0:
        raise ValueError("and_reward_weight + lev_reward_weight must be positive")

    normalized_and_weight = and_reward_weight / weight_sum
    normalized_lev_weight = lev_reward_weight / weight_sum
    return (
        normalized_and_weight * (float(candidate_nodes) / max(reference_nodes, 1))
        + normalized_lev_weight * (float(candidate_level) / max(reference_level, 1))
    )


def is_better_result(
    reference_nodes: int,
    reference_level: Optional[int],
    candidate_nodes: int,
    candidate_level: Optional[int],
    incumbent_nodes: int,
    incumbent_level: Optional[int],
    and_reward_weight: float = AND_REWARD_WEIGHT,
    lev_reward_weight: float = LEV_REWARD_WEIGHT,
) -> bool:
    candidate_cost = compute_weighted_result_cost(
        reference_nodes=reference_nodes,
        candidate_nodes=candidate_nodes,
        reference_level=reference_level,
        candidate_level=candidate_level,
        and_reward_weight=and_reward_weight,
        lev_reward_weight=lev_reward_weight,
    )
    incumbent_cost = compute_weighted_result_cost(
        reference_nodes=reference_nodes,
        candidate_nodes=incumbent_nodes,
        reference_level=reference_level,
        candidate_level=incumbent_level,
        and_reward_weight=and_reward_weight,
        lev_reward_weight=lev_reward_weight,
    )

    if candidate_cost is None:
        return False
    if incumbent_cost is None:
        return True
    if candidate_cost != incumbent_cost:
        return candidate_cost < incumbent_cost


@dataclass
class PrefixCacheEntry:
    prefix: Tuple[str, ...]
    nodes: Optional[int]
    level: Optional[int]
    snapshot_path: Optional[str]
    success: bool
    runtime_sec: float
    log: str
    cache_hit: bool
    peak_memory_kb: float | None


class PrefixCache:
    """Prefix cache that persists each evaluated prefix as an AIG snapshot."""

    def __init__(self, design_path: str, abc_bin: str, timeout_sec: int):
        self.design_path = os.path.abspath(design_path)
        self.abc_bin = abc_bin
        self.timeout_sec = timeout_sec
        cache_root = result_utils.get_cache_root(METHOD_NAME)
        self.cache_dir = os.path.join(cache_root, uuid.uuid4().hex)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.entries: Dict[Tuple[str, ...], PrefixCacheEntry] = {}

    def close(self) -> None:
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def get_or_build(self, prefix: Tuple[str, ...]) -> PrefixCacheEntry:
        cached = self.entries.get(prefix)
        if cached is not None:
            if cached.snapshot_path is None or os.path.exists(cached.snapshot_path):
                return cached

        key = self._cache_key(prefix)
        snapshot_path = os.path.join(self.cache_dir, f"{key}.aig")
        if prefix:
            parent = self.get_or_build(prefix[:-1])
            if (not parent.success) or (parent.snapshot_path is None):
                entry = PrefixCacheEntry(
                    prefix=prefix,
                    nodes=None,
                    level=None,
                    snapshot_path=None,
                    success=False,
                    runtime_sec=0.0,
                    log="parent prefix failed",
                    cache_hit=False,
                    peak_memory_kb=None,
                )
                self.entries[prefix] = entry
                return entry
            read_command = f"read_aiger {shlex.quote(parent.snapshot_path)}"
            commands = [prefix[-1], "print_stats", f"write_aiger {shlex.quote(snapshot_path)}"]
        else:
            read_command = f"read_blif {shlex.quote(self.design_path)}"
            commands = ["strash", "print_stats", f"write_aiger {shlex.quote(snapshot_path)}"]

        command_string = "; ".join([read_command, *commands])
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [self.abc_bin, "-c", command_string],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
            peak_memory_kb = None
            success = completed.returncode == 0
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            success = False
            peak_memory_kb = None
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        except OSError as exc:
            success = False
            peak_memory_kb = None
            stdout = ""
            stderr = str(exc)

        runtime_sec = time.perf_counter() - started
        payload = "\n".join(part for part in (stdout, stderr) if part).strip()
        nodes, level = parse_abc_stats(payload)
        snapshot_exists = os.path.exists(snapshot_path)
        entry = PrefixCacheEntry(
            prefix=prefix,
            nodes=nodes,
            level=level,
            snapshot_path=snapshot_path if snapshot_exists else None,
            success=bool(success and nodes is not None and level is not None and snapshot_exists),
            runtime_sec=runtime_sec,
            log=payload,
            cache_hit=False,
            peak_memory_kb=peak_memory_kb,
        )
        self.entries[prefix] = entry
        return entry

    def _cache_key(self, prefix: Tuple[str, ...]) -> str:
        payload = json.dumps(
            {
                "design": self.design_path,
                "prefix": list(prefix),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return uuid.uuid5(uuid.NAMESPACE_URL, payload).hex


def optimize_one_benchmark(
    blif_path: str,
    benchmark_name: Optional[str] = None,
    bandit_c: float = DEFAULT_BANDIT_C,
    abc_bin: Optional[str] = None,
    debug_search: bool = False,
) -> Dict:
    if benchmark_name is None:
        benchmark_name = derive_benchmark_name(blif_path)
    logging.info("开始处理: %s", benchmark_name)
    started = time.perf_counter()
    peak_memory_kb: float | None = None
    if abc_bin is None:
        abc_bin = DEFAULT_ABC_BIN

    prefix_cache = PrefixCache(
        blif_path,
        abc_bin=abc_bin,
        timeout_sec=ABC_TIMEOUT_SEC,
    )
    try:
        root_entry = prefix_cache.get_or_build(())
        if (not root_entry.success) or (root_entry.nodes is None):
            logging.error("初始评估失败: %s", benchmark_name)
            logging.debug("初始输出: %s", root_entry.log)
            return {
                "benchmark": benchmark_name,
                "status": "failed_init",
                "error": "failed to get initial stats",
                "runtime_sec": time.perf_counter() - started,
                "peak_memory_kb": peak_memory_kb,
            }

        init_nodes = root_entry.nodes
        init_level = root_entry.level

        bandits = [
            UCBBandit(
                actions,
                c=bandit_c,
                rng=random.Random(RANDOM_SEED + step_idx),
            )
            for step_idx in range(K_STEPS)
        ]

        best_nodes = init_nodes
        best_level = init_level
        best_recipe: List[str] = []
        debug_trace: List[Dict[str, object]] = []

        for ep in range(1, N_EPISODES + 1):
            current_prefix: Tuple[str, ...] = ()
            current_nodes = init_nodes
            current_level = init_level
            current_seq: List[str] = []
            episode_reward = 0.0
            episode_success = True
            episode_log_tail = ""

            for step in range(1, K_STEPS + 1):
                step_bandit = bandits[step - 1]
                action_state_rows = snapshot_action_state(step_bandit) if debug_search else []
                action_idx = step_bandit.select_action()
                action_cmd = actions[action_idx]

                step_bandit.total_steps += 1
                step_bandit.counts[action_idx] += 1

                next_prefix = current_prefix + (action_cmd,)
                next_entry = prefix_cache.get_or_build(next_prefix)

                step_reward = compute_reward(
                    current_nodes,
                    next_entry.nodes,
                    current_level,
                    next_entry.level,
                    next_entry.success,
                    severe_negative_reward=SEVERE_NEGATIVE_REWARD,
                    spike_factor=SPIKE_FACTOR,
                    and_reward_weight=AND_REWARD_WEIGHT,
                    lev_reward_weight=LEV_REWARD_WEIGHT,
                )
                step_bandit.update([action_idx], step_reward)

                current_seq.append(action_cmd)
                episode_reward += step_reward
                episode_log_tail = next_entry.log

                if debug_search:
                    sequence_prefix = "; ".join(current_seq)
                    for action_state in action_state_rows:
                        debug_trace.append(
                            {
                                "episode": ep,
                                "step": step,
                                "action_index": action_state["action_index"],
                                "action": action_state["action"],
                                "selected": int(action_state["action_index"] == action_idx),
                                "count_before": action_state["count_before"],
                                "q_before": action_state["q_before"],
                                "bonus_before": action_state["bonus_before"],
                                "score_before": action_state["score_before"],
                                "cold_start": int(bool(action_state["cold_start"])),
                                "count_after_select": step_bandit.counts[int(action_state["action_index"])],
                                "count_after_update": step_bandit.counts[int(action_state["action_index"])],
                                "q_after_update": step_bandit.q_values[int(action_state["action_index"])],
                                "sequence_prefix": sequence_prefix,
                                "step_reward": step_reward,
                                "step_nodes": next_entry.nodes,
                                "step_level": next_entry.level,
                                "step_cache_hit": int(next_entry.cache_hit),
                            }
                        )

                if (not next_entry.success) or (next_entry.nodes is None) or (next_entry.level is None):
                    episode_success = False
                    logging.warning(
                        "[%s][ep=%d step=%d] 前缀评估失败: %s, reward=%.4f",
                        benchmark_name,
                        ep,
                        step,
                        action_cmd,
                        step_reward,
                    )
                    logging.debug("ABC输出(失败): %s", next_entry.log)
                    break

                current_prefix = next_prefix
                current_nodes = next_entry.nodes
                current_level = next_entry.level

            episode_completed = episode_success and len(current_seq) == K_STEPS
            if episode_completed and is_better_result(
                reference_nodes=init_nodes,
                reference_level=init_level,
                candidate_nodes=current_nodes,
                candidate_level=current_level,
                incumbent_nodes=best_nodes,
                incumbent_level=best_level,
                and_reward_weight=AND_REWARD_WEIGHT,
                lev_reward_weight=LEV_REWARD_WEIGHT,
            ):
                best_nodes = current_nodes
                best_level = current_level
                best_recipe = list(current_seq)
                logging.info(
                    "[%s] 新最优: nd %d -> %d (ep=%d), seq=%s",
                    benchmark_name,
                    init_nodes,
                    best_nodes,
                    ep,
                    "; ".join(best_recipe),
                )

            if debug_search:
                episode_sequence = "; ".join(current_seq)
                for row in debug_trace[-(len(actions) * len(current_seq)):]:
                    if int(row["episode"]) != ep:
                        continue
                    row["episode_reward"] = episode_reward
                    row["episode_success"] = int(episode_success)
                    row["episode_nodes"] = current_nodes
                    row["episode_level"] = current_level
                    row["episode_peak_memory_kb"] = peak_memory_kb
                    row["episode_sequence"] = episode_sequence
                    row["best_nodes_after_episode"] = best_nodes
                    row["best_level_after_episode"] = best_level

            logging.info(
                "[%s] Episode %d/%d 完成, final_and=%d final_lev=%d reward=%.4f best_and=%d best_lev=%d",
                benchmark_name,
                ep,
                N_EPISODES,
                current_nodes,
                current_level,
                episode_reward,
                best_nodes,
                best_level,
            )
            logging.debug("[%s][ep=%d] tail output: %s", benchmark_name, ep, episode_log_tail)

        improvement = init_nodes - best_nodes
        improvement_ratio = (improvement / init_nodes) if init_nodes > 0 else 0.0

        return {
            "benchmark": benchmark_name,
            "benchmark_path": os.path.abspath(blif_path),
            "status": "ok",
            "initial": {
                "nodes": init_nodes,
                "level": init_level,
            },
            "best": {
                "nodes": best_nodes,
                "level": best_level,
                "recipe_str": "; ".join(best_recipe),
            },
            "improvement": {
                "nodes_reduced": improvement,
                "ratio": improvement_ratio,
            },
            "config": {
                "episodes": N_EPISODES,
                "steps_per_episode": K_STEPS,
                "ucb_c": bandit_c,
                "and_reward_weight": AND_REWARD_WEIGHT,
                "lev_reward_weight": LEV_REWARD_WEIGHT,
                "debug_search": debug_search,
                "seed": RANDOM_SEED,
                "step_reward_mode": "prefix_delta",
                "prefix_cache": True,
            },
            "runtime_sec": time.perf_counter() - started,
            "peak_memory_kb": peak_memory_kb,
            "debug_trace": debug_trace if debug_search else [],
        }
    finally:
        prefix_cache.close()


def _resolve_local_path(path: Path) -> Path:
    cwd = Path.cwd().resolve()
    candidate = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
    if not candidate.is_relative_to(cwd):
        raise SystemExit(f"Path '{path}' must stay under the current working directory: {cwd}")
    return candidate


def _discover_designs_or_exit(dataset_root: Path) -> dict[str, str]:
    files = find_blif_files(str(dataset_root))
    if not files:
        raise SystemExit(f"未找到 .blif 文件，请检查目录: {dataset_root}")
    return {
        os.path.relpath(os.path.abspath(blif_path), str(dataset_root.resolve())).replace("\\", "/"): blif_path
        for blif_path in files
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=METHOD_NAME)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-search", help="Run prefix-evaluated baseline MAB search.")
    run.add_argument("--workdir", type=Path, default=Path(result_utils.DEFAULT_WORKDIR_NAME))
    run.add_argument("--abc-bin", default=DEFAULT_ABC_BIN)
    run.add_argument("--dataset-root", type=Path, default=Path("tc_public"))
    run.add_argument("--design", help="Single .blif filename under dataset-root. Default: run all.")
    run.add_argument("--steps", type=int, default=K_STEPS)
    run.add_argument("--episodes", type=int, default=N_EPISODES)
    run.add_argument("--timeout", type=int, default=ABC_TIMEOUT_SEC)
    run.add_argument("--spike-factor", type=float, default=SPIKE_FACTOR)
    run.add_argument("--severe-negative-reward", type=float, default=SEVERE_NEGATIVE_REWARD)
    run.add_argument("--and-weight", type=float, default=AND_REWARD_WEIGHT)
    run.add_argument("--lev-weight", type=float, default=LEV_REWARD_WEIGHT)
    run.add_argument("--ucb-c", type=float, default=DEFAULT_BANDIT_C)
    run.add_argument("--seed", type=int, default=RANDOM_SEED)
    run.add_argument("--actions", default=None)
    run.add_argument("--result-json", type=Path, default=None)
    run.add_argument("--log-level", default="INFO")
    run.add_argument(
        "--external-monitor",
        action="store_true",
        help="Run each selected design under the external monitor and patch result JSON metrics.",
    )
    run.add_argument("--debug-search", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Aggregate JSON search results into CSV.")
    summarize.add_argument("--workdir", type=Path, default=Path(result_utils.DEFAULT_WORKDIR_NAME))
    return parser


def _configure_runtime_from_args(args: argparse.Namespace) -> None:
    global K_STEPS, N_EPISODES, ABC_TIMEOUT_SEC, SPIKE_FACTOR
    global SEVERE_NEGATIVE_REWARD, RANDOM_SEED, AND_REWARD_WEIGHT, LEV_REWARD_WEIGHT
    K_STEPS = args.steps
    N_EPISODES = args.episodes
    ABC_TIMEOUT_SEC = args.timeout
    SPIKE_FACTOR = args.spike_factor
    SEVERE_NEGATIVE_REWARD = args.severe_negative_reward
    AND_REWARD_WEIGHT = args.and_weight
    LEV_REWARD_WEIGHT = args.lev_weight
    RANDOM_SEED = args.seed
    if args.actions:
        parsed_actions = [x.strip() for x in args.actions.split(",") if x.strip()]
        if parsed_actions:
            actions[:] = parsed_actions


def _write_summary_csv(workdir: str) -> str:
    summarize_results.main([METHOD_NAME, "--workdir", workdir])
    return os.path.join(result_utils.get_results_root(), "summary.csv")


def _command_run_search(args: argparse.Namespace) -> int:
    args.workdir = _resolve_local_path(args.workdir)
    result_utils.set_workdir(str(args.workdir))
    result_json = _resolve_local_path(args.result_json) if args.result_json else None
    _configure_runtime_from_args(args)

    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    logging.info("ABC二进制: %s", args.abc_bin)
    logging.info("Benchmark目录: %s", os.path.abspath(args.dataset_root))

    designs = _discover_designs_or_exit(args.dataset_root)
    if args.design:
        if args.design not in designs:
            available = ", ".join(sorted(designs))
            raise SystemExit(f"Unknown design '{args.design}'. Available designs: {available}")
        selected = {args.design: designs[args.design]}
    else:
        selected = designs

    all_results = {
        "meta": {
            "abc_bin": args.abc_bin,
            "benchmark_dir": os.path.abspath(args.dataset_root),
            "num_benchmarks": len(selected),
        },
        "results": [],
    }

    for design_name, design_path in selected.items():
        result = optimize_one_benchmark(
            design_path,
            benchmark_name=design_name,
            bandit_c=args.ucb_c,
            abc_bin=args.abc_bin,
            debug_search=args.debug_search,
        )
        result["meta"] = {
            "mode": "single_design" if args.design else "batch",
            "abc_bin": args.abc_bin,
        }
        all_results["results"].append(result)
        result_path = write_result_artifacts(result, method_name=METHOD_NAME)
        logging.info("完成: %s -> %s", design_name, os.path.abspath(result_path))

    all_results["summary"] = build_summary(all_results["results"])
    summary_path = result_utils.write_method_summary(METHOD_NAME, all_results)
    if result_json:
        with open(result_json, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2, ensure_ascii=False)

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
    return 0


def _command_summarize(args: argparse.Namespace) -> int:
    args.workdir = _resolve_local_path(args.workdir)
    result_utils.set_workdir(str(args.workdir))
    summarize_results.main([METHOD_NAME, "--workdir", str(args.workdir)])
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    if (
        args.command == "run-search"
        and args.external_monitor
        and not is_external_monitor_active()
    ):
        args.workdir = _resolve_local_path(args.workdir)
        result_utils.set_workdir(str(args.workdir))
        base_argv = remove_flag(raw_argv, "--external-monitor")
        designs = _discover_designs_or_exit(args.dataset_root)
        selected_designs = [args.design] if args.design else sorted(designs, key=natural_sort_key)
        for design_name in selected_designs:
            patch_json = Path(
                result_utils.get_method_results_dir(METHOD_NAME)
            ) / result_utils.benchmark_to_result_filename(design_name)
            design_argv = upsert_option(base_argv, "--design", design_name)
            exit_code = run_under_external_monitor(
                raw_argv=design_argv,
                patch_json=patch_json,
                module_name="MABSyn.baseline_mab_prefix",
            )
            if exit_code != 0:
                return exit_code
        return 0
    if args.command == "run-search":
        return _command_run_search(args)
    if args.command == "summarize":
        return _command_summarize(args)
    parser.exit(status=2, message="Unknown command.\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
