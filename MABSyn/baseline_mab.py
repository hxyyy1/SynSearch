#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
纯 MAB（UCB1）逻辑综合 Baseline
- 通过 subprocess 调用 ABC
- 对 .blif 电路进行固定长度命令序列探索
- 优化目标：最小化 AIG 节点数（nd / and）
"""

import json
import argparse
import csv
import logging
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from external_monitoring import (
    is_external_monitor_active,
    remove_flag,
    run_under_external_monitor,
    upsert_option,
)

from . import result_utils, summarize_results

# ------------------------------
# 1) 动作空间（老虎机摇臂）
# ------------------------------
actions = [
    "rewrite",
    "rewrite -z",
    "refactor",
    "refactor -z",
    "balance",
    # "dc2",
    "resub",
    "resub -z",
]

# ------------------------------
# 可调参数
# ------------------------------
ABC_BIN = os.environ.get("ABC_BIN", "abc")
BENCHMARK_DIR = "./tc_public"
RESULT_JSON = "baseline_results.json"

N_EPISODES = 50
K_STEPS = 10
ABC_TIMEOUT_SEC = 30

# 当节点数异常突增时，给予重罚
SPIKE_FACTOR = 1.8
SEVERE_NEGATIVE_REWARD = -1
AND_REWARD_WEIGHT = 0.5
LEV_REWARD_WEIGHT = 0.2

# 随机种子（用于并列时随机打破平局）
RANDOM_SEED = 0


def natural_sort_key(text: str) -> List[object]:
    """
    按数字自然排序，避免 tc_public_10 排在 tc_public_2 前面。
    """
    parts = re.split(r"(\d+)", text)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def parse_optional_kv_args(args: List[str]) -> Dict[str, str]:
    """
    解析可选参数，支持:
    - --key value
    - --flag
    """
    boolean_flags = {"debug-search", "debug", "external-monitor"}
    kv: Dict[str, str] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            raise ValueError(f"无效参数: {token}（可选参数需为 --key value）")
        key = token[2:]
        if key in boolean_flags and (i + 1 >= len(args) or args[i + 1].startswith("--")):
            kv[key] = "true"
            i += 1
            continue
        if i + 1 >= len(args):
            raise ValueError(f"参数 {token} 缺少值")
        kv[key] = args[i + 1]
        i += 2
    return kv


def parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"无法解析布尔值: {value}")


# ------------------------------
# 2) ABC 交互模块
# ------------------------------
def run_abc_command(
    blif_path: str,
    command_sequence: List[str],
    abc_bin: Optional[str] = None,
    timeout_sec: int = ABC_TIMEOUT_SEC,
) -> Tuple[bool, str, str, float | None]:
    """
    参数:
    - blif_path: 输入 blif 路径
    - command_sequence: 需要执行的 ABC 命令列表
    - abc_bin: ABC 可执行文件路径（默认取环境变量 ABC_BIN 或 abc）
    - timeout_sec: 子进程超时时间

    返回:
    - (success, stdout, stderr)
      success=True 仅代表进程成功返回（returncode==0），不代表 parse 一定成功。
    """
    seq = "; ".join(command_sequence) if command_sequence else ""

    # 每次都从原始 blif 重新读入，并先 strash 转换到 AIG
    # 结尾 print_stats 用于解析 nd / lev
    if seq:
        abc_cmd = f"read_blif {blif_path}; strash; {seq}; print_stats"
    else:
        abc_cmd = f"read_blif {blif_path}; strash; print_stats"

    if abc_bin is None:
        abc_bin = ABC_BIN

    try:
        completed = subprocess.run(
            [abc_bin, "-c", abc_cmd],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        peak_memory_kb = None
        success = completed.returncode == 0
        return success, completed.stdout or "", completed.stderr or "", peak_memory_kb

    except subprocess.TimeoutExpired as exc:
        # 超时：返回失败，并将已捕获输出带回上层记录
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        timeout_msg = f"[Timeout] ABC timed out after {timeout_sec}s"
        return False, stdout, f"{stderr}\n{timeout_msg}".strip(), None

    except Exception as exc:  # 防止调用层卡死
        return False, "", f"[Exception] {exc}", None


# ------------------------------
# 3) 统计解析与奖励
# ------------------------------
def parse_abc_stats(abc_output_string: str) -> Tuple[Optional[int], Optional[int]]:
    """
    从 ABC 输出中解析节点数与层级。

    支持两类常见字段：
    - nd = <int>（常见）
    - and = <int>（部分版本写法）
    层级字段通常为 lev = <int>。

    返回:
    - (nodes, level)，若解析失败则对应项为 None。
    """
    if not abc_output_string:
        return None, None

    text = abc_output_string.lower()

    # 取最后一次出现的统计，避免中间日志干扰
    nd_matches = re.findall(r"\bnd\s*=\s*(\d+)", text)
    and_matches = re.findall(r"\band\s*=\s*(\d+)", text)
    lev_matches = re.findall(r"\blev\s*=\s*(\d+)", text)

    nodes = None
    level = None

    if nd_matches:
        nodes = int(nd_matches[-1])
    elif and_matches:
        nodes = int(and_matches[-1])

    if lev_matches:
        level = int(lev_matches[-1])

    return nodes, level


def _signed_sqrt_reward(previous_value: int, current_value: int, baseline: float) -> float:
    reward = math.sqrt(abs(previous_value - current_value) / baseline)
    return reward if previous_value > current_value else -reward


def compute_reward(
    old_nodes: int,
    new_nodes: Optional[int],
    old_level: Optional[int],
    new_level: Optional[int],
    is_success: bool,
    severe_negative_reward: float = SEVERE_NEGATIVE_REWARD,
    spike_factor: float = SPIKE_FACTOR,
    and_reward_weight: float = AND_REWARD_WEIGHT,
    lev_reward_weight: float = LEV_REWARD_WEIGHT,
) -> float:
    """
    Reward = w_and * and_gain + w_lev * lev_gain
    """
    if (not is_success) or (new_nodes is None) or (old_level is None) or (new_level is None):
        return severe_negative_reward

    # 异常突增保护：可视为坏动作，给重罚
    if old_nodes > 0 and new_nodes > int(old_nodes * spike_factor):
        return severe_negative_reward

    weight_sum = and_reward_weight + lev_reward_weight
    if weight_sum <= 0:
        raise ValueError("and_reward_weight + lev_reward_weight must be positive")

    normalized_and_weight = and_reward_weight / weight_sum
    normalized_lev_weight = lev_reward_weight / weight_sum
    and_gain = _signed_sqrt_reward(old_nodes, new_nodes, max(old_nodes, 1))
    lev_gain = _signed_sqrt_reward(old_level, new_level, max(old_level, 1))
    return (normalized_and_weight * and_gain) + (normalized_lev_weight * lev_gain)


def compute_weighted_result_cost(
    reference_nodes: int,
    candidate_nodes: int,
    reference_level: Optional[int],
    candidate_level: Optional[int],
    and_reward_weight: float = AND_REWARD_WEIGHT,
    lev_reward_weight: float = LEV_REWARD_WEIGHT,
) -> Optional[float]:
    """Lower cost is better and matches the weighted objective used by reward."""
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


# ------------------------------
# 4) 纯 MAB（UCB1）
# ------------------------------
class UCBBandit:
    """基础 UCB1 老虎机"""

    def __init__(
        self,
        action_list: List[str],
        c: float = 0.4,
        rng: Optional[random.Random] = None,
    ):
        self.actions = list(action_list)
        self.c = c
        self.rng = rng or random.Random()
        self.counts = [0 for _ in self.actions]    # 每个动作被选次数
        self.q_values = [0.0 for _ in self.actions]  # 每个动作平均奖励
        self.total_steps = 0

    def select_action(self) -> int:
        """
        UCB1 选臂逻辑：
        1) 优先选择尚未尝试的动作（冷启动）
        2) 否则最大化 q_i + c * sqrt(ln(t) / n_i)
        """
        # 冷启动：每个动作至少试一次
        untried = [i for i, cnt in enumerate(self.counts) if cnt == 0]
        if untried:
            return self.rng.choice(untried)

        t = max(1, self.total_steps)
        ucb_scores = []
        for i in range(len(self.actions)):
            bonus = self.c * math.sqrt(math.log(t) / self.counts[i])
            ucb_scores.append(self.q_values[i] + bonus)

        max_score = max(ucb_scores)
        candidates = [i for i, s in enumerate(ucb_scores) if s == max_score]
        return self.rng.choice(candidates)

    def update(self, trial_index: List[int], reward: float) -> None:
        """增量方式更新平均奖励"""

        for action_index in set(trial_index):
            n = self.counts[action_index]
            q = self.q_values[action_index]

            # Q_n = Q_{n-1} + (R - Q_{n-1}) / n
            self.q_values[action_index] = q + (reward - q) / n
            # self.q_values[action_index] = q + (reward - q) * 0.1


def snapshot_action_state(bandit: UCBBandit) -> List[Dict[str, object]]:
    t = max(1, bandit.total_steps)
    rows: List[Dict[str, object]] = []
    for action_index, action_name in enumerate(bandit.actions):
        count_before = bandit.counts[action_index]
        q_before = bandit.q_values[action_index]
        if count_before == 0:
            bonus_before = None
            score_before = None
            cold_start = True
        else:
            bonus_before = bandit.c * math.sqrt(math.log(t) / count_before)
            score_before = q_before + bonus_before
            cold_start = False
        rows.append(
            {
                "action_index": action_index,
                "action": action_name,
                "count_before": count_before,
                "q_before": q_before,
                "bonus_before": bonus_before,
                "score_before": score_before,
                "cold_start": cold_start,
            }
        )
    return rows


# ------------------------------
# 5) 主循环与评估流水线
# ------------------------------
def evaluate_sequence(
    blif_path: str,
    seq: List[str],
    abc_bin: Optional[str] = None,
) -> Tuple[bool, Optional[int], Optional[int], str, float | None]:
    """执行命令序列并返回 (success, nodes, level, raw_output, peak_memory_kb)"""
    success, stdout, stderr, peak_memory_kb = run_abc_command(
        blif_path,
        seq,
        abc_bin=abc_bin,
    )

    # ABC 的统计通常在 stdout；少数情况下可能输出到 stderr，统一拼接解析
    combined = "\n".join([stdout, stderr]).strip()
    nodes, level = parse_abc_stats(combined)
    return success, nodes, level, combined, peak_memory_kb


def find_blif_files(benchmark_dir: str) -> List[str]:
    """递归收集目录下的 .blif 文件。"""
    if not os.path.isdir(benchmark_dir):
        return []

    files = []
    for root, _, names in os.walk(benchmark_dir):
        for name in names:
            if name.lower().endswith(".blif"):
                files.append(os.path.join(root, name))

    files.sort(key=natural_sort_key)
    return files


def derive_benchmark_name(blif_path: str) -> str:
    abs_path = os.path.abspath(blif_path)
    benchmark_root = os.path.abspath(BENCHMARK_DIR)
    try:
        rel_path = os.path.relpath(abs_path, benchmark_root)
    except ValueError:
        return os.path.basename(abs_path)
    if not rel_path.startswith(".."):
        return rel_path
    return os.path.basename(abs_path)


def optimize_one_benchmark(
    blif_path: str,
    benchmark_name: Optional[str] = None,
    bandit_c: float = 2.0,
    abc_bin: Optional[str] = None,
    debug_search: bool = False,
) -> Dict:
    """
    对单个电路做 MAB 优化。

    返回一个可序列化 dict，包括：
    - 初始 nd/lev
    - 最优 nd/lev 及 recipe
    - 每轮最优曲线（best-so-far）
    """
    if benchmark_name is None:
        benchmark_name = derive_benchmark_name(blif_path)
    logging.info("开始处理: %s", benchmark_name)
    started = time.perf_counter()
    peak_memory_kb: float | None = None

    # 先拿初始（仅 read_blif + strash）指标
    init_success, init_nodes, init_level, raw, init_peak_memory_kb = evaluate_sequence(
        blif_path,
        [],
        abc_bin=abc_bin,
    )
    if init_peak_memory_kb is not None:
        peak_memory_kb = init_peak_memory_kb
    if (not init_success) or (init_nodes is None):
        logging.error("初始评估失败: %s", benchmark_name)
        logging.debug("初始输出: %s", raw)
        return {
            "benchmark": benchmark_name,
            "status": "failed_init",
            "error": "failed to get initial stats",
            "runtime_sec": time.perf_counter() - started,
            "peak_memory_kb": peak_memory_kb,
        }

    bandit = UCBBandit(
        actions,
        c=bandit_c,
        rng=random.Random(RANDOM_SEED),
    )

    best_nodes = init_nodes
    best_level = init_level
    best_recipe: List[str] = []

    episode_best_trace = []
    debug_trace: List[Dict[str, object]] = []

    for ep in range(1, N_EPISODES + 1):
        current_nodes = init_nodes
        current_level = init_level
        current_seq: List[str] = []
        current_idx: List[int] = []

        for step in range(1, K_STEPS + 1):
            action_state_rows = snapshot_action_state(bandit) if debug_search else []
            action_idx = bandit.select_action()
            action_cmd = actions[action_idx]

            current_seq += [action_cmd]
            current_idx += [action_idx]      # 序列里动作对应的 index
            bandit.total_steps += 1
            bandit.counts[action_idx] += 1
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
                            "count_after_select": bandit.counts[int(action_state["action_index"])],
                            "sequence_prefix": sequence_prefix,
                        }
                    )

        success, new_nodes, new_level, output, eval_peak_memory_kb = evaluate_sequence(
            blif_path,
            current_seq,
            abc_bin=abc_bin,
        )
        if eval_peak_memory_kb is not None and (peak_memory_kb is None or eval_peak_memory_kb > peak_memory_kb):
            peak_memory_kb = eval_peak_memory_kb

        reward = compute_reward(
            current_nodes,
            new_nodes,
            current_level,
            new_level,
            success,
            and_reward_weight=AND_REWARD_WEIGHT,
            lev_reward_weight=LEV_REWARD_WEIGHT,
        )
        bandit.update(current_idx, reward)
        if debug_search:
            episode_sequence = "; ".join(current_seq)
            for row in debug_trace[-(len(actions) * K_STEPS):]:
                if int(row["episode"]) != ep:
                    continue
                action_index = int(row["action_index"])
                row["count_after_update"] = bandit.counts[action_index]
                row["q_after_update"] = bandit.q_values[action_index]
                row["episode_reward"] = reward
                row["episode_success"] = int(success)
                row["episode_nodes"] = new_nodes
                row["episode_level"] = new_level
                row["episode_peak_memory_kb"] = eval_peak_memory_kb
                row["episode_sequence"] = episode_sequence

        # 若命令失败或统计异常，不推进序列状态（但该动作仍受惩罚）
        if (not success) or (new_nodes is None):
            logging.warning(
                "[%s][ep=%d step=%d] 动作失败: %s, reward=%.1f",
                benchmark_name,
                ep,
                step,
                action_cmd,
                reward,
            )
            logging.debug("ABC输出(失败): %s", output)
            continue

        # 异常突增时，也不给状态推进，避免坏状态污染后续
        if current_nodes > 0 and new_nodes > int(current_nodes * SPIKE_FACTOR):
            logging.warning(
                "[%s][ep=%d step=%d] 节点异常突增 old=%d new=%d, action=%s, reward=%.1f",
                benchmark_name,
                ep,
                step,
                current_nodes,
                new_nodes,
                action_cmd,
                reward,
            )
            continue

        # 正常推进当前序列状态
        current_nodes = new_nodes
        current_level = new_level

        # logging.info(bandit.q_values)

        # 更新全局最优
        if is_better_result(
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

        # logging.info(current_seq)

        episode_best_trace.append(best_nodes)
        if debug_search:
            for row in debug_trace[-(len(actions) * K_STEPS):]:
                if int(row["episode"]) != ep:
                    continue
                row["best_nodes_after_episode"] = best_nodes
                row["best_level_after_episode"] = best_level
        logging.info(
            "[%s] Episode %d/%d 完成, 当前全局最优 and=%d lev=%d",
            benchmark_name,
            ep,
            N_EPISODES,
            best_nodes,
            best_level
        )

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
            # "recipe": best_recipe,
            "recipe_str": "; ".join(best_recipe),
        },
        "improvement": {
            "nodes_reduced": improvement,
            "ratio": improvement_ratio,
        },
        "config": {
            "episodes": N_EPISODES,
            "steps_per_episode": K_STEPS,
            # "abc_timeout_sec": ABC_TIMEOUT_SEC,
            # "spike_factor": SPIKE_FACTOR,
            # "severe_negative_reward": SEVERE_NEGATIVE_REWARD,
            "ucb_c": bandit_c,
            "and_reward_weight": AND_REWARD_WEIGHT,
            "lev_reward_weight": LEV_REWARD_WEIGHT,
            "debug_search": debug_search,
            "seed": RANDOM_SEED,
        },
        "runtime_sec": time.perf_counter() - started,
        "peak_memory_kb": peak_memory_kb,
        "debug_trace": debug_trace if debug_search else [],
        "bandit": {
            # "actions": actions,
            # "counts": bandit.counts,
            # "q_values": bandit.q_values,
            # "total_steps": bandit.total_steps,
        },
    }


def build_summary(results: List[Dict]) -> Dict:
    """汇总批量结果，便于快速比较。"""
    total = len(results)
    ok_items = [r for r in results if r.get("status") == "ok"]
    failed = total - len(ok_items)

    initial_sum = sum(r["initial"]["nodes"] for r in ok_items)
    best_sum = sum(r["best"]["nodes"] for r in ok_items)
    reduced_sum = sum(r["improvement"]["nodes_reduced"] for r in ok_items)

    avg_ratio = (
        sum(r["improvement"]["ratio"] for r in ok_items) / len(ok_items)
        if ok_items
        else 0.0
    )

    global_ratio = (reduced_sum / initial_sum) if initial_sum > 0 else 0.0

    return {
        "total_cases": total,
        "success_cases": len(ok_items),
        "failed_cases": failed,
        "sum_initial_nodes": initial_sum,
        "sum_best_nodes": best_sum,
        "sum_reduced_nodes": reduced_sum,
        "avg_improvement_ratio": avg_ratio,
        "global_improvement_ratio": global_ratio,
    }


def write_result_artifacts(
    result: Dict,
    method_name: str = "baseline_mab",
    aggregate_payload: Optional[Dict] = None,
) -> str:
    result_path = result_utils.write_benchmark_result(method_name, result)
    _write_debug_trace_csv(method_name, result)
    _write_debug_trace_json(method_name, result)
    if aggregate_payload is not None:
        result_utils.write_method_summary(method_name, aggregate_payload)
    return result_path


def _write_debug_trace_csv(method_name: str, result: Dict) -> str | None:
    debug_rows = result.get("debug_trace")
    if not isinstance(debug_rows, list) or not debug_rows:
        return None
    benchmark_name = str(result.get("benchmark", "unknown"))
    output_path = _debug_trace_csv_path(method_name, benchmark_name)
    fieldnames = list(debug_rows[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in debug_rows:
            writer.writerow(row)
    return output_path


def _write_debug_trace_json(method_name: str, result: Dict) -> str | None:
    debug_rows = result.get("debug_trace")
    if not isinstance(debug_rows, list) or not debug_rows:
        return None
    benchmark_name = str(result.get("benchmark", "unknown"))
    output_path = _debug_trace_json_path(method_name, benchmark_name)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_build_debug_episode_payload(result, debug_rows), handle, indent=2, ensure_ascii=False)
    return output_path


def _debug_trace_csv_path(method_name: str, benchmark_name: str) -> str:
    result_dir = result_utils.get_method_results_dir(method_name)
    filename = result_utils.benchmark_to_result_filename(benchmark_name).removesuffix(".json") + ".debug.csv"
    return os.path.join(result_dir, filename)


def _debug_trace_json_path(method_name: str, benchmark_name: str) -> str:
    result_dir = result_utils.get_method_results_dir(method_name)
    filename = result_utils.benchmark_to_result_filename(benchmark_name).removesuffix(".json") + ".debug.json"
    return os.path.join(result_dir, filename)


def _build_debug_episode_payload(result: Dict, debug_rows: List[Dict[str, object]]) -> Dict[str, object]:
    config = result.get("config", {})
    if not isinstance(config, dict):
        config = {}

    episodes: List[Dict[str, object]] = []
    rows_by_episode: Dict[int, List[Dict[str, object]]] = {}
    for row in debug_rows:
        episode = int(row["episode"])
        rows_by_episode.setdefault(episode, []).append(row)

    for episode in sorted(rows_by_episode):
        episode_rows = rows_by_episode[episode]
        rows_by_step: Dict[int, List[Dict[str, object]]] = {}
        for row in episode_rows:
            step = int(row["step"])
            rows_by_step.setdefault(step, []).append(row)

        first_row = episode_rows[0]
        steps_payload: List[Dict[str, object]] = []
        for step in sorted(rows_by_step):
            step_rows = rows_by_step[step]
            selected_row = next((row for row in step_rows if int(row["selected"]) == 1), step_rows[0])
            actions_payload = [
                {
                    "action": row["action"],
                    "selected": bool(int(row["selected"])),
                    "count_before": row["count_before"],
                    "q_before": row["q_before"],
                    "bonus_before": row["bonus_before"],
                    "score_before": row["score_before"],
                    "count_after_update": row.get("count_after_update"),
                    "q_after_update": row.get("q_after_update"),
                }
                for row in sorted(step_rows, key=lambda item: int(item["action_index"]))
            ]
            steps_payload.append(
                {
                    "step": step,
                    "selected_action": selected_row["action"],
                    "actions": actions_payload,
                }
            )

        episodes.append(
            {
                "episode": episode,
                "reward": first_row.get("episode_reward"),
                "success": bool(int(first_row.get("episode_success", 0))),
                "nodes": first_row.get("episode_nodes"),
                "level": first_row.get("episode_level"),
                "best_nodes": first_row.get("best_nodes_after_episode"),
                "best_level": first_row.get("best_level_after_episode"),
                "sequence": first_row.get("episode_sequence"),
                "steps": steps_payload,
            }
        )

    return {
        "benchmark": result.get("benchmark"),
        "config": {
            "episodes": config.get("episodes"),
            "steps_per_episode": config.get("steps_per_episode"),
            "ucb_c": config.get("ucb_c"),
            "and_reward_weight": config.get("and_reward_weight"),
            "lev_reward_weight": config.get("lev_reward_weight"),
            "seed": config.get("seed"),
        },
        "episodes": episodes,
    }


def setup_logging() -> None:
    """统一日志配置。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


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
    parser = argparse.ArgumentParser(prog="baseline_mab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-search", help="Run baseline MAB search.")
    run.add_argument("--workdir", type=Path, default=Path(result_utils.DEFAULT_WORKDIR_NAME))
    run.add_argument("--abc-bin", default=ABC_BIN)
    run.add_argument("--dataset-root", type=Path, default=Path("tc_public"))
    run.add_argument("--design", help="Single .blif filename under dataset-root. Default: run all.")
    run.add_argument("--steps", type=int, default=K_STEPS)
    run.add_argument("--episodes", type=int, default=N_EPISODES)
    run.add_argument("--timeout", type=int, default=ABC_TIMEOUT_SEC)
    run.add_argument("--spike-factor", type=float, default=SPIKE_FACTOR)
    run.add_argument("--severe-negative-reward", type=float, default=SEVERE_NEGATIVE_REWARD)
    run.add_argument("--and-weight", type=float, default=AND_REWARD_WEIGHT)
    run.add_argument("--lev-weight", type=float, default=LEV_REWARD_WEIGHT)
    run.add_argument("--ucb-c", type=float, default=2.0)
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
        result_path = write_result_artifacts(result)
        logging.info("完成: %s -> %s", design_name, os.path.abspath(result_path))

    all_results["summary"] = build_summary(all_results["results"])
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
    return 0


def _command_summarize(args: argparse.Namespace) -> int:
    args.workdir = _resolve_local_path(args.workdir)
    summarize_results.main(["baseline_mab", "--workdir", str(args.workdir)])
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
                result_utils.get_method_results_dir("baseline_mab")
            ) / result_utils.benchmark_to_result_filename(design_name)
            design_argv = upsert_option(base_argv, "--design", design_name)
            exit_code = run_under_external_monitor(
                raw_argv=design_argv,
                patch_json=patch_json,
                module_name="MABSyn.baseline_mab",
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
