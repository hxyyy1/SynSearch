#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
纯 MAB（UCB1）逻辑综合 Baseline
- 通过 subprocess 调用 ABC
- 对 .blif 电路进行固定长度命令序列探索
- 优化目标：最小化 AIG 节点数（nd / and）
"""

import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from memory_utils import run_command_with_peak_memory

from . import result_utils

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

# 随机种子（用于并列时随机打破平局）
RANDOM_SEED = 2


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
    解析 --key value 形式的可选参数。
    """
    kv: Dict[str, str] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            raise ValueError(f"无效参数: {token}（可选参数需为 --key value）")
        if i + 1 >= len(args):
            raise ValueError(f"参数 {token} 缺少值")
        kv[token[2:]] = args[i + 1]
        i += 2
    return kv


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
        completed, peak_memory_kb = run_command_with_peak_memory(
            [abc_bin, "-c", abc_cmd],
            text=True,
            timeout=timeout_sec,
            check=False,
        )
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


def compute_reward(
    old_nodes: int,
    new_nodes: Optional[int],
    is_success: bool,
    severe_negative_reward: float = SEVERE_NEGATIVE_REWARD,
    spike_factor: float = SPIKE_FACTOR,
) -> float:
    """
    - Reward = N_old - N_new
    - 节点数下降 => 正奖励
    - 节点数上升 => 负奖励
    - 失败/解析失败/异常突增 => 严重负奖励
    """
    if (not is_success) or (new_nodes is None):
        return severe_negative_reward

    # 异常突增保护：可视为坏动作，给重罚
    if old_nodes > 0 and new_nodes > int(old_nodes * spike_factor):
        return severe_negative_reward

    return float((old_nodes - new_nodes)/old_nodes)


# ------------------------------
# 4) 纯 MAB（UCB1）
# ------------------------------
class UCBBandit:
    """基础 UCB1 老虎机"""

    def __init__(
        self,
        action_list: List[str],
        c: float = 2.0,
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


# ------------------------------
# 5) 主循环与评估流水线
# ------------------------------
def evaluate_sequence(
    blif_path: str,
    seq: List[str],
    abc_bin: Optional[str] = None,
) -> Tuple[bool, Optional[int], Optional[int], str, float | None]:
    """执行命令序列并返回 (success, nodes, level, raw_output, peak_memory_kb)"""
    success, stdout, stderr, peak_memory_kb = run_abc_command(blif_path, seq, abc_bin=abc_bin)

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

    for ep in range(1, N_EPISODES + 1):
        current_nodes = init_nodes
        current_level = init_level
        current_seq: List[str] = []
        current_idx: List[int] = []

        for step in range(1, K_STEPS + 1):
            action_idx = bandit.select_action()
            action_cmd = actions[action_idx]

            current_seq += [action_cmd]
            current_idx += [action_idx]      # 序列里动作对应的 index
            bandit.total_steps += 1
            bandit.counts[action_idx] += 1

        success, new_nodes, new_level, output, eval_peak_memory_kb = evaluate_sequence(
            blif_path,
            current_seq,
            abc_bin=abc_bin,
        )
        if eval_peak_memory_kb is not None and (peak_memory_kb is None or eval_peak_memory_kb > peak_memory_kb):
            peak_memory_kb = eval_peak_memory_kb

        reward = compute_reward(current_nodes, new_nodes, success)
        bandit.update(current_idx, reward)

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
        if current_nodes < best_nodes:
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
            "seed": RANDOM_SEED,
        },
        "runtime_sec": time.perf_counter() - started,
        "peak_memory_kb": peak_memory_kb,
        # "episode_best_trace": episode_best_trace,
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
    if aggregate_payload is not None:
        result_utils.write_method_summary(method_name, aggregate_payload)
    return result_path


def setup_logging() -> None:
    """统一日志配置。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    global K_STEPS, N_EPISODES, ABC_TIMEOUT_SEC, SPIKE_FACTOR
    global SEVERE_NEGATIVE_REWARD, RANDOM_SEED

    # 单电路模式：python baseline_mab.py <design> [--key value...]
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("--"):
        design = sys.argv[1]
        kv = parse_optional_kv_args(sys.argv[2:])

        abc_bin = kv.get("abc-bin", ABC_BIN)
        workdir = kv.get("workdir", result_utils.DEFAULT_WORKDIR_NAME)
        result_utils.set_workdir(workdir)
        result_json = result_utils.optional_output_path(
            kv,
            "result-json",
        )
        ucb_c = float(kv.get("ucb-c", "2.0"))
        log_level = kv.get("log-level", "INFO").upper()

        K_STEPS = int(kv.get("steps", str(K_STEPS)))
        N_EPISODES = int(kv.get("episodes", str(N_EPISODES)))
        ABC_TIMEOUT_SEC = int(kv.get("timeout", str(ABC_TIMEOUT_SEC)))
        SPIKE_FACTOR = float(kv.get("spike-factor", str(SPIKE_FACTOR)))
        SEVERE_NEGATIVE_REWARD = float(
            kv.get("severe-negative-reward", str(SEVERE_NEGATIVE_REWARD))
        )
        RANDOM_SEED = int(kv.get("seed", str(RANDOM_SEED)))

        if "actions" in kv:
            parsed_actions = [x.strip() for x in kv["actions"].split(",") if x.strip()]
            if parsed_actions:
                actions[:] = parsed_actions

        setup_logging()
        logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

        if not os.path.isfile(design):
            logging.error("design 文件不存在: %s", design)
            return

        logging.info("单电路模式: design=%s", os.path.abspath(design))
        logging.info(
            "参数: steps=%d episodes=%d timeout=%d ucb_c=%.3f",
            K_STEPS,
            N_EPISODES,
            ABC_TIMEOUT_SEC,
            ucb_c,
        )

        result = optimize_one_benchmark(design, bandit_c=ucb_c, abc_bin=abc_bin)
        result["meta"] = {
            "mode": "single_design",
            "abc_bin": abc_bin,
        }

        result_path = write_result_artifacts(result)
        if result_json:
            with open(result_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        logging.info("完成，结果已写入: %s", os.path.abspath(result_path))
        return

    setup_logging()
    logging.error("批量模式已拆分，请使用 baseline_mab_batch.py")


if __name__ == "__main__":
    main()
