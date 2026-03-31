#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 Contextual Bandit（LinUCB）的逻辑综合 baseline。

相较于原始全局 UCB1 版本，本脚本做了这些关键改动：
1) 为每个 step index 单独维护一个 LinUCB policy。
2) context 来自当前电路状态，包含 nodes / levels / fanout 统计。
3) reward 改为逐步贪心下的 per-arm 短期 reward，并在每个 step 内固定。
4) 每个 step 独立训练一个 LinUCB bandit，迭代若干次后固定该步动作。
"""

import json
import argparse
import logging
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import hashlib
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from memory_utils import read_linux_peak_memory_kb
from . import result_utils, summarize_results


# ------------------------------
# 1) 动作空间（保持与原脚本一致）
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
RESULT_JSON = "linucb_results.json"

N_EPISODES = 150
K_STEPS = 10
ABC_TIMEOUT_SEC = 100

SPIKE_FACTOR = 1.8
SEVERE_NEGATIVE_REWARD = -1.0
TIMEOUT_NEGATIVE_REWARD = -1.5
LEVEL_REWARD_WEIGHT = 0

ALPHA = 1.5
REG_LAMBDA = 1.0

RANDOM_SEED = 0
EARLY_STOP_SAME_ARM_STREAK = 3
LONG_TERM_ROLLOUTS = 1              # 条数
LONG_TERM_HORIZON = 3               # 长度
RETURN_BACK_THRESHOLD = 0.02


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


def parse_runtime_args(argv: Sequence[str]) -> Tuple[Optional[str], Dict[str, str]]:
    """
    兼容原脚本：
    - 有位置参数时为单电路模式
    - 否则为批量模式
    两种模式都允许继续传 --key value 形式的参数。
    """
    if len(argv) >= 2 and not argv[1].startswith("--"):
        return argv[1], parse_optional_kv_args(list(argv[2:]))
    return None, parse_optional_kv_args(list(argv[1:]))


# ------------------------------
# 2) ABC 交互与统计解析
# ------------------------------
class ABCSessionError(RuntimeError):
    pass
class ABCSessionTimeout(ABCSessionError):
    pass

@dataclass
class CircuitStats:
    nodes: int
    level: int
    ave_fanout: float
    max_fanout: float

class ABCSession:
    """
    在单个 episode 内维护一个连续的 ABC 交互 session。
    说明：
    - 初始状态：read_blif + strash + stats
    - 每一步：执行 action 后立即 print_stats + pfan
    - 通过 echo marker 截取每个 step 对应输出块
    """

    def __init__(self, design_path: str, abc_bin: str, timeout_sec: int):
        self.design_path = design_path
        self.abc_bin = abc_bin
        self.timeout_sec = timeout_sec
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.block_index = 0
        self.stdout_queue: "queue.Queue[object]" = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self.reader_error: Optional[BaseException] = None
        self.recv_buffer = bytearray()
        self._queue_eof = object()
        self.peak_memory_kb: float | None = None

    def __enter__(self) -> "ABCSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            return

        self.process = subprocess.Popen(
            [self.abc_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        self.stdout_queue = queue.Queue()
        self.reader_error = None
        self.recv_buffer = bytearray()
        self.reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"abc-reader-{os.path.basename(self.design_path)}",
            daemon=True,
        )
        self.reader_thread.start()
        self._update_peak_memory_kb()

    def close(self) -> None:
        if self.process is None:
            return
        self._update_peak_memory_kb()

        try:
            if self.process.poll() is None and self.process.stdin is not None:
                self.process.stdin.write(b"quit;\n")
                self.process.stdin.flush()
        except Exception:
            pass

        try:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=1.0)
        except Exception:
            try:
                if self.process.poll() is None:
                    self.process.kill()
            except Exception:
                pass

        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1.0)

        self.process = None
        self.reader_thread = None

    def initialize(self) -> Tuple[CircuitStats, str]:
        read_cmd = self._build_read_command()
        block = self._run_block(
            [
                read_cmd,
                "strash",
                "print_stats",
                "pfan",
            ],
            label="init",
        )
        return parse_stats_snapshot(block), block

    def apply_action(self, action_cmd: str, step_idx: int) -> Tuple[CircuitStats, str]:
        block = self._run_block(
            [
                action_cmd,
                "print_stats",
                "pfan",
            ],
            label=f"step_{step_idx}",
        )
        return parse_stats_snapshot(block), block

    def write_current_design(self, output_path: str, label: str) -> None:
        write_cmd = self._build_write_command(output_path)
        self._run_block(
            [write_cmd],
            label=label,
        )

    def _build_read_command(self) -> str:
        lowered = self.design_path.lower()
        if lowered.endswith(".aig") or lowered.endswith(".aiger"):
            return f"read_aiger {self.design_path}"
        if lowered.endswith(".blif"):
            return f"read_blif {self.design_path}"
        raise ABCSessionError(f"不支持的设计文件类型: {self.design_path}")

    def _build_write_command(self, output_path: str) -> str:
        lowered = output_path.lower()
        if lowered.endswith(".aig") or lowered.endswith(".aiger"):
            return f"write_aiger {output_path}"
        if lowered.endswith(".blif"):
            return f"write_blif {output_path}"
        raise ABCSessionError(f"不支持的输出文件类型: {output_path}")

    def _run_block(self, commands: Sequence[str], label: str) -> str:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise ABCSessionError("ABC session 未启动")

        self.block_index += 1
        token = f"__MABSYN_{label}_{self.block_index}_{uuid.uuid4().hex[:8]}__"
        begin_marker = f"{token}_BEGIN"
        end_marker = f"{token}_END"
        begin_marker_bytes = begin_marker.encode("utf-8")
        end_marker_bytes = end_marker.encode("utf-8")

        script_lines = [f"echo {begin_marker}", *commands, f"echo {end_marker}"]
        try:
            payload = (";\n".join(script_lines) + ";\n").encode("utf-8")
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            raise ABCSessionError("ABC session 已异常退出") from exc

        seen_begin = False
        deadline = time.monotonic() + self.timeout_sec
        captured = bytearray()

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                partial_output = captured.decode("utf-8", errors="replace")
                if self.recv_buffer:
                    partial_output += self.recv_buffer.decode("utf-8", errors="replace")
                self.close()
                raise ABCSessionTimeout(
                    f"ABC step 超时（>{self.timeout_sec}s），label={label}, "
                    f"partial_output={partial_output[-2000:]}"
                )

            if not seen_begin:
                begin_idx = self.recv_buffer.find(begin_marker_bytes)
                if begin_idx != -1:
                    line_end_idx = self.recv_buffer.find(b"\n", begin_idx)
                    if line_end_idx != -1:
                        del self.recv_buffer[: line_end_idx + 1]
                        seen_begin = True
                        continue
                elif len(self.recv_buffer) > len(begin_marker_bytes):
                    del self.recv_buffer[: len(self.recv_buffer) - len(begin_marker_bytes)]
            else:
                end_idx = self.recv_buffer.find(end_marker_bytes)
                if end_idx != -1:
                    captured.extend(self.recv_buffer[:end_idx])
                    line_end_idx = self.recv_buffer.find(b"\n", end_idx)
                    if line_end_idx != -1:
                        del self.recv_buffer[: line_end_idx + 1]
                    else:
                        del self.recv_buffer[:]
                    self._update_peak_memory_kb()
                    return captured.decode("utf-8", errors="replace")

                safe_cut = max(0, len(self.recv_buffer) - len(end_marker_bytes))
                if safe_cut > 0:
                    captured.extend(self.recv_buffer[:safe_cut])
                    del self.recv_buffer[:safe_cut]

            chunk = self._get_next_chunk(remaining)
            if chunk is None:
                return_code = self.process.poll() if self.process is not None else None
                partial_output = captured.decode("utf-8", errors="replace")
                if self.recv_buffer:
                    partial_output += self.recv_buffer.decode("utf-8", errors="replace")
                raise ABCSessionError(
                    "ABC session 在读取输出时结束，"
                    f"returncode={return_code}, partial_output={partial_output[-2000:]}"
                )
            self.recv_buffer.extend(chunk)

    def _update_peak_memory_kb(self) -> None:
        if self.process is None:
            return
        current_peak = read_linux_peak_memory_kb(self.process.pid)
        if current_peak is None:
            return
        if self.peak_memory_kb is None or current_peak > self.peak_memory_kb:
            self.peak_memory_kb = current_peak

    def _reader_loop(self) -> None:
        try:
            if self.process is None or self.process.stdout is None:
                return

            fd = self.process.stdout.fileno()
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                self.stdout_queue.put(chunk)
        except BaseException as exc:
            self.reader_error = exc
        finally:
            self.stdout_queue.put(self._queue_eof)

    def _get_next_chunk(self, timeout_sec: float) -> Optional[bytes]:
        try:
            item = self.stdout_queue.get(timeout=max(timeout_sec, 1e-3))
        except queue.Empty:
            return b""

        if item is self._queue_eof:
            if self.reader_error is not None:
                raise ABCSessionError(f"ABC reader 线程异常: {self.reader_error}")
            return None

        return item if isinstance(item, bytes) else None


def parse_abc_stats(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    从 ABC 输出中解析节点数与层级。
    """
    if not text:
        return None, None

    lowered = text.lower()
    nd_matches = re.findall(r"\bnd\s*=\s*(\d+)", lowered)
    and_matches = re.findall(r"\band\s*=\s*(\d+)", lowered)
    lev_matches = re.findall(r"\blev\s*=\s*(\d+)", lowered)

    nodes = None
    level = None
    if nd_matches:
        nodes = int(nd_matches[-1])
    elif and_matches:
        nodes = int(and_matches[-1])

    if lev_matches:
        level = int(lev_matches[-1])

    return nodes, level


def parse_fanout_stats(text: str) -> Tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None

    lowered = text.lower()

    m = re.search(r"fanouts?\s*:\s*(.*)", lowered)
    if not m:
        return None, None
    fanouts_part = m.group(1)

    max_fanout = None
    avg_fanout = None

    mmax = re.search(r"max\s*=\s*([0-9]+(?:\.[0-9]+)?)", fanouts_part)
    if mmax:
        max_fanout = float(mmax.group(1))

    mavg = re.search(r"(?:ave|avg|average)\s*=\s*([0-9]+(?:\.[0-9]+)?)", fanouts_part)
    if mavg:
        avg_fanout = float(mavg.group(1))

    return avg_fanout, max_fanout


def parse_stats_snapshot(output: str) -> CircuitStats:
    """
    将单次 block 输出解析为电路状态。
    """
    nodes, level = parse_abc_stats(output)
    avg_fanout, max_fanout = parse_fanout_stats(output)

    if nodes is None:
        raise ABCSessionError(f"无法从 ABC 输出解析 nodes:\n{output}")

    return CircuitStats(
        nodes=nodes,
        level=level if level is not None else 0,
        ave_fanout=avg_fanout if avg_fanout is not None else 0.0,
        max_fanout=max_fanout if max_fanout is not None else 0.0,
    )


@dataclass
class PrefixCacheEntry:
    prefix: Tuple[str, ...]
    stats: CircuitStats
    snapshot_path: str


class PrefixCache:
    """
    模仿 ABCBackend 的前缀缓存：
    - key 使用 (design_path, prefix) 的哈希
    - 每个 prefix 对应一个持久化的 AIG 快照
    - 构建较长前缀时递归复用父前缀快照
    """

    def __init__(self, root_design_path: str, abc_bin: str, timeout_sec: int):
        self.root_design_path = os.path.abspath(root_design_path)
        self.abc_bin = abc_bin
        self.timeout_sec = timeout_sec
        cache_root = result_utils.get_cache_root("linucb")
        os.makedirs(cache_root, exist_ok=True)
        self.cache_dir = os.path.join(cache_root, uuid.uuid4().hex)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.entries: Dict[Tuple[str, ...], PrefixCacheEntry] = {}
        self.peak_memory_kb: float | None = None

    def close(self) -> None:
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def get_or_build(self, prefix: Tuple[str, ...]) -> PrefixCacheEntry:
        cached = self.entries.get(prefix)
        if cached is not None and os.path.exists(cached.snapshot_path):
            return cached

        snapshot_path = os.path.join(self.cache_dir, f"{self._cache_key(prefix)}.aig")
        if prefix:
            parent = self.get_or_build(prefix[:-1])
            with ABCSession(parent.snapshot_path, abc_bin=self.abc_bin, timeout_sec=self.timeout_sec) as session:
                _, _ = session.initialize()
                current_stats, _ = session.apply_action(prefix[-1], len(prefix))
                session.write_current_design(snapshot_path, f"prefix_{len(prefix)}")
                self._observe_peak_memory_kb(session.peak_memory_kb)
        else:
            with ABCSession(self.root_design_path, abc_bin=self.abc_bin, timeout_sec=self.timeout_sec) as session:
                current_stats, _ = session.initialize()
                session.write_current_design(snapshot_path, "prefix_root")
                self._observe_peak_memory_kb(session.peak_memory_kb)

        entry = PrefixCacheEntry(
            prefix=prefix,
            stats=CircuitStats(
                nodes=current_stats.nodes,
                level=current_stats.level,
                ave_fanout=current_stats.ave_fanout,
                max_fanout=current_stats.max_fanout,
            ),
            snapshot_path=snapshot_path,
        )
        self.entries[prefix] = entry
        return entry

    def _observe_peak_memory_kb(self, peak_memory_kb: float | None) -> None:
        if peak_memory_kb is None:
            return
        if self.peak_memory_kb is None or peak_memory_kb > self.peak_memory_kb:
            self.peak_memory_kb = peak_memory_kb

    def trace_for_actions(
        self,
        base_prefix: Tuple[str, ...],
        extra_actions: Sequence[str],
    ) -> List[PrefixCacheEntry]:
        entries: List[PrefixCacheEntry] = []
        prefix = base_prefix
        for action_cmd in extra_actions:
            prefix = prefix + (action_cmd,)
            entries.append(self.get_or_build(prefix))
        return entries

    def _cache_key(self, prefix: Tuple[str, ...]) -> str:
        payload = json.dumps(
            {
                "design": self.root_design_path,
                "prefix": list(prefix),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------
# 3) LinUCB 与 context
# ------------------------------
class LinUCBPolicy:
    """
    标准 LinUCB：
    - A[a] = lambda * I
    - b[a] = 0
    - p[a] = theta_a^T x + alpha * sqrt(x^T A_a^-1 x)
    - update: A[a] += x x^T, b[a] += r x
    """

    def __init__(
        self,
        num_actions: int,
        context_dim: int,
        alpha: float = ALPHA,
        reg_lambda: float = REG_LAMBDA,
        seed: int = RANDOM_SEED,
    ):
        if reg_lambda <= 0:
            raise ValueError("LinUCB 的 lambda 必须大于 0")

        self.num_actions = num_actions
        self.context_dim = context_dim
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.rng = np.random.default_rng(seed)

        self.A = np.array(
            [reg_lambda * np.eye(context_dim, dtype=np.float64) for _ in range(num_actions)],
            dtype=np.float64,
        )
        self.b = np.zeros((num_actions, context_dim), dtype=np.float64)
        self.counts = np.zeros(num_actions, dtype=np.int64)

    def select_action(
        self,
        arm_contexts: Sequence[np.ndarray],
        allowed_action_indices: Optional[Set[int]] = None,
    ) -> int:
        scores = np.full(self.num_actions, -np.inf, dtype=np.float64)
        for action_idx, context in enumerate(arm_contexts):
            if allowed_action_indices is not None and action_idx not in allowed_action_indices:
                continue
            a_inv = np.linalg.inv(self.A[action_idx])
            theta = a_inv @ self.b[action_idx]              # d × 1
            mean = float(theta @ context)
            variance = float(context @ a_inv @ context)
            variance = max(variance, 0.0)
            scores[action_idx] = mean + self.alpha * np.sqrt(variance)

        best_score = float(np.max(scores))
        if np.isneginf(best_score):
            raise ValueError("当前 step 没有可选动作")
        candidates = np.flatnonzero(np.isclose(scores, best_score))
        return int(self.rng.choice(candidates))

    def update(self, action_idx: int, context: np.ndarray, reward: float) -> None:
        self.A[action_idx] += np.outer(context, context)
        self.b[action_idx] += reward * context              # d × 1
        self.counts[action_idx] += 1

class ContextEncoder:
    """
    可解释 context:
    - bias
    - nodes / nodes0
    - level / level0
    - avg fanout
    - max fanout
    """

    def __init__(self, initial_stats: CircuitStats):
        self.nodes0 = max(initial_stats.nodes, 1)
        self.level0 = max(initial_stats.level, 1)
        self.avg_fanout0 = max(initial_stats.ave_fanout, 1.0)
        self.max_fanout0 = max(initial_stats.max_fanout, 1.0)

    @property
    def short_dimension(self) -> int:
        return 4

    @property
    def long_dimension(self) -> int:
        return 6

    @property
    def dimension(self) -> int:
        return self.short_dimension + self.long_dimension

    def encode_short(self, stats: CircuitStats) -> np.ndarray:
        nodes_ratio = float(stats.nodes / self.nodes0)
        level_ratio = float(stats.level / self.level0)
        avg_fanout_ratio = float(stats.ave_fanout / self.avg_fanout0)
        return np.array(
            [
                1.0,
                nodes_ratio,
                level_ratio,
                avg_fanout_ratio,
            ],
            dtype=np.float64,
        )

    def encode_long(
        self,
        base_stats: CircuitStats,
        rollout_stats_list: Sequence[CircuitStats],
    ) -> np.ndarray:
        if not rollout_stats_list:
            return np.zeros(self.long_dimension, dtype=np.float64)

        base_nodes = max(base_stats.nodes, 1)
        base_level = max(base_stats.level, 1)

        node_gains = [
            float(base_nodes - stats.nodes) / base_nodes for stats in rollout_stats_list
        ]
        level_gains = [
            float(base_level - stats.level) / base_level for stats in rollout_stats_list
        ]
        return np.array(
            [
                float(np.mean(node_gains)),
                float(np.max(node_gains)),
                float(np.std(node_gains)),
                float(np.mean(level_gains)),
                float(np.max(level_gains)),
                float(np.std(level_gains)),
            ],
            dtype=np.float64,
        )

    def encode(self, short_stats: CircuitStats, long_stats: Sequence[CircuitStats]) -> np.ndarray:
        return np.concatenate(
            [
                self.encode_short(short_stats),
                self.encode_long(short_stats, long_stats),
            ]
        )


def compute_step_reward(
    prev_stats: CircuitStats,
    next_stats: CircuitStats,
    level_reward_weight: float = LEVEL_REWARD_WEIGHT,
) -> float:
    node_term = float(prev_stats.nodes - next_stats.nodes) / max(prev_stats.nodes, 1)
    level_term = float(prev_stats.level - next_stats.level) / max(prev_stats.level, 1)
    return node_term * (1 -level_reward_weight) + level_reward_weight * level_term


def is_better_stats(candidate: CircuitStats, incumbent: CircuitStats) -> bool:
    if candidate.nodes != incumbent.nodes:
        return candidate.nodes < incumbent.nodes
    return candidate.level < incumbent.level


def normalized_nodes_gap(reference: CircuitStats, incumbent: CircuitStats, nodes0: int) -> float:
    return max(float(incumbent.nodes - reference.nodes) / max(nodes0, 1), 0.0)


def update_rollout_best_by_depth(
    rollout_best_by_depth: Dict[int, "RolloutBestEntry"],
    stats: CircuitStats,
    origin_step: int,
    origin_action_idx: int,
    search_depth: int,
) -> None:
    current_best = rollout_best_by_depth.get(search_depth)
    if current_best is None or is_better_stats(stats, current_best.stats):
        rollout_best_by_depth[search_depth] = RolloutBestEntry(
            stats=CircuitStats(
                nodes=stats.nodes,
                level=stats.level,
                ave_fanout=stats.ave_fanout,
                max_fanout=stats.max_fanout,
            ),
            origin_step=origin_step,
            origin_action_idx=origin_action_idx,
            search_depth=search_depth,
        )


@dataclass
class ArmEvaluation:
    action_idx: int
    action_cmd: str
    next_stats: Optional[CircuitStats]
    reward_raw: float
    short_context: np.ndarray
    is_valid: bool
    error: Optional[str] = None


@dataclass
class RolloutBestEntry:
    stats: CircuitStats
    origin_step: int
    origin_action_idx: int
    search_depth: int


@dataclass
class RolloutSample:
    action_sequence: List[str]
    stats_trace: List[CircuitStats]

    @property
    def final_stats(self) -> CircuitStats:
        return self.stats_trace[-1]

def evaluate_step_arms(
    current_prefix: Tuple[str, ...],
    prefix_cache: PrefixCache,
    current_stats: CircuitStats,
    context_encoder: ContextEncoder,
    abc_bin: str,
    excluded_action_indices: Optional[Set[int]] = None,
) -> List[ArmEvaluation]:
    arm_results: List[ArmEvaluation] = []
    excluded_action_indices = excluded_action_indices or set()

    for action_idx, action_cmd in enumerate(actions):
        fallback_context = context_encoder.encode_short(current_stats)
        if action_idx in excluded_action_indices:
            arm_results.append(
                ArmEvaluation(
                    action_idx=action_idx,
                    action_cmd=action_cmd,
                    next_stats=None,
                    reward_raw=SEVERE_NEGATIVE_REWARD,
                    short_context=fallback_context,
                    is_valid=False,
                    error="excluded",
                )
            )
            continue
        try:
            next_entry = prefix_cache.get_or_build(
                current_prefix + (action_cmd,)
            )
            next_stats = next_entry.stats
            if next_stats.nodes > int(max(current_stats.nodes, 1) * SPIKE_FACTOR):
                reward_raw = SEVERE_NEGATIVE_REWARD
                arm_results.append(
                    ArmEvaluation(
                        action_idx=action_idx,
                        action_cmd=action_cmd,
                        next_stats=next_stats,
                        reward_raw=reward_raw,
                        short_context=fallback_context,
                        is_valid=False,
                        error="node_spike",
                    )
                )
            else:
                reward_raw = compute_step_reward(current_stats, next_stats)
                arm_results.append(
                    ArmEvaluation(
                        action_idx=action_idx,
                        action_cmd=action_cmd,
                        next_stats=next_stats,
                        reward_raw=reward_raw,
                        short_context=context_encoder.encode_short(next_stats),
                        is_valid=True,
                    )
                )

        except ABCSessionTimeout as exc:
            reward_raw = TIMEOUT_NEGATIVE_REWARD
            arm_results.append(
                ArmEvaluation(
                    action_idx=action_idx,
                    action_cmd=action_cmd,
                    next_stats=None,
                    reward_raw=reward_raw,
                    short_context=fallback_context,
                    is_valid=False,
                    error=str(exc),
                )
            )
        except Exception as exc:
            reward_raw = SEVERE_NEGATIVE_REWARD
            arm_results.append(
                ArmEvaluation(
                    action_idx=action_idx,
                    action_cmd=action_cmd,
                    next_stats=None,
                    reward_raw=reward_raw,
                    short_context=fallback_context,
                    is_valid=False,
                    error=str(exc),
                )
            )

    return arm_results


def sample_long_term_rollout_stats(
    current_prefix: Tuple[str, ...],
    prefix_cache: PrefixCache,
    first_action_idx: int,
    first_action: str,
    step_idx: int,
    abc_bin: str,
    rng: random.Random,
    rollout_best_by_depth: Dict[int, "RolloutBestEntry"],
) -> List[RolloutSample]:
    rollout_samples: List[RolloutSample] = []
    rollout_length = min(LONG_TERM_HORIZON, K_STEPS)
    if rollout_length <= 0:
        return rollout_samples

    future_length = max(rollout_length - 1, 0)
    for _ in range(LONG_TERM_ROLLOUTS):
        sampled_suffix = [rng.choice(actions) for _ in range(future_length)]
        sampled_actions = [first_action] + sampled_suffix
        try:
            rollout_entries = prefix_cache.trace_for_actions(
                base_prefix=current_prefix,
                extra_actions=sampled_actions,
            )
            rollout_trace = [entry.stats for entry in rollout_entries]
            if not rollout_trace:
                continue
            for depth_offset, rollout_stats in enumerate(rollout_trace, start=1):
                update_rollout_best_by_depth(
                    rollout_best_by_depth=rollout_best_by_depth,
                    stats=rollout_stats,
                    origin_step=step_idx,
                    origin_action_idx=first_action_idx,
                    search_depth=step_idx + depth_offset,
                )
            rollout_samples.append(
                RolloutSample(
                    action_sequence=list(sampled_actions),
                    stats_trace=rollout_trace,
                )
            )
        except Exception:
            continue

    return rollout_samples


def build_arm_contexts_for_iteration(
    current_prefix: Tuple[str, ...],
    prefix_cache: PrefixCache,
    current_stats: CircuitStats,
    step_idx: int,
    arm_evaluations: Sequence[ArmEvaluation],
    context_encoder: ContextEncoder,
    abc_bin: str,
    rng: random.Random,
    rollout_best_by_depth: Dict[int, "RolloutBestEntry"],
) -> Tuple[List[np.ndarray], List[List[RolloutSample]]]:
    arm_contexts: List[np.ndarray] = []
    arm_rollout_samples: List[List[RolloutSample]] = []
    for arm in arm_evaluations:
        if not arm.is_valid:
            arm_rollout_samples.append([])
            arm_contexts.append(
                np.concatenate(
                    [arm.short_context, np.zeros(context_encoder.long_dimension, dtype=np.float64)]
                )
            )
            continue
        rollout_samples = sample_long_term_rollout_stats(
            current_prefix=current_prefix,
            prefix_cache=prefix_cache,
            first_action_idx=arm.action_idx,
            first_action=arm.action_cmd,
            step_idx=step_idx,
            abc_bin=abc_bin,
            rng=rng,
            rollout_best_by_depth=rollout_best_by_depth,
        )
        arm_rollout_samples.append(rollout_samples)
        long_context = context_encoder.encode_long(
            current_stats,
            [sample.final_stats for sample in rollout_samples],
        )
        arm_contexts.append(np.concatenate([arm.short_context, long_context]))
    return arm_contexts, arm_rollout_samples


# ------------------------------
# 4) 优化主循环
# ------------------------------
def find_blif_files(benchmark_dir: str) -> List[str]:
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
    alpha: float = ALPHA,
    reg_lambda: float = REG_LAMBDA,
    abc_bin: Optional[str] = None,
    seed: int = RANDOM_SEED,
) -> Dict:
    if benchmark_name is None:
        benchmark_name = derive_benchmark_name(blif_path)

    logging.info("开始处理: %s", benchmark_name)
    abc_bin = abc_bin or ABC_BIN
    started = time.perf_counter()
    peak_memory_kb: float | None = None

    try:
        with ABCSession(blif_path, abc_bin=abc_bin, timeout_sec=ABC_TIMEOUT_SEC) as session:
            initial_stats, _ = session.initialize()
            peak_memory_kb = session.peak_memory_kb
    except Exception as exc:
        logging.error("初始评估失败: %s", benchmark_name)
        logging.debug("初始输出异常: %s", exc)
        return {
            "benchmark": benchmark_name,
            "benchmark_path": os.path.abspath(blif_path),
            "status": "failed_init",
            "error": str(exc),
            "runtime_sec": time.perf_counter() - started,
            "peak_memory_kb": peak_memory_kb,
        }

    context_encoder = ContextEncoder(initial_stats)
    prefix_cache = PrefixCache(
        root_design_path=blif_path,
        abc_bin=abc_bin,
        timeout_sec=ABC_TIMEOUT_SEC,
    )
    root_entry = prefix_cache.get_or_build(())
    prefix_cache.entries[()] = PrefixCacheEntry(
        prefix=(),
        stats=CircuitStats(
            nodes=initial_stats.nodes,
            level=initial_stats.level,
            ave_fanout=initial_stats.ave_fanout,
            max_fanout=initial_stats.max_fanout,
        ),
        snapshot_path=root_entry.snapshot_path,
    )

    def build_policy(step_idx: int) -> LinUCBPolicy:
        return LinUCBPolicy(
            num_actions=len(actions),
            context_dim=context_encoder.dimension,
            alpha=alpha,
            reg_lambda=reg_lambda,
            seed=seed + step_idx,
        )

    policies = [build_policy(step_idx) for step_idx in range(K_STEPS)]

    best_stats = CircuitStats(
        nodes=initial_stats.nodes,
        level=initial_stats.level,
        ave_fanout=initial_stats.ave_fanout,
        max_fanout=initial_stats.max_fanout,
    )
    best_recipe: List[str] = []
    current_stats = CircuitStats(
        nodes=initial_stats.nodes,
        level=initial_stats.level,
        ave_fanout=initial_stats.ave_fanout,
        max_fanout=initial_stats.max_fanout,
    )
    current_recipe: List[str] = []
    rollout_rng = random.Random(seed)
    fixed_rollout_horizon = min(LONG_TERM_HORIZON, K_STEPS)
    tail_finalize_step = max(K_STEPS - fixed_rollout_horizon, 0)
    rollout_best_by_depth: Dict[int, RolloutBestEntry] = {}
    excluded_action_indices: List[Set[int]] = [set() for _ in range(K_STEPS)]
    step_backtracked: List[bool] = [False for _ in range(K_STEPS)]
    selected_action_indices: List[Optional[int]] = [None for _ in range(K_STEPS)]
    selected_recipe_by_step: List[Optional[str]] = [None for _ in range(K_STEPS)]
    stats_by_depth: List[Optional[CircuitStats]] = [None for _ in range(K_STEPS + 1)]
    prefixes_by_depth: List[Optional[Tuple[str, ...]]] = [None for _ in range(K_STEPS + 1)]

    stats_by_depth[0] = CircuitStats(
        nodes=initial_stats.nodes,
        level=initial_stats.level,
        ave_fanout=initial_stats.ave_fanout,
        max_fanout=initial_stats.max_fanout,
    )
    prefixes_by_depth[0] = ()

    try:
        step_idx = 0
        while step_idx < K_STEPS:
            policy = policies[step_idx]
            current_prefix = prefixes_by_depth[step_idx]
            step_base_stats = stats_by_depth[step_idx]
            if current_prefix is None or step_base_stats is None:
                logging.warning(
                    "[%s][step=%d] 缺少回退快照，提前终止",
                    benchmark_name,
                    step_idx + 1,
                )
                break

            arm_evaluations = evaluate_step_arms(
                current_prefix=current_prefix,
                prefix_cache=prefix_cache,
                current_stats=step_base_stats,
                context_encoder=context_encoder,
                abc_bin=abc_bin,
                excluded_action_indices=excluded_action_indices[step_idx],
            )
            allowed_action_indices = {
                arm.action_idx for arm in arm_evaluations if arm.is_valid
            }
            if not allowed_action_indices:
                logging.warning(
                    "[%s][step=%d] 没有可用候选动作，提前终止",
                    benchmark_name,
                    step_idx + 1,
                )
                break

            last_action_idx = 0
            same_arm_streak = 0
            prev_action_idx: Optional[int] = None
            arm_rollout_samples: List[List[RolloutSample]] = []
            tail_best_sample_over_iters: Optional[RolloutSample] = None
            for iter_idx in range(1, N_EPISODES + 1):
                arm_contexts, arm_rollout_samples = build_arm_contexts_for_iteration(
                    current_prefix=current_prefix,
                    prefix_cache=prefix_cache,
                    current_stats=step_base_stats,
                    step_idx=step_idx,
                    arm_evaluations=arm_evaluations,
                    context_encoder=context_encoder,
                    abc_bin=abc_bin,
                    rng=rollout_rng,
                    rollout_best_by_depth=rollout_best_by_depth,
                )
                if step_idx == tail_finalize_step:
                    for rollout_samples in arm_rollout_samples:
                        for sample in rollout_samples:
                            if tail_best_sample_over_iters is None or is_better_stats(
                                sample.final_stats,
                                tail_best_sample_over_iters.final_stats,
                            ):
                                tail_best_sample_over_iters = sample
                last_action_idx = policy.select_action(
                    arm_contexts,
                    allowed_action_indices=allowed_action_indices,
                )
                selected_arm = arm_evaluations[last_action_idx]
                policy.update(last_action_idx, arm_contexts[last_action_idx], selected_arm.reward_raw)

                if prev_action_idx == last_action_idx:
                    same_arm_streak += 1
                else:
                    same_arm_streak = 1
                    prev_action_idx = last_action_idx

                logging.debug(
                    "[%s][step=%d iter=%d] action=%s raw_reward=%.4f streak=%d",
                    benchmark_name,
                    step_idx + 1,
                    iter_idx,
                    selected_arm.action_cmd,
                    selected_arm.reward_raw,
                    same_arm_streak,
                )

                if same_arm_streak >= EARLY_STOP_SAME_ARM_STREAK:
                    logging.info(
                        "[%s][step=%d] 提前收敛: action=%s iter=%d",
                        benchmark_name,
                        step_idx + 1,
                        selected_arm.action_cmd,
                        iter_idx,
                    )
                    break

            if step_idx == tail_finalize_step:
                if tail_best_sample_over_iters is None:
                    logging.warning(
                        "[%s][step=%d] 未找到可用的 long-term rollout，提前终止",
                        benchmark_name,
                        step_idx + 1,
                    )
                    break

                tail_prefix = current_prefix + tuple(tail_best_sample_over_iters.action_sequence)
                final_entry = prefix_cache.get_or_build(tail_prefix)
                prefixes_by_depth[K_STEPS] = tail_prefix

                for offset, action_cmd in enumerate(tail_best_sample_over_iters.action_sequence):
                    target_step = step_idx + offset
                    if target_step >= K_STEPS:
                        break
                    selected_recipe_by_step[target_step] = action_cmd
                    try:
                        selected_action_indices[target_step] = actions.index(action_cmd)
                    except ValueError:
                        selected_action_indices[target_step] = None
                    if offset < len(tail_best_sample_over_iters.stats_trace):
                        trace_stats = tail_best_sample_over_iters.stats_trace[offset]
                        stats_by_depth[target_step + 1] = CircuitStats(
                            nodes=trace_stats.nodes,
                            level=trace_stats.level,
                            ave_fanout=trace_stats.ave_fanout,
                            max_fanout=trace_stats.max_fanout,
                        )

                current_recipe = [
                    action_cmd for action_cmd in selected_recipe_by_step[:K_STEPS]
                    if action_cmd is not None
                ]
                current_stats = final_entry.stats

                if current_stats.nodes < best_stats.nodes:
                    best_stats = CircuitStats(
                        nodes=current_stats.nodes,
                        level=current_stats.level,
                        ave_fanout=current_stats.ave_fanout,
                        max_fanout=current_stats.max_fanout,
                    )
                    best_recipe = list(current_recipe)
                    logging.info(
                        "[%s] 新最优: and %d -> %d, lev %d -> %d (tail_from_step=%d), seq=%s",
                        benchmark_name,
                        initial_stats.nodes,
                        best_stats.nodes,
                        initial_stats.level,
                        best_stats.level,
                        step_idx + 1,
                        "; ".join(best_recipe),
                    )

                logging.info(
                    "[%s] 从 step %d 直接采用最优 rollout tail，final_and=%d best_and=%d final_lev=%d best_lev=%d",
                    benchmark_name,
                    step_idx + 1,
                    current_stats.nodes,
                    best_stats.nodes,
                    current_stats.level,
                    best_stats.level,
                )
                break

            best_arm = arm_evaluations[last_action_idx]
            if (not best_arm.is_valid) or (best_arm.next_stats is None):
                logging.warning(
                    "[%s][step=%d] 最终选中动作不可用: action=%s err=%s",
                    benchmark_name,
                    step_idx + 1,
                    best_arm.action_cmd,
                    best_arm.error,
                )
                break

            next_prefix = current_prefix + (best_arm.action_cmd,)
            prefixes_by_depth[step_idx + 1] = next_prefix
            next_entry = prefix_cache.get_or_build(next_prefix)
            stats_by_depth[step_idx + 1] = CircuitStats(
                nodes=next_entry.stats.nodes,
                level=next_entry.stats.level,
                ave_fanout=next_entry.stats.ave_fanout,
                max_fanout=next_entry.stats.max_fanout,
            )
            selected_action_indices[step_idx] = best_arm.action_idx
            selected_recipe_by_step[step_idx] = best_arm.action_cmd
            current_recipe = [
                action_cmd for action_cmd in selected_recipe_by_step[: step_idx + 1]
                if action_cmd is not None
            ]
            current_stats = next_entry.stats

            if current_stats.nodes < best_stats.nodes:
                best_stats = CircuitStats(
                    nodes=current_stats.nodes,
                    level=current_stats.level,
                    ave_fanout=current_stats.ave_fanout,
                    max_fanout=current_stats.max_fanout,
                )
                best_recipe = list(current_recipe)
                logging.info(
                    "[%s] 新最优: and %d -> %d, lev %d -> %d (step=%d), seq=%s",
                    benchmark_name,
                    initial_stats.nodes,
                    best_stats.nodes,
                    initial_stats.level,
                    best_stats.level,
                    step_idx + 1,
                    "; ".join(best_recipe),
                )

            logging.info(
                "[%s] Step %d/%d 完成, final_action=%s final_and=%d best_and=%d final_lev=%d best_lev=%d",
                benchmark_name,
                step_idx + 1,
                K_STEPS,
                best_arm.action_cmd,
                current_stats.nodes,
                best_stats.nodes,
                    current_stats.level,
                    best_stats.level,
                )

            current_depth = step_idx + 1
            best_rollout_entry = rollout_best_by_depth.get(current_depth)
            if best_rollout_entry is not None:
                gap = normalized_nodes_gap(
                    reference=best_rollout_entry.stats,
                    incumbent=current_stats,
                    nodes0=initial_stats.nodes,
                )
                target_step = best_rollout_entry.origin_step
                selected_action_idx = (
                    selected_action_indices[target_step]
                    if 0 <= target_step < len(selected_action_indices)
                    else None
                )
                if (
                    gap > RETURN_BACK_THRESHOLD
                    and target_step < step_idx
                    and selected_action_idx is not None
                    and not step_backtracked[target_step]
                ):
                    step_backtracked[target_step] = True
                    excluded_action_indices[target_step].add(selected_action_idx)
                    logging.info(
                        "[%s][depth=%d] 触发回退: current_and=%d best_rollout_and=%d gap=%.4f "
                        "target_step=%d exclude=%s",
                        benchmark_name,
                        current_depth,
                        current_stats.nodes,
                        best_rollout_entry.stats.nodes,
                        gap,
                        target_step + 1,
                        actions[selected_action_idx],
                    )

                    for depth_idx in range(target_step + 1, K_STEPS + 1):
                        prefix_to_clear = prefixes_by_depth[depth_idx]
                        if prefix_to_clear is not None:
                            prefixes_by_depth[depth_idx] = None
                        stats_by_depth[depth_idx] = None

                    for reset_step in range(target_step, K_STEPS):
                        selected_action_indices[reset_step] = None
                        selected_recipe_by_step[reset_step] = None
                        policies[reset_step] = build_policy(reset_step)

                    stale_depths = [
                        depth for depth in rollout_best_by_depth
                        if depth > target_step
                    ]
                    for depth in stale_depths:
                        del rollout_best_by_depth[depth]

                    current_recipe = [
                        action_cmd for action_cmd in selected_recipe_by_step[:target_step]
                        if action_cmd is not None
                    ]
                    base_stats = stats_by_depth[target_step]
                    if base_stats is not None:
                        current_stats = CircuitStats(
                            nodes=base_stats.nodes,
                            level=base_stats.level,
                            ave_fanout=base_stats.ave_fanout,
                            max_fanout=base_stats.max_fanout,
                        )
                    step_idx = target_step
                    continue

            step_idx += 1
    finally:
        if prefix_cache.peak_memory_kb is not None and (
            peak_memory_kb is None or prefix_cache.peak_memory_kb > peak_memory_kb
        ):
            peak_memory_kb = prefix_cache.peak_memory_kb
        prefix_cache.close()

    improvement = initial_stats.nodes - current_stats.nodes
    improvement_ratio = (improvement / initial_stats.nodes) if initial_stats.nodes > 0 else 0.0

    return {
        "benchmark": benchmark_name,
        "benchmark_path": os.path.abspath(blif_path),
        "status": "ok",
        "initial": asdict(initial_stats),
        "best": {
            **asdict(current_stats),
            "recipe_str": "; ".join(current_recipe),
            "recipe_best": "; ".join(best_recipe),
        },
        "improvement": {
            "nodes_reduced": improvement,
            "ratio": improvement_ratio,
        },
        "config": {
            "iterations_per_step": N_EPISODES,
            "steps_per_sequence": K_STEPS,
            "early_stop_same_arm_streak": EARLY_STOP_SAME_ARM_STREAK,
            "long_term_rollouts": LONG_TERM_ROLLOUTS,
            "long_term_horizon": LONG_TERM_HORIZON,
            "return_back_threshold": RETURN_BACK_THRESHOLD,
            "abc_timeout_sec": ABC_TIMEOUT_SEC,
            "spike_factor": SPIKE_FACTOR,
            "severe_negative_reward": SEVERE_NEGATIVE_REWARD,
            "timeout_negative_reward": TIMEOUT_NEGATIVE_REWARD,
            "alpha": alpha,
            "lambda": reg_lambda,
            "level_reward_weight": LEVEL_REWARD_WEIGHT,
            "seed": seed,
        },
        "runtime_sec": time.perf_counter() - started,
        "peak_memory_kb": peak_memory_kb,
    }


def build_summary(results: List[Dict]) -> Dict:
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
    method_name: str = "linucb",
    aggregate_payload: Optional[Dict] = None,
) -> str:
    result_path = result_utils.write_benchmark_result(method_name, result)
    if aggregate_payload is not None:
        result_utils.write_method_summary(method_name, aggregate_payload)
    return result_path


def setup_logging() -> None:
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
    parser = argparse.ArgumentParser(prog="linucb")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-search", help="Run LinUCB search.")
    run.add_argument("--workdir", type=Path, default=Path(result_utils.DEFAULT_WORKDIR_NAME))
    run.add_argument("--abc-bin", default=ABC_BIN)
    run.add_argument("--dataset-root", type=Path, default=Path("tc_public"))
    run.add_argument("--design", help="Single .blif filename under dataset-root. Default: run all.")
    run.add_argument("--steps", type=int, default=K_STEPS)
    run.add_argument("--episodes", type=int, default=N_EPISODES)
    run.add_argument("--timeout", type=int, default=ABC_TIMEOUT_SEC)
    run.add_argument("--spike-factor", type=float, default=SPIKE_FACTOR)
    run.add_argument("--severe-negative-reward", type=float, default=SEVERE_NEGATIVE_REWARD)
    run.add_argument("--timeout-negative-reward", type=float, default=TIMEOUT_NEGATIVE_REWARD)
    run.add_argument("--seed", type=int, default=RANDOM_SEED)
    run.add_argument("--alpha", type=float, default=ALPHA)
    run.add_argument("--linucb-alpha", type=float, default=None)
    run.add_argument("--ucb-c", type=float, default=None)
    run.add_argument("--lambda", dest="reg_lambda", type=float, default=REG_LAMBDA)
    run.add_argument("--linucb-lambda", type=float, default=None)
    run.add_argument("--long-term-rollouts", type=int, default=LONG_TERM_ROLLOUTS)
    run.add_argument("--long-term-horizon", type=int, default=LONG_TERM_HORIZON)
    run.add_argument("--return-back-threshold", type=float, default=RETURN_BACK_THRESHOLD)
    run.add_argument("--actions", default=None)
    run.add_argument("--result-json", type=Path, default=None)
    run.add_argument("--log-level", default="INFO")

    summarize = subparsers.add_parser("summarize", help="Aggregate JSON search results into CSV.")
    summarize.add_argument("--workdir", type=Path, default=Path(result_utils.DEFAULT_WORKDIR_NAME))
    return parser


def _configure_runtime_from_args(args: argparse.Namespace) -> None:
    global K_STEPS, N_EPISODES, ABC_TIMEOUT_SEC, SPIKE_FACTOR
    global SEVERE_NEGATIVE_REWARD, TIMEOUT_NEGATIVE_REWARD, RANDOM_SEED
    global ALPHA, REG_LAMBDA, LONG_TERM_ROLLOUTS, LONG_TERM_HORIZON
    global RETURN_BACK_THRESHOLD
    K_STEPS = args.steps
    N_EPISODES = args.episodes
    ABC_TIMEOUT_SEC = args.timeout
    SPIKE_FACTOR = args.spike_factor
    SEVERE_NEGATIVE_REWARD = args.severe_negative_reward
    TIMEOUT_NEGATIVE_REWARD = args.timeout_negative_reward
    RANDOM_SEED = args.seed
    ALPHA = args.alpha
    if args.linucb_alpha is not None:
        ALPHA = args.linucb_alpha
    if args.ucb_c is not None:
        ALPHA = args.ucb_c
    REG_LAMBDA = args.reg_lambda
    if args.linucb_lambda is not None:
        REG_LAMBDA = args.linucb_lambda
    LONG_TERM_ROLLOUTS = args.long_term_rollouts
    LONG_TERM_HORIZON = args.long_term_horizon
    RETURN_BACK_THRESHOLD = args.return_back_threshold
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
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    logging.info("ABC二进制: %s", args.abc_bin)
    logging.info("Benchmark目录: %s", os.path.abspath(args.dataset_root))
    logging.info(
        "参数: steps=%d iters_per_step=%d timeout=%d alpha=%.3f lambda=%.3f "
        "rollouts=%d horizon=%d return_back_threshold=%.4f seed=%d",
        K_STEPS,
        N_EPISODES,
        ABC_TIMEOUT_SEC,
        ALPHA,
        REG_LAMBDA,
        LONG_TERM_ROLLOUTS,
        LONG_TERM_HORIZON,
        RETURN_BACK_THRESHOLD,
        RANDOM_SEED,
    )

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
            "alpha": ALPHA,
            "lambda": REG_LAMBDA,
            "long_term_rollouts": LONG_TERM_ROLLOUTS,
            "long_term_horizon": LONG_TERM_HORIZON,
            "return_back_threshold": RETURN_BACK_THRESHOLD,
            "seed": RANDOM_SEED,
        },
        "results": [],
    }

    for design_name, design_path in selected.items():
        result = optimize_one_benchmark(
            design_path,
            benchmark_name=design_name,
            alpha=ALPHA,
            reg_lambda=REG_LAMBDA,
            abc_bin=args.abc_bin,
            seed=RANDOM_SEED,
        )
        result["meta"] = {
            "mode": "single_design" if args.design else "batch",
            "abc_bin": args.abc_bin,
        }
        all_results["results"].append(result)
        result_path = write_result_artifacts(result)
        logging.info("完成: %s -> %s", design_name, os.path.abspath(result_path))

    all_results["summary"] = build_summary(all_results["results"])
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
    return 0


def _command_summarize(args: argparse.Namespace) -> int:
    args.workdir = _resolve_local_path(args.workdir)
    summarize_results.main(["linucb", "--workdir", str(args.workdir)])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-search":
        return _command_run_search(args)
    if args.command == "summarize":
        return _command_summarize(args)
    parser.exit(status=2, message="Unknown command.\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
