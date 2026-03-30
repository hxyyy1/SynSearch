from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from .backend import immediate_reward
from .types import (
    BackendResult,
    BaselineInfo,
    SearchConfig,
    SearchResult,
    StepResult,
    SynthesisBackend,
)


@dataclass
class Edge:
    action: str
    prior: float
    reward: float
    child: "Node"
    visits: int = 0
    q_value: float = 0.0


@dataclass
class Node:
    prefix: tuple[str, ...]
    and_count: int
    backend_result: BackendResult
    children: dict[str, Edge] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return False


def _selection_score(parent: Node, edge: Edge, cpuct: float) -> float:
    exploration = cpuct * edge.prior * math.sqrt(max(parent.visits, 1)) / (edge.visits + 1)
    return edge.q_value + edge.reward + exploration


def _exploration_score(parent: Node, edge: Edge, cpuct: float) -> float:
    return cpuct * edge.prior * math.sqrt(max(parent.visits, 1)) / (edge.visits + 1)


def _node_debug_snapshot(node: Node, config: SearchConfig) -> dict[str, object]:
    return {
        "prefix": list(node.prefix),
        "and": node.and_count,
        "lev": node.backend_result.lev_count,
        "visits": node.visits,
        "value": node.value,
        "children": {
            action: {
                "q": edge.q_value,
                "r": edge.reward,
                "u": _exploration_score(node, edge, config.cpuct),
                "q_plus_r": edge.q_value + edge.reward,
                "selection_score": _selection_score(node, edge, config.cpuct),
                "visits": edge.visits,
                "prior": edge.prior,
                "child_and": edge.child.and_count,
                "child_lev": edge.child.backend_result.lev_count,
            }
            for action, edge in node.children.items()
        },
    }


def _expand_node(
    node: Node,
    config: SearchConfig,
    baseline: BaselineInfo,
    backend: SynthesisBackend,
) -> None:
    if node.children or len(node.prefix) >= config.sequence_length:
        _recompute_value(node, config.mu_discount)
        return

    uniform_prior = 1.0 / len(config.action_space)
    for action in config.action_space:
        child_prefix = node.prefix + (action,)
        backend_result = backend.evaluate_prefix(config.design_path, child_prefix)
        child = Node(
            prefix=child_prefix,
            and_count=backend_result.and_count,
            backend_result=backend_result,
        )
        reward = immediate_reward(
            node.and_count,
            child.and_count,
            node.backend_result.lev_count,
            child.backend_result.lev_count,
            baseline.and_baseline,
            baseline.lev_baseline,
        )
        node.children[action] = Edge(
            action=action,
            prior=uniform_prior,
            reward=reward,
            child=child,
        )
    _recompute_value(node, config.mu_discount)


def _recompute_value(node: Node, mu_discount: float) -> None:
    if not node.children:
        node.value = 0.0
        return

    best_child_return = max(edge.reward + edge.child.value for edge in node.children.values())
    for edge in node.children.values():
        edge.q_value = edge.child.value
    node.value = mu_discount * best_child_return


def _backpropagate(path: list[tuple[Node, Edge]], config: SearchConfig) -> None:
    if not path:
        return

    visited_nodes = [path[0][0]]
    for _, edge in path:
        edge.visits += 1
        visited_nodes.append(edge.child)

    for node in visited_nodes:
        node.visits += 1

    for node, edge in reversed(path):
        _recompute_value(edge.child, config.mu_discount)
        edge.q_value = edge.child.value
        _recompute_value(node, config.mu_discount)


def _run_iteration(
    root: Node,
    config: SearchConfig,
    baseline: BaselineInfo,
    backend: SynthesisBackend,
    iteration_index: int,
) -> dict[str, object] | None:
    node = root
    path: list[tuple[Node, Edge]] = []
    selected_actions: list[str] = []
    expanded_node: Node | None = None

    while True:
        if len(node.prefix) >= config.sequence_length:
            break
        if not node.children:
            _expand_node(node, config, baseline, backend)
            expanded_node = node
            break
        edge = max(
            node.children.values(),
            key=lambda current: (_selection_score(node, current, config.cpuct), current.action),
        )
        path.append((node, edge))
        selected_actions.append(edge.action)
        node = edge.child

    _backpropagate(path, config)
    _recompute_value(root, config.mu_discount)
    if not config.debug_search:
        return None

    traced_nodes: list[Node] = []
    seen_node_ids: set[int] = set()
    for current_node, edge in path:
        if id(current_node) not in seen_node_ids:
            traced_nodes.append(current_node)
            seen_node_ids.add(id(current_node))
        if id(edge.child) not in seen_node_ids:
            traced_nodes.append(edge.child)
            seen_node_ids.add(id(edge.child))
    if expanded_node is not None and id(expanded_node) not in seen_node_ids:
        traced_nodes.append(expanded_node)
        seen_node_ids.add(id(expanded_node))
    if not traced_nodes:
        traced_nodes.append(root)

    return {
        "iteration_index": iteration_index,
        "selected_actions": selected_actions,
        "expanded_prefix": list(expanded_node.prefix) if expanded_node is not None else None,
        "root_value": root.value,
        "root_visits": root.visits,
        "nodes": [_node_debug_snapshot(traced_node, config) for traced_node in traced_nodes],
    }


def _step_summary(
    step_index: int,
    root: Node,
    selected_action: str,
    config: SearchConfig,
    iteration_traces: tuple[dict[str, object], ...],
) -> StepResult:
    action_scores = {
        action: edge.q_value + edge.reward
        for action, edge in root.children.items()
    }
    action_visits = {
        action: edge.visits
        for action, edge in root.children.items()
    }
    action_debug = {
        action: {
            "q": edge.q_value,
            "r": edge.reward,
            "u": _exploration_score(root, edge, config.cpuct),
            "q_plus_r": edge.q_value + edge.reward,
            "selection_score": _selection_score(root, edge, config.cpuct),
            "visits": edge.visits,
            "prior": edge.prior,
            "child_and": edge.child.and_count,
            "child_lev": edge.child.backend_result.lev_count,
        }
        for action, edge in root.children.items()
    }
    selected_edge = root.children[selected_action]
    return StepResult(
        step_index=step_index,
        selected_action=selected_action,
        prefix=selected_edge.child.prefix,
        and_count=selected_edge.child.and_count,
        lev_count=selected_edge.child.backend_result.lev_count,
        root_value=root.value,
        root_visits=root.visits,
        search_iterations=config.search_iterations,
        action_scores=action_scores,
        action_visits=action_visits,
        action_debug=action_debug,
        iteration_traces=iteration_traces,
    )


def run_search(config: SearchConfig, backend: SynthesisBackend) -> SearchResult:
    started = time.perf_counter()
    random.seed(config.seed)
    baseline = backend.compute_baseline(config.design_path)
    root_backend_result = backend.evaluate_prefix(config.design_path, ())
    root = Node(prefix=(), and_count=root_backend_result.and_count, backend_result=root_backend_result)
    steps: list[StepResult] = []

    for step_index in range(1, config.sequence_length + 1):
        iteration_traces: list[dict[str, object]] = []
        for iteration_index in range(1, config.search_iterations + 1):
            trace = _run_iteration(root, config, baseline, backend, iteration_index)
            if trace is not None:
                iteration_traces.append(trace)
        if not root.children:
            break
        selected_edge = max(
            root.children.values(),
            key=lambda edge: (edge.q_value + edge.reward, edge.action),
        )
        steps.append(
            _step_summary(
                step_index,
                root,
                selected_edge.action,
                config,
                tuple(iteration_traces),
            )
        )
        root = selected_edge.child

    peak_memory_getter = getattr(backend, "get_peak_memory_kb", None)
    return SearchResult(
        design_name=config.design_name,
        design_path=config.design_path,
        abc_version=backend.version(),
        seed=config.seed,
        sequence=root.prefix,
        final_and_count=root.and_count,
        final_lev_count=root.backend_result.lev_count,
        total_runtime_sec=time.perf_counter() - started,
        peak_memory_kb=peak_memory_getter() if callable(peak_memory_getter) else None,
        baseline=baseline,
        steps=tuple(steps),
        metadata={
            "config": config.to_json_dict(),
            "implementation": "AlphaSyn w/o nn core only",
        },
    )
