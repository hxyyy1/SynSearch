from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from MABSyn.baseline_mab import SEVERE_NEGATIVE_REWARD, UCBBandit, snapshot_action_state
from alphasyn.backend import (
    AND_REWARD_WEIGHT,
    LEV_REWARD_WEIGHT,
    BackendError,
    immediate_reward,
)
from alphasyn.mcts import build_root_node, search_from_root
from alphasyn.types import DEFAULT_ACTION_SPACE, SearchConfig, SearchResult, StepResult, SynthesisBackend


@dataclass(frozen=True)
class HybridSearchConfig:
    design_name: str
    design_path: Path
    action_space: tuple[str, ...] = DEFAULT_ACTION_SPACE
    sequence_length: int = 24
    warmup_steps: int = 4
    warmup_episodes: int = 64
    warmup_top_k: int = 1
    ucb_c: float = 0.4
    search_iterations: int = 64
    cpuct: float = 1.0
    mu_discount: float = 0.9
    seed: int = 0
    debug_search: bool = False
    workdir: Path = Path(".hybridsyn_work")

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["design_path"] = str(self.design_path)
        payload["workdir"] = str(self.workdir)
        return payload

    def to_mcts_config(self) -> SearchConfig:
        return SearchConfig(
            design_name=self.design_name,
            design_path=self.design_path,
            action_space=self.action_space,
            sequence_length=self.sequence_length,
            search_iterations=self.search_iterations,
            cpuct=self.cpuct,
            mu_discount=self.mu_discount,
            seed=self.seed,
            debug_search=self.debug_search,
            workdir=self.workdir,
        )


ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class WarmupCandidate:
    prefix: tuple[str, ...]
    and_count: int
    lev_count: int
    cost: float


def _weighted_result_cost(
    reference_and: int,
    candidate_and: int,
    reference_lev: int,
    candidate_lev: int,
) -> float:
    weight_sum = AND_REWARD_WEIGHT + LEV_REWARD_WEIGHT
    and_weight = AND_REWARD_WEIGHT / weight_sum
    lev_weight = LEV_REWARD_WEIGHT / weight_sum
    return (
        and_weight * (float(candidate_and) / max(reference_and, 1))
        + lev_weight * (float(candidate_lev) / max(reference_lev, 1))
    )


def _is_better_handoff(
    candidate_cost: float,
    candidate_prefix: tuple[str, ...],
    best_cost: float | None,
    best_prefix: tuple[str, ...],
) -> bool:
    if best_cost is None:
        return True
    if candidate_cost != best_cost:
        return candidate_cost < best_cost
    return candidate_prefix < best_prefix


def _sorted_candidates(candidates: dict[tuple[str, ...], WarmupCandidate]) -> list[WarmupCandidate]:
    return sorted(
        candidates.values(),
        key=lambda candidate: (candidate.cost, candidate.prefix),
    )


def _candidate_to_json(candidate: WarmupCandidate) -> dict[str, object]:
    return {
        "prefix": list(candidate.prefix),
        "and": candidate.and_count,
        "lev": candidate.lev_count,
        "cost": candidate.cost,
    }


def _run_warmup(
    config: HybridSearchConfig,
    backend: SynthesisBackend,
    baseline_and: float,
    baseline_lev: float,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    warmup_steps = min(max(config.warmup_steps, 0), config.sequence_length)
    root = build_root_node(config.design_path, (), backend)
    top_k = max(1, config.warmup_top_k)
    retained_candidates: dict[tuple[str, ...], WarmupCandidate] = {}
    debug_trace: list[dict[str, object]] = []

    if warmup_steps == 0:
        root_candidate = WarmupCandidate(
            prefix=(),
            and_count=root.and_count,
            lev_count=root.backend_result.lev_count,
            cost=1.0,
        )
        if progress_callback is not None:
            progress_callback(
                "warmup_skipped",
                {
                    "effective_warmup_steps": warmup_steps,
                    "selected_prefix": [],
                    "selected_and": root_candidate.and_count,
                    "selected_lev": root_candidate.lev_count,
                },
            )
        return {
            "requested_warmup_steps": config.warmup_steps,
            "effective_warmup_steps": warmup_steps,
            "warmup_episodes": config.warmup_episodes,
            "warmup_top_k": top_k,
            "ucb_c": config.ucb_c,
            "selected_prefix": [],
            "selected_and": root_candidate.and_count,
            "selected_lev": root_candidate.lev_count,
            "selected_cost": root_candidate.cost,
            "top_candidates": [_candidate_to_json(root_candidate)],
            "debug_trace": debug_trace,
        }

    bandits = [
        UCBBandit(
            list(config.action_space),
            c=config.ucb_c,
            rng=random.Random(config.seed + step_index),
        )
        for step_index in range(warmup_steps)
    ]
    progress_interval = max(1, config.warmup_episodes // 10)

    for episode in range(1, config.warmup_episodes + 1):
        current_prefix: tuple[str, ...] = ()
        current_and = root.and_count
        current_lev = root.backend_result.lev_count
        episode_success = True

        for step_index in range(1, warmup_steps + 1):
            bandit = bandits[step_index - 1]
            action_state_rows = snapshot_action_state(bandit) if config.debug_search else []
            action_index = bandit.select_action()
            action_name = bandit.actions[action_index]
            bandit.total_steps += 1
            bandit.counts[action_index] += 1

            next_prefix = current_prefix + (action_name,)
            evaluation_error = ""
            try:
                next_result = backend.evaluate_prefix(config.design_path, next_prefix)
                reward = immediate_reward(
                    current_and,
                    next_result.and_count,
                    current_lev,
                    next_result.lev_count,
                    baseline_and,
                    baseline_lev,
                )
                step_and = next_result.and_count
                step_lev = next_result.lev_count
                cache_hit = int(next_result.cache_hit)
            except BackendError as exc:
                next_result = None
                reward = SEVERE_NEGATIVE_REWARD
                step_and = None
                step_lev = None
                cache_hit = 0
                episode_success = False
                evaluation_error = str(exc)

            bandit.update([action_index], reward)

            if config.debug_search:
                sequence_prefix = "; ".join(next_prefix)
                for action_state in action_state_rows:
                    debug_trace.append(
                        {
                            "episode": episode,
                            "step": step_index,
                            "action_index": action_state["action_index"],
                            "action": action_state["action"],
                            "selected": int(action_state["action_index"] == action_index),
                            "count_before": action_state["count_before"],
                            "q_before": action_state["q_before"],
                            "bonus_before": action_state["bonus_before"],
                            "score_before": action_state["score_before"],
                            "cold_start": int(bool(action_state["cold_start"])),
                            "count_after_select": bandit.counts[int(action_state["action_index"])],
                            "count_after_update": bandit.counts[int(action_state["action_index"])],
                            "q_after_update": bandit.q_values[int(action_state["action_index"])],
                            "sequence_prefix": sequence_prefix,
                            "step_reward": reward,
                            "step_and": step_and,
                            "step_lev": step_lev,
                            "step_cache_hit": cache_hit,
                            "evaluation_error": evaluation_error,
                        }
                    )

            if next_result is None:
                break

            current_prefix = next_prefix
            current_and = next_result.and_count
            current_lev = next_result.lev_count

        if episode_success and len(current_prefix) == warmup_steps:
            current_cost = _weighted_result_cost(
                root.and_count,
                current_and,
                root.backend_result.lev_count,
                current_lev,
            )
            candidate = WarmupCandidate(
                prefix=current_prefix,
                and_count=current_and,
                lev_count=current_lev,
                cost=current_cost,
            )
            retained_candidates[current_prefix] = candidate
            sorted_candidates = _sorted_candidates(retained_candidates)
            retained_candidates = {
                item.prefix: item
                for item in sorted_candidates[:top_k]
            }
        best_candidate = _sorted_candidates(retained_candidates)[0] if retained_candidates else None
        if progress_callback is not None and (
            episode == 1
            or episode == config.warmup_episodes
            or episode % progress_interval == 0
        ):
            progress_callback(
                "warmup_progress",
                {
                    "episode": episode,
                    "warmup_episodes": config.warmup_episodes,
                    "best_prefix": list(best_candidate.prefix) if best_candidate is not None else [],
                    "best_and": best_candidate.and_count if best_candidate is not None else root.and_count,
                    "best_lev": best_candidate.lev_count if best_candidate is not None else root.backend_result.lev_count,
                },
            )

    if not retained_candidates:
        retained_candidates[()] = WarmupCandidate(
            prefix=(),
            and_count=root.and_count,
            lev_count=root.backend_result.lev_count,
            cost=1.0,
        )
    sorted_candidates = _sorted_candidates(retained_candidates)
    best_candidate = sorted_candidates[0]
    return {
        "requested_warmup_steps": config.warmup_steps,
        "effective_warmup_steps": warmup_steps,
        "warmup_episodes": config.warmup_episodes,
        "warmup_top_k": top_k,
        "ucb_c": config.ucb_c,
        "selected_prefix": list(best_candidate.prefix),
        "selected_and": best_candidate.and_count,
        "selected_lev": best_candidate.lev_count,
        "selected_cost": best_candidate.cost,
        "top_candidates": [_candidate_to_json(candidate) for candidate in sorted_candidates],
        "debug_trace": debug_trace,
    }


def run_search(
    config: HybridSearchConfig,
    backend: SynthesisBackend,
    progress_callback: ProgressCallback | None = None,
) -> SearchResult:
    started = time.perf_counter()
    random.seed(config.seed)

    baseline = backend.compute_baseline(config.design_path)
    if progress_callback is not None:
        progress_callback(
            "design_start",
            {
                "design_name": config.design_name,
                "initial_and": baseline.initial_and_count,
                "initial_lev": baseline.initial_lev_count,
                "sequence_length": config.sequence_length,
                "warmup_steps": min(max(config.warmup_steps, 0), config.sequence_length),
                "warmup_episodes": config.warmup_episodes,
                "warmup_top_k": max(1, config.warmup_top_k),
                "search_iterations": config.search_iterations,
            },
        )
    warmup = _run_warmup(
        config,
        backend,
        baseline.and_baseline,
        baseline.lev_baseline,
        progress_callback=progress_callback,
    )
    warmup_candidates = [
        WarmupCandidate(
            prefix=tuple(str(action) for action in candidate["prefix"]),
            and_count=int(candidate["and"]),
            lev_count=int(candidate["lev"]),
            cost=float(candidate["cost"]),
        )
        for candidate in warmup["top_candidates"]
    ]

    branch_results: list[tuple[WarmupCandidate, object, tuple[StepResult, ...]]] = []
    for branch_index, candidate in enumerate(warmup_candidates, start=1):
        handoff_root = build_root_node(config.design_path, candidate.prefix, backend)
        if progress_callback is not None:
            progress_callback(
                "handoff",
                {
                    "branch_index": branch_index,
                    "branch_count": len(warmup_candidates),
                    "prefix": list(candidate.prefix),
                    "and": handoff_root.and_count,
                    "lev": handoff_root.backend_result.lev_count,
                },
            )

        mcts_steps = ()
        final_root = handoff_root
        if len(candidate.prefix) < config.sequence_length:
            def on_mcts_step(step_result: StepResult, *, current_branch_index: int = branch_index) -> None:
                if progress_callback is None:
                    return
                progress_callback(
                    "mcts_step",
                    {
                        "branch_index": current_branch_index,
                        "branch_count": len(warmup_candidates),
                        "step_index": step_result.step_index,
                        "sequence_length": config.sequence_length,
                        "selected_action": step_result.selected_action,
                        "and": step_result.and_count,
                        "lev": step_result.lev_count,
                    },
                )

            final_root, mcts_steps = search_from_root(
                handoff_root,
                config.to_mcts_config(),
                baseline,
                backend,
                start_step_index=len(candidate.prefix) + 1,
                step_callback=on_mcts_step,
            )
        branch_results.append((candidate, final_root, mcts_steps))

    best_candidate, final_root, mcts_steps = min(
        branch_results,
        key=lambda item: (
            _weighted_result_cost(
                baseline.initial_and_count,
                item[1].and_count,
                baseline.initial_lev_count,
                item[1].backend_result.lev_count,
            ),
            item[1].prefix,
        ),
    )

    peak_memory_getter = getattr(backend, "get_peak_memory_kb", None)
    metadata_warmup = dict(warmup)
    if not config.debug_search:
        metadata_warmup.pop("debug_trace", None)

    if progress_callback is not None:
        progress_callback(
            "design_done",
            {
                "design_name": config.design_name,
                "selected_branch_prefix": list(best_candidate.prefix),
                "final_and": final_root.and_count,
                "final_lev": final_root.backend_result.lev_count,
                "runtime_sec": time.perf_counter() - started,
                "sequence": list(final_root.prefix),
            },
        )

    return SearchResult(
        design_name=config.design_name,
        design_path=config.design_path,
        abc_version=backend.version(),
        seed=config.seed,
        sequence=final_root.prefix,
        final_and_count=final_root.and_count,
        final_lev_count=final_root.backend_result.lev_count,
        total_runtime_sec=time.perf_counter() - started,
        peak_memory_kb=peak_memory_getter() if callable(peak_memory_getter) else None,
        baseline=baseline,
        steps=mcts_steps,
        metadata={
            "config": config.to_json_dict(),
            "implementation": "HybridSyn: UCB1 warm-start plus MCTS continuation",
            "warmup": metadata_warmup,
            "selected_warmup_candidate": _candidate_to_json(best_candidate),
        },
    )
