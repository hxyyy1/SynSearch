from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .core_types import (
    BackendResult,
    BaselineInfo,
    IterationResult,
    RESYN2_SEED_ACTIONS,
    SearchConfig,
    SearchResult,
    SynthesisBackend,
)


@dataclass(frozen=True)
class EvaluatedSequence:
    sequence: tuple[str, ...]
    backend_result: BackendResult
    energy: float


def energy_from_counts(
    and_count: int,
    lev_count: int,
    baseline: BaselineInfo,
    and_weight: float,
    lev_weight: float,
) -> float:
    initial_and = max(baseline.initial_and_count, 1)
    initial_lev = max(baseline.initial_lev_count, 1)
    weight_sum = and_weight + lev_weight
    normalized_and_weight = and_weight / weight_sum
    normalized_lev_weight = lev_weight / weight_sum
    return (normalized_and_weight * (and_count / initial_and)) + (normalized_lev_weight * (lev_count / initial_lev))


def initial_sequence(config: SearchConfig) -> tuple[str, ...]:
    if config.sequence_length <= 0:
        return ()
    allowed_actions = set(config.action_space)
    seed_actions = tuple(action for action in RESYN2_SEED_ACTIONS if action in allowed_actions)
    if not seed_actions:
        seed_actions = config.action_space
    repeated = []
    while len(repeated) < config.sequence_length:
        repeated.extend(seed_actions)
    return tuple(repeated[: config.sequence_length])


def _evaluate_sequence(
    config: SearchConfig,
    baseline: BaselineInfo,
    backend: SynthesisBackend,
    sequence: tuple[str, ...],
) -> EvaluatedSequence:
    backend_result = backend.evaluate_sequence(config.design_path, sequence)
    return EvaluatedSequence(
        sequence=sequence,
        backend_result=backend_result,
        energy=energy_from_counts(
            backend_result.and_count,
            backend_result.lev_count,
            baseline,
            config.and_weight,
            config.lev_weight,
        ),
    )


def _propose_neighbor(
    sequence: tuple[str, ...],
    action_space: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, ...]:
    if not sequence:
        return sequence

    move = rng.choices(
        population=("replace", "swap", "block-rewrite"),
        weights=(0.6, 0.2, 0.2),
        k=1,
    )[0]
    data = list(sequence)

    if move == "replace" or len(sequence) == 1:
        index = rng.randrange(len(data))
        current_action = data[index]
        candidates = [action for action in action_space if action != current_action]
        data[index] = rng.choice(candidates or list(action_space))
        return tuple(data)

    if move == "swap":
        left, right = sorted(rng.sample(range(len(data)), 2))
        data[left], data[right] = data[right], data[left]
        return tuple(data)

    block_size = min(len(data), rng.randint(2, min(4, len(data))))
    start = rng.randint(0, len(data) - block_size)
    for offset in range(block_size):
        data[start + offset] = rng.choice(action_space)
    return tuple(data)


def _move_type(
    current_sequence: tuple[str, ...],
    candidate_sequence: tuple[str, ...],
) -> str:
    if current_sequence == candidate_sequence:
        return "replace"
    differing = [
        index for index, pair in enumerate(zip(current_sequence, candidate_sequence)) if pair[0] != pair[1]
    ]
    if len(differing) == 2:
        left, right = differing
        if (
            current_sequence[left] == candidate_sequence[right]
            and current_sequence[right] == candidate_sequence[left]
        ):
            return "swap"
    if len(differing) > 2:
        return "block-rewrite"
    return "replace"


def _estimate_initial_temperature(
    config: SearchConfig,
    baseline: BaselineInfo,
    backend: SynthesisBackend,
    sequence: tuple[str, ...],
    current_energy: float,
    rng: random.Random,
) -> float:
    uphill_deltas: list[float] = []
    for _ in range(32):
        neighbor_sequence = _propose_neighbor(sequence, config.action_space, rng)
        neighbor = _evaluate_sequence(config, baseline, backend, neighbor_sequence)
        delta = neighbor.energy - current_energy
        if delta > 0:
            uphill_deltas.append(delta)
    if uphill_deltas:
        average_uphill = sum(uphill_deltas) / len(uphill_deltas)
        return max(-average_uphill / math.log(0.8), config.min_temperature)
    return max(0.05, config.min_temperature)


def _temperature_at(
    iteration: int,
    search_iterations: int,
    initial_temperature: float,
    min_temperature: float,
) -> float:
    if search_iterations <= 1:
        return max(min_temperature, initial_temperature)
    ratio = iteration / max(search_iterations - 1, 1)
    return initial_temperature * ((min_temperature / initial_temperature) ** ratio)


def run_search(config: SearchConfig, backend: SynthesisBackend) -> SearchResult:
    started = time.perf_counter()
    rng = random.Random(config.seed)
    baseline = backend.compute_baseline(config.design_path)
    current = _evaluate_sequence(config, baseline, backend, initial_sequence(config))
    best = current

    if config.initial_temperature is None:
        initial_temperature_value = _estimate_initial_temperature(
            config,
            baseline,
            backend,
            current.sequence,
            current.energy,
            rng,
        )
    else:
        initial_temperature_value = max(config.initial_temperature, config.min_temperature)

    iterations: list[IterationResult] = []
    for iteration in range(1, config.search_iterations + 1):
        temperature = _temperature_at(
            iteration - 1,
            config.search_iterations,
            initial_temperature_value,
            config.min_temperature,
        )
        previous_sequence = current.sequence
        candidate_sequence = _propose_neighbor(previous_sequence, config.action_space, rng)
        move_type = _move_type(previous_sequence, candidate_sequence)
        candidate = _evaluate_sequence(config, baseline, backend, candidate_sequence)
        delta_energy = candidate.energy - current.energy
        accepted = delta_energy <= 0 or rng.random() < math.exp(-delta_energy / max(temperature, 1e-12))
        if accepted:
            current = candidate
        if current.energy < best.energy:
            best = current
        iterations.append(
            IterationResult(
                iteration=iteration,
                temperature=temperature,
                move_type=move_type,
                accepted=accepted,
                delta_energy=delta_energy,
                current_and_count=current.backend_result.and_count,
                current_lev_count=current.backend_result.lev_count,
                current_energy=current.energy,
                best_and_count=best.backend_result.and_count,
                best_lev_count=best.backend_result.lev_count,
                best_energy=best.energy,
                current_sequence=current.sequence,
                best_sequence=best.sequence,
            )
        )

    peak_memory_getter = getattr(backend, "get_peak_memory_kb", None)
    return SearchResult(
        design_name=config.design_name,
        design_path=config.design_path,
        abc_version=backend.version(),
        seed=config.seed,
        sequence=best.sequence,
        final_and_count=best.backend_result.and_count,
        final_lev_count=best.backend_result.lev_count,
        total_runtime_sec=time.perf_counter() - started,
        peak_memory_kb=peak_memory_getter() if callable(peak_memory_getter) else None,
        baseline=baseline,
        iterations=tuple(iterations),
        metadata={
            "config": config.to_json_dict(),
            "implementation": "Simulated annealing full-sequence search",
            "initial_temperature_used": initial_temperature_value,
        },
    )
