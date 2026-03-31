from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


DEFAULT_ACTION_SPACE = (
    "balance",
    "rewrite",
    "rewrite-z",
    "refactor",
    "refactor-z",
    "resub",
    "resub-z",
)

ACTION_TO_ABC_COMMAND = {
    "balance": "balance",
    "rewrite": "rewrite",
    "rewrite-z": "rewrite -z",
    "refactor": "refactor",
    "refactor-z": "refactor -z",
    "resub": "resub",
    "resub-z": "resub -z",
}

RESYN2_EXPANDED_COMMANDS = (
    "balance",
    "rewrite",
    "refactor",
    "balance",
    "rewrite",
    "rewrite -z",
    "balance",
    "refactor -z",
    "rewrite -z",
    "balance",
)

RESYN2_SEED_ACTIONS = (
    "balance",
    "rewrite",
    "refactor",
    "balance",
    "rewrite",
    "rewrite-z",
    "balance",
    "refactor-z",
    "rewrite-z",
    "balance",
)


def format_sequence_for_abc(sequence: tuple[str, ...]) -> str:
    return "; ".join(ACTION_TO_ABC_COMMAND[action] for action in sequence)


@dataclass(frozen=True)
class SearchConfig:
    design_name: str
    design_path: Path
    action_space: tuple[str, ...] = DEFAULT_ACTION_SPACE
    sequence_length: int = 24
    search_iterations: int = 1000
    and_weight: float = 0.5
    lev_weight: float = 0.2
    initial_temperature: float | None = None
    min_temperature: float = 1e-3
    seed: int = 0
    debug_search: bool = False
    workdir: Path = Path(".sasyn_work")

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["design_path"] = str(self.design_path)
        payload["workdir"] = str(self.workdir)
        return payload


@dataclass(frozen=True)
class BackendResult:
    and_count: int
    lev_count: int
    runtime_sec: float
    log: str
    peak_memory_kb: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "and": self.and_count,
            "lev": self.lev_count,
            "runtime_sec": self.runtime_sec,
            "log": self.log,
            "peak_memory_kb": self.peak_memory_kb,
        }


@dataclass(frozen=True)
class BaselineInfo:
    initial_and_count: int
    initial_lev_count: int
    and_baseline: float
    lev_baseline: float
    heuristic_and_count: int
    heuristic_lev_count: int
    heuristic_steps: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "initial_and": self.initial_and_count,
            "initial_lev": self.initial_lev_count,
            "and_baseline": self.and_baseline,
            "lev_baseline": self.lev_baseline,
            "heuristic_and": self.heuristic_and_count,
            "heuristic_lev": self.heuristic_lev_count,
            "heuristic_steps": self.heuristic_steps,
        }


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    temperature: float
    move_type: str
    accepted: bool
    delta_energy: float
    current_and_count: int
    current_lev_count: int
    current_energy: float
    best_and_count: int
    best_lev_count: int
    best_energy: float
    current_sequence: tuple[str, ...]
    best_sequence: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "temperature": self.temperature,
            "move_type": self.move_type,
            "accepted": self.accepted,
            "delta_energy": self.delta_energy,
            "current_and": self.current_and_count,
            "current_lev": self.current_lev_count,
            "current_energy": self.current_energy,
            "best_and": self.best_and_count,
            "best_lev": self.best_lev_count,
            "best_energy": self.best_energy,
            "current_sequence": format_sequence_for_abc(self.current_sequence),
            "best_sequence": format_sequence_for_abc(self.best_sequence),
        }


@dataclass(frozen=True)
class SearchResult:
    design_name: str
    design_path: Path
    abc_version: str
    seed: int
    sequence: tuple[str, ...]
    final_and_count: int
    final_lev_count: int
    total_runtime_sec: float
    peak_memory_kb: float | None
    baseline: BaselineInfo
    iterations: tuple[IterationResult, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "design_name": self.design_name,
            "design_path": str(self.design_path),
            "abc_version": self.abc_version,
            "seed": self.seed,
            "sequence": format_sequence_for_abc(self.sequence),
            "final_and": self.final_and_count,
            "final_lev": self.final_lev_count,
            "total_runtime_sec": self.total_runtime_sec,
            "peak_memory_kb": self.peak_memory_kb,
            "baseline": self.baseline.to_json_dict(),
            "iterations": [iteration.to_json_dict() for iteration in self.iterations],
            "metadata": self.metadata,
        }


class SynthesisBackend(Protocol):
    def version(self) -> str: ...

    def compute_baseline(self, design_path: Path) -> BaselineInfo: ...

    def evaluate_sequence(self, design_path: Path, sequence: tuple[str, ...]) -> BackendResult: ...
