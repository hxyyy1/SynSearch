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

RESYN2_EXPANDED_ACTIONS = (
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


def format_sequence_for_abc(sequence: tuple[str, ...]) -> str:
    return "; ".join(ACTION_TO_ABC_COMMAND[action] for action in sequence)


@dataclass(frozen=True)
class SearchConfig:
    design_name: str
    design_path: Path
    action_space: tuple[str, ...] = DEFAULT_ACTION_SPACE
    sequence_length: int = 24
    search_iterations: int = 64
    cpuct: float = 1.0
    mu_discount: float = 0.9
    seed: int = 0
    workdir: Path = Path(".alphasyn_work")

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
    snapshot_path: Path
    log: str
    cache_hit: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "and": self.and_count,
            "lev": self.lev_count,
            "runtime_sec": self.runtime_sec,
            "snapshot_path": str(self.snapshot_path),
            "log": self.log,
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True)
class BaselineInfo:
    initial_and_count: int
    initial_lev_count: int
    baseline: float
    heuristic_and_count: int
    heuristic_lev_count: int
    heuristic_steps: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "initial_and": self.initial_and_count,
            "initial_lev": self.initial_lev_count,
            "baseline": self.baseline,
            "heuristic_and": self.heuristic_and_count,
            "heuristic_lev": self.heuristic_lev_count,
            "heuristic_steps": self.heuristic_steps,
        }


@dataclass(frozen=True)
class StepResult:
    step_index: int
    selected_action: str
    prefix: tuple[str, ...]
    and_count: int
    lev_count: int
    root_value: float
    search_iterations: int
    action_scores: dict[str, float]
    action_visits: dict[str, int]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "selected_action": self.selected_action,
            "prefix": list(self.prefix),
            "and": self.and_count,
            "lev": self.lev_count,
            "root_value": self.root_value,
            "search_iterations": self.search_iterations,
            "action_scores": self.action_scores,
            "action_visits": self.action_visits,
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
    baseline: BaselineInfo
    steps: tuple[StepResult, ...] = field(default_factory=tuple)
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
            "baseline": self.baseline.to_json_dict(),
            "steps": [step.to_json_dict() for step in self.steps],
            "metadata": self.metadata,
        }


class SynthesisBackend(Protocol):
    """Backend interface for ABC-driven prefix evaluation."""

    def version(self) -> str: ...

    def compute_baseline(self, design_path: Path) -> BaselineInfo: ...

    def evaluate_prefix(self, design_path: Path, prefix: tuple[str, ...]) -> BackendResult: ...
