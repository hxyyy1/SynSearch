from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from memory_utils import run_command_with_peak_memory

from .core_types import (
    ACTION_TO_ABC_COMMAND,
    BackendResult,
    BaselineInfo,
    RESYN2_EXPANDED_COMMANDS,
)

OBJECTIVE_PATTERNS = (
    re.compile(r"\band\s*=\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bnd\s*=\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bnodes?\s*=\s*(\d+)\b", re.IGNORECASE),
)
LEVEL_PATTERNS = (
    re.compile(r"\blev\s*=\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\blevels?\s*=\s*(\d+)\b", re.IGNORECASE),
)


class BackendError(RuntimeError):
    """Raised when the ABC backend fails."""


def parse_abc_and_count(output: str) -> int:
    for pattern in OBJECTIVE_PATTERNS:
        match = pattern.search(output)
        if match:
            return int(match.group(1))
    raise BackendError("Unable to parse and-count from ABC output.")


def parse_abc_lev_count(output: str) -> int:
    for pattern in LEVEL_PATTERNS:
        match = pattern.search(output)
        if match:
            return int(match.group(1))
    raise BackendError("Unable to parse lev-count from ABC output.")


def parse_abc_stats(output: str) -> tuple[int, int]:
    return parse_abc_and_count(output), parse_abc_lev_count(output)


class ABCBackend:
    def __init__(self, abc_bin: str | None, workdir: Path) -> None:
        self.workdir = workdir.resolve()
        self.base_aig_dir = self.workdir / "base_aig"
        self.base_aig_dir.mkdir(parents=True, exist_ok=True)
        self.abc_bin = abc_bin or self._discover_abc()
        self._version: str | None = None
        self._prepared_bases: dict[Path, Path] = {}
        self._peak_memory_kb: float | None = None

    def _discover_abc(self) -> str:
        for candidate in ("abc", "yosys-abc", "berkeley-abc"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        raise BackendError(
            "ABC executable not found. Pass --abc-bin or set ABC_BIN to a valid executable."
        )

    def version(self) -> str:
        if self._version is None:
            self._version = self._probe_version()
        return self._version

    def get_peak_memory_kb(self) -> float | None:
        return self._peak_memory_kb

    def _observe_peak_memory_kb(self, peak_memory_kb: float | None) -> None:
        if peak_memory_kb is None:
            return
        if self._peak_memory_kb is None or peak_memory_kb > self._peak_memory_kb:
            self._peak_memory_kb = peak_memory_kb

    def _probe_version(self) -> str:
        for command in ("version", "print_stats", "help"):
            try:
                completed, _ = run_command_with_peak_memory(
                    [self.abc_bin, "-c", command],
                    check=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError):
                continue
            payload = (completed.stdout + completed.stderr).strip()
            if payload:
                return payload.splitlines()[0]
        return "unknown"

    def compute_baseline(self, design_path: Path) -> BaselineInfo:
        root = self.evaluate_sequence(design_path, ())
        heuristic = self._evaluate_script(design_path, RESYN2_EXPANDED_COMMANDS)
        and_delta = root.and_count - heuristic.and_count
        lev_delta = root.lev_count - heuristic.lev_count
        return BaselineInfo(
            initial_and_count=root.and_count,
            initial_lev_count=root.lev_count,
            and_baseline=and_delta if and_delta > 0 else max(root.and_count / 10.0, 1.0),
            lev_baseline=lev_delta if lev_delta > 0 else max(root.lev_count / 10.0, 1.0),
            heuristic_and_count=heuristic.and_count,
            heuristic_lev_count=heuristic.lev_count,
            heuristic_steps=len(RESYN2_EXPANDED_COMMANDS),
        )

    def evaluate_sequence(self, design_path: Path, sequence: tuple[str, ...]) -> BackendResult:
        commands = tuple(ACTION_TO_ABC_COMMAND[action] for action in sequence)
        return self._run_sequence(design_path, commands)

    def _evaluate_script(
        self,
        design_path: Path,
        commands: tuple[str, ...],
    ) -> BackendResult:
        return self._run_sequence(design_path, commands)

    def _run_sequence(self, design_path: Path, commands: tuple[str, ...] | list[str]) -> BackendResult:
        base_aig_path = self._prepare_base_aig(design_path)
        command_string = "; ".join(
            [
                f"read_aiger {shlex.quote(str(base_aig_path))}",
                *commands,
                "print_stats",
            ]
        )
        started = time.perf_counter()
        try:
            completed, peak_memory_kb = run_command_with_peak_memory(
                [self.abc_bin, "-c", command_string],
                check=True,
                text=True,
            )
        except OSError as exc:
            raise BackendError(f"Failed to execute ABC: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            payload = (exc.stdout or "") + "\n" + (exc.stderr or "")
            raise BackendError(
                "ABC command failed while evaluating a full sequence.\n"
                f"Command: {command_string}\n"
                f"Output:\n{payload.strip()}"
            ) from exc
        runtime_sec = time.perf_counter() - started
        self._observe_peak_memory_kb(peak_memory_kb)
        payload = (completed.stdout + "\n" + completed.stderr).strip()
        and_count, lev_count = parse_abc_stats(payload)
        return BackendResult(
            and_count=and_count,
            lev_count=lev_count,
            runtime_sec=runtime_sec,
            log=payload,
            peak_memory_kb=peak_memory_kb,
        )

    def _prepare_base_aig(self, design_path: Path) -> Path:
        resolved_design = design_path.resolve()
        cached_base = self._prepared_bases.get(resolved_design)
        if cached_base is not None and cached_base.exists():
            return cached_base

        design_hash = hashlib.sha256(str(resolved_design).encode("ascii")).hexdigest()[:16]
        base_aig_path = self.base_aig_dir / f"{resolved_design.stem}_{design_hash}.aig"
        command_string = "; ".join(
            [
                f"read_blif {shlex.quote(str(resolved_design))}",
                "strash",
                f"write_aiger {shlex.quote(str(base_aig_path))}",
            ]
        )
        try:
            _, peak_memory_kb = run_command_with_peak_memory(
                [self.abc_bin, "-c", command_string],
                check=True,
                text=True,
            )
        except OSError as exc:
            raise BackendError(f"Failed to execute ABC: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            payload = (exc.stdout or "") + "\n" + (exc.stderr or "")
            raise BackendError(
                "ABC command failed while preparing the base AIG.\n"
                f"Command: {command_string}\n"
                f"Output:\n{payload.strip()}"
            ) from exc

        self._observe_peak_memory_kb(peak_memory_kb)
        self._prepared_bases[resolved_design] = base_aig_path
        return base_aig_path
