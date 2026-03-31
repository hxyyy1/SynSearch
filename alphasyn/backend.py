from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import time
import uuid
import math
from pathlib import Path

from memory_utils import run_command_with_peak_memory

from .types import (
    ACTION_TO_ABC_COMMAND,
    BackendResult,
    BaselineInfo,
    RESYN2_EXPANDED_ACTIONS,
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


AND_REWARD_WEIGHT = 0.5
LEV_REWARD_WEIGHT = 0.2


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


def _signed_sqrt_reward(previous_value: int, current_value: int, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    reward = math.sqrt(abs(previous_value - current_value) / baseline)
    return reward if previous_value > current_value else -reward


def immediate_reward(
    previous_and_count: int,
    current_and_count: int,
    previous_lev_count: int,
    current_lev_count: int,
    and_baseline: float,
    lev_baseline: float,
) -> float:
    and_reward = _signed_sqrt_reward(previous_and_count, current_and_count, and_baseline)
    lev_reward = _signed_sqrt_reward(previous_lev_count, current_lev_count, lev_baseline)
    # Keep the total reward in the same range as the current sqrt-normalized reward.
    weight_sum = AND_REWARD_WEIGHT + LEV_REWARD_WEIGHT
    and_weight = AND_REWARD_WEIGHT / weight_sum
    lev_weight = LEV_REWARD_WEIGHT / weight_sum
    return (and_weight * and_reward) + (lev_weight * lev_reward)


class ABCBackend:
    def __init__(self, abc_bin: str | None, workdir: Path) -> None:
        self.workdir = workdir.resolve()
        self.cache_root = self.workdir / "cache"
        self.cache_namespace = uuid.uuid4().hex
        self.cache_dir = self.cache_root / self.cache_namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.abc_bin = abc_bin or self._discover_abc()
        self._version: str | None = None
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
        commands = ("version", "print_stats", "help")
        for command in commands:
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
        root = self.evaluate_prefix(design_path, ())
        heuristic = self._evaluate_script(design_path, "baseline_resyn2", RESYN2_EXPANDED_ACTIONS)
        and_delta = root.and_count - heuristic.and_count
        lev_delta = root.lev_count - heuristic.lev_count
        heuristic_steps = len(RESYN2_EXPANDED_ACTIONS)
        raw_baseline = and_delta / heuristic_steps if heuristic_steps else 0.0
        # baseline = raw_baseline if raw_baseline > 0 else max(root.and_count / 1000.0, 1.0)
        and_baseline = and_delta if and_delta > 0 else max(root.and_count / 10.0, 1.0)
        lev_baseline = lev_delta if lev_delta > 0 else max(root.lev_count / 10.0, 1.0)
        return BaselineInfo(
            initial_and_count=root.and_count,
            initial_lev_count=root.lev_count,
            and_baseline=and_baseline,
            lev_baseline=lev_baseline,
            heuristic_and_count=heuristic.and_count,
            heuristic_lev_count=heuristic.lev_count,
            heuristic_steps=heuristic_steps,
        )

    def evaluate_prefix(self, design_path: Path, prefix: tuple[str, ...]) -> BackendResult:
        key = self._cache_key(design_path, prefix, script_name=None)
        cached = self._load_cached_result(key)
        if cached is not None:
            return cached

        if prefix:
            parent = self.evaluate_prefix(design_path, prefix[:-1])
            input_path = parent.snapshot_path
            read_command = f"read_aiger {shlex.quote(str(input_path))}"
            commands = [ACTION_TO_ABC_COMMAND[prefix[-1]]]
        else:
            read_command = f"read_blif {shlex.quote(str(design_path.resolve()))}"
            commands = ["strash"]

        return self._run_and_cache(
            key=key,
            read_command=read_command,
            commands=commands,
        )

    def _evaluate_script(
        self,
        design_path: Path,
        script_name: str,
        commands: tuple[str, ...],
    ) -> BackendResult:
        key = self._cache_key(design_path, prefix=(), script_name=script_name)
        cached = self._load_cached_result(key)
        if cached is not None:
            return cached
        return self._run_and_cache(
            key=key,
            read_command=f"read_blif {shlex.quote(str(design_path.resolve()))}",
            commands=("strash",) + commands,
        )

    def _run_and_cache(
        self,
        *,
        key: str,
        read_command: str,
        commands: tuple[str, ...] | list[str],
    ) -> BackendResult:
        snapshot_path = self.cache_dir / f"{key}.aig"
        command_string = "; ".join(
            [
                read_command,
                *commands,
                "print_stats",
                f"write_aiger {shlex.quote(str(snapshot_path))}",
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
                "ABC command failed while evaluating a prefix.\n"
                f"Command: {command_string}\n"
                f"Output:\n{payload.strip()}"
            ) from exc
        runtime_sec = time.perf_counter() - started
        self._observe_peak_memory_kb(peak_memory_kb)
        payload = (completed.stdout + "\n" + completed.stderr).strip()
        and_count, lev_count = parse_abc_stats(payload)
        result = BackendResult(
            and_count=and_count,
            lev_count=lev_count,
            runtime_sec=runtime_sec,
            snapshot_path=snapshot_path,
            log=payload,
            cache_hit=False,
            peak_memory_kb=peak_memory_kb,
        )
        self._store_cached_result(key, result)
        return result

    def _cache_key(self, design_path: Path, prefix: tuple[str, ...], script_name: str | None) -> str:
        payload = {
            "design": str(design_path.resolve()),
            "prefix": list(prefix),
            "script_name": script_name,
            "abc_version": self.version(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("ascii")).hexdigest()

    def _metadata_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cached_result(self, key: str) -> BackendResult | None:
        metadata_path = self._metadata_path(key)
        if not metadata_path.exists():
            return None
        payload = json.loads(metadata_path.read_text(encoding="ascii"))
        if "and" not in payload or "lev" not in payload:
            return None
        snapshot_path = Path(payload["snapshot_path"])
        if not snapshot_path.exists():
            return None
        result = BackendResult(
            and_count=int(payload["and"]),
            lev_count=int(payload["lev"]),
            runtime_sec=float(payload["runtime_sec"]),
            snapshot_path=snapshot_path,
            log=str(payload["log"]),
            cache_hit=True,
            peak_memory_kb=float(payload["peak_memory_kb"]) if payload.get("peak_memory_kb") is not None else None,
        )
        self._observe_peak_memory_kb(result.peak_memory_kb)
        return result

    def _store_cached_result(self, key: str, result: BackendResult) -> None:
        metadata_path = self._metadata_path(key)
        metadata_path.write_text(
            json.dumps(result.to_json_dict(), indent=2),
            encoding="ascii",
        )

    def cleanup_cache(self) -> None:
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        if self.cache_root.exists() and not any(self.cache_root.iterdir()):
            self.cache_root.rmdir()
