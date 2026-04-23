from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


EXTERNAL_MONITOR_ACTIVE_ENV = "MCTSYN_EXTERNAL_MONITOR_ACTIVE"


def is_external_monitor_active() -> bool:
    return os.environ.get(EXTERNAL_MONITOR_ACTIVE_ENV) == "1"


def run_under_external_monitor(
    *,
    raw_argv: list[str],
    patch_json: Path,
    module_name: str,
) -> int:
    monitor_script = Path(__file__).resolve().parent / "scripts" / "external_monitor.py"
    env = os.environ.copy()
    env[EXTERNAL_MONITOR_ACTIVE_ENV] = "1"
    command = [
        sys.executable,
        str(monitor_script),
        f"--patch-json={patch_json}",
        "--",
        sys.executable,
        "-m",
        module_name,
        *raw_argv,
    ]
    completed = subprocess.run(command, env=env, check=False)
    return completed.returncode


def remove_flag(raw_argv: list[str], flag: str) -> list[str]:
    return [item for item in raw_argv if item != flag]


def upsert_option(raw_argv: list[str], option: str, value: str) -> list[str]:
    updated: list[str] = []
    index = 0
    replaced = False
    while index < len(raw_argv):
        token = raw_argv[index]
        if token == option:
            updated.extend([option, value])
            index += 2
            replaced = True
            continue
        if token.startswith(f"{option}="):
            updated.append(f"{option}={value}")
            index += 1
            replaced = True
            continue
        updated.append(token)
        index += 1
    if not replaced:
        updated.extend([option, value])
    return updated
