from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from typing import Sequence


_PEAK_MEMORY_MARKER = "__CODEX_PEAK_MEMORY_KB__="


def _find_time_binary() -> str | None:
    for candidate in ("/usr/bin/time", shutil.which("gtime"), shutil.which("time")):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def run_command_with_peak_memory(
    command: Sequence[str],
    *,
    text: bool = False,
    timeout: float | None = None,
    check: bool = False,
) -> tuple[subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes], float | None]:
    time_bin = _find_time_binary()
    if time_bin is None:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=text,
            timeout=timeout,
            check=False,
        )
        if check and completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                list(command),
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return completed, None

    wrapped_command = [time_bin, "-f", f"{_PEAK_MEMORY_MARKER}%M", *command]
    process = subprocess.Popen(
        wrapped_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        preexec_fn=os.setsid if os.name == "posix" else None,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            list(command),
            timeout,
            output=stdout if exc.output is None else exc.output,
            stderr=stderr if exc.stderr is None else exc.stderr,
        ) from exc

    stderr_text = stderr if isinstance(stderr, str) else stderr.decode("utf-8", errors="replace")
    peak_memory_kb, cleaned_stderr_text = _extract_peak_memory(stderr_text)
    if text:
        cleaned_stderr: str | bytes = cleaned_stderr_text
    else:
        cleaned_stderr = cleaned_stderr_text.encode("utf-8")

    completed = subprocess.CompletedProcess(
        list(command),
        process.returncode,
        stdout,
        cleaned_stderr,
    )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            list(command),
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed, peak_memory_kb


def _extract_peak_memory(stderr_text: str) -> tuple[float | None, str]:
    peak_memory_kb: float | None = None
    kept_lines: list[str] = []
    for line in stderr_text.splitlines():
        match = re.fullmatch(rf"{re.escape(_PEAK_MEMORY_MARKER)}(\d+)", line.strip())
        if match:
            peak_memory_kb = float(match.group(1))
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines)
    if stderr_text.endswith("\n") and cleaned:
        cleaned += "\n"
    return peak_memory_kb, cleaned


def read_linux_peak_memory_kb(pid: int) -> float | None:
    status_path = f"/proc/{pid}/status"
    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            status_text = handle.read()
    except OSError:
        return None

    for field_name in ("VmHWM", "VmRSS"):
        match = re.search(rf"^{field_name}:\s+(\d+)\s+kB$", status_text, re.MULTILINE)
        if match:
            return float(match.group(1))
    return None
