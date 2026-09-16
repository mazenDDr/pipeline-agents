"""Minimal script runner for the model probe: a subprocess in its own dir with a timeout.

Not the agents' sandbox (T4 adds memory limits, no network and escape tests). Good enough here because
the probe scripts are short, the hidden answers live outside the work dir, and nothing else runs on the box.
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    exit_code: int | None  # None on timeout
    stdout: str
    stderr: str
    seconds: float


def run_script(code: str, workdir: Path, timeout_s: float = 120.0) -> RunResult:
    script = workdir / "solution.py"
    script.write_text(code)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, script.name], cwd=workdir, capture_output=True, text=True, timeout=timeout_s
        )
        return RunResult(
            proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:], time.perf_counter() - start
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"")[-4000:]
        return RunResult(
            None,
            out.decode(errors="replace") if isinstance(out, bytes) else out,
            "timeout",
            time.perf_counter() - start,
        )
