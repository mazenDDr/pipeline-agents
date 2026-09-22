"""Run agent-written Python in a sandbox, and classify failures as infra or logic (T4).

On the GPU machine (Linux) a run is wrapped in three layers:

1. `systemd-run --user --scope`: a cgroup with MemoryMax (no swap) and TasksMax, so a memory bomb is killed
  and
   a fork bomb stops at the task limit without touching the model servers' processes.
2. `unshare --user --net --mount --pid --fork`: no network at all, a private process table, and a private
  mount
   table in which the whole home directory, /tmp and /mnt (the Windows drives under WSL) are replaced by empty
   tmpfs mounts. Only the Python environment (read-only), the step's `data/` (read-only) and its `output/`
   (writable) are mounted back. The hidden benchmark answers, SSH keys, other runs and the repository are
     simply
   not there.
3. `prlimit`: a maximum file size and number of open files.

A wall-clock timeout stops the whole cgroup, children included.

Classification: a timeout, an out-of-memory kill, the file size limit or a sandbox setup error is an infra
failure (retry or stop, but never charge a revision); a traceback or any other non-zero exit is a logic
  failure.

`UnsandboxedRunner` runs a plain subprocess with a timeout, for the Mac and for tests; it isolates nothing and
says so in every result.
"""

import functools
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Reason = Literal["ok", "timeout", "oom", "file_size", "sandbox_error", "traceback", "nonzero_exit"]
INFRA_REASONS = {"timeout", "oom", "file_size", "sandbox_error"}
OUTPUT_TAIL = 8000


@dataclass(frozen=True)
class Limits:
    timeout_s: float = 300.0
    memory_mb: int = 4096
    tasks: int = 64  # processes + threads in the cgroup
    file_size_mb: int = 1024
    open_files: int = 256
    threads_per_library: int = 4  # OMP/MKL/OpenBLAS threads, so numeric code stays inside the task limit


@dataclass(frozen=True)
class RunResult:
    exit_code: int | None
    stdout: str
    stderr: str
    seconds: float
    reason: Reason
    sandboxed: bool

    @property
    def ok(self) -> bool:
        return self.reason == "ok"

    @property
    def failure(self) -> Literal["infra", "logic"] | None:
        if self.ok:
            return None
        return "infra" if self.reason in INFRA_REASONS else "logic"


def classify(exit_code: int | None, stderr: str, timed_out: bool) -> Reason:
    if timed_out:
        return "timeout"
    if exit_code == 0:
        return "ok"
    if exit_code in (-signal.SIGKILL, 128 + signal.SIGKILL) or "MemoryError" in stderr:
        return "oom"  # the cgroup's OOM killer is the only SIGKILL the runner does not send itself
    if exit_code in (-signal.SIGXFSZ, 128 + signal.SIGXFSZ) or "File too large" in stderr:
        return "file_size"
    if "SANDBOX SETUP FAILED" in stderr:
        return "sandbox_error"
    if "Traceback (most recent call last)" in stderr:
        return "traceback"
    return "nonzero_exit"


def _thread_env(limits: Limits) -> dict[str, str]:
    n = str(limits.threads_per_library)
    return {
        k: n for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    }


class UnsandboxedRunner:
    """A plain subprocess with a timeout. No isolation: for the Mac and for tests only."""

    sandboxed = False

    def __init__(self, limits: Limits | None = None, python: str = sys.executable) -> None:
        self.limits = limits or Limits()
        self.python = python

    def run(self, workdir: Path, argv: list[str]) -> RunResult:
        start = time.perf_counter()
        env = {**os.environ, **_thread_env(self.limits), "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [self.python, *argv],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=self.limits.timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            timed_out = True
        code = None if timed_out else proc.returncode
        return RunResult(
            code,
            out[-OUTPUT_TAIL:],
            err[-OUTPUT_TAIL:],
            time.perf_counter() - start,
            classify(code, err, timed_out),
            sandboxed=False,
        )


# Runs inside `unshare`, as the namespace's root. $1 = staging dir, $2 = the real work dir, $3 = python env
# root, $4 = home, $5 = file size limit (bytes), $6 = open files limit; the rest is the command.
MOUNT_SCRIPT = r"""
set -e
fail() { echo "SANDBOX SETUP FAILED: $1" >&2; exit 90; }
SB="$1"; WORK="$2"; ENVROOT="$3"; HOMEDIR="$4"; FSIZE="$5"; NOFILE="$6"; shift 6
mount --make-rprivate / || fail "make-rprivate"
mount --rbind "$ENVROOT" "$SB/env" || fail "bind env"
mount --rbind "$WORK" "$SB/work" || fail "bind work"  # before home is hidden: the work dir may live under it
mount -t tmpfs -o size=16m,mode=755 tmpfs "$HOMEDIR" || fail "hide home"
mkdir -p "$ENVROOT" "$HOMEDIR/work"
mount --rbind "$SB/env" "$ENVROOT" || fail "restore env"
mount -o remount,bind,ro "$ENVROOT" || fail "env read-only"
mount --rbind "$SB/work" "$HOMEDIR/work" || fail "restore work"
if [ -d "$HOMEDIR/work/data" ]; then
  mount --rbind "$HOMEDIR/work/data" "$HOMEDIR/work/data" && mount -o remount,bind,ro "$HOMEDIR/work/data" \
    || fail "data read-only"
fi
mount -t tmpfs -o size=512m tmpfs /tmp || fail "hide /tmp"
[ -d /mnt ] && { mount -t tmpfs -o size=1m tmpfs /mnt || fail "hide /mnt"; }
mount -t proc proc /proc || fail "proc"
cd "$HOMEDIR/work"
exec prlimit --fsize="$FSIZE" --nofile="$NOFILE" -- "$@"
"""


class LinuxSandbox:
    """The real sandbox (see the module docstring). Needs Linux with systemd --user, unshare and prlimit."""

    sandboxed = True

    def __init__(
        self, limits: Limits | None = None, python: str = sys.executable, staging: Path = Path("/tmp")
    ) -> None:
        self.limits = limits or Limits()
        self.python = python
        self.env_root = str(Path(python).resolve().parents[1])
        self.staging = staging

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def available() -> bool:
        """Linux, the three tools, and a kernel that actually grants a user namespace.

        The tools can all be present while the host still refuses to map uids, which is the case
        inside many containers and on GitHub-hosted runners. Probing once is cheaper than letting
        every step fail as an infra error.
        """
        if sys.platform != "linux":
            return False
        if not all(shutil.which(tool) for tool in ("systemd-run", "unshare", "prlimit")):
            return False
        try:
            probe = subprocess.run(
                ["unshare", "--user", "--map-root-user", "true"],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return probe.returncode == 0

    def run(self, workdir: Path, argv: list[str]) -> RunResult:
        unit = f"pa-sandbox-{uuid.uuid4().hex[:12]}"
        sb = self.staging / unit
        (sb / "env").mkdir(parents=True)
        (sb / "work").mkdir()
        start = time.perf_counter()
        home = str(Path.home())
        lim = self.limits
        inner = [
            "unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--mount",
            "--pid",
            "--fork",
            "--kill-child",
            "sh",
            "-c",
            MOUNT_SCRIPT,
            "sandbox",
            str(sb),
            str(workdir.resolve()),
            self.env_root,
            home,
            str(lim.file_size_mb * 1024 * 1024),
            str(lim.open_files),
            "env",
            "-i",
            f"PATH={self.env_root}/bin:/usr/bin:/bin",
            f"HOME={home}",
            "PYTHONUNBUFFERED=1",
            "MPLBACKEND=Agg",
            *[f"{k}={v}" for k, v in _thread_env(lim).items()],
            self.python,
            *argv,
        ]
        cmd = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            f"--unit={unit}",
            "-p",
            f"MemoryMax={lim.memory_mb}M",
            "-p",
            "MemorySwapMax=0",
            "-p",
            f"TasksMax={lim.tasks}",
            *inner,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = proc.communicate(timeout=lim.timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["systemctl", "--user", "kill", "--signal=SIGKILL", f"{unit}.scope"], capture_output=True
            )
            out, err = proc.communicate()
            timed_out = True
        finally:
            shutil.rmtree(sb, ignore_errors=True)
        code = None if timed_out else proc.returncode
        reason = classify(code, err, timed_out)
        # When the OOM killer takes the step, unshare exits with 1 ("sigprocmask unblock failed") instead
        # of passing
        # the signal on, so the exit code cannot tell. systemd records the scope's result: ask it, then
        # clear it.
        scope_result = self._scope_result(unit)
        if not timed_out and scope_result == "oom-kill":
            reason = "oom"
        return RunResult(
            code,
            out[-OUTPUT_TAIL:],
            err[-OUTPUT_TAIL:],
            time.perf_counter() - start,
            reason,
            sandboxed=True,
        )

    @staticmethod
    def _scope_result(unit: str) -> str:
        shown = subprocess.run(
            ["systemctl", "--user", "show", "-p", "Result", "--value", f"{unit}.scope"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(["systemctl", "--user", "reset-failed", f"{unit}.scope"], capture_output=True)
        return shown


def default_runner(limits: Limits | None = None) -> LinuxSandbox | UnsandboxedRunner:
    return LinuxSandbox(limits) if LinuxSandbox.available() else UnsandboxedRunner(limits)
