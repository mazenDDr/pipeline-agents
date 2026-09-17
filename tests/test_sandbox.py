"""Sandbox: failure classification everywhere; escape attempts on Linux, where the real sandbox runs.

Run the escape tests on the GPU machine: ./gpu exec python -m pytest -q tests/test_sandbox.py
"""

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from pipeline_agents.sandbox.runner import Limits, LinuxSandbox, UnsandboxedRunner, classify

linux_only = pytest.mark.skipif(
    not LinuxSandbox.available(), reason="the real sandbox needs Linux + systemd --user"
)


def _work(tmp_path: Path, code: str) -> Path:
    work = tmp_path / "work"
    (work / "data").mkdir(parents=True)
    (work / "output").mkdir()
    (work / "data" / "input.csv").write_text("a,b\n1,2\n3,4\n")
    (work / "output" / "step.py").write_text(textwrap.dedent(code))
    return work


# --- classification (all platforms) ----------------------------------------------------------


@pytest.mark.parametrize(
    ("exit_code", "stderr", "timed_out", "reason"),
    [
        (0, "", False, "ok"),
        (None, "", True, "timeout"),
        (-9, "", False, "oom"),
        (137, "", False, "oom"),
        (1, "Traceback (most recent call last):\nMemoryError", False, "oom"),
        (-25, "", False, "file_size"),
        (90, "SANDBOX SETUP FAILED: hide home", False, "sandbox_error"),
        (1, "Traceback (most recent call last):\nKeyError: 'duration'", False, "traceback"),
        (3, "", False, "nonzero_exit"),
    ],
)
def test_classify(exit_code, stderr, timed_out, reason) -> None:
    assert classify(exit_code, stderr, timed_out) == reason


def test_unsandboxed_runner_timeout_kills_children(tmp_path: Path) -> None:
    code = """
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open("output/child.pid", "w").write(str(child.pid))
    time.sleep(60)
    """
    work = _work(tmp_path, code)
    result = UnsandboxedRunner(Limits(timeout_s=2)).run(work, ["output/step.py"])
    assert result.reason == "timeout" and result.failure == "infra" and not result.sandboxed
    assert not _alive(int((work / "output" / "child.pid").read_text()))


def _alive(pid: int) -> bool:
    """A killed child may linger as an unreaped zombie on Linux; that is not running."""
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# --- the real sandbox ------------------------------------------------------------------------


def _run(tmp_path: Path, code: str, **limits) -> tuple:
    work = _work(tmp_path, code)
    return LinuxSandbox(Limits(**{"timeout_s": 60, **limits})).run(work, ["output/step.py"]), work


@linux_only
def test_normal_step_runs_and_writes_output(tmp_path: Path) -> None:
    result, work = _run(
        tmp_path,
        """
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        df = pd.read_csv("data/input.csv")
        LinearRegression().fit(df[["a"]], df["b"])
        df.to_csv("output/clean.csv", index=False)
        print("rows", len(df))
    """,
    )
    assert result.ok and result.sandboxed, result.stderr
    assert "rows 2" in result.stdout and (work / "output" / "clean.csv").exists()


@linux_only
def test_traceback_is_a_logic_failure(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, "import pandas as pd\npd.read_csv('data/input.csv')['duration']\n")
    assert result.reason == "traceback" and result.failure == "logic"


@linux_only
def test_no_network(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path,
        """
        import socket
        for target in [("1.1.1.1", 53), ("huggingface.co", 443)]:
            try:
                socket.create_connection(target, timeout=3)
                print("CONNECTED", target)
            except OSError as e:
                print("blocked", target, type(e).__name__)
    """,
    )
    assert result.ok and "CONNECTED" not in result.stdout and result.stdout.count("blocked") == 2


@linux_only
def test_home_repository_ssh_and_windows_drives_are_hidden(tmp_path: Path) -> None:
    secret = Path.home() / "pipeline-agents-sandbox-test-secret.txt"
    secret.write_text("hidden answer")
    try:
        result, _ = _run(
            tmp_path,
            f"""
            import os
            from pathlib import Path
            for p in [{str(secret)!r}, {str(Path.home() / ".ssh")!r}, {str(Path.cwd())!r}, "/mnt/c"]:
                visible = Path(p).exists() and (not Path(p).is_dir() or os.listdir(p))
                print(p, "EXISTS" if visible else "absent")
            print("home contains", sorted(os.listdir(Path.home())))
        """,
        )
    finally:
        secret.unlink()
    assert result.ok, result.stderr
    assert "EXISTS" not in result.stdout, result.stdout
    assert "home contains ['miniconda3', 'work']" in result.stdout


@linux_only
def test_data_is_read_only_output_is_writable(tmp_path: Path) -> None:
    result, work = _run(
        tmp_path,
        """
        try:
            open("data/input.csv", "a").write("5,6\\n")
            print("DATA WRITTEN")
        except OSError as e:
            print("data read-only:", type(e).__name__)
        open("output/ok.txt", "w").write("fine")
    """,
    )
    assert result.ok and "DATA WRITTEN" not in result.stdout
    assert (work / "data" / "input.csv").read_text() == "a,b\n1,2\n3,4\n" and (
        work / "output" / "ok.txt"
    ).exists()


@linux_only
def test_other_processes_are_invisible(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path, "import os\nprint(sorted(int(p) for p in os.listdir('/proc') if p.isdigit()))\n"
    )
    assert result.ok and max(eval(result.stdout.strip())) < 20  # the host has hundreds of pids


@linux_only
def test_memory_bomb_is_an_infra_failure(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path, "import numpy as np\nx = np.ones((1024, 1024, 256))\nprint('ALLOCATED')\n", memory_mb=512
    )
    assert result.reason == "oom" and result.failure == "infra" and "ALLOCATED" not in result.stdout


@linux_only
def test_fork_bomb_is_contained(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path,
        """
        import os, time
        children = 0
        try:
            while True:
                if os.fork() == 0:
                    time.sleep(30)
                    os._exit(0)
                children += 1
        except OSError as e:
            print("stopped after", children, type(e).__name__)
    """,
        tasks=32,
        timeout_s=15,
    )
    assert "stopped after" in result.stdout
    assert int(result.stdout.split("stopped after")[1].split()[0]) < 32


@linux_only
def test_timeout_kills_the_whole_tree(tmp_path: Path) -> None:
    marker = f"pa-sandbox-sleeper-{os.getpid()}"
    result, _ = _run(
        tmp_path,
        f"""
        import subprocess, time
        subprocess.Popen(["sh", "-c", "exec -a {marker} sleep 120"])
        time.sleep(120)
    """,
        timeout_s=3,
    )
    assert result.reason == "timeout" and result.failure == "infra"
    left = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout
    assert left == "", f"orphans left: {left}"


@linux_only
def test_file_size_limit(tmp_path: Path) -> None:
    result, work = _run(
        tmp_path,
        """
        block = b"x" * (1024 * 1024)
        with open("output/huge.bin", "wb") as f:
            for _ in range(200):
                f.write(block)
    """,
        file_size_mb=50,
    )
    assert result.reason == "file_size" and result.failure == "infra"
    assert (work / "output" / "huge.bin").stat().st_size <= 50 * 1024 * 1024
