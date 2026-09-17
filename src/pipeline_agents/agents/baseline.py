"""The single-agent baseline (T7): one model call plans, writes the whole pipeline and checks itself.

What it shares with the multi-agent system, so the comparison is about structure and not about retries:
the same model tier, prompt registry, Observer, budget, sandbox, dataset profile, delivery checks and
hidden checker. Like a multi-agent run, each attempt runs in the workspace first (leaving its outputs
there) and then passes the delivery checks from a clean copy. When an attempt fails either, the model sees
what went wrong and replies again, up to `attempts_allowed(loop)`: the most attempts the multi-agent loop
can spend on one step (a first try and its revisions, plus the same again after each re-plan).

No Critic, Reviser or per-step validation: that is the difference being measured.
"""

import re
from pathlib import Path

from pydantic import BaseModel

from pipeline_agents.agents.common import Deps, InfraError, ParseError, call_role
from pipeline_agents.config import LoopConfig
from pipeline_agents.graph.checks import deliver_checks, output_tail
from pipeline_agents.harness.budget import BudgetExceeded
from pipeline_agents.schemas import Finding, Ledger, RunState, TaskContext
from pipeline_agents.tools.profile import profile_dir

FILE_BLOCK = re.compile(r"###\s*output/([\w.\-]+\.py)\s*```(?:python|py)?\s*\n(.*?)```", re.S)


class BaselineAttempt(BaseModel):
    attempt: int
    files: dict[str, str]
    target_column: str | None
    self_check: str
    findings: list[Finding]
    passed: bool


class BaselineResult(BaseModel):
    run_id: str
    task_id: str
    status: str  # succeeded, failed
    stop_reason: str
    attempts: list[BaselineAttempt]
    ledger: Ledger
    model_calls: int


def attempts_allowed(loop: LoopConfig) -> int:
    per_plan = 1 + loop.max_revisions_per_step
    return per_plan * (1 + loop.max_replans)


def parse_reply(text: str) -> tuple[dict[str, str], str | None, str]:
    files = {name: code for name, code in FILE_BLOCK.findall(text)}
    if "pipeline.py" not in files:
        raise ParseError("no '### output/pipeline.py' heading followed by a ```python block")
    target = re.search(r"^TARGET:\s*(.+)$", text, re.M)
    target_column = target.group(1).strip().strip("`'\"") if target else None
    if target_column and target_column.lower() in {"none", "n/a", "null", "-"}:
        target_column = None
    self_check = text.split("SELF-CHECK:", 1)[1].strip()[:2000] if "SELF-CHECK:" in text else ""
    return files, target_column, self_check


def _problems(findings: list[Finding]) -> list[str]:
    return [f"{f.tool}: {f.detail}" for f in findings if not f.passed]


def run_baseline(deps: Deps, task: TaskContext, workspace: Path, run_id: str) -> BaselineResult:
    output = workspace / "output"
    profile = profile_dir(workspace / "data")
    attempts: list[BaselineAttempt] = []
    allowed = attempts_allowed(deps.config.loop)
    infra_failures = 0
    status, reason = "failed", f"no passing attempt in {allowed}"

    while len(attempts) < allowed:
        previous = attempts[-1] if attempts else None
        context = {
            "task_md": task.task_md,
            "data_readme": task.data_readme,
            "dataset_profile": profile,
            "memory_hits": [],
            "previous": None
            if previous is None
            else {"files": previous.files, "problems": _problems(previous.findings)},
        }
        try:
            reply = call_role(deps, "baseline", context, parse_reply, "high", None, len(attempts) + 1)
        except BudgetExceeded as e:
            status, reason = "failed", f"budget: {e}"
            break
        except InfraError as e:
            infra_failures += 1
            if infra_failures > deps.config.loop.infra_retries:
                status, reason = "failed", f"infra: model endpoint failed {infra_failures} times: {e}"
                break
            continue
        except ParseError as e:
            status, reason = "failed", f"baseline output unusable after a re-ask: {e}"
            break
        files, target_column, self_check = reply.value
        for stale in output.glob("*.py"):
            stale.unlink()
        for name, code in files.items():
            (output / name).write_text(code)
        state = RunState(
            run_id=run_id,
            task_id=task.task_id,
            goal=task.task_md,
            task=task,
            workspace=str(workspace),
            ledger=deps.observer.ledger,
        )
        in_workspace = deps.runner.run(workspace, ["output/pipeline.py"])
        if in_workspace.ok:
            findings = deliver_checks(state, deps.runner, target_column)
        else:
            findings = [
                Finding(
                    tool="pipeline_run",
                    passed=False,
                    detail=f"pipeline.py failed ({in_workspace.reason}): {output_tail(in_workspace)}",
                )
            ]
        passed = all(f.passed for f in findings)
        attempts.append(
            BaselineAttempt(
                attempt=len(attempts) + 1,
                files=files,
                target_column=target_column,
                self_check=self_check,
                findings=findings,
                passed=passed,
            )
        )
        if passed:
            status, reason = "succeeded", "delivered"
            break
        if deps.observer.ledger.total.calls >= deps.config.loop.max_model_calls:
            status, reason = "failed", f"cap: {deps.config.loop.max_model_calls} model calls reached"
            break

    ledger = deps.observer.ledger.model_copy(deep=True)
    return BaselineResult(
        run_id=run_id,
        task_id=task.task_id,
        status=status,
        stop_reason=reason,
        attempts=attempts,
        ledger=ledger,
        model_calls=ledger.total.calls,
    )
