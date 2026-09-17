"""The four roles (T6). Each builds its own context from the run state, calls its model and returns a typed
result; the graph decides what happens next.

- Planner: goal, data dictionary, profile, memory, (on re-plan) the failed plan and why -> `Plan`.
- Executor: one step -> a script, run in the sandbox inside the run's workspace -> `StepAttempt`.
- Critic: the attempt plus validation tool findings -> `Verdict` with a confidence.
- Reviser: the rejected attempt and the verdict -> `Revision` (fix, or escalate to the Planner).
"""

import re
from pathlib import Path

from pipeline_agents.agents.common import Deps, ParseError, call_role, parse_model
from pipeline_agents.schemas import Plan, PlanStep, Revision, RunState, StepAttempt, Verdict

TAIL = 3000


def step_file(state: RunState, step: PlanStep) -> str:
    index = [s.id for s in state.plan.steps].index(step.id) + 1
    return f"step_{index:02d}_{step.kind}.py"


def _output_files(workspace: Path) -> dict[str, float]:
    out = workspace / "output"
    return {p.name: p.stat().st_mtime_ns for p in out.iterdir() if p.is_file()} if out.exists() else {}


# --- Planner ---------------------------------------------------------------------------------


def plan(deps: Deps, state: RunState) -> Plan:
    # On a re-plan the failed plan is still the current one; the graph archives it after the new one is made.
    failed = state.plan.model_dump_json(indent=1) if state.escalation_reason and state.plan else None
    context = {
        "task_md": state.task.task_md,
        "goal": state.goal,
        "data_readme": state.task.data_readme,
        "dataset_profile": state.dataset_profile,
        "memory_hits": [h for h in state.memory_hits if h.step_id is None],
        "failed_plan": failed,
        "escalation_reason": state.escalation_reason,
    }
    reply = call_role(deps, "planner", context, parse_model(Plan), "high", None, state.replans)
    new_plan: Plan = reply.value
    ids = [s.id for s in new_plan.steps]
    if len(set(ids)) != len(ids):
        raise ParseError(f"duplicate step ids {ids}")
    return new_plan


# --- Executor --------------------------------------------------------------------------------


def parse_script(text: str) -> tuple[str, str]:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        raise ParseError("no ```python code block found")
    claims = re.search(r"CLAIMS:\s*(.+)", text.split("```")[-1] if "```" in text else text, re.S)
    return max(blocks, key=len), (claims.group(1).strip()[:600] if claims else "")


def execute(deps: Deps, state: RunState) -> StepAttempt:
    step = state.current_step
    workspace = Path(state.workspace)
    record = state.steps.get(step.id)
    previous = record.attempts[-1] if record and record.attempts else None
    done = []
    for s in state.plan.steps[: state.cursor]:
        r = state.steps[s.id]
        last = r.attempts[-1]
        done.append(
            {
                "step_id": s.id,
                "file": step_file(state, s),
                "claims": last.claims,
                "stdout": last.stdout[-800:],
            }
        )
    file = step_file(state, step)
    context = {
        "task_md": state.task.task_md,
        "data_readme": state.task.data_readme,
        "dataset_profile": state.dataset_profile,
        "plan": state.plan,
        "current": step,
        "done": done,
        "output_files": sorted(_output_files(workspace)),
        "file": file,
        "is_last": state.cursor == len(state.plan.steps) - 1,
        # The Executor gets facts about the data and the skills retrieved for this step; past episodes are for
        # the Planner.
        "memory_hits": [h for h in state.memory_hits if h.store == "semantic" or h.step_id == step.id],
        "previous": None
        if previous is None
        else {
            "code": previous.code,
            "exit_code": previous.exit_code,
            "reason": previous.failure or "ok",
            "stdout": previous.stdout[-1500:],
            "stderr": previous.stderr[-1500:],
        },
        "revision": state.pending_revision or "",
    }
    attempt_no = len(record.attempts) + 1 if record else 1
    reply = call_role(deps, "executor", context, parse_script, "high", step.id, attempt_no)
    code, claims = reply.value
    script = workspace / "output" / file
    script.write_text(code)
    before = _output_files(workspace)
    result = deps.runner.run(workspace, [f"output/{file}"])
    after = _output_files(workspace)
    written = sorted(n for n, t in after.items() if before.get(n) != t and n != file)
    return StepAttempt(
        attempt=attempt_no,
        code=code,
        exit_code=result.exit_code,
        stdout=result.stdout[-TAIL:],
        stderr=result.stderr[-TAIL:],
        failure=result.failure,
        artifacts=written,
        claims=claims,
        seconds=result.seconds,
    )


# --- Critic and Reviser ----------------------------------------------------------------------


def critique(deps: Deps, state: RunState, stakes: str) -> Verdict:
    """The Critic sees the attempt and the findings the validate node stored in `state.step_findings`."""
    step = state.current_step
    attempt = state.steps[step.id].attempts[-1]
    context = {
        "task_md": state.task.task_md,
        "data_readme": state.task.data_readme,
        "step": step,
        "attempt": attempt,
        "reason": attempt.failure or "ok",
        "findings": state.step_findings,
    }
    verdict = call_role(deps, "critic", context, parse_model(Verdict), stakes, step.id, attempt.attempt).value
    verdict.findings = list(state.step_findings)
    return verdict


def revise(deps: Deps, state: RunState) -> Revision:
    step = state.current_step
    record = state.steps[step.id]
    context = {
        "task_md": state.task.task_md,
        "step": step,
        "attempt": record.attempts[-1],
        "reason": record.attempts[-1].failure or "ok",
        "verdict": record.verdicts[-1],
        "findings": record.verdicts[-1].findings,
        "history": record.revisions,
    }
    return call_role(
        deps, "reviser", context, parse_model(Revision), "low", step.id, len(record.revisions) + 1
    ).value
