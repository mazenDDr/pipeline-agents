"""The orchestration graph (T6): an explicit LangGraph state machine over `RunState`, checkpointed to SQLite.

    init -> plan -> execute -> validate -> critic -> advance -> ... -> deliver -> END
                       ^          |           |
                       |          | crash     | revise / escalate / low confidence
                       |          v           v
                       +------- revise <------+          (escalate or revisions used up) -> plan
                                                          (re-plans used up or low confidence) -> human

Every node is a plain function of the state that returns the fields it changed. Routing lives in the `route_*`
functions below, and nothing else decides where a run goes. Stops always record `status` and `stop_reason`.

Failure policy (numbers from RunConfig.loop):
- a crash (traceback) skips the Critic: the error is its own evidence, and the Reviser gets it directly;
- an infra failure of a step (timeout, out of memory) retries the step without spending a revision, up to
  `infra_retries`; the same for a model endpoint failure in any node;
- a step whose `max_revisions_per_step` fixes are used up escalates without asking the Reviser;
- escalations re-plan up to `max_replans` times, then stop as `needs_human`;
- a Critic confidence below `human_review_threshold` stops as `needs_human` instead of looping;
- caps on node visits and model calls, and the budget, stop the run with the reason.
"""

import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from pipeline_agents.agents import roles
from pipeline_agents.agents.common import Deps, InfraError, ParseError
from pipeline_agents.graph.checks import assemble_pipeline, deliver_checks, step_checks
from pipeline_agents.harness.budget import BudgetExceeded
from pipeline_agents.schemas import RunState, StepRecord, Verdict
from pipeline_agents.tools.profile import profile_dir, profile_file

HUMAN = "human"


def _stop(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "stop_reason": reason}


def node(name: str, deps: Deps) -> Callable[[Callable[[Deps, RunState], dict]], Callable[[RunState], dict]]:
    """Wrap a node body: count the visit, enforce the caps, sync the ledger, and turn budget and endpoint
    failures into state instead of exceptions."""

    def wrap(body: Callable[[Deps, RunState], dict]) -> Callable[[RunState], dict]:
        def run(state: RunState) -> dict:
            loop = deps.config.loop
            visits = state.node_visits + 1
            base: dict[str, Any] = {"node_visits": visits, "retry_node": None}
            if visits > loop.max_node_visits:
                return {
                    **base,
                    **_stop("failed", f"cap: {loop.max_node_visits} node visits reached in {name}"),
                }
            if deps.observer.ledger.total.calls >= loop.max_model_calls:
                return {
                    **base,
                    **_stop("failed", f"cap: {loop.max_model_calls} model calls reached before {name}"),
                }
            try:
                update = body(deps, state)
            except BudgetExceeded as e:
                update = _stop("failed", f"budget: {e}")
            except InfraError as e:
                retries = state.infra_retries + 1
                if retries > loop.infra_retries:
                    update = _stop("failed", f"infra: model endpoint failed {retries} times in {name}: {e}")
                else:
                    update = {"infra_retries": retries, "retry_node": name}
            ledger = deps.observer.ledger.model_copy(deep=True)
            return {**base, **update, "ledger": ledger, "model_calls": ledger.total.calls}

        return run

    return wrap


def build_graph(deps: Deps, checkpointer: SqliteSaver | None = None):
    loop = deps.config.loop
    graph = StateGraph(RunState)

    @node("init", deps)
    def init(deps: Deps, state: RunState) -> dict:
        data = Path(state.workspace) / "data"
        files = [
            p
            for p in sorted(data.iterdir())
            if p.is_file()
            and not p.name.lower().startswith("readme")
            and p.suffix.lower() in {".csv", ".data", ".txt", ".tsv"}
        ]
        return {
            "dataset_profile": state.dataset_profile or profile_dir(data),
            "raw_rows": {p.name: profile_file(p).rows for p in files},
        }

    @node("plan", deps)
    def plan(deps: Deps, state: RunState) -> dict:
        replanning = state.plan is not None
        try:
            new_plan = roles.plan(deps, state)
            new_plan = new_plan.model_copy(
                update={"version": len(state.plan_history) + (2 if replanning else 1)}
            )
        except ParseError as e:
            return _stop("failed", f"planner output unusable after a re-ask: {e}")
        update: dict[str, Any] = {
            "plan": new_plan,
            "cursor": 0,
            "steps": {},
            "pending_revision": None,
            "step_findings": [],
            "deliverable_findings": [],
            "infra_retries": 0,
        }
        if replanning:
            update |= {
                "plan_history": [*state.plan_history, state.plan],
                "archived_steps": [*state.archived_steps, state.steps],
                "replans": state.replans + 1,
            }
            output = Path(state.workspace) / "output"  # a new plan starts from clean outputs
            shutil.rmtree(output)
            output.mkdir()
        return update

    @node("execute", deps)
    def execute(deps: Deps, state: RunState) -> dict:
        step = state.current_step
        try:
            attempt = roles.execute(deps, state)
        except ParseError as e:
            return _stop("failed", f"executor output unusable after a re-ask on {step.id}: {e}")
        record = state.steps.get(step.id, StepRecord(step_id=step.id)).model_copy(deep=True)
        record.attempts.append(attempt)
        record.status = "running"
        infra = attempt.failure == "infra"
        if infra and state.infra_retries + 1 > loop.infra_retries:
            reason = f"infra: {step.id} failed {state.infra_retries + 1} times ({attempt.stderr[-200:]})"
            return {"steps": {**state.steps, step.id: record}, **_stop("failed", reason)}
        return {
            "steps": {**state.steps, step.id: record},
            "pending_revision": None if not infra else state.pending_revision,
            "infra_retries": state.infra_retries + 1 if infra else 0,
            "step_findings": [],
        }

    @node("validate", deps)
    def validate(deps: Deps, state: RunState) -> dict:
        step = state.current_step
        return {"step_findings": step_checks(state, step, state.steps[step.id].attempts[-1])}

    @node("critic", deps)
    def critic(deps: Deps, state: RunState) -> dict:
        step = state.current_step
        # The critique of the last step decides whether the deliverables are right: high stakes.
        stakes = "high" if state.cursor == len(state.plan.steps) - 1 else "low"
        try:
            verdict = roles.critique(deps, state, stakes)
        except ParseError as e:
            return _stop("failed", f"critic output unusable after a re-ask on {step.id}: {e}")
        record = state.steps[step.id].model_copy(deep=True)
        record.verdicts.append(verdict)
        return {"steps": {**state.steps, step.id: record}}

    @node("revise", deps)
    def revise(deps: Deps, state: RunState) -> dict:
        step = state.current_step
        record = state.steps[step.id].model_copy(deep=True)
        attempt = record.attempts[-1]
        if attempt.failure == "logic" and (
            not record.verdicts or len(record.verdicts) < len(record.attempts)
        ):
            record.verdicts.append(
                Verdict(
                    decision="revise",
                    confidence=1.0,
                    issues=[f"the script crashed ({attempt.exit_code}): {attempt.stderr[-800:]}"],
                )
            )
        if record.revision_count >= loop.max_revisions_per_step:
            record.status = "escalated"
            issues = record.verdicts[-1].issues[:2]
            reason = f"{step.id} still rejected after {record.revision_count} revisions: {issues}"
            return {"steps": {**state.steps, step.id: record}, "escalation_reason": reason}
        try:
            revision = roles.revise(
                deps, state.model_copy(update={"steps": {**state.steps, step.id: record}})
            )
        except ParseError as e:
            return _stop("failed", f"reviser output unusable after a re-ask on {step.id}: {e}")
        record.revisions.append(revision)
        if revision.action == "escalate":
            record.status = "escalated"
            return {
                "steps": {**state.steps, step.id: record},
                "escalation_reason": f"{step.id}: {revision.instructions} ({revision.reason})",
            }
        return {"steps": {**state.steps, step.id: record}, "pending_revision": revision.instructions}

    @node("advance", deps)
    def advance(deps: Deps, state: RunState) -> dict:
        step = state.current_step
        record = state.steps[step.id].model_copy(deep=True)
        record.status = "accepted"
        return {"steps": {**state.steps, step.id: record}, "cursor": state.cursor + 1, "step_findings": []}

    @node("deliver", deps)
    def deliver(deps: Deps, state: RunState) -> dict:
        files = [roles.step_file(state, s) for s in state.plan.steps]
        assemble_pipeline(state, files)
        findings = deliver_checks(state, deps.runner)
        if all(f.passed for f in findings):
            return {"deliverable_findings": findings, "status": "succeeded", "stop_reason": "delivered"}
        last = state.plan.steps[-1]
        record = state.steps[last.id].model_copy(deep=True)
        record.status = "running"
        record.verdicts.append(
            Verdict(
                decision="revise",
                confidence=1.0,
                findings=findings,
                issues=[f"delivery check {f.tool} failed: {f.detail}" for f in findings if not f.passed],
            )
        )
        return {
            "deliverable_findings": findings,
            "cursor": len(state.plan.steps) - 1,
            "steps": {**state.steps, last.id: record},
        }

    @node(HUMAN, deps)
    def human(deps: Deps, state: RunState) -> dict:
        step = state.current_step
        verdict = state.steps[step.id].verdicts[-1] if step and state.steps.get(step.id) else None
        if verdict is not None and verdict.confidence < loop.human_review_threshold:
            reason = f"low critic confidence ({verdict.confidence:.2f}) on {step.id}: {verdict.issues[:2]}"
        else:
            reason = f"re-plans used up ({state.replans}): {state.escalation_reason}"
        return _stop("needs_human", reason)

    for name, fn in [
        ("init", init),
        ("plan", plan),
        ("execute", execute),
        ("validate", validate),
        ("critic", critic),
        ("revise", revise),
        ("advance", advance),
        ("deliver", deliver),
        (HUMAN, human),
    ]:
        graph.add_node(name, fn)
    graph.set_entry_point("init")
    graph.add_conditional_edges("init", route_simple("plan"))
    graph.add_conditional_edges("plan", route_simple("execute"))
    graph.add_conditional_edges("execute", route_after_execute(loop))
    graph.add_conditional_edges("validate", route_simple("critic"))
    graph.add_conditional_edges("critic", route_after_critic(loop))
    graph.add_conditional_edges("revise", route_after_revise(loop))
    graph.add_conditional_edges("advance", route_after_advance)
    graph.add_conditional_edges("deliver", route_after_deliver)
    graph.add_edge(HUMAN, END)
    return graph.compile(checkpointer=checkpointer)


# --- routing ---------------------------------------------------------------------------------


def _halted(state: RunState) -> str | None:
    if state.status != "running":
        return END
    return state.retry_node


def route_simple(next_node: str):
    def route(state: RunState) -> str:
        return _halted(state) or next_node

    return route


def route_after_execute(loop):
    def route(state: RunState) -> str:
        if halted := _halted(state):
            return halted
        attempt = state.steps[state.current_step.id].attempts[-1]
        if attempt.failure == "infra":
            return "execute"  # the execute node stops the run once the retries are used up
        return "revise" if attempt.failure == "logic" else "validate"

    return route


def route_after_critic(loop):
    def route(state: RunState) -> str:
        if halted := _halted(state):
            return halted
        verdict = state.steps[state.current_step.id].verdicts[-1]
        if verdict.confidence < loop.human_review_threshold:
            return HUMAN
        return "advance" if verdict.decision == "accept" else "revise"

    return route


def route_after_revise(loop):
    def route(state: RunState) -> str:
        if halted := _halted(state):
            return halted
        record = state.steps[state.current_step.id]
        if record.status == "escalated":
            return "plan" if state.replans < loop.max_replans else HUMAN
        return "execute"

    return route


def route_after_advance(state: RunState) -> str:
    if halted := _halted(state):
        return halted
    return "execute" if state.cursor < len(state.plan.steps) else "deliver"


def route_after_deliver(state: RunState) -> str:
    if halted := _halted(state):
        return halted
    return "revise"


# --- running ---------------------------------------------------------------------------------


def open_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def run(deps: Deps, state: RunState | None, db_path: Path, recursion_limit: int = 1000) -> RunState:
    """Start a run (state given) or resume it from its last checkpoint (state None)."""
    saver = open_checkpointer(db_path)
    graph = build_graph(deps, saver)
    config = {
        "configurable": {"thread_id": state.run_id if state else deps.observer.run_id},
        "recursion_limit": recursion_limit,
    }
    if state is None:
        saved = graph.get_state(config).values
        deps.observer.ledger = RunState.model_validate(saved).ledger.model_copy(deep=True)
    result = graph.invoke(state, config)
    return RunState.model_validate(result)
