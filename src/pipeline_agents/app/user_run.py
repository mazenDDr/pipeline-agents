"""Turn an upload into a run the agents can take, and read a running one back (T13).

A benchmark run starts from a prepared workspace. A run started from the app has to build one: the uploaded
files become `data/`, the goal becomes `task.md` with the same deliverables contract the benchmark uses, and
the data dictionary is whatever the user pasted (or a note that there is none). Nothing here talks to a model
or to Streamlit, so it can be tested.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pipeline_agents.agents.common import Deps
from pipeline_agents.config import RunConfig, build_observer
from pipeline_agents.graph.build import load_state, run
from pipeline_agents.harness.registry import PromptRegistry
from pipeline_agents.sandbox.runner import default_runner
from pipeline_agents.schemas import Ledger, RunState, TaskContext

ANALYTICAL = """# Task

{goal}

## Deliverables

Write `output/pipeline.py`. It must run from this directory with `python output/pipeline.py`, read its
inputs only from `data/`, and write everything it produces into `output/`. It will be re-run from a clean
copy of `data/` and your `.py` files, so do not rely on files created by hand.

`output/answer.json` must hold the answer as JSON: one key per question asked above.
"""

PREDICTIVE = """# Task

{goal}

## Deliverables

Write `output/pipeline.py`. It must run from this directory with `python output/pipeline.py`, read its
inputs only from `data/`, and write everything it produces into `output/`. It will be re-run from a clean
copy of `data/` and your `.py` files, so do not rely on files created by hand.

The id column is `{id_column}`; `prediction` is the predicted `{target}`.

For this predictive task the pipeline must also:
- write `output/metrics.json` containing your own validation estimate, `{{"validation_{metric}": <number>}}`,
  measured on data the model was not trained on, in a way that reflects how the model will be used;
- leave a `output/predict.py` that runs as `python output/predict.py <features.csv> <predictions.csv>` and
  writes a CSV with a header and columns `{id_column},prediction` for every row of the features file. The
  features file has the same format and columns as the training data, minus the target and anything that
  would not be known at prediction time.
"""

NO_README = "No data dictionary was provided with these files. Read the data itself before trusting a column."


@dataclass
class UserTask:
    goal: str
    kind: str  # "analytical" or "predictive"
    files: dict[str, bytes]  # file name -> contents, as uploaded
    data_readme: str = ""
    target: str | None = None
    id_column: str | None = None
    metric: str = "mae"

    def task_md(self) -> str:
        if self.kind == "analytical":
            return ANALYTICAL.format(goal=self.goal.strip())
        return PREDICTIVE.format(
            goal=self.goal.strip(),
            id_column=self.id_column or "row_id",
            target=self.target or "target",
            metric=self.metric,
        )


def new_run_id(prefix: str = "app") -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}_{prefix}"


def prepare(task: UserTask, run_dir: Path) -> RunState:
    """Write the workspace and the starting state. Raises if the run directory already exists."""
    if not task.files:
        raise ValueError("a run needs at least one data file")
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists")
    workspace = run_dir / "workspace"
    (workspace / "data").mkdir(parents=True)
    (workspace / "output").mkdir()
    for name, content in task.files.items():
        (workspace / "data" / Path(name).name).write_bytes(content)
    (workspace / "task.md").write_text(task.task_md())
    (workspace / "data" / "README.md").write_text(task.data_readme.strip() or NO_README)
    return RunState(
        run_id=run_dir.name,
        task_id=run_dir.name,
        goal=task.goal.strip(),
        task=TaskContext(
            task_id=run_dir.name,
            task_md=task.task_md(),
            data_readme=task.data_readme.strip() or NO_README,
            kind=task.kind,
            id_column=task.id_column,
            metric=task.metric if task.kind == "predictive" else None,
        ),
        workspace=str(workspace.resolve()),
        ledger=Ledger(cap_usd=0.0),  # replaced by start(), which knows the config
    )


@dataclass
class RunHandle:
    """A run going on in a thread, watched by reading its checkpoints and its call log."""

    run_dir: Path
    run_id: str
    thread: threading.Thread
    error: list[str] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def state(self) -> RunState | None:
        try:
            return load_state(self.run_dir / "state.sqlite", self.run_id)
        except (FileNotFoundError, KeyError, ValueError):
            return None  # the first node has not been checkpointed yet

    def calls(self) -> list[dict]:
        return read_calls(self.run_dir)


def read_calls(run_dir: Path) -> list[dict]:
    path = run_dir / "calls.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            break  # a line still being written
    return out


def start(task: UserTask, cfg: RunConfig, runs_root: Path = Path("outputs/app")) -> RunHandle:
    """Prepare the workspace and run the graph in a background thread."""
    run_dir = runs_root / new_run_id()
    state = prepare(task, run_dir)
    state = state.model_copy(update={"ledger": Ledger(cap_usd=cfg.budget_usd, degrade_at=cfg.degrade_at)})
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    (run_dir / "task.json").write_text(
        json.dumps({"goal": task.goal, "kind": task.kind, "files": sorted(task.files)}, indent=2)
    )
    errors: list[str] = []

    def go() -> None:
        observer = build_observer(cfg, state.run_id, run_dir, cache_dir=run_dir / "cache")
        deps = Deps(
            observer=observer,
            registry=PromptRegistry("prompts"),
            config=cfg,
            runner=default_runner(),
        )
        try:
            final = run(deps, state, run_dir / "state.sqlite")
            (run_dir / "final.json").write_text(final.model_dump_json(indent=2))
        except Exception as e:  # noqa: BLE001 - shown in the page instead of dying in a thread
            errors.append(f"{type(e).__name__}: {e}")

    thread = threading.Thread(target=go, daemon=True)
    handle = RunHandle(run_dir=run_dir, run_id=state.run_id, thread=thread, error=errors)
    thread.start()
    return handle


def wait_for(handle: RunHandle, seconds: float, poll: float = 0.5) -> RunState | None:
    """For scripts and tests: the state once the run has one, or None when the wait runs out."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = handle.state()
        if state is not None:
            return state
        if not handle.running:
            return None
        time.sleep(poll)
    return None


def trace(state: RunState) -> list[dict]:
    """The run so far as rows: one per step attempt, with the verdict and the revision that followed."""
    rows = []
    plans = [*state.plan_history, state.plan] if state.plan else list(state.plan_history)
    archived = [*state.archived_steps, state.steps]
    for version, (plan, steps) in enumerate(zip(plans, archived, strict=False), start=1):
        for step in plan.steps:
            record = steps.get(step.id)
            if record is None:
                continue
            for n, attempt in enumerate(record.attempts):
                verdict = record.verdicts[n] if n < len(record.verdicts) else None
                revision = record.revisions[n] if n < len(record.revisions) else None
                rows.append(
                    {
                        "plan": version,
                        "step": step.id,
                        "kind": step.kind,
                        "intent": step.intent,
                        "attempt": attempt.attempt,
                        "exit": attempt.exit_code,
                        "failure": attempt.failure or "",
                        "artifacts": ", ".join(attempt.artifacts),
                        "decision": verdict.decision
                        if verdict
                        else ("running" if record.status == "running" else ""),
                        "confidence": verdict.confidence if verdict else None,
                        "issues": "; ".join(verdict.issues[:2]) if verdict else "",
                        "failed_checks": ", ".join(
                            f.tool for f in (verdict.findings if verdict else []) if not f.passed
                        ),
                        "revision": revision.instructions[:200] if revision else "",
                        "status": record.status,
                    }
                )
    return rows


def spend(state: RunState) -> list[dict]:
    """What each role has cost so far."""
    return [
        {
            "role": role,
            "calls": used.calls,
            "input tokens": used.input_tokens,
            "output tokens": used.output_tokens,
            "shadow $": round(used.shadow_usd, 5),
            "seconds": round(used.seconds, 1),
        }
        for role, used in sorted(state.ledger.by_role.items())
    ]
