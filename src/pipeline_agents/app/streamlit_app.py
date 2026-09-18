"""The app (T13): start a run from a CSV and a goal, watch it, and read the measured results.

    streamlit run src/pipeline_agents/app/streamlit_app.py

The models must be reachable from wherever this runs: set PIPELINE_MODEL_HOST to the GPU machine's address
(the run config's default is 127.0.0.1). Nothing on this page computes a number: the run tab reads the run's
own checkpoints and call log, and the results tab reads the committed run summaries.
"""

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline_agents.analysis.failures import CATEGORIES, failures
from pipeline_agents.app.user_run import UserTask, read_calls, spend, start, trace
from pipeline_agents.config import load_run_config
from pipeline_agents.graph.build import load_state

RUNS = Path("outputs/runs")
APP_RUNS = Path("outputs/app")
GRIDS = ["t11-test-core", "t11-test-stores", "t11-dev2", "t11-dev", "t9-dev"]
POLL_SECONDS = 3

st.set_page_config(page_title="pipeline-agents", layout="wide")


def _config() -> tuple:
    names = sorted(p.stem for p in Path("configs/run").glob("*.yaml"))
    st.sidebar.header("Run settings")
    name = st.sidebar.selectbox("Config", names, index=names.index("default") if "default" in names else 0)
    cfg = load_run_config(f"configs/run/{name}.yaml")
    budget = st.sidebar.number_input("Budget, shadow $", 0.005, 1.0, float(cfg.budget_usd), step=0.005)
    seed = st.sidebar.number_input("Seed (0 = unseeded)", 0, 10_000, 1)
    host = st.sidebar.text_input("Model host", cfg.model_host)
    return cfg.model_copy(update={"budget_usd": float(budget), "seed": int(seed) or None, "model_host": host})


def _status_line(state, handle) -> None:
    columns = st.columns(5)
    columns[0].metric("Status", state.status if state else "starting")
    columns[1].metric(
        "Step",
        f"{min(state.cursor + 1, len(state.plan.steps))}/{len(state.plan.steps)}"
        if state and state.plan
        else "-",
    )
    columns[2].metric("Model calls", state.model_calls if state else 0)
    columns[3].metric("Shadow $", f"{state.ledger.total.shadow_usd:.4f}" if state else "0.0000")
    columns[4].metric("Re-plans", state.replans if state else 0)
    if state and state.ledger.cap_usd:
        st.progress(min(state.ledger.fraction_spent, 1.0), text="budget used")
    if handle is not None and handle.error:
        st.error(handle.error[-1])


def _run_view(state, run_dir: Path) -> None:
    if state is None:
        st.info("Waiting for the first checkpoint: the Planner is writing the plan.")
        return
    if state.plan:
        st.subheader(f"Plan v{state.plan.version}")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "step": s.id,
                        "kind": s.kind,
                        "intent": s.intent,
                        "checks": "; ".join(s.acceptance_checks),
                    }
                    for s in state.plan.steps
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    rows = trace(state)
    if rows:
        st.subheader("Steps, verdicts and revisions")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if state.memory_hits:
        st.subheader("Memory the run retrieved")
        st.dataframe(
            pd.DataFrame(
                [h.model_dump(include={"store", "step_id", "score", "text"}) for h in state.memory_hits]
            ),
            width="stretch",
            hide_index=True,
        )
    left, right = st.columns(2)
    with left:
        st.subheader("Spend by role")
        st.dataframe(pd.DataFrame(spend(state)), width="stretch", hide_index=True)
    with right:
        st.subheader("Model calls")
        calls = read_calls(run_dir)
        if calls:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "role": c["role"],
                            "step": c["step_id"] or "",
                            "tier": c["tier"],
                            "in": c["input_tokens"],
                            "out": c["output_tokens"],
                            "s": round(c["total_s"], 1),
                            "$": round(c["shadow_usd"], 5),
                        }
                        for c in calls[-25:]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
    if state.stop_reason:
        st.info(f"Stopped: {state.stop_reason}")
    files = sorted((Path(state.workspace) / "output").glob("*"))
    if files:
        st.subheader("Files the run wrote")
        pick = st.selectbox("File", [f.name for f in files])
        chosen = Path(state.workspace) / "output" / pick
        if chosen.suffix in {".py", ".json", ".txt", ".md"}:
            st.code(chosen.read_text()[:20_000], language="python" if chosen.suffix == ".py" else "json")
        else:
            st.write(f"{chosen.stat().st_size:,} bytes")


def tab_new_run(cfg) -> None:
    st.header("Run the agents on your data")
    # One form, so the goal and the columns are committed together with the button: a single click starts
    # the run instead of one click to commit the text and another to press the button.
    with st.form("new_run"):
        uploads = st.file_uploader("Data files (CSV or text)", accept_multiple_files=True)
        goal = st.text_area(
            "What should the pipeline do?",
            placeholder="Summarise revenue per month, and count the cancelled orders.",
            height=90,
        )
        kind = st.radio("Kind of task", ["analytical", "predictive"], horizontal=True)
        columns = st.columns(3)
        target = columns[0].text_input("Target column", help="Predictive tasks only")
        id_column = columns[1].text_input("Id column", help="Predictive tasks only")
        metric = columns[2].selectbox("Metric", ["mae", "rmse", "roc_auc"], help="Predictive tasks only")
        st.caption("Target, id and metric are read only for a predictive task.")
        readme = st.text_area("Data dictionary (optional but worth pasting)", height=90)
        submitted = st.form_submit_button("Start the run", type="primary")
    if submitted:
        missing = []
        if not uploads:
            missing.append("a data file")
        if not goal.strip():
            missing.append("a goal")
        if kind == "predictive" and not (target.strip() and id_column.strip()):
            missing.append("the target and id columns")
        if missing:
            st.warning("Still needed: " + ", ".join(missing))
        else:
            task = UserTask(
                goal=goal,
                kind=kind,
                files={u.name: u.getvalue() for u in uploads},
                data_readme=readme,
                target=target.strip() or None,
                id_column=id_column.strip() or None,
                metric=metric,
            )
            st.session_state["handle"] = start(task, cfg, APP_RUNS)
    handle = st.session_state.get("handle")
    if handle is None:
        st.caption(
            "Upload at least one file and describe the goal. The run starts when you press the button."
        )
        return
    st.divider()
    st.caption(f"Run `{handle.run_id}` in `{handle.run_dir}`")
    state = handle.state()
    _status_line(state, handle)
    _run_view(state, handle.run_dir)
    if handle.running:
        st.caption(f"Running: this page reloads every {POLL_SECONDS} seconds until the run stops.")
        time.sleep(POLL_SECONDS)
        st.rerun()


def _finished_runs() -> list[Path]:
    seen = [p.parent for p in APP_RUNS.glob("*/state.sqlite")]
    seen += [p.parent for p in RUNS.glob("*/*/state.sqlite")]
    return sorted(seen, key=lambda p: p.stat().st_mtime, reverse=True)[:100]


def tab_past_run() -> None:
    st.header("Look at a finished run")
    runs = _finished_runs()
    if not runs:
        st.info(
            "No runs with checkpoints on this machine yet. Runs made on the GPU machine can be pulled "
            "with `./gpu pull`."
        )
        return
    labels = [f"{p.parent.name}/{p.name}" for p in runs]
    chosen = runs[labels.index(st.selectbox("Run", labels))]
    try:
        state = load_state(chosen / "state.sqlite", chosen.name)
    except Exception as e:  # noqa: BLE001 - a half-written checkpoint is worth a message, not a crash
        st.error(f"Could not read that run: {e}")
        return
    result_path = chosen / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        st.caption(
            f"{result['task']} | {result['system']} | seed {result['seed']} | "
            f"{'passed' if result['success'] else 'failed'} the hidden check"
            + (f" at `{(result.get('checker') or {}).get('first_failure')}`" if not result["success"] else "")
        )
    _status_line(state, None)
    _run_view(state, chosen)


def tab_results() -> None:
    st.header("What the measurements say")
    grids = [RUNS / g for g in GRIDS if (RUNS / g).exists()]
    if not grids:
        st.info("No grid results in outputs/runs yet.")
        return
    chosen = st.multiselect("Grids", [g.name for g in grids], default=[g.name for g in grids[:2]])
    picked = [RUNS / name for name in chosen]
    results = [json.loads(p.read_text()) for g in picked for p in g.glob("*/result.json")]
    if not results:
        st.info("Those grids hold no results.")
        return
    frame = pd.DataFrame(
        [
            {
                "arm": r["run_id"].split("_", 1)[0],
                "task": r["task"],
                "seed": r["seed"],
                "success": float(r["success"]),
                "model calls": r.get("model_calls") or 0,
                "shadow $": r.get("shadow_usd") or 0.0,
                "seconds": r.get("seconds") or 0.0,
                "status": r["status"],
            }
            for r in results
        ]
    )
    by_arm = (
        frame.groupby("arm")
        .agg(
            runs=("success", "size"),
            success=("success", "mean"),
            calls=("model calls", "mean"),
            shadow=("shadow $", "mean"),
            seconds=("seconds", "mean"),
        )
        .round({"success": 2, "calls": 1, "shadow": 4, "seconds": 0})
        .sort_values("success", ascending=False)
    )
    st.subheader("Per arm")
    st.dataframe(by_arm, width="stretch")
    st.caption(
        "Means over runs, without intervals: the intervals that matter are in docs/ablations.md, which "
        "resamples tasks and then seeds."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Success by arm")
        st.bar_chart(by_arm["success"])
    with right:
        st.subheader("Shadow $ per run by arm")
        st.bar_chart(by_arm["shadow"])
    st.subheader("Success per task")
    st.dataframe(
        frame.pivot_table(index="task", columns="arm", values="success", aggfunc="mean").round(2),
        width="stretch",
    )
    st.subheader("How the failures split")
    cases_path = Path("outputs/trust/t12/answer_cases.json")
    cases = json.loads(cases_path.read_text()) if cases_path.exists() else {}
    found = failures(picked, cases)
    if found:
        counts = pd.DataFrame(
            [{"class": f.category, "arm": "baseline" if f.arm == "baseline" else "team"} for f in found]
        )
        table = counts.pivot_table(index="class", columns="arm", aggfunc="size", fill_value=0)
        table["what it is"] = [CATEGORIES.get(c, "") for c in table.index]
        st.dataframe(table.sort_values(table.columns[0], ascending=False), width="stretch")
    else:
        st.write("No failures in the selected grids.")


def main() -> None:
    st.title("pipeline-agents")
    st.caption(
        "A planner, an executor, a critic and a reviser build a pandas/sklearn pipeline for your data, in a "
        "sandbox, with every model call recorded. Costs are shadow prices, never money spent."
    )
    cfg = _config()
    new_run, past, results = st.tabs(["New run", "Finished run", "Results"])
    with new_run:
        tab_new_run(cfg)
    with past:
        tab_past_run()
    with results:
        tab_results()


main()
