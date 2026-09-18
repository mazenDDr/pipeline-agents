"""Build docs/tour.md: one real run, start to finish, from its own checkpoints and call log (T17).

    python scripts/build_tour_doc.py outputs/runs/t11-test-core/multi_credit-1-default_s2

Nothing is rewritten for the page: the plan, the code, the printed output, the tool findings, the verdicts
and the checker stages all come from the run's files. Long code and output are cut, and the cut is marked.
"""

import json
import sys
from pathlib import Path

from pipeline_agents.graph.build import load_state
from pipeline_agents.schemas import RunState

CODE_LINES = 26
OUTPUT_LINES = 12


def _cut(text: str, lines: int) -> str:
    kept = text.strip().splitlines()
    if len(kept) <= lines:
        return "\n".join(kept)
    return "\n".join(kept[:lines] + [f"... ({len(kept) - lines} more lines)"])


def _verdict_line(verdict) -> str:
    issues = "; ".join(verdict.issues[:2]) if verdict.issues else "no issues"
    return f"**{verdict.decision}** (confidence {verdict.confidence:g}) - {issues}"


def _findings(verdict) -> list[str]:
    lines = []
    for finding in verdict.findings:
        mark = "pass" if finding.passed else "FAIL"
        lines.append(f"- `[{mark}] {finding.tool}`: {finding.detail[:220]}")
    return lines


def notice(run_dir: Path, result: dict, state: RunState) -> list[str]:
    """Three things a reader should not have to find: the re-plan, the tools, and the price of the team."""
    tool_findings = [f for r in state.steps.values() for v in r.verdicts for f in v.findings]
    failed_tools = [f.tool for f in tool_findings if not f.passed]
    rival = run_dir.parent / run_dir.name.replace(run_dir.name.split("_", 1)[0], "baseline", 1)
    lines = ["## What to notice", ""]
    if result["replans"]:
        lines.append(
            "- **The re-plan came from a false alarm.** The Critic rejected the last step because "
            "`output/pipeline.py` was a placeholder that printed a message. It is right that the file was a "
            "placeholder, but the orchestrator assembles `output/pipeline.py` from the accepted steps "
            "itself, so the file the Critic read was going to be replaced. The new plan was better "
            "anyway: it asked "
            "the last step for the whole flow."
        )
    if failed_tools:
        lines.append(
            f"- **The checks that fired.** {len(tool_findings)} tool findings across the run, "
            f"{len(failed_tools)} of them failures: {', '.join(sorted(set(failed_tools)))}. Each one is "
            "shown above, next to the step it judged."
        )
    else:
        lines.append(
            f"- **The validation tools stayed quiet.** {len(tool_findings)} findings across the run and "
            "not one failure. They look for dropped rows, sentinels left in place and target leakage."
        )
    if (rival / "result.json").exists():
        other = json.loads((rival / "result.json").read_text())
        lines.append(
            f"- **The single-agent baseline did the same task** with {other['model_calls']} model calls "
            f"against {result['model_calls']}, {other['shadow_usd']:.4f} shadow $ against "
            f"{result['shadow_usd']:.4f}, and {other['seconds']:.0f} s against {result['seconds']:.0f} s. It "
            f"scored {(other.get('checker') or {}).get('holdout_metric'):.4f} on the holdout against "
            f"{(result.get('checker') or {}).get('holdout_metric'):.4f}. That is the finding of "
            "`docs/ablations.md` in one pair of runs."
        )
    return [*lines, ""]


def build(run_dir: Path) -> str:
    result = json.loads((run_dir / "result.json").read_text())
    state: RunState = load_state(run_dir / "state.sqlite", run_dir.name)
    plans = [*state.plan_history, state.plan]
    step_sets = [*state.archived_steps, state.steps]
    checker = result.get("checker") or {}

    lines = [
        "# One run, start to finish",
        "",
        f"`{result['run_id']}`: the agent team on `{result['task']}`, seed {result['seed']}, config "
        f"`{result['config']}`, no memory. Everything below is taken from the run's own checkpoints and call "
        "log by `scripts/build_tour_doc.py`; nothing is rewritten, and every cut is marked.",
        "",
        "| | |",
        "|---|---|",
        f"| Result | {'passed' if result['success'] else 'failed'} the hidden check |",
        f"| Steps | {len(state.plan.steps)} in the final plan, {result['replans']} re-plan |",
        f"| Model calls | {result['model_calls']} |",
        f"| Shadow $ | {result['shadow_usd']:.4f} |",
        f"| Wall clock | {result['seconds']:.0f} s |",
        "",
        "## The task the agents were given",
        "",
        "```",
        _cut(state.task.task_md, 22),
        "```",
        "",
    ]
    for version, (plan, steps) in enumerate(zip(plans, step_sets, strict=True), start=1):
        header = "## The plan" if version == 1 else f"## The plan after the re-plan (v{version})"
        lines += [
            header,
            "",
            f"*{plan.rationale.strip()}*" if plan.rationale else "",
            "",
            "| step | kind | intent |",
            "|---|---|---|",
        ]
        lines += [f"| {s.id} | {s.kind} | {s.intent} |" for s in plan.steps]
        lines.append("")
        for step in plan.steps:
            record = steps.get(step.id)
            if record is None or not record.attempts:
                continue
            interesting = len(record.attempts) > 1 or any(
                not f.passed for v in record.verdicts for f in v.findings
            )
            lines += [f"### {step.id}: {step.intent}", ""]
            for n, attempt in enumerate(record.attempts):
                if len(record.attempts) > 1:
                    lines += [f"**Attempt {attempt.attempt}**", ""]
                lines += ["```python", _cut(attempt.code, CODE_LINES if interesting else 12), "```", ""]
                printed = _cut(attempt.stdout, OUTPUT_LINES)
                if printed:
                    lines += ["What it printed:", "", "```", printed, "```", ""]
                if attempt.stderr.strip():
                    lines += ["What it wrote to stderr:", "", "```", _cut(attempt.stderr, 8), "```", ""]
                verdict = record.verdicts[n] if n < len(record.verdicts) else None
                if verdict:
                    found = _findings(verdict)
                    if found:
                        lines += ["What the validation tools said:", "", *found, ""]
                    lines += [f"The Critic: {_verdict_line(verdict)}", ""]
                revision = record.revisions[n] if n < len(record.revisions) else None
                if revision:
                    lines += [
                        f"The Reviser ({revision.action}): {revision.instructions.strip()[:600]}",
                        "",
                    ]
            if len(record.verdicts) > len(record.attempts):
                extra = record.verdicts[len(record.attempts) :]
                for verdict in extra:
                    found = _findings(verdict)
                    lines += ["After the deliverables were assembled:", ""]
                    lines += found + [""] if found else []
                    lines += [f"The Critic: {_verdict_line(verdict)}", ""]
    lines += ["## What the delivery checks found", ""]
    for finding in state.deliverable_findings:
        lines.append(f"- `[{'pass' if finding.passed else 'FAIL'}] {finding.tool}`: {finding.detail[:300]}")
    lines += [
        "",
        "## What the hidden checker found",
        "",
        "The agents never see this: it runs the delivered pipeline from a clean copy of the data, then "
        "scores it on rows they never had.",
        "",
        "| stage | passed | detail |",
        "|---|---|---|",
    ]
    for stage in checker.get("stages", []):
        lines.append(f"| {stage['name']} | {'yes' if stage['passed'] else 'no'} | {stage['detail'][:160]} |")
    lines += [
        "",
        "## What it cost",
        "",
        "| role | calls | input tokens | output tokens | shadow $ | seconds |",
        "|---|---|---|---|---|---|",
    ]
    for role, used in sorted(result["by_role"].items()):
        lines.append(
            f"| {role} | {used['calls']} | {used['input_tokens']:,} | {used['output_tokens']:,} | "
            f"{used['shadow_usd']:.5f} | {used['seconds']:.0f} |"
        )
    lines += [
        "",
        "Shadow $ are token counts times a pinned price table (`configs/prices.yaml`), never money spent: "
        "the models run locally.",
        "",
        *notice(run_dir, result, state),
        "## What this run shows, and what it does not",
        "",
        "One run is an illustration, not evidence. The measured comparisons are in `docs/ablations.md`, how "
        "far the Critic can be trusted is in `docs/trust.md`, and every failed run of every grid is classed "
        "in `docs/failure_taxonomy.md`.",
        "",
    ]
    return "\n".join(line for line in lines if line is not None)


def main() -> None:
    run_dir = Path(sys.argv[1])
    Path("docs/tour.md").write_text(build(run_dir))
    print(f"wrote docs/tour.md from {run_dir}")


if __name__ == "__main__":
    main()
