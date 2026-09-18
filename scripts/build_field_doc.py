"""Build docs/field_test.md from the field runs (T16).

    python scripts/build_field_doc.py outputs/field

Counts and outcomes come from outputs/field/{summary,reference}.json; the notes come from
outputs/field/notes.json, written after reading the delivered pipelines.
"""

import json
import sys
from pathlib import Path

from pipeline_agents.bench.field import BY_ID

SOURCE = {
    "wine-quality": "UCI Wine Quality (white), 4,898 wines",
    "student-grades": "UCI Student Performance (mathematics), 395 students",
    "household-power": "UCI Individual Household Electric Power Consumption, 2,075,259 minutes",
}


def build(root: Path) -> str:
    results = json.loads((root / "summary.json").read_text())
    reference = json.loads((root / "reference.json").read_text())
    notes = json.loads((root / "notes.json").read_text())
    by_task: dict[str, dict] = {}
    for r in results:
        by_task.setdefault(r["task"], {})[r["system"]] = r
    passed = sum(r["success"] for r in results)

    lines = [
        "# Field test: data the benchmark never saw",
        "",
        "The benchmark tasks were built for this system: the traps were planted on purpose and the "
        "thresholds were set from reference solutions. A field test asks a different question. Three public "
        "datasets that are not in the benchmark, a goal written the way someone would actually ask for it, "
        "and an answer computed here in pandas and never shown to the agents. Both systems ran once on each, "
        f"on the same data, with the same budget. **{passed} of {len(results)} runs got it right.**",
        "",
        "| task | what was asked | the team | the single-agent baseline |",
        "|---|---|---|---|",
    ]
    for task_id, runs in by_task.items():
        task = BY_ID[task_id]
        ask = (
            "a number per group, checked against pandas"
            if task.kind == "analytical"
            else (
                f"a model, scored by {task.metric} on {reference[task_id]['rows_held_back']:,} held-back rows"
            )
        )
        cells = []
        for system in ("multi", "baseline"):
            r = runs[system]
            mark = "passed" if r["success"] else "failed"
            cells.append(
                f"{mark} - {r['model_calls']} model call{'s' if r['model_calls'] != 1 else ''}, "
                f"{r['shadow_usd']:.4f} shadow $, {r['seconds']:.0f} s"
            )
        lines.append(f"| `{task_id}` | {ask} | {cells[0]} | {cells[1]} |")
    lines += ["", "## The three tasks", ""]
    for task_id, runs in by_task.items():
        task = BY_ID[task_id]
        note = notes.get(task_id, {})
        lines += [
            f"### {task_id}",
            "",
            f"*{SOURCE.get(task_id, '')}.* {note.get('why', '')}",
            "",
            "The goal, as the agents received it:",
            "",
            "> " + task.goal.strip().replace("\n\n", "\n>\n> ").replace("\n- ", "\n> - "),
            "",
        ]
        if task.kind == "analytical":
            lines += [
                "The reference answer, computed in pandas by `pipeline_agents.bench.field` and never shown "
                "to the agents (every number must match within "
                f"{task.tolerance:.1%} of itself):",
                "",
                "```json",
                json.dumps(reference[task_id]["reference"], indent=1)[:900].rstrip() + "\n...",
                "```",
                "",
            ]
        else:
            lines += [
                f"Measured on this exact split before any agent ran: {note.get('baselines', '')}. The run "
                f"had to reach {task.threshold} to pass, so a delivered model has to beat a one-line "
                "regression.",
                "",
            ]
        for system in ("multi", "baseline"):
            r = runs[system]
            name = "The team" if system == "multi" else "The baseline"
            lines.append(f"- **{name}**: {r['score']['detail']}")
        if note.get("what_happened"):
            lines += ["", note["what_happened"], ""]
        else:
            lines.append("")
    lines += [
        "## What this says",
        "",
        *[f"- {line}" for line in notes.get("conclusions", [])],
        "",
        "## What it does not say",
        "",
        "- One run per system per task, and three tasks. This is a sanity check on unseen data, not a "
        "measurement; the measured comparison is `docs/ablations.md`.",
        "- I wrote these goals, so they are clear and they ask for things the data can answer. A goal "
        "that is vague, or asks for something the data cannot support, is a different test.",
        "- The references are my own pandas code. They are unit-tested on small frames, and the agents' "
        "answers matching them to the last digit is evidence for both.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/field")
    Path("docs/field_test.md").write_text(build(root))
    print("wrote docs/field_test.md")


if __name__ == "__main__":
    main()
