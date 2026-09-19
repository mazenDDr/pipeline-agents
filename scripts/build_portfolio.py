"""Build README, six SVGs, field guide, field test and tour from committed evidence.

    pip install -e '.[docs]'
    python scripts/build_portfolio.py

Templates are in site/templates. Run summaries and generated research documents are the
inputs; building this portfolio never calls a model or reads a private checkpoint.
"""

import html
import json
import re
import shutil
import textwrap
from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
REPO = "https://github.com/mazenDDr/pipeline-agents"
LABELS = {
    "baseline": "Single agent",
    "multi": "Team · no memory",
    "memall": "Team · all memories",
    "memproc": "Procedural only",
    "memsem": "Semantic only",
    "memepi": "Episodic only",
}
STYLE = """
.bg{fill:#ECEEF1}.card{fill:#FFFFFF;stroke:#DCE0E6}.accent{stroke:#2D5BD6;stroke-width:1.6}
.lab{font:600 9.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#6B7480;letter-spacing:.1em}
.hd{font:600 14px -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;fill:#171B21}
.body{font:400 12px -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;fill:#4A535F}
.tiny{font:400 10px -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;fill:#6B7480}
.mono{font:400 11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#171B21}
.big{font:600 30px -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;fill:#171B21}
.blue{fill:#2D5BD6}.ok{fill:#2E7D4F}.bad{fill:#C0392B}
.rule{stroke:#DCE0E6;stroke-width:1}.ci{stroke:#4A535F;stroke-width:1.5}
.bar{fill:#2D5BD6;transform-box:fill-box;transform-origin:left center;animation:grow 11s infinite}
.bar.other{fill:#8A93A1}.reveal{animation:show 11s infinite}.late{animation:late 11s infinite}
@keyframes grow{0%,8%{transform:scaleX(0)}35%,100%{transform:scaleX(1)}}
@keyframes show{0%,10%{opacity:0}28%,100%{opacity:1}}
@keyframes late{0%,32%{opacity:0}48%,100%{opacity:1}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;opacity:1!important}}
"""


def read(path: str) -> dict | list:
    return json.loads((ROOT / path).read_text())


def text(x: float, y: float, value: object, css: str = "body") -> str:
    return f'<text x="{x}" y="{y}" class="{css}">{html.escape(str(value))}</text>'


def lines(x: int, y: int, value: str, width: int = 35, css: str = "body") -> str:
    return "".join(text(x, y + i * 19, line, css) for i, line in enumerate(textwrap.wrap(value, width)))


def card(x: int, y: int, width: int, height: int, accent: bool = False) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="7" '
        f'class="card{" accent" if accent else ""}"/>'
    )


def svg(name: str, title: str, height: int, content: str) -> None:
    markup = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 '
        f'{height}" width="880" height="{height}" role="img" aria-labelledby="title">'
        f'<title id="title">{html.escape(title)}</title><style>{STYLE}</style>'
        f'<rect class="bg" width="880" height="{height}" rx="14"/>{content}</svg>\n'
    )
    (ASSETS / f"{name}.svg").write_text(markup)


def figures(arms: dict, field: list, trust: dict, reference: dict, comparisons: dict) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    wine = {r["system"]: r for r in field if r["task"] == "wine-quality"}
    body = text(28, 28, "ONE REAL TASK · WINE QUALITY · RECORDED FIELD TEST", "lab")
    for x, label in ((28, "THE INPUT"), (308, "THE TEAM"), (588, "ONE MODEL")):
        body += card(x, 44, 264, 228, x == 588) + text(x + 16, 68, label, "lab")
    body += text(44, 99, "Predict quality before tasting", "hd")
    body += lines(44, 125, "Eleven laboratory measurements. Predict the panel’s quality score.", 30)
    body += text(44, 194, f"{reference['wine-quality']['rows_given']:,} wines given", "mono")
    body += text(44, 217, f"{reference['wine-quality']['rows_held_back']:,} wines held back", "mono")
    body += text(44, 248, "Hidden labels stay outside the run.", "tiny")
    for x, system in ((324, "multi"), (604, "baseline")):
        row = wine[system]
        body += '<g class="reveal">' if system == "multi" else '<g class="late">'
        body += text(x, 111, f"{row['score']['metric']:.4f}", "big")
        body += text(x, 132, "mean absolute error · lower is better", "tiny")
        body += text(x, 167, f"{row['model_calls']} model calls · {row['seconds']:.0f} s", "mono")
        body += text(x, 192, f"{row['shadow_usd']:.4f} shadow $", "mono")
        body += text(x, 241, "✓ PASSED THE HIDDEN CHECK", "lab ok") + "</g>"
    body += text(
        28,
        302,
        "Both built a working model. The team used eight times the calls for a similar error.",
        "body",
    )
    body += text(
        28,
        324,
        "One run per system here. The controlled comparison below uses eight held-out tasks and three seeds.",
        "tiny",
    )
    svg(
        "hero", "One wine-quality task: team and single-agent results from the recorded field test", 344, body
    )

    stages = [
        (
            "01 · PROFILE",
            "Read the task",
            "Data dictionary, row counts, columns and memory.",
            "init → RunState",
        ),
        ("02 · PLAN", "Make a plan", "Ordered steps, acceptance checks and a typed reply.", "plan → Plan"),
        (
            "03 · EXECUTE",
            "Write one step",
            "Run Python with Linux isolation and resource limits.",
            "execute → files",
        ),
        (
            "04 · CHECK",
            "Inspect the files",
            "Tools inspect artifacts. Critic judges findings.",
            "validate → critic",
        ),
        (
            "05 · DELIVER",
            "Run it clean",
            "Assemble the steps. Smoke-test predictions.",
            "deliver → pipeline.py",
        ),
    ]
    body = text(28, 28, "EACH STEP LEAVES EVIDENCE THE NEXT ONE CAN INSPECT", "lab")
    for i, (label, title, detail, artifact) in enumerate(stages):
        x = 28 + 168 * i
        body += f'<g class="reveal" style="animation-delay:{i * 0.45}s">'
        body += card(x, 44, 152, 132, i == 4)
        body += text(x + 12, 66, label, "lab blue") + text(x + 12, 91, title, "hd")
        body += lines(x + 12, 114, detail, 22, "tiny") + text(x + 12, 196, artifact, "tiny")
        if i < 4:
            body += text(x + 155, 116, "→", "body")
        body += "</g>"
    for x, name, detail in [
        (28, "revise", "Repair → execute; escalate → plan"),
        (308, "advance", "Accepted → next step or deliver"),
        (588, "human", "Low confidence / exhausted re-plans"),
    ]:
        body += card(x, 218, 264, 65) + text(x + 12, 241, name, "mono blue")
        body += text(x + 12, 264, detail, "tiny")
    body += text(
        28,
        312,
        "Logic failure → revise. Infrastructure failure → retry. Delivery failure → revise. "
        "Every node checkpoints its state.",
        "tiny",
    )
    body += text(28, 334, "OUTSIDE THE GRAPH", "lab blue")
    body += text(
        174, 334, "Benchmark checker → hidden labels and scores. New uploads have no hidden checker.", "tiny"
    )
    svg(
        "pipeline",
        "All nine graph nodes, grouped by stage; benchmark scoring runs outside the graph",
        354,
        body,
    )

    body = text(28, 28, "THE CONTROLLED TEST · EIGHT TASKS × THREE SEEDS", "lab")
    for i, key in enumerate(("baseline", "memall", "memepi", "memsem", "multi", "memproc")):
        row = arms[key]["success"]
        y = 64 + i * 42
        body += text(28, y + 14, LABELS[key], "body")
        body += (
            f'<rect class="bar{" other" if key != "baseline" else ""}" x="235" y="{y}" '
            f'width="{row["mean"] * 480}" height="22" rx="3"/>'
        )
        lo, hi = (235 + v * 480 for v in row["ci"])
        body += f'<path class="ci" d="M{lo} {y + 6}v10m0 -5H{hi}m0 -5v10"/>'
        body += text(735, y + 16, f"{row['mean']:.2f}", "mono")
    body += text(
        28,
        340,
        "Whiskers: 95% bootstrap intervals over tasks, then seeds. Comparisons are paired on task and seed.",
        "tiny",
    )
    svg("results", "Held-out success rates with 95% intervals, from the committed T11 summaries", 360, body)

    body = text(28, 28, "MEMORY HAS THREE DIFFERENT JOBS", "lab")
    for x, title, detail, route in [
        (
            28,
            "Semantic facts",
            "File name + first line identify the dataset. Retrieve facts about its columns and units.",
            "init → Planner + Executor",
        ),
        (
            308,
            "Procedural skills",
            "Mine accepted steps from passing dev runs. Retrieve by similarity to the current step.",
            "execute → Executor",
        ),
        (
            588,
            "Past episodes",
            "Retrieve similar goals and profiles. Exclude episodes from this same task.",
            "init → Planner",
        ),
    ]:
        body += card(x, 48, 264, 175) + text(x + 16, 76, title, "hd")
        body += lines(x + 16, 104, detail, 33)
        body += '<g class="late">' + text(x + 16, 201, route, "mono blue") + "</g>"
    delta = comparisons["memall_multi"]
    body += text(
        28,
        252,
        f"All stores: {arms['memall']['success']['mean']:.2f} vs "
        f"{arms['multi']['success']['mean']:.2f} without memory. Paired "
        f"{delta['difference']:+.2f} [{delta['ci'][0]:.2f}, {delta['ci'][1]:.2f}] includes zero.",
        "tiny",
    )
    svg("memory", "Three separate memory stores with different retrieval rules and recipients", 274, body)

    confusion = trust["confusion"]
    body = text(28, 28, "CHECK THE CRITIC BEFORE TRUSTING ITS VERDICTS", "lab")
    for x, title, key, color in [
        (28, "Agreed with labels", None, "blue"),
        (308, "Too strict", "ok->not_ok", "bad"),
        (588, "Too lenient", "not_ok->ok", "bad"),
    ]:
        count = confusion[key] if key else confusion["ok->ok"] + confusion["not_ok->not_ok"]
        body += card(x, 48, 264, 128) + text(x + 16, 78, title, "hd")
        body += '<g class="reveal">' + text(x + 16, 126, count, f"big {color}") + "</g>"
        body += text(x + 75, 126, f"of {trust['labelled']} reviewed verdicts", "tiny")
    low, high = trust["kappa_ci"]
    body += text(
        28,
        210,
        f"Cohen’s κ {trust['kappa']:.2f} [{low:.2f}, {high:.2f}]. "
        "One labeller; stratified sample, not a random population estimate.",
        "body",
    )
    body += text(
        28,
        236,
        "A confident Critic is not ground truth. "
        "The hidden checker and the saved artifacts are separate evidence.",
        "tiny",
    )
    svg(
        "trust",
        "Critic audit: agreement, overly strict and overly lenient verdicts from reference labels",
        258,
        body,
    )

    body = text(28, 28, "FRESH DATA · THREE TASKS NEITHER SYSTEM WAS TUNED ON", "lab")
    for i, task in enumerate(dict.fromkeys(r["task"] for r in field)):
        rows = {r["system"]: r for r in field if r["task"] == task}
        x = 28 + i * 280
        body += card(x, 48, 264, 174) + text(x + 16, 80, task, "hd")
        for j, system in enumerate(("multi", "baseline")):
            row = rows[system]
            label = "Team" if system == "multi" else "Single"
            body += text(
                x + 16, 115 + j * 38, f"{label}: {row['model_calls']} calls · {row['seconds']:.0f} s", "mono"
            )
        body += '<g class="late">' + text(x + 16, 198, "✓ BOTH PASSED", "lab ok") + "</g>"
    body += text(
        28,
        250,
        "Six runs, six passes. A sanity check on unseen datasets, not evidence of a universal success rate.",
        "tiny",
    )
    svg(
        "field",
        "Six recorded field runs: team and single-agent calls and latency on three unseen datasets",
        272,
        body,
    )


def main() -> None:
    arms = {}
    comparisons = {}
    for grid in ("t11-test-core", "t11-test-stores"):
        summary = read(f"outputs/runs/{grid}/summary.json")
        arms.update(summary["arms"])
        comparisons.update(
            {f"{row['a']}_{row['b']}": row for row in summary["comparisons"] if row["metric"] == "success"}
        )
    field = read("outputs/field/summary.json")
    trust = read("outputs/trust/t10/scores.json")["reference"]
    reference = read("outputs/field/reference.json")
    figures(arms, field, trust, reference, comparisons)
    for asset in ASSETS.glob("*.svg"):
        shutil.copyfile(asset, ROOT / "site/assets" / asset.name)
    env = Environment(loader=FileSystemLoader(ROOT / "site/templates"), undefined=StrictUndefined)
    context = {
        "arms": arms,
        "field": field,
        "trust": trust,
        "labels": LABELS,
        "repo": REPO,
        "comparisons": comparisons,
    }
    (ROOT / "README.md").write_text(env.get_template("README.md.j2").render(**context) + "\n")
    tree = []
    for path in sorted((ROOT / "src/pipeline_agents").rglob("*.py")):
        if path.name != "__init__.py":
            tree.append({"path": str(path.relative_to(ROOT)), "lines": len(path.read_text().splitlines())})
    pages = [
        ("index.html", "Field guide", "guide.md.j2"),
        ("field-test/index.html", "Field test", "field.md.j2"),
        ("tour/index.html", "One run, start to finish", "tour.md.j2"),
    ]
    for destination, title, template in pages:
        prefix = "../" if "/" in destination else ""
        source = env.get_template(template).render(**context, prefix=prefix, tree=tree)
        if template == "tour.md.j2":
            source += "\n\n" + (ROOT / "docs/tour.md").read_text().split("\n", 1)[1]
        if template == "field.md.j2":
            source += "\n\n" + (ROOT / "docs/field_test.md").read_text().split("\n", 1)[1]
        # Research documents keep relative source references; public pages link back to GitHub.
        source = re.sub(r"\]\((docs/[^)]+)\)", rf"]({REPO}/blob/main/\1)", source)
        content = markdown.markdown(source, extensions=["tables", "fenced_code", "toc", "md_in_html"])
        content = re.sub(r"(<table>.*?</table>)", r'<div class="table-wrap">\1</div>', content, flags=re.S)
        output = ROOT / "site" / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        page = env.get_template("page.html.j2").render(
            title=title,
            content=content,
            prefix=prefix,
            repo=REPO,
        )
        output.write_text("\n".join(line.rstrip() for line in page.splitlines()) + "\n")
    print("Built README, six SVGs, field guide, field test and recorded tour.")


if __name__ == "__main__":
    main()
