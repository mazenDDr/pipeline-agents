"""Build the README's two SVGs (T14): the loop, and the measured result.

    python scripts/build_svgs.py

GitHub renders an SVG in an <img>: no JavaScript and no page CSS, but CSS inside the file works, so both
files are self-contained and animate with keyframes. They hold at a readable resting state, use system fonts
only, and stop animating when the reader asks for reduced motion.

The result chart is drawn from outputs/runs/t11-test-core/summary.json and the stores grid: no number in it
is typed here.
"""

import json
from pathlib import Path

IMG = Path("docs/img")
CORE = Path("outputs/runs/t11-test-core/summary.json")
STORES = Path("outputs/runs/t11-test-stores/summary.json")

ARM_LABELS = {
    "baseline": "one model, whole pipeline",
    "memall": "agent team, all 3 memories",
    "memepi": "agent team, episodic only",
    "memsem": "agent team, semantic only",
    "multi": "agent team, no memory",
    "memproc": "agent team, procedural only",
}

FONT = "ui-sans-serif, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
INK = "#1d2b24"
MUTED = "#5d6f66"
LINE = "#d2ddd6"
TEAM = "#2f6f57"
BASE = "#b4632c"


def loop_svg() -> str:
    """The real LangGraph topology, plus the hidden checker that wraps benchmark runs."""
    boxes = [
        ("init", 24, 58, "profile + run memory", "plain"),
        ("plan", 178, 58, "Planner → Plan", "model"),
        ("execute", 332, 58, "Executor → script", "model"),
        ("validate", 486, 58, "deterministic tools", "plain"),
        ("critic", 640, 58, "Critic → Verdict", "model"),
        ("advance", 794, 58, "accept step", "plain"),
        ("deliver", 640, 190, "assemble + clean run", "plain"),
        ("END", 794, 190, "status: succeeded", "success"),
        ("revise", 332, 306, "Reviser → Revision", "model"),
        ("human", 640, 306, "needs_human", "warn"),
        ("checker", 794, 306, "hidden holdout", "outer"),
    ]
    width, height = 940, 444
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="The exact pipeline-agents graph: init, plan, execute, '
        'validate, critic, advance, deliver, revise and human, followed by the benchmark checker">',
        "<style>",
        f".box{{fill:#fff;stroke:{LINE};stroke-width:1.5}}",
        f".box.model,.box.success{{fill:#edf6f1;stroke:{TEAM}}}",
        f".box.warn{{fill:#fff5ed;stroke:{BASE}}}",
        f".box.outer{{fill:none;stroke:{MUTED};stroke-dasharray:5 4}}",
        f".name{{font:600 14px {FONT};fill:{INK}}}",
        f".sub{{font:11px {FONT};fill:{MUTED}}}",
        f".edge{{fill:none;stroke:{LINE};stroke-width:2;marker-end:url(#arrow)}}",
        f".edge.loop{{stroke:{BASE};stroke-dasharray:5 4;marker-end:url(#arrow-warn)}}",
        f".edge.outer{{stroke:{MUTED};stroke-dasharray:5 4;marker-end:url(#arrow-muted)}}",
        f".label{{font:10px {FONT};fill:{MUTED};paint-order:stroke;stroke:#f5f8f6;stroke-width:4px}}",
        f".label.loop{{fill:{BASE}}}",
        f".section{{font:600 11px {FONT};fill:{MUTED};letter-spacing:.08em}}",
        f".token{{fill:{TEAM}}}",
        "@keyframes travel{"
        "0%{transform:translate(85px,88px)} 12%{transform:translate(239px,88px)}"
        "24%{transform:translate(393px,88px)} 36%{transform:translate(547px,88px)}"
        "48%{transform:translate(701px,88px)} 60%{transform:translate(855px,88px)}"
        "72%{transform:translate(701px,220px)} 84%{transform:translate(855px,220px)}"
        "100%{transform:translate(855px,336px)}}",
        ".token{animation:travel 10s ease-in-out infinite}",
        "@media (prefers-reduced-motion: reduce){.token{animation:none;transform:translate(855px,336px)}}",
        "</style>",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{LINE}"/></marker>'
        '<marker id="arrow-warn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{BASE}"/></marker>'
        '<marker id="arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker></defs>',
        f'<rect width="{width}" height="{height}" fill="#f5f8f6"/>',
        '<text class="section" x="24" y="28">LANGGRAPH — src/pipeline_agents/graph/build.py</text>',
        '<text class="section" x="794" y="290">BENCHMARK WRAPPER</text>',
    ]
    for name, x, y, sub, css in boxes:
        parts += [
            f'<rect class="box {css}" x="{x}" y="{y}" width="122" height="60" rx="9"/>',
            f'<text class="name" x="{x + 12}" y="{y + 24}">{name}</text>',
            f'<text class="sub" x="{x + 12}" y="{y + 43}" textLength="98" '
            f'lengthAdjust="spacingAndGlyphs">{sub}</text>',
        ]
    edges = [
        ("M146 88 H178", "", "edge"),
        ("M300 88 H332", "", "edge"),
        ("M454 88 H486", "", "edge"),
        ("M608 88 H640", "", "edge"),
        ("M762 88 H794", "accept", "edge"),
        ("M855 118 V150 H393 V118", "next step", "edge"),
        ("M855 118 V220 H762", "plan complete", "edge"),
        ("M762 220 H794", "checks pass", "edge"),
        ("M855 250 V306", "benchmark runs only", "edge outer"),
        ("M393 118 V306", "script crash", "edge loop"),
        ("M701 118 V150 H454 V306", "revise / escalate", "edge loop"),
        ("M721 118 V306", "low confidence", "edge loop"),
        ("M640 220 H547 V336 H454", "delivery fails", "edge loop"),
        ("M393 306 V250 H300 V118", "escalate → re-plan", "edge loop"),
        ("M332 336 H270 V150 H393 V118", "fix", "edge loop"),
        ("M454 336 H640", "re-plans used", "edge loop"),
        ("M762 336 H794", "", "edge outer"),
    ]
    labels = {
        "accept": (766, 79),
        "next step": (564, 143),
        "plan complete": (789, 174),
        "checks pass": (766, 211),
        "benchmark runs only": (803, 271),
        "script crash": (399, 212),
        "revise / escalate": (533, 172),
        "low confidence": (726, 172),
        "delivery fails": (513, 326),
        "escalate → re-plan": (210, 229),
        "fix": (275, 325),
        "re-plans used": (508, 327),
    }
    for path, label, css in edges:
        parts.append(f'<path class="{css}" d="{path}"/>')
        if label:
            x, y = labels[label]
            klass = "label loop" if "loop" in css else "label"
            parts.append(f'<text class="{klass}" x="{x}" y="{y}">{label}</text>')
    parts += [
        '<circle class="token" r="6" cx="0" cy="0"/>',
        f'<text class="sub" x="24" y="{height - 42}">Green fills call a model; a green outline marks '
        "success. Plain nodes are deterministic. Orange paths recover or stop.</text>",
        f'<text class="sub" x="24" y="{height - 20}">Every graph node checkpoints RunState to SQLite; '
        "the Observer wraps each model call. The hidden checker never enters the agent workspace.</text>",
        "</svg>",
    ]
    return "\n".join(parts)


def results_svg() -> str:
    """Success on the test split, per arm, with the bootstrap interval. Real numbers, from the summaries."""
    arms: dict[str, dict] = {}
    for path in (CORE, STORES):
        arms.update(json.loads(path.read_text())["arms"])
    rows = sorted(
        ((arm, entry["success"]["mean"], entry["success"]["ci"]) for arm, entry in arms.items()),
        key=lambda row: -row[1],
    )
    width, left, row_height = 720, 250, 44
    height = 70 + row_height * len(rows) + 40
    scale = width - left - 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="Success rate per arm on the test split">',
        "<style>",
        f".title{{font:600 16px {FONT};fill:{INK}}}",
        f".arm{{font:13px {FONT};fill:{INK}}}",
        f".note{{font:12px {FONT};fill:{MUTED}}}",
        f".value{{font:600 13px {FONT};fill:{INK}}}",
        f".axis{{stroke:{LINE};stroke-width:1}}",
        f".bar{{fill:{TEAM};transform-origin:{left}px 0}}",
        f".bar.baseline{{fill:{BASE}}}",
        f".ci{{stroke:{MUTED};stroke-width:1.5;opacity:.65}}",
        "@keyframes grow{from{transform:scaleX(0)} to{transform:scaleX(1)}}",
        ".bar{animation:grow 1.1s cubic-bezier(.2,.7,.3,1) both}",
        "@media (prefers-reduced-motion: reduce){.bar{animation:none}}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#f5f8f6"/>',
        '<text class="title" x="24" y="32">Success on the 8 test tasks, 3 seeds each</text>',
    ]
    for index, (arm, mean, (low, high)) in enumerate(rows):
        y = 64 + index * row_height
        bar = max(2, int(mean * scale))
        css = "bar baseline" if arm == "baseline" else "bar"
        delay = 0.08 * index
        parts += [
            f'<text class="arm" x="24" y="{y + 18}">{ARM_LABELS.get(arm, arm)}</text>',
            f'<rect class="{css}" x="{left}" y="{y + 4}" width="{bar}" height="20" rx="3" '
            f'style="animation-delay:{delay:.2f}s"/>',
            f'<line class="ci" x1="{left + int(low * scale)}" y1="{y + 14}" '
            f'x2="{left + int(high * scale)}" y2="{y + 14}"/>',
            f'<line class="ci" x1="{left + int(low * scale)}" y1="{y + 8}" '
            f'x2="{left + int(low * scale)}" y2="{y + 20}"/>',
            f'<line class="ci" x1="{left + int(high * scale)}" y1="{y + 8}" '
            f'x2="{left + int(high * scale)}" y2="{y + 20}"/>',
            f'<text class="value" x="{left + int(high * scale) + 8}" y="{y + 19}">{mean:.2f}</text>',
        ]
    baseline_y = 64 + row_height * len(rows)
    parts += [
        f'<line class="axis" x1="{left}" y1="56" x2="{left}" y2="{baseline_y}"/>',
        f'<text class="note" x="24" y="{baseline_y + 24}">Bars are the mean over tasks of the mean over '
        "seeds; the whiskers are 95% bootstrap intervals (tasks, then seeds).</text>",
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    (IMG / "loop.svg").write_text(loop_svg())
    (IMG / "results.svg").write_text(results_svg())
    print(f"wrote {IMG}/loop.svg and {IMG}/results.svg")


if __name__ == "__main__":
    main()
