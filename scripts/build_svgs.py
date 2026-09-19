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
    """The graph as it runs: a step is executed, checked, rejected, fixed, accepted, then delivered."""
    boxes = [
        ("Planner", 40, 40, "writes the ordered plan"),
        ("Executor", 250, 40, "writes one step's script"),
        ("Sandbox", 460, 40, "runs it, no network"),
        ("Tools", 460, 150, "check the files it wrote"),
        ("Critic", 250, 150, "accept / revise / escalate"),
        ("Reviser", 40, 150, "what to change, or re-plan"),
        ("Deliver", 250, 260, "assemble, re-run clean, smoke test"),
    ]
    width, height = 700, 340
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="The agent loop: planner, executor, sandbox, tools, '
        'critic, reviser, deliver">',
        "<style>",
        f".box{{fill:#fff;stroke:{LINE};stroke-width:1.5;rx:10}}",
        f".name{{font:600 15px {FONT};fill:{INK}}}",
        f".sub{{font:12px {FONT};fill:{MUTED}}}",
        f".edge{{fill:none;stroke:{LINE};stroke-width:2}}",
        f".edge.revise{{stroke:{BASE};stroke-dasharray:5 4}}",
        f".label{{font:11px {FONT};fill:{MUTED}}}",
        f".label.revise{{fill:{BASE}}}",
        f".token{{fill:{TEAM}}}",
        "@keyframes travel{"
        "0%{transform:translate(150px,70px)} 12%{transform:translate(340px,70px)}"
        "24%{transform:translate(550px,70px)} 36%{transform:translate(550px,180px)}"
        "48%{transform:translate(340px,180px)} 58%{transform:translate(150px,180px)}"
        "68%{transform:translate(340px,70px)} 80%{transform:translate(340px,180px)}"
        "92%{transform:translate(340px,290px)} 100%{transform:translate(340px,290px)}}",
        ".token{animation:travel 9s ease-in-out infinite}",
        "@media (prefers-reduced-motion: reduce){.token{animation:none;transform:translate(340px,180px)}}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#f5f8f6"/>',
    ]
    for name, x, y, sub in boxes:
        parts += [
            f'<rect class="box" x="{x}" y="{y}" width="200" height="60"/>',
            f'<text class="name" x="{x + 16}" y="{y + 26}">{name}</text>',
            f'<text class="sub" x="{x + 16}" y="{y + 45}">{sub}</text>',
        ]
    edges = [
        ("M240 70 H250", ""),
        ("M450 70 H460", ""),
        ("M560 100 V150", "writes files"),
        ("M450 180 H460", ""),
        ("M240 180 H250", ""),
        ("M140 150 V100", "fix, up to 2"),
        ("M350 150 V100", "accept: next step"),
        ("M350 210 V260", "last step done"),
    ]
    for path, label in edges:
        css = "edge revise" if label.startswith("fix") or label == "rejected" else "edge"
        parts.append(f'<path class="{css}" d="{path}" marker-end="url(#arrow)"/>')
        if label:
            x, y = _label_position(path)
            klass = "label revise" if css == "edge revise" else "label"
            parts.append(f'<text class="{klass}" x="{x}" y="{y}">{label}</text>')
    parts += [
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{LINE}"/></marker></defs>',
        '<circle class="token" r="7" cx="0" cy="0"/>',
        f'<text class="sub" x="40" y="{height - 16}">One step at a time. Two revisions, then the Planner; '
        "one re-plan, then a human.</text>",
        "</svg>",
    ]
    return "\n".join(parts)


def _label_position(path: str) -> tuple[int, int]:
    """A label just off the middle of a straight edge."""
    numbers = [int(n) for n in path.replace("M", " ").replace("H", " ").replace("V", " ").split()]
    if "H" in path:
        x0, y0, x1 = numbers[0], numbers[1], numbers[2]
        return (x0 + x1) // 2 - 24, y0 - 8
    x0, y0, y1 = numbers[0], numbers[1], numbers[2]
    return x0 + 8, (y0 + y1) // 2


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
