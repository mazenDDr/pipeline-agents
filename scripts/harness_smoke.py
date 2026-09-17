"""End-to-end check of the harness on the real served models (T3). Run on the GPU machine with the tiered
layout serving (bash scripts/serve_models.sh tiered):

    python scripts/harness_smoke.py

1. Render planner v1 for bike-1-forecast and call the strong model (high stakes); parse the plan.
2. Push the ledger past degrade_at and make a low-stakes call: it must go to the cheap model.
3. Replay call 1 from the cache with no model available: same content, same cost.
Writes outputs/runs/t3-smoke/{calls.jsonl, replay/calls.jsonl, summary.json}. Lock the prompts it used
on the Mac afterwards: python scripts/lock_prompts.py outputs/runs/t3-smoke
"""

import json
import shutil
from pathlib import Path

from pipeline_agents.bench.families.bike import FORECAST
from pipeline_agents.config import build_observer, load_run_config
from pipeline_agents.harness.budget import record
from pipeline_agents.harness.client import FakeModelClient
from pipeline_agents.harness.registry import PromptRegistry
from pipeline_agents.harness.render import render
from pipeline_agents.schemas import Plan, Spend

run_dir = Path("outputs/runs/t3-smoke")
shutil.rmtree(run_dir, ignore_errors=True)
cfg = load_run_config("configs/run/default.yaml")
registry = PromptRegistry("prompts")
workspace = Path("data/benchmark") / FORECAST.id / "workspace"
head = "".join((workspace / "data" / "rentals_hourly.csv").open().readlines()[:6])
context = {
    "goal": FORECAST.goal,
    "data_readme": (workspace / "data" / "README.md").read_text(),
    "dataset_profile": f"rentals_hourly.csv, first lines:\n{head}",
    "memory_hits": [],
    "failed_plan": None,
    "escalation_reason": None,
}
planner = cfg.role("planner")
prompt = render(registry, "planner", context, planner.system_version, planner.user_version)
params = {"temperature": planner.temperature, "max_tokens": planner.max_tokens, "seed": 0}

obs = build_observer(cfg, "t3-smoke", run_dir, cache_dir=run_dir / "cache")
first = obs.call(prompt, planner.tier, "high", params)
print(
    f"1. strong: ok={first.ok} error={first.error} {first.output_tokens} tokens, ttft {first.ttft_s}s, "
    f"total {first.total_s:.1f}s"
)
text = first.content.strip().removeprefix("```json").removesuffix("```").strip()
try:
    plan = Plan.model_validate_json(text)
    plan_note = f"valid plan, {len(plan.steps)} steps: {[s.kind for s in plan.steps]}"
except ValueError as e:
    plan_note = f"plan did not validate: {str(e).splitlines()[0]}"
print("   ", plan_note)

record(obs.ledger, "test", None, "strong", Spend(shadow_usd=cfg.degrade_at * cfg.budget_usd))
second = obs.call(prompt, "strong", "low", {**params, "max_tokens": 64})
print(f"2. low stakes after degrade_at: model={second.model} ok={second.ok}")

replay = build_observer(
    cfg,
    "t3-smoke-replay",
    run_dir / "replay",
    cache_dir=run_dir / "cache",
    replay=True,
    clients={"strong": FakeModelClient([], "gemma-4-26b-a4b"), "cheap": FakeModelClient([], "gemma-4-e4b")},
)
again = replay.call(prompt, planner.tier, "high", params)
print(f"3. replay: cached={again.cached} same content={again.content == first.content}")

rows = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines()]
summary = {
    "calls": [
        {
            k: r[k]
            for k in (
                "role",
                "stakes",
                "tier",
                "degraded",
                "model",
                "input_tokens",
                "output_tokens",
                "ttft_s",
                "total_s",
                "shadow_usd",
                "error",
            )
        }
        for r in rows
    ],
    "plan": plan_note,
    "replay_cached": again.cached,
    "replay_same_content": again.content == first.content,
    "replay_shadow_usd": replay.ledger.total.shadow_usd,
    "original_first_call_shadow_usd": rows[0]["shadow_usd"],
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
