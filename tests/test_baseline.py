"""The single-agent baseline: one reply per attempt, the same delivery checks, bounded repair attempts."""

import json
from pathlib import Path

from pipeline_agents.agents.baseline import attempts_allowed, parse_reply, run_baseline
from pipeline_agents.agents.common import Deps
from pipeline_agents.config import LoopConfig, RoleConfig, RunConfig
from pipeline_agents.harness.budget import Price
from pipeline_agents.harness.client import FakeModelClient
from pipeline_agents.harness.observer import Observer
from pipeline_agents.harness.registry import PromptRegistry
from pipeline_agents.sandbox.runner import Limits, UnsandboxedRunner
from pipeline_agents.schemas import Ledger, TaskContext

PRICE = Price(input_per_mtok=0.1, output_per_mtok=0.3, source="test", retrieved="2026-09-17")

GOOD = """PLAN:
1. read sales, mark -999 as missing
2. write the mean

TARGET: none

### output/pipeline.py
```python
import json, pandas as pd
df = pd.read_csv("data/sales.csv")
amount = df["amount"].where(df["amount"] != -999)
json.dump({"mean_amount": float(amount.mean())}, open("output/answer.json", "w"))
print("rows", len(df))
```

SELF-CHECK:
- sentinel -999 handled"""
CRASH = GOOD.replace('df["amount"]', 'df["amt"]', 1)
PRINTS_AND_EXITS = GOOD.replace(
    'print("rows", len(df))', 'print("found no amounts to average"); raise SystemExit(2)'
)

PREDICTIVE = """PLAN:
1. fit the mean amount

TARGET: amount

### output/pipeline.py
```python
import json, pandas as pd
df = pd.read_csv("data/sales.csv")
json.dump({"mean": float(df["amount"].mean())}, open("output/model.json", "w"))
json.dump({"validation_mae": 5.0}, open("output/metrics.json", "w"))
```

### output/predict.py
```python
import json, sys, pandas as pd
X = pd.read_csv(sys.argv[1])
assert "amount" not in X.columns, "features must not contain the target"
mean = json.load(open("output/model.json"))["mean"]
pd.DataFrame({"order_id": X["order_id"], "prediction": mean}).to_csv(sys.argv[2], index=False)
```

SELF-CHECK:
- no target in features"""


def _setup(
    tmp_path: Path,
    replies: list,
    kind: str = "analytical",
    budget: float = 10.0,
    loop: LoopConfig | None = None,
) -> tuple[Deps, TaskContext, Path, FakeModelClient]:
    ws = tmp_path / "workspace"
    (ws / "data").mkdir(parents=True)
    (ws / "output").mkdir()
    (ws / "data" / "sales.csv").write_text(
        "order_id,amount\n" + "".join(f"{i},{-999 if i % 5 == 0 else 10 + i}\n" for i in range(1, 41))
    )
    (ws / "data" / "README.md").write_text("amount in dollars; -999 means not recorded")
    client = FakeModelClient(replies, model="m")
    cfg = RunConfig(
        name="t", budget_usd=budget, loop=loop or LoopConfig(), roles={"baseline": RoleConfig(max_tokens=512)}
    )
    observer = Observer(
        "b1", tmp_path / "run", Ledger(cap_usd=budget), {"strong": client, "cheap": client}, {"m": PRICE}
    )
    deps = Deps(
        observer=observer,
        registry=PromptRegistry("prompts"),
        config=cfg,
        runner=UnsandboxedRunner(Limits(timeout_s=60)),
    )
    task = TaskContext(
        task_id="toy",
        task_md="Report the mean known amount in output/answer.json.",
        data_readme="amount in dollars; -999 means not recorded",
        kind=kind,
        id_column="order_id" if kind == "predictive" else None,
        metric="mae" if kind == "predictive" else None,
    )
    return deps, task, ws, client


def test_parse_reply() -> None:
    files, target, check = parse_reply(PREDICTIVE)
    assert set(files) == {"pipeline.py", "predict.py"} and target == "amount" and "no target" in check
    assert parse_reply(GOOD)[1] is None


def test_attempts_match_the_multi_agent_repair_allowance() -> None:
    assert attempts_allowed(LoopConfig(max_revisions_per_step=2, max_replans=1)) == 6
    assert attempts_allowed(LoopConfig(max_revisions_per_step=1, max_replans=0)) == 2


def test_passes_first_time(tmp_path: Path) -> None:
    deps, task, ws, client = _setup(tmp_path, [GOOD])
    result = run_baseline(deps, task, ws, "b1")
    assert result.status == "succeeded" and len(result.attempts) == 1 and result.model_calls == 1
    assert json.loads((ws / "output" / "answer.json").read_text())["mean_amount"] > 0


def test_repairs_with_the_error_it_saw(tmp_path: Path) -> None:
    deps, task, ws, client = _setup(tmp_path, [CRASH, PRINTS_AND_EXITS, GOOD])
    result = run_baseline(deps, task, ws, "b1")
    assert result.status == "succeeded" and [a.passed for a in result.attempts] == [False, False, True]
    second, third = client.calls[1][0][1]["content"], client.calls[2][0][1]["content"]
    assert "KeyError" in second and "did not pass" in second
    assert "found no amounts to average" in third  # printed to stdout, still shown


def test_attempts_run_out(tmp_path: Path) -> None:
    loop = LoopConfig(max_revisions_per_step=1, max_replans=0)
    deps, task, ws, client = _setup(tmp_path, [CRASH] * 5, loop=loop)
    result = run_baseline(deps, task, ws, "b1")
    assert (
        result.status == "failed"
        and len(result.attempts) == 2
        and "no passing attempt in 2" in result.stop_reason
    )


def test_unusable_reply_after_a_reask(tmp_path: Path) -> None:
    deps, task, ws, client = _setup(tmp_path, ["here is the code: print(1)", "still no headings"])
    result = run_baseline(deps, task, ws, "b1")
    assert result.status == "failed" and "unusable" in result.stop_reason and result.model_calls == 2


def test_budget_stops_it(tmp_path: Path) -> None:
    deps, task, ws, client = _setup(tmp_path, [CRASH] * 6, budget=0.00001)
    result = run_baseline(deps, task, ws, "b1")
    assert result.status == "failed" and result.stop_reason.startswith("budget:")


def test_predictive_smoke_uses_its_declared_target(tmp_path: Path) -> None:
    deps, task, ws, client = _setup(tmp_path, [PREDICTIVE], kind="predictive")
    result = run_baseline(deps, task, ws, "b1")
    assert result.status == "succeeded", [f.detail for f in result.attempts[-1].findings]
    assert [f.tool for f in result.attempts[0].findings] == ["clean_rerun", "deliverables", "predict_smoke"]
