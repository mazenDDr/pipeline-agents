"""Grid runner, statistics and report: on known inputs, with a fake cell runner (no models)."""

import json
from pathlib import Path

import pytest

from pipeline_agents.bench.grid import Arm, Grid, cells, estimate_seconds, run_grid
from pipeline_agents.eval.report import render, summarize
from pipeline_agents.eval.stats import is_real, nested_bootstrap, paired_difference

ARMS = [
    Arm(label="multi", system="multi", config="configs/run/default.yaml"),
    Arm(label="baseline", system="baseline", config="configs/run/default.yaml"),
]


# --- statistics ------------------------------------------------------------------------------


def test_nested_bootstrap_on_constant_and_varied_values() -> None:
    assert nested_bootstrap({"a": [1, 1], "b": [1, 1]}) == (1.0, 1.0, 1.0)
    mean, lo, hi = nested_bootstrap({"a": [1, 1, 1], "b": [0, 0, 0], "c": [1, 0, 1]})
    assert mean == pytest.approx((1 + 0 + 2 / 3) / 3) and lo < mean < hi


def test_task_is_the_unit_not_the_run() -> None:
    """Three seeds of one easy task are not three independent successes: the interval stays wide."""
    _, lo, hi = nested_bootstrap({"easy": [1, 1, 1], "hard": [0, 0, 0]})
    assert lo == 0.0 and hi == 1.0


def test_paired_difference_uses_only_shared_cells() -> None:
    a = {"t1": {1: 1.0, 2: 1.0}, "t2": {1: 1.0}}
    b = {"t1": {1: 0.0, 2: 0.0}, "t2": {2: 0.0}}  # t2 shares no seed
    mean, lo, hi, pairs = paired_difference(a, b)
    assert pairs == 2 and mean == 1.0 and is_real(lo, hi)
    assert not is_real(-0.1, 0.2)


# --- grid ------------------------------------------------------------------------------------


def test_cells_order_seeds_then_arms_then_warmup_before_followup() -> None:
    grid = Grid(name="g", tasks=["bike-2-weather", "bike-1-forecast"], seeds=[1, 2], arms=ARMS)
    ids = [c.run_id for c in cells(grid)]
    assert ids[:4] == [
        "multi_bike-1-forecast_s1",
        "multi_bike-2-weather_s1",
        "baseline_bike-1-forecast_s1",
        "baseline_bike-2-weather_s1",
    ]
    assert len(ids) == 8 and ids[4].endswith("_s2")
    assert Grid(name="g", tasks="dev", seeds=[1], arms=ARMS[:1]).task_ids() == [
        "bike-1-forecast",
        "bike-2-weather",
        "adult-1-income",
        "adult-2-education",
        "air-1-co-daily",
        "air-2-co-estimate",
    ]


class FakeRunner:
    def __init__(self, crash_on: str | None = None) -> None:
        self.calls: list[tuple] = []
        self.crash_on = crash_on

    def __call__(self, task_id, system, cfg, run_dir, memory_dir, record_memory):
        self.calls.append((task_id, system, cfg.seed, memory_dir, record_memory))
        if task_id == self.crash_on:
            raise RuntimeError("boom")
        run_dir.mkdir(parents=True)
        result = {
            "run_id": run_dir.name,
            "task": task_id,
            "system": system,
            "seed": cfg.seed,
            "kind": "analytical",
            "status": "succeeded",
            "success": system == "multi",
            "revisions": 1,
            "model_calls": 5,
            "shadow_usd": 0.001,
            "seconds": 10.0,
            "checker": {"first_failure": None if system == "multi" else "answer"},
        }
        (run_dir / "result.json").write_text(json.dumps(result))
        return result


def _grid(**kw) -> Grid:
    return Grid(
        **{"name": "g", "tasks": ["bike-2-weather", "adult-2-education"], "seeds": [1], "arms": ARMS, **kw}
    )


def test_grid_resumes_and_records_crashes(tmp_path: Path) -> None:
    runner = FakeRunner(crash_on="adult-2-education")
    results = run_grid(
        _grid(), root=tmp_path, runner=runner, wait_for_servers=lambda: True, log=lambda _: None
    )
    assert [r["status"] for r in results] == ["succeeded", "error", "succeeded", "error"]
    assert (
        "boom"
        in json.loads((tmp_path / "g" / "multi_adult-2-education_s1" / "result.json").read_text())[
            "stop_reason"
        ]
    )
    again = FakeRunner()
    run_grid(_grid(), root=tmp_path, runner=again, wait_for_servers=lambda: True, log=lambda _: None)
    assert again.calls == []  # errors count as done unless retried
    run_grid(
        _grid(),
        root=tmp_path,
        runner=again,
        retry_errors=True,
        wait_for_servers=lambda: True,
        log=lambda _: None,
    )
    assert [c[0] for c in again.calls] == ["adult-2-education", "adult-2-education"]


def test_seeds_and_memory_directories(tmp_path: Path) -> None:
    runner = FakeRunner()
    mined = tmp_path / "mined"
    mined.mkdir()
    (mined / "procedural.jsonl").write_text('{"id": "k1"}\n')
    run_grid(
        _grid(seeds=[4, 5], memory="sequence", seed_memory=str(mined)),
        root=tmp_path,
        runner=runner,
        wait_for_servers=lambda: True,
        log=lambda _: None,
    )
    assert {c[2] for c in runner.calls} == {4, 5}
    dirs = {(c[1], c[2]): c[3] for c in runner.calls}
    assert dirs[("multi", 4)] == tmp_path / "g" / "memory" / "multi_s4" != dirs[("baseline", 4)]
    assert all(c[4] for c in runner.calls)
    for directory in set(dirs.values()):  # dev-mined skills are seeded, facts and episodes start empty
        assert (directory / "procedural.jsonl").read_text() == '{"id": "k1"}\n'
        assert not (directory / "semantic.jsonl").exists()


def test_grid_waits_for_servers_then_gives_up(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("pipeline_agents.bench.grid.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="not answering"):
        run_grid(
            _grid(),
            root=tmp_path,
            runner=FakeRunner(),
            wait_for_servers=lambda: False,
            max_wait_s=60,
            log=lambda _: None,
        )


def test_estimate_uses_past_runs_by_system_and_kind() -> None:
    grid = Grid(name="g", tasks=["bike-2-weather", "bike-1-forecast"], seeds=[1], arms=ARMS[:1])
    history = [
        {"system": "multi", "kind": "analytical", "seconds": 100},
        {"system": "multi", "kind": "predictive", "seconds": 700},
    ]
    assert estimate_seconds(cells(grid), history) == 800


# --- seeds in model calls ----------------------------------------------------------------------


def test_each_call_gets_a_seed_from_the_run_seed() -> None:
    from pipeline_agents.agents.common import Deps, call_role
    from pipeline_agents.config import RunConfig
    from pipeline_agents.harness.budget import Price
    from pipeline_agents.harness.client import FakeModelClient
    from pipeline_agents.harness.observer import Observer
    from pipeline_agents.harness.registry import PromptRegistry
    from pipeline_agents.schemas import Ledger

    def seeds_for(seed: int | None, tmp: Path) -> list:
        client = FakeModelClient(['{"facts": []}'] * 2, "m")
        price = Price(input_per_mtok=1, output_per_mtok=1, source="t", retrieved="2026-09-17")
        observer = Observer("r", tmp, Ledger(cap_usd=1), {"strong": client, "cheap": client}, {"m": price})
        deps = Deps(
            observer=observer,
            registry=PromptRegistry("prompts"),
            config=RunConfig(name="t", seed=seed),
            runner=None,
        )
        context = {"files": "", "data_readme": "", "rationale": "", "steps": [], "outcome": ""}
        for _ in range(2):
            call_role(deps, "memory_writer", context, lambda t: t, "low", None, 0)
        return [params.get("seed") for _, params in client.calls]

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        one, two, none = (
            seeds_for(1, Path(tmp) / "a"),
            seeds_for(2, Path(tmp) / "b"),
            seeds_for(None, Path(tmp) / "c"),
        )
    assert one[0] != one[1] and one != two and none == [None, None]


# --- report ----------------------------------------------------------------------------------


def test_report_summarises_arms_tasks_and_pairs(tmp_path: Path) -> None:
    results = run_grid(
        _grid(seeds=[1, 2]),
        root=tmp_path,
        runner=FakeRunner(),
        wait_for_servers=lambda: True,
        log=lambda _: None,
    )
    summary = summarize(results)
    assert (
        summary["arms"]["multi"]["success"]["mean"] == 1.0
        and summary["arms"]["baseline"]["success"]["mean"] == 0.0
    )
    success = next(c for c in summary["comparisons"] if c["metric"] == "success")
    assert (success["a"], success["b"], success["difference"], success["pairs"]) == (
        "multi",
        "baseline",
        1.0,
        4,
    )
    assert summary["tasks"]["bike-2-weather"]["baseline"] == {
        "success": 0,
        "cells": 2,
        "failures": {"answer": 2},
    }
    text = render(summary, "g")
    assert "| multi | 4 |" in text and "answer 2" in text


def test_earlier_grid_arms_join_under_a_prefix(tmp_path: Path) -> None:
    from pipeline_agents.eval.report import load_results

    for name in ("old", "new"):
        run_grid(
            _grid(name=name),
            root=tmp_path,
            runner=FakeRunner(),
            wait_for_servers=lambda: True,
            log=lambda _: None,
        )
    results = load_results(tmp_path / "new") + load_results(tmp_path / "old", "t9-")
    summary = summarize(results)
    assert list(summary["arms"]) == ["baseline", "multi", "t9-baseline", "t9-multi"]
    pair = next(
        c for c in summary["comparisons"] if (c["a"], c["b"], c["metric"]) == ("multi", "t9-multi", "success")
    )
    assert pair["pairs"] == 2 and pair["difference"] == 0.0
