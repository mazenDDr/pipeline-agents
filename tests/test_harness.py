"""Harness core: prompt registry and locking, rendering, budget policy, the Observer, the streaming client."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from jinja2 import UndefinedError

from pipeline_agents.config import RunConfig, build_observer
from pipeline_agents.harness.budget import BudgetExceeded, Price, choose_tier, ensure_room, record, shadow_usd
from pipeline_agents.harness.client import Completion, FakeModelClient, OpenAICompatibleClient
from pipeline_agents.harness.observer import Observer
from pipeline_agents.harness.registry import PromptRegistry
from pipeline_agents.harness.render import render
from pipeline_agents.schemas import Ledger, Spend

PRICES = {
    "strong-model": Price(input_per_mtok=1.0, output_per_mtok=2.0, source="test", retrieved="2026-09-17"),
    "cheap-model": Price(input_per_mtok=0.1, output_per_mtok=0.2, source="test", retrieved="2026-09-17"),
}


def _registry(tmp_path: Path) -> PromptRegistry:
    (tmp_path / "planner").mkdir(parents=True)
    (tmp_path / "planner" / "system.v1.md").write_text("You plan.")
    (tmp_path / "planner" / "system.v2.md").write_text("You plan carefully.")
    (tmp_path / "planner" / "user.v1.j2").write_text(
        "Goal: {{ goal }}\n{% for hit in memory_hits %}- {{ hit.text }}\n{% endfor %}"
    )
    return PromptRegistry(tmp_path)


# --- registry and rendering ------------------------------------------------------------------


def test_registry_versions_and_latest(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert registry.versions("planner", "system") == [1, 2]
    assert registry.get("planner", "system").text == "You plan carefully."
    assert registry.get("planner", "system", 1).ref == "planner/system.v1"
    with pytest.raises(FileNotFoundError):
        registry.latest("critic", "system")


def test_locked_prompt_cannot_change(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.lock(registry.get("planner", "system", 1))
    registry.lock(registry.get("planner", "system", 1))  # same content: no-op
    assert registry.check_locked() == []
    (tmp_path / "planner" / "system.v1.md").write_text("You plan, edited in place.")
    assert registry.check_locked() == ["planner/system.v1"]
    with pytest.raises(ValueError, match="new version"):
        registry.lock(registry.get("planner", "system", 1))


def test_repository_prompts_match_their_lock() -> None:
    assert PromptRegistry("prompts").check_locked() == []


def test_render_keeps_template_and_instance_apart(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    prompt = render(
        registry, "planner", {"goal": "predict churn", "memory_hits": [{"text": "pdays 999 = never"}]}
    )
    assert prompt.refs == {"system": "planner/system.v2", "user": "planner/user.v1"}
    assert prompt.messages[1]["content"] == "Goal: predict churn\n- pdays 999 = never"
    assert "{{" in prompt.user.text  # the template itself is kept, unfilled
    assert prompt.context_keys == ["goal", "memory_hits"]


def test_render_is_strict(tmp_path: Path) -> None:
    with pytest.raises(UndefinedError):
        render(_registry(tmp_path), "planner", {"memory_hits": []})  # goal missing


def test_planner_v1_renders_first_plan_and_replan() -> None:
    registry = PromptRegistry("prompts")
    base = {
        "goal": "g",
        "data_readme": "r",
        "dataset_profile": "p",
        "memory_hits": [],
        "failed_plan": None,
        "escalation_reason": None,
    }
    first = render(registry, "planner", base, 1, 1).messages[1]["content"]
    assert "previous plan failed" not in first and "memory" not in first
    replan = render(registry, "planner", {**base, "failed_plan": "{...}", "escalation_reason": "leak"}, 1, 1)
    assert "Reason: leak" in replan.messages[1]["content"]


# --- budget ----------------------------------------------------------------------------------


def test_shadow_price() -> None:
    assert shadow_usd(PRICES["strong-model"], 1_000_000, 500_000) == pytest.approx(2.0)


def test_degradation_only_for_low_stakes_and_only_past_threshold() -> None:
    ledger = Ledger(cap_usd=1.0, degrade_at=0.5)
    record(ledger, "critic", "s1", "strong", Spend(calls=1, shadow_usd=0.49))
    assert choose_tier(ledger, "strong", "low") == "strong"
    record(ledger, "critic", "s1", "strong", Spend(calls=1, shadow_usd=0.02))
    assert choose_tier(ledger, "strong", "low") == "cheap"
    assert choose_tier(ledger, "strong", "high") == "strong"
    assert ledger.by_role["critic"].calls == 2 and ledger.by_step["s1"].shadow_usd == pytest.approx(0.51)


def test_cap_refuses_the_next_call() -> None:
    ledger = Ledger(cap_usd=0.1)
    record(ledger, "planner", None, "strong", Spend(calls=1, shadow_usd=0.1))
    with pytest.raises(BudgetExceeded, match="cap reached"):
        ensure_room(ledger)


# --- the Observer ----------------------------------------------------------------------------


def _observer(tmp_path: Path, strong: list, cheap: list, cap: float = 1.0, **kwargs) -> Observer:
    return Observer(
        run_id="r1",
        run_dir=tmp_path / "run",
        ledger=Ledger(cap_usd=cap, degrade_at=0.5),
        clients={
            "strong": FakeModelClient(strong, "strong-model"),
            "cheap": FakeModelClient(cheap, "cheap-model"),
        },
        prices=PRICES,
        tier_params={"strong": {"chat_template_kwargs": {"enable_thinking": False}}},
        **kwargs,
    )


def test_observer_logs_every_field(tmp_path: Path) -> None:
    obs = _observer(tmp_path, ["plan text"], [])
    prompt = render(_registry(tmp_path / "p"), "planner", {"goal": "g", "memory_hits": []})
    completion = obs.call(prompt, "strong", "high", {"temperature": 0.7}, step_id=None, iteration=0)
    row = json.loads((tmp_path / "run" / "calls.jsonl").read_text())
    for field in (
        "run_id",
        "role",
        "step_id",
        "iteration",
        "tier",
        "model",
        "prompt_refs",
        "prompt_sha256",
        "messages",
        "params",
        "content",
        "reasoning",
        "input_tokens",
        "output_tokens",
        "ttft_s",
        "total_s",
        "shadow_usd",
        "error",
        "degraded",
    ):
        assert field in row, field
    assert row["content"] == completion.content == "plan text"
    assert row["params"] == {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.7}
    assert row["shadow_usd"] == pytest.approx(
        shadow_usd(PRICES["strong-model"], row["input_tokens"], row["output_tokens"])
    )


def test_observer_degrades_then_stops_at_cap(tmp_path: Path) -> None:
    big = Completion(model="strong-model", content="x", input_tokens=300_000, output_tokens=0)
    obs = _observer(tmp_path, [big, "strong again"], ["cheap answer"], cap=0.6)
    prompt = render(_registry(tmp_path / "p"), "planner", {"goal": "g", "memory_hits": []})
    obs.call(prompt, "strong", "low")  # 0.30 of 0.60: at the 0.5 threshold
    assert obs.call(prompt, "strong", "low").content == "cheap answer"  # low stakes: degraded
    assert obs.call(prompt, "strong", "high").content == "strong again"  # high stakes: never degraded
    rows = [json.loads(line) for line in (tmp_path / "run" / "calls.jsonl").read_text().splitlines()]
    assert [r["degraded"] for r in rows] == [False, True, False]
    obs.ledger.total.shadow_usd = 0.6
    with pytest.raises(BudgetExceeded):
        obs.call(prompt, "strong", "high")


def test_replay_uses_the_cache_and_costs_the_same(tmp_path: Path) -> None:
    prompt = render(_registry(tmp_path / "p"), "planner", {"goal": "g", "memory_hits": []})
    first = _observer(tmp_path / "a", ["cached answer"], [], cache_dir=tmp_path / "cache")
    first.call(prompt, "strong", "high", {"seed": 1})
    replay = _observer(
        tmp_path / "b", [], [], cache_dir=tmp_path / "cache", replay=True
    )  # no script: must not call
    completion = replay.call(prompt, "strong", "high", {"seed": 1})
    assert completion.cached and completion.content == "cached answer"
    assert replay.ledger.total.shadow_usd == pytest.approx(first.ledger.total.shadow_usd)


def test_errors_are_logged_but_not_cached(tmp_path: Path) -> None:
    down = Completion(model="strong-model", content="", error="APIConnectionError: refused", infra_error=True)
    obs = _observer(tmp_path, [down], [], cache_dir=tmp_path / "cache")
    prompt = render(_registry(tmp_path / "p"), "planner", {"goal": "g", "memory_hits": []})
    assert obs.call(prompt, "strong", "high").infra_error
    assert not (tmp_path / "cache").exists()
    assert json.loads((tmp_path / "run" / "calls.jsonl").read_text())["infra_error"] is True


# --- the streaming client against a local server ---------------------------------------------


class _SSEHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        chunks = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "hel"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for c in chunks:
            payload = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m", **c}
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args) -> None:
        pass


def test_streaming_client_parses_content_reasoning_usage() -> None:
    server = HTTPServer(("127.0.0.1", 0), _SSEHandler)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = OpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}/v1", "m", timeout_s=10)
        out = client.complete(
            [{"role": "user", "content": "hi"}],
            {"temperature": 0.1, "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
        )
    finally:
        server.shutdown()
    assert out.ok and out.content == "hello" and out.reasoning == "thinking" and out.finish_reason == "stop"
    assert (out.input_tokens, out.output_tokens) == (12, 3) and out.ttft_s is not None
    sent = server.requests[0]
    assert (
        sent["temperature"] == 0.1
        and sent["top_k"] == 20
        and sent["chat_template_kwargs"] == {"enable_thinking": False}
    )


def test_unreachable_server_is_an_infra_error() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed again: nothing listens here
    out = OpenAICompatibleClient(f"http://127.0.0.1:{port}/v1", "m", timeout_s=5).complete(
        [{"role": "user", "content": "hi"}], {}
    )
    assert not out.ok and out.infra_error


# --- config ----------------------------------------------------------------------------------


def test_build_observer_from_the_repository_configs(tmp_path: Path) -> None:
    obs = build_observer(RunConfig(name="t", model_host="10.0.0.5"), "r", tmp_path)
    assert obs.clients["strong"].model == "gemma-4-26b-a4b" and obs.clients["cheap"].model == "gemma-4-e4b"
    assert str(obs.clients["strong"]._client.base_url).startswith("http://10.0.0.5:8081/v1")
    assert obs.tier_params["strong"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert set(obs.prices) >= {"gemma-4-26b-a4b", "gemma-4-e4b"}
