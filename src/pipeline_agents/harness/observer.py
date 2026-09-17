"""The Observer (T3): every model call goes through here. Cross-cutting instrumentation, not an agent.

For each call it picks the tier (budget policy), refuses the call if the cap is reached, sends the rendered
prompt, and appends one line to `calls.jsonl` with: run, role, step, loop iteration, tier, model, prompt
template refs and hashes, the rendered messages, the raw response (content and reasoning), input and output
tokens, time to first token, total latency, shadow $, and any error.

Replay: responses are cached on disk by a hash of (model, messages, params, seed). With `replay=True` a cached
response is returned without calling the model, so a run can be re-executed exactly.
"""

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline_agents.harness.budget import Price, choose_tier, ensure_room, record, shadow_usd
from pipeline_agents.harness.client import Completion, ModelClient
from pipeline_agents.harness.render import RenderedPrompt
from pipeline_agents.schemas import Ledger, Spend, Stakes, Tier


def cache_key(model: str, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
    blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class Observer:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        ledger: Ledger,
        clients: dict[Tier, ModelClient],
        prices: dict[str, Price],
        tier_params: dict[Tier, dict[str, Any]] | None = None,
        cache_dir: Path | None = None,
        replay: bool = False,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.ledger = ledger
        self.clients = clients
        self.prices = prices
        self.tier_params = tier_params or {}
        self.cache_dir = cache_dir
        self.replay = replay
        run_dir.mkdir(parents=True, exist_ok=True)

    def call(
        self,
        prompt: RenderedPrompt,
        role_tier: Tier,
        stakes: Stakes,
        params: dict[str, Any] | None = None,
        step_id: str | None = None,
        iteration: int = 0,
    ) -> Completion:
        ensure_room(self.ledger)
        tier = choose_tier(self.ledger, role_tier, stakes)
        client = self.clients[tier]
        full_params = {
            **self.tier_params.get(tier, {}),
            **(params or {}),
        }  # call params win over tier defaults
        key = cache_key(client.model, prompt.messages, full_params)

        completion = self._cached(key) if self.replay else None
        if completion is None:
            completion = client.complete(prompt.messages, full_params)
            if completion.ok and self.cache_dir is not None:
                self._store(key, completion)

        price = self.prices[client.model]
        # A replayed call costs what the original cost: tier choices depend on spend, and a replay that spent
        # less would degrade later, take different paths and miss its own cache.
        cost = shadow_usd(price, completion.input_tokens, completion.output_tokens)
        spend = Spend(
            calls=1,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            shadow_usd=cost,
            seconds=completion.total_s,
        )
        record(self.ledger, prompt.role, step_id, tier, spend)
        self._log(
            {
                "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "run_id": self.run_id,
                "role": prompt.role,
                "step_id": step_id,
                "iteration": iteration,
                "stakes": stakes,
                "role_tier": role_tier,
                "tier": tier,
                "degraded": tier != role_tier,
                "model": client.model,
                "prompt_refs": prompt.refs,
                "prompt_sha256": {"system": prompt.system.sha256, "user": prompt.user.sha256},
                "messages": prompt.messages,
                "params": full_params,
                "cache_key": key,
                **asdict(completion),
                "shadow_usd": cost,
                "run_shadow_usd": self.ledger.total.shadow_usd,
            }
        )
        return completion

    def _log(self, row: dict[str, Any]) -> None:
        with (self.run_dir / "calls.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _cached(self, key: str) -> Completion | None:
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path is None or not path.exists():
            return None
        return Completion(**{**json.loads(path.read_text()), "cached": True})

    def _store(self, key: str, completion: Completion) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(json.dumps(asdict(completion)))
