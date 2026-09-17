"""Run configuration (T3): the model layout, each role's tier and sampling, the budget and the loop caps.

Loaded from configs/run/<name>.yaml and configs/models.yaml. Every run writes its resolved config next to its
results, so any number can be traced back to the settings that produced it.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from pipeline_agents.harness.budget import load_prices
from pipeline_agents.harness.client import ModelClient, OpenAICompatibleClient
from pipeline_agents.harness.observer import Observer
from pipeline_agents.schemas import Ledger, Tier


class RoleConfig(BaseModel):
    tier: Tier = "strong"
    temperature: float = 0.6
    max_tokens: int = 4096
    system_version: int | None = None  # None: the latest version in the registry
    user_version: int | None = None


class LoopConfig(BaseModel):
    max_revisions_per_step: int = 2
    max_replans: int = 1
    human_review_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_node_visits: int = 200
    max_model_calls: int = 120
    infra_retries: int = 3


class MemoryConfig(BaseModel):
    procedural: bool = False
    semantic: bool = False
    episodic: bool = False
    top_k: int = 3


class RunConfig(BaseModel):
    name: str
    layout: str = "tiered"
    model_host: str = "127.0.0.1"  # the GPU machine's address as seen from where the run executes
    roles: dict[str, RoleConfig] = {}
    budget_usd: float = 0.05
    degrade_at: float = 0.7
    loop: LoopConfig = LoopConfig()
    memory: MemoryConfig = MemoryConfig()

    def role(self, name: str) -> RoleConfig:
        return self.roles.get(name, RoleConfig())


def load_run_config(path: Path | str) -> RunConfig:
    cfg = RunConfig(**yaml.safe_load(Path(path).read_text()))
    if host := os.environ.get("PIPELINE_MODEL_HOST"):
        cfg = cfg.model_copy(update={"model_host": host})
    return cfg


def load_models(path: Path | str = "configs/models.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def build_observer(
    cfg: RunConfig,
    run_id: str,
    run_dir: Path,
    clients: dict[Tier, ModelClient] | None = None,
    models_path: Path | str = "configs/models.yaml",
    prices_path: Path | str = "configs/prices.yaml",
    cache_dir: Path | None = None,
    replay: bool = False,
) -> Observer:
    """Clients come from the layout in configs/models.yaml unless given (tests pass fakes)."""
    models = load_models(models_path)
    layout = models["layouts"][cfg.layout]
    tier_params: dict[Tier, dict[str, Any]] = {}
    built: dict[Tier, ModelClient] = {}
    for tier in ("strong", "cheap"):
        spec = layout[tier]
        tier_params[tier] = {
            **models.get("sampling", {}),
            **models["models"][spec["model"]].get("request", {}),
        }
        built[tier] = OpenAICompatibleClient(f"http://{cfg.model_host}:{spec['port']}/v1", spec["model"])
    return Observer(
        run_id=run_id,
        run_dir=run_dir,
        ledger=Ledger(cap_usd=cfg.budget_usd, degrade_at=cfg.degrade_at),
        clients=clients or built,
        prices=load_prices(prices_path),
        tier_params=tier_params,
        cache_dir=cache_dir,
        replay=replay,
    )
