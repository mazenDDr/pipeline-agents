"""Shadow-price budget (T3): cost per call, a ledger per role / step / tier, degradation and a hard cap.

Costs are shadow dollars: real token counts times a hosted provider's list price for the same open weights
(configs/prices.yaml). Nothing is billed.

Policy: every call has stakes. High-stakes calls (planning, the final critique) always use their role's
tier. Once the run has spent `degrade_at` of its cap, low-stakes calls move to the cheap tier. A call is
refused once the cap is reached, so a run stops with a reason instead of overspending.
"""

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

from pipeline_agents.schemas import Ledger, Spend, Stakes, Tier


class Price(BaseModel):
    input_per_mtok: float
    output_per_mtok: float
    source: str
    retrieved: date


class BudgetExceeded(RuntimeError):
    pass


def load_prices(path: Path | str = "configs/prices.yaml") -> dict[str, Price]:
    return {name: Price(**p) for name, p in yaml.safe_load(Path(path).read_text())["models"].items()}


def shadow_usd(price: Price, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000


def choose_tier(ledger: Ledger, role_tier: Tier, stakes: Stakes) -> Tier:
    if stakes == "low" and ledger.fraction_spent >= ledger.degrade_at:
        return "cheap"
    return role_tier


def ensure_room(ledger: Ledger) -> None:
    if ledger.total.shadow_usd >= ledger.cap_usd:
        raise BudgetExceeded(
            f"budget cap reached: {ledger.total.shadow_usd:.4f} of {ledger.cap_usd:.4f} shadow $ spent"
        )


def record(ledger: Ledger, role: str, step_id: str | None, tier: Tier, spend: Spend) -> None:
    ledger.total.add(spend)
    for bucket, key in ((ledger.by_role, role), (ledger.by_tier, tier), (ledger.by_step, step_id or "run")):
        bucket.setdefault(key, Spend()).add(spend)
