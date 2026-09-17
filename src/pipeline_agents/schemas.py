"""State and message contracts shared by the harness, the agents and the graph (T3).

Everything a run knows lives in one `RunState`, checkpointed after every node, so a run can be paused,
resumed and replayed, and a failure can be traced to the first step that went wrong.
"""

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["planner", "executor", "critic", "reviser", "baseline"]
Tier = Literal["strong", "cheap"]
Stakes = Literal["high", "low"]
StepKind = Literal["load", "profile", "clean", "feature", "split", "train", "evaluate", "analyze", "report"]
FailureKind = Literal["infra", "logic"]
RunStatus = Literal["running", "needs_human", "succeeded", "failed"]


# --- what the agents produce -----------------------------------------------------------------


class PlanStep(BaseModel):
    id: str  # "s1", "s2", ...
    kind: StepKind
    intent: str = Field(min_length=5)  # what the step does and why
    depends_on: list[str] = []
    acceptance_checks: list[str] = Field(min_length=1)  # concrete checks that show the step worked


class Plan(BaseModel):
    version: int = 1  # bumped on every re-plan
    goal_type: Literal["predictive", "analytical"]
    target_column: str | None
    steps: list[PlanStep] = Field(min_length=1, max_length=15)
    rationale: str = ""


class StepAttempt(BaseModel):
    attempt: int  # 1-based, per step
    code: str
    exit_code: int | None  # None when the sandbox stopped it (timeout, memory)
    stdout: str = ""
    stderr: str = ""
    failure: FailureKind | None = None
    artifacts: list[str] = []  # files the step wrote under output/
    claims: str = ""  # what the Executor says the step did; the Critic checks it, never trusts it
    seconds: float = 0.0


class Finding(BaseModel):
    """One result from a Critic validation tool, which inspects artifacts directly."""

    tool: str
    passed: bool
    detail: str


class Verdict(BaseModel):
    decision: Literal["accept", "revise", "escalate"]
    issues: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    findings: list[Finding] = []


class Revision(BaseModel):
    action: Literal["fix", "escalate"]  # escalate sends the run back to the Planner
    instructions: str
    reason: str = ""


class StepRecord(BaseModel):
    step_id: str
    attempts: list[StepAttempt] = []
    verdicts: list[Verdict] = []
    revisions: list[Revision] = []
    status: Literal["pending", "running", "accepted", "escalated", "failed"] = "pending"

    @property
    def revision_count(self) -> int:
        return sum(r.action == "fix" for r in self.revisions)


# --- memory ----------------------------------------------------------------------------------


class MemoryHit(BaseModel):
    store: Literal["procedural", "semantic", "episodic"]
    key: str
    text: str
    score: float | None = None  # similarity, for stores that rank
    step_id: str | None = None  # the step the hit was retrieved for; None for run-level retrieval


# --- budget ----------------------------------------------------------------------------------


class Spend(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    shadow_usd: float = 0.0
    seconds: float = 0.0

    def add(self, other: "Spend") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.shadow_usd += other.shadow_usd
        self.seconds += other.seconds


class Ledger(BaseModel):
    cap_usd: float
    degrade_at: float = Field(default=0.7, ge=0.0, le=1.0)  # fraction of the cap
    total: Spend = Spend()
    by_role: dict[str, Spend] = {}
    by_step: dict[str, Spend] = {}
    by_tier: dict[str, Spend] = {}

    @property
    def fraction_spent(self) -> float:
        return self.total.shadow_usd / self.cap_usd if self.cap_usd else 0.0


# --- the run ---------------------------------------------------------------------------------


class RunState(BaseModel):
    run_id: str
    task_id: str
    goal: str
    dataset_profile: str = ""
    plan: Plan | None = None
    plan_history: list[Plan] = []  # every superseded plan, for the re-plan trace
    cursor: int = 0  # index of the current step in plan.steps
    steps: dict[str, StepRecord] = {}
    replans: int = 0
    memory_hits: list[MemoryHit] = []
    ledger: Ledger
    node_visits: int = 0
    model_calls: int = 0
    status: RunStatus = "running"
    stop_reason: str | None = None
