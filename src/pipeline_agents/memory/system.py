"""The three memories and their rules (T8).

| store      | written                                         | retrieved                                   |
|------------|-------------------------------------------------|---------------------------------------------|
| semantic   | after a run: facts about its data files         | exact fingerprint (file name + header)      |
| procedural | mined from dev runs whose checker passed        | top-k by similarity to the step             |
| episodic   | after every run: task, plan, outcome, problems  | top-k by similarity to goal + data profile  |

Writers raise ParseError when a reply is unusable; callers record it rather than treating it as "nothing to
remember".

Leakage rules:
- Writers never see the hidden checker's details (they can contain the answer); only whether it passed and the
  stage that failed.
- Procedural skills come only from dev-split runs; `mine_skills` refuses anything else.
- Episodic retrieval never returns an episode of the same task.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from pipeline_agents.agents.common import Deps, ParseError, call_role, extract_json
from pipeline_agents.config import MemoryConfig
from pipeline_agents.memory.store import Embedder, JsonlStore, MemoryItem, item_id
from pipeline_agents.schemas import MemoryHit, PlanStep, RunState

DATA_SUFFIXES = {".csv", ".data", ".txt", ".tsv"}
MAX_FACTS = 12


class Outcome(BaseModel):
    """What memory may know about how a run ended. No checker details."""

    status: str
    checker_passed: bool | None = None
    first_failure: str | None = None  # a stage name, e.g. "score", never the stage's detail

    def describe(self) -> str:
        checked = (
            ""
            if self.checker_passed is None
            else (
                " The hidden check passed."
                if self.checker_passed
                else f" The hidden check failed at: {self.first_failure}."
            )
        )
        return f"The run ended with status {self.status}.{checked}"


def data_files(workspace: Path) -> list[Path]:
    data = workspace / "data"
    return [
        p
        for p in sorted(data.iterdir())
        if p.is_file() and p.suffix.lower() in DATA_SUFFIXES and not p.name.lower().startswith("readme")
    ]


def fingerprint(path: Path) -> str:
    """A dataset file's identity for semantic memory: its name and first line, not its rows (a warm-up and a
    follow-up task can hold different slices of the same file)."""
    with path.open("rb") as f:
        first_line = f.readline()
    return hashlib.sha256(path.name.encode() + b"|" + first_line.strip()).hexdigest()[:16]


def _issues(state: RunState, step_id: str) -> list[str]:
    record = state.steps.get(step_id)
    if record is None:
        return []
    issues = [i for v in record.verdicts if v.decision != "accept" for i in v.issues]
    return list(dict.fromkeys(issues))[:5]


@dataclass
class MemorySystem:
    root: Path
    embedder: Embedder
    config: MemoryConfig
    split: str  # the split of the runs this system serves: dev, test or user

    def __post_init__(self) -> None:
        self.semantic = JsonlStore(self.root / "semantic.jsonl")
        self.procedural = JsonlStore(self.root / "procedural.jsonl")
        self.episodic = JsonlStore(self.root / "episodic.jsonl")

    # --- retrieval ---------------------------------------------------------------------------

    def run_hits(self, task_id: str, workspace: Path, goal: str, profile: str) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        if self.config.semantic:
            prints = {fingerprint(p) for p in data_files(workspace)}
            facts = [i for i in self.semantic.items() if i.key in prints]
            hits += [MemoryHit(store="semantic", key=i.key, text=i.text) for i in facts[-MAX_FACTS:]]
        if self.config.episodic:
            query = f"{goal}\n{profile[:1500]}"
            found = self.episodic.search(
                query, self.embedder, self.config.top_k, where=lambda i: i.source_task != task_id
            )
            hits += [MemoryHit(store="episodic", key=i.key, text=i.text, score=s) for i, s in found]
        return hits

    def step_hits(self, step: PlanStep) -> list[MemoryHit]:
        if not self.config.procedural:
            return []
        query = f"{step.kind}: {step.intent}\n" + "; ".join(step.acceptance_checks)
        found = self.procedural.search(
            query, self.embedder, self.config.top_k, where=lambda i: i.source_split == "dev"
        )
        return [
            MemoryHit(store="procedural", key=i.key, text=i.text, score=s, step_id=step.id) for i, s in found
        ]

    # --- writing -----------------------------------------------------------------------------

    def record_episode(self, state: RunState, outcome: Outcome) -> int:
        plan = state.plan
        steps = "\n".join(f"- {s.id} [{s.kind}] {s.intent}" for s in plan.steps) if plan else "(no plan)"
        problems = [i for s in (plan.steps if plan else []) for i in _issues(state, s.id)][:6]
        text = (
            f"Task {state.task_id}: {state.goal[:400]}\n"
            f"Data files: {[p.name for p in data_files(Path(state.workspace))]}\n"
            f"Plan (version {plan.version if plan else 0}, {state.replans} re-plans):\n{steps}\n"
            f"Outcome: {outcome.describe()}\n"
            + ("Problems met: " + " | ".join(problems) if problems else "No step was rejected.")
        )
        item = MemoryItem(
            id=item_id("episodic", state.run_id, text),
            store="episodic",
            key=state.task_id,
            text=text,
            source_run=state.run_id,
            source_task=state.task_id,
            source_split=self.split,
        )
        return self.episodic.add([item], self.embedder)

    def extract_facts(self, deps: Deps, state: RunState, outcome: Outcome) -> int:
        if state.plan is None:
            return 0
        workspace = Path(state.workspace)
        files = data_files(workspace)
        steps = []
        for s in state.plan.steps:
            record = state.steps.get(s.id)
            if record and record.attempts:
                steps.append(
                    {
                        "id": s.id,
                        "kind": s.kind,
                        "intent": s.intent,
                        "claims": record.attempts[-1].claims,
                        "stdout": record.attempts[-1].stdout[-600:],
                        "issues": _issues(state, s.id),
                    }
                )
        context = {
            "files": "\n".join(
                f"- {p.name}: first line {p.open(errors='replace').readline().strip()[:200]!r}" for p in files
            ),
            "data_readme": state.task.data_readme if state.task else "",
            "rationale": state.plan.rationale,
            "steps": steps,
            "outcome": outcome.describe(),
        }
        # Schema-enforced: in the first real mining run a free-form reply broke its JSON twice and was lost.
        reply = call_role(
            deps, "memory_writer", context, _facts, "low", None, 0, schema=FactsReply.model_json_schema()
        )
        prints = [fingerprint(p) for p in files]
        items = []
        for fact in reply.value[:8]:
            # A fact names its file; store it under that file's fingerprint (all files when it names none).
            named = [fingerprint(p) for p in files if p.name in fact] or prints
            items += [
                MemoryItem(
                    id=item_id("semantic", key, fact),
                    store="semantic",
                    key=key,
                    text=fact,
                    source_run=state.run_id,
                    source_task=state.task_id,
                    source_split=self.split,
                )
                for key in named
            ]
        return self.semantic.add(items, self.embedder)

    def mine_skills(self, deps: Deps, state: RunState, outcome: Outcome) -> int:
        if self.split != "dev":
            raise ValueError(f"procedural skills are mined from dev runs only, not {self.split!r}")
        if not outcome.checker_passed or state.plan is None:
            return 0
        steps = [
            {
                "id": s.id,
                "kind": s.kind,
                "intent": s.intent,
                "code": state.steps[s.id].attempts[-1].code[-3000:],
                "issues": _issues(state, s.id),
            }
            for s in state.plan.steps
            if s.id in state.steps and state.steps[s.id].status == "accepted"
        ]
        reply = call_role(
            deps,
            "skill_miner",
            {"goal": state.goal, "steps": steps},
            _skills,
            "low",
            None,
            0,
            schema=SkillsReply.model_json_schema(),
        )
        items = []
        for skill in reply.value[:4]:
            pattern = f"```python\n{skill.code_pattern}\n```"
            text = f"When: {skill.situation}\nDo: {skill.strategy}\nPattern:\n{pattern}"
            kind = next((s["kind"] for s in steps if s["issues"]), steps[0]["kind"] if steps else "clean")
            items.append(
                MemoryItem(
                    id=item_id("procedural", kind, text),
                    store="procedural",
                    key=kind,
                    text=text,
                    source_run=state.run_id,
                    source_task=state.task_id,
                    source_split="dev",
                )
            )
        return self.procedural.add(items, self.embedder)

    def digest(self) -> dict[str, object]:
        return {
            name: store.digest()
            for name, store in (
                ("semantic", self.semantic),
                ("procedural", self.procedural),
                ("episodic", self.episodic),
            )
        }


class Skill(BaseModel):
    situation: str
    strategy: str
    code_pattern: str


class SkillsReply(BaseModel):
    skills: list[Skill]


class FactsReply(BaseModel):
    facts: list[str]


def _facts(text: str) -> list[str]:
    data = extract_json(text)
    facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(facts, list) or not all(isinstance(f, str) for f in facts):
        raise ParseError('expected {"facts": ["...", ...]}')
    return [f.strip() for f in facts if f.strip()]


def _skills(text: str) -> list[Skill]:
    data = extract_json(text)
    try:
        return [Skill.model_validate(s) for s in data.get("skills", [])]
    except (AttributeError, ValueError) as e:
        raise ParseError(f'expected {{"skills": [...]}}: {e}') from e
