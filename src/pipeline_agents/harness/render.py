"""Render a role's prompt from the run state (T3).

The template and the filled instance are kept apart: a `RenderedPrompt` carries the template refs and hashes
next to the messages, so every logged call can answer "was this a prompt problem or a model problem?".
Rendering is strict: a variable the template uses but the context lacks is an error, not an empty string.
"""

from dataclasses import dataclass, field
from typing import Any

from jinja2 import Environment, StrictUndefined

from pipeline_agents.harness.registry import PromptRegistry, PromptTemplate

_ENV = Environment(
    undefined=StrictUndefined, keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True
)


@dataclass(frozen=True)
class RenderedPrompt:
    role: str
    system: PromptTemplate
    user: PromptTemplate
    messages: list[dict[str, str]]
    context_keys: list[str] = field(default_factory=list)

    @property
    def refs(self) -> dict[str, str]:
        return {"system": self.system.ref, "user": self.user.ref}


def render(
    registry: PromptRegistry,
    role: str,
    context: dict[str, Any],
    system_version: int | None = None,
    user_version: int | None = None,
) -> RenderedPrompt:
    system = registry.get(role, "system", system_version)
    user = registry.get(role, "user", user_version)
    messages = [
        {"role": "system", "content": _ENV.from_string(system.text).render(**context).strip()},
        {"role": "user", "content": _ENV.from_string(user.text).render(**context).strip()},
    ]
    return RenderedPrompt(role, system, user, messages, sorted(context))
