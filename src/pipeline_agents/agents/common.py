"""What every role shares (T6): render the role's prompt, call the model through the Observer, parse the
reply, and re-ask once with the parse error before giving up.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from pipeline_agents.config import RunConfig
from pipeline_agents.harness.client import Completion
from pipeline_agents.harness.observer import Observer
from pipeline_agents.harness.registry import PromptRegistry
from pipeline_agents.harness.render import render
from pipeline_agents.schemas import Stakes

T = TypeVar("T")
Model = TypeVar("Model", bound=BaseModel)

REASK = (
    "Your previous reply could not be used: {error}\n\nReply again, following the output format exactly. "
    "Here is your previous reply:\n\n{reply}"
)


class ParseError(ValueError):
    pass


class InfraError(RuntimeError):
    """The model endpoint failed (unreachable, timeout, 5xx): retry, and do not count it against the agent."""


@dataclass
class Deps:
    """Everything the graph's nodes need besides the state. Not checkpointed."""

    observer: Observer
    registry: PromptRegistry
    config: RunConfig
    runner: Any  # a sandbox runner (LinuxSandbox or UnsandboxedRunner)


@dataclass
class RoleReply:
    value: Any
    completion: Completion
    attempts: int


def extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    if not candidate:
        raise ParseError("no JSON object found")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e}") from e


def parse_model(model: type[Model]) -> Callable[[str], Model]:
    def parse(text: str) -> Model:
        try:
            return model.model_validate(extract_json(text))
        except ValidationError as e:
            raise ParseError(f"JSON does not match the schema: {str(e).splitlines()[0:3]}") from e

    return parse


def call_role(
    deps: Deps,
    role: str,
    context: dict[str, Any],
    parse: Callable[[str], T],
    stakes: Stakes,
    step_id: str | None,
    iteration: int,
) -> RoleReply:
    role_cfg = deps.config.role(role)
    prompt = render(deps.registry, role, context, role_cfg.system_version, role_cfg.user_version)
    params = {"temperature": role_cfg.temperature, "max_tokens": role_cfg.max_tokens}
    completion = deps.observer.call(
        prompt, role_cfg.tier, stakes, params, step_id=step_id, iteration=iteration
    )
    for attempt in (1, 2):
        if completion.infra_error:
            raise InfraError(completion.error or "model endpoint failed")
        try:
            return RoleReply(parse(completion.content), completion, attempt)
        except ParseError as e:
            if attempt == 2:
                raise
            retry = prompt.__class__(
                prompt.role,
                prompt.system,
                prompt.user,
                [
                    *prompt.messages,
                    {"role": "assistant", "content": completion.content},
                    {"role": "user", "content": REASK.format(error=e, reply=completion.content[-2000:])},
                ],
                prompt.context_keys,
            )
            completion = deps.observer.call(
                retry, role_cfg.tier, stakes, params, step_id=step_id, iteration=iteration
            )
    raise AssertionError("unreachable")
