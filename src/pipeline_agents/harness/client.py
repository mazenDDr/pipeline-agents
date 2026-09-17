"""Model clients (T3): an OpenAI-compatible streaming client for real servers, a scripted fake for tests.

A call never raises for a model or network problem: the `Completion` carries the error and whether it was an
infra failure (the server was unreachable, timed out or returned 5xx) worth retrying without spending a
revision, so the orchestrator can decide.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import openai
from openai import OpenAI

# Other params (top_k, chat_template_kwargs, cache_prompt, ...) are llama-server specific: sent in extra_body.
OPENAI_PARAMS = {"temperature", "top_p", "max_tokens", "seed", "response_format", "stop"}


@dataclass
class Completion:
    model: str
    content: str
    reasoning: str = ""
    finish_reason: str | None = None
    ttft_s: float | None = None  # first token of any kind, reasoning included
    total_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    infra_error: bool = False
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


class ModelClient(Protocol):
    model: str

    def complete(self, messages: list[dict[str, str]], params: dict[str, Any]) -> Completion: ...


class OpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 600.0, api_key: str = "none") -> None:
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)

    def complete(self, messages: list[dict[str, str]], params: dict[str, Any]) -> Completion:
        start = time.perf_counter()
        out = Completion(model=self.model, content="")
        content, reasoning = [], []
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={k: v for k, v in params.items() if k not in OPENAI_PARAMS} or None,
                **{k: v for k, v in params.items() if k in OPENAI_PARAMS},
            )
            for chunk in stream:
                if chunk.usage:
                    out.input_tokens, out.output_tokens = (
                        chunk.usage.prompt_tokens,
                        chunk.usage.completion_tokens,
                    )
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = (delta.model_extra or {}).get("reasoning_content") or ""
                if (piece or delta.content) and out.ttft_s is None:
                    out.ttft_s = time.perf_counter() - start
                reasoning.append(piece)
                content.append(delta.content or "")
                out.finish_reason = chunk.choices[0].finish_reason or out.finish_reason
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            out.error, out.infra_error = f"{type(e).__name__}: {e}", True
        except openai.APIError as e:
            out.error = f"{type(e).__name__}: {e}"
        out.content, out.reasoning = "".join(content), "".join(reasoning)
        out.total_s = time.perf_counter() - start
        return out


Script = str | Completion | Callable[[list[dict[str, str]], dict[str, Any]], str | Completion]


class FakeModelClient:
    """Replays a script of responses, one per call, for tests that must not touch a real model.

    Each item is the response text, a full `Completion` (to simulate errors or truncation), or a function of
    the messages and params that returns either. Token counts are estimated as characters / 4.
    """

    def __init__(self, script: list[Script], model: str = "fake") -> None:
        self.model = model
        self.script = list(script)
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, str]], params: dict[str, Any]) -> Completion:
        self.calls.append((messages, params))
        if not self.script:
            raise AssertionError(
                f"FakeModelClient({self.model}) has no scripted response for call {len(self.calls)}"
            )
        item = self.script.pop(0)
        if callable(item):
            item = item(messages, params)
        if isinstance(item, Completion):
            return item
        prompt_chars = sum(len(m["content"]) for m in messages)
        return Completion(
            model=self.model,
            content=item,
            finish_reason="stop",
            ttft_s=0.01,
            total_s=0.05,
            input_tokens=max(1, prompt_chars // 4),
            output_tokens=max(1, len(item) // 4),
        )
