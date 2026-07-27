"""Claude (spec M8.3b).

Kept as its own provider rather than folded into the OpenAI-compatible one
because the wire format genuinely differs: the system prompt is a top-level
field, not a message, and the response is a list of content blocks. The logic
here is the one that was already running in ``llm.py``, moved rather than
rewritten.

Claude's independence matters structurally: the ``JUDGE`` role is required to
use a *different* provider from the one that produced the artefact under
review, and with Kimi as the default author, Claude is what makes that possible.
"""
from __future__ import annotations

import asyncio
import logging

from aegis.clock import CLOCK
from aegis.cortex.providers.base import CallParams, Completion, Provider, estimate_tokens

logger = logging.getLogger("aegis.cortex.anthropic")


def _build_client(api_key: str, timeout: float):
    from anthropic import AsyncAnthropic
    return AsyncAnthropic(api_key=api_key, timeout=timeout)


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Separate the system prompt from the conversation.

    Multiple system messages are joined rather than last-one-wins: a caller that
    layered a role instruction on top of a base prompt means both, and dropping
    the first would quietly change what the model was told.
    """
    system_parts: list[str] = []
    conversation: list[dict] = []
    for message in messages or []:
        if message.get("role") == "system":
            system_parts.append(str(message.get("content", "")))
        else:
            conversation.append({"role": message.get("role", "user"),
                                 "content": message.get("content", "")})
    if not conversation:
        # The API rejects an empty conversation; a system prompt alone is a
        # legitimate request ("just think"), so it gets a minimal turn.
        conversation = [{"role": "user", "content": "Think."}]
    return "\n\n".join(system_parts), conversation


class AnthropicProvider(Provider):
    kind = "anthropic"

    def __init__(self, name: str, model: str, api_key: str = "",
                 *, timeout: float = 60.0, client=None, client_factory=None):
        super().__init__(name, model)
        self.api_key = api_key
        self.timeout = timeout
        self._client = client
        self._client_factory = client_factory

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.model)

    def unavailable_reason(self) -> str:
        if not self.model:
            return f"{self.name}: no model id configured"
        if not self.api_key:
            return f"{self.name}: no ANTHROPIC_API_KEY configured"
        return ""

    def client(self):
        if self._client is None:
            factory = self._client_factory or _build_client
            self._client = factory(self.api_key, self.timeout)
        return self._client

    async def _invoke(self, messages: list[dict], params: CallParams) -> Completion:
        system_text, conversation = split_system(messages)
        request: dict = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }
        if system_text:
            request["system"] = system_text
        if params.top_p is not None:
            request["top_p"] = params.top_p
        if params.stop:
            request["stop_sequences"] = list(params.stop)

        started = CLOCK.monotonic()
        response = await asyncio.wait_for(
            self.client().messages.create(**request), timeout=params.timeout)
        elapsed_ms = (CLOCK.monotonic() - started) * 1000

        blocks = getattr(response, "content", None) or []
        # Concatenate every text block: a response split across blocks would
        # otherwise be silently truncated to its first paragraph.
        text = "".join(getattr(b, "text", "") or "" for b in blocks)

        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "input_tokens", None) if usage else None
        tokens_out = getattr(usage, "output_tokens", None) if usage else None
        if tokens_in is None:
            tokens_in = estimate_tokens(messages)
        if tokens_out is None:
            tokens_out = max(1, len(text) // 4)

        return Completion(
            text=text, provider=self.name, model=self.model,
            tokens_in=int(tokens_in), tokens_out=int(tokens_out),
            latency_ms=elapsed_ms,
        )
