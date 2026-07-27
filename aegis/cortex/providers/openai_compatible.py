"""Any endpoint that speaks ``/v1/chat/completions`` (spec M8.3b).

One provider class covers Kimi, DeepSeek, GPT and every local server — Ollama,
llama.cpp, vLLM — because they all expose the same wire format. That is the
whole reason the spec routes the cheap, high-frequency ``FAST`` role to a local
quantized model: it is not a different integration, only a different base URL,
so it costs no API tokens and needs no extra code.

The model identifier always comes from configuration. Moving to a newer
generation of any family is an environment variable, never an edit here.
"""
from __future__ import annotations

import asyncio
import logging

from aegis.clock import CLOCK
from aegis.cortex.providers.base import CallParams, Completion, Provider, estimate_tokens

logger = logging.getLogger("aegis.cortex.openai")


def _build_client(api_key: str, base_url: str, timeout: float):
    """Construct an AsyncOpenAI client. Imported lazily — the SDK is heavy and
    a deployment that only uses the local HF path should not need it."""
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=api_key or "unused", base_url=base_url,
                       timeout=timeout)


class OpenAICompatibleProvider(Provider):
    """Chat completions against an OpenAI-shaped endpoint."""

    kind = "openai_compatible"

    def __init__(self, name: str, model: str, api_key: str = "",
                 base_url: str = "", *, requires_key: bool = True,
                 timeout: float = 60.0, client=None, client_factory=None):
        super().__init__(name, model)
        self.api_key = api_key
        self.base_url = base_url
        # Local servers (Ollama, llama.cpp) ignore the key entirely; demanding
        # one would make the offline path impossible to configure.
        self.requires_key = requires_key
        self.timeout = timeout
        self._client = client
        self._client_factory = client_factory

    # ── availability ─────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        if not self.model:
            return False        # nothing to ask
        if not self.base_url:
            return False        # nowhere to ask
        if self.requires_key and not self.api_key:
            return False        # not allowed to ask
        return True

    def unavailable_reason(self) -> str:
        """Why this provider was dropped, in words an operator can act on."""
        if not self.model:
            return f"{self.name}: no model id configured"
        if not self.base_url:
            return f"{self.name}: no base URL configured"
        if self.requires_key and not self.api_key:
            return f"{self.name}: no API key configured"
        return ""

    # ── client ───────────────────────────────────────────────────────

    def client(self):
        """The chat client, built on first use.

        Built lazily rather than in ``__init__`` so that constructing the whole
        router costs nothing when a provider is never actually called — which is
        the normal case for the failover tail of every route.
        """
        if self._client is None:
            factory = self._client_factory or _build_client
            self._client = factory(self.api_key, self.base_url, self.timeout)
        return self._client

    # ── call ─────────────────────────────────────────────────────────

    async def _invoke(self, messages: list[dict], params: CallParams) -> Completion:
        request: dict = {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }
        if params.top_p is not None:
            request["top_p"] = params.top_p
        if params.seed is not None:
            request["seed"] = params.seed
        if params.stop:
            request["stop"] = list(params.stop)

        started = CLOCK.monotonic()
        # The SDK carries its own timeout, but a hung connection setup or a
        # stalled stream can still outlive it; the outer bound is what keeps a
        # cognitive phase from waiting forever.
        response = await asyncio.wait_for(
            self.client().chat.completions.create(**request),
            timeout=params.timeout,
        )
        elapsed_ms = (CLOCK.monotonic() - started) * 1000

        text = ""
        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            text = (getattr(message, "content", None) or "") if message else ""

        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
        tokens_out = getattr(usage, "completion_tokens", None) if usage else None
        # Servers differ on whether they report usage at all (many local ones do
        # not). Falling back to an estimate keeps the resource accounting from
        # silently reading zero for every local call.
        if tokens_in is None:
            tokens_in = estimate_tokens(messages)
        if tokens_out is None:
            tokens_out = max(1, len(text) // 4)

        return Completion(
            text=text, provider=self.name, model=self.model,
            tokens_in=int(tokens_in), tokens_out=int(tokens_out),
            latency_ms=elapsed_ms,
        )

    def status(self) -> dict:
        return {**super().status(), "base_url": self.base_url,
                "requires_key": self.requires_key}
