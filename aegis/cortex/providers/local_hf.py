"""The in-process transformers model (spec M8.3).

This provider exists for two narrow jobs, and it is worth being explicit about
which, because conflating them is the mistake the development text complains
about ("1.5B is too small"):

* it is the **trainable** model — the one place the system rewrites its own
  weights, where being small is a feature, not a limitation;
* it is the **offline last resort** when no OpenAI-compatible server is running
  at all.

It is *not* the fast inference path. That is a quantized 7–8B served over
``/v1`` by Ollama or llama.cpp, which is several times faster on CPU than
``transformers`` in fp32 and needs no VRAM — and which reaches the router
through :mod:`~aegis.cortex.providers.openai_compatible`, not this module.

Loading is delegated to :class:`~aegis.layers.weight_modifier.WeightModifier`,
so the memory and quantization guards of §M8.3a apply here too.
"""
from __future__ import annotations

import asyncio
import logging

from aegis.clock import CLOCK
from aegis.cortex.providers.base import CallParams, Completion, Provider, estimate_tokens

logger = logging.getLogger("aegis.cortex.local_hf")

# Prompt characters kept when the conversation is longer than the model can
# take. The tail is kept, not the head: the most recent turn is the request.
MAX_PROMPT_CHARS = 4000


def flatten_messages(messages: list[dict]) -> str:
    """Render a chat conversation as the plain prompt a base model expects."""
    parts = []
    for message in messages or []:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        parts.append(f"{role.capitalize()}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


class LocalHFProvider(Provider):
    """Generation through a locally held ``transformers`` model."""

    kind = "local_hf"

    def __init__(self, name: str = "local_hf", weight_modifier=None,
                 *, max_prompt_chars: int = MAX_PROMPT_CHARS):
        super().__init__(name, model="local")
        self.weight_modifier = weight_modifier
        self.max_prompt_chars = max_prompt_chars

    @property
    def available(self) -> bool:
        return bool(self.weight_modifier is not None
                    and getattr(self.weight_modifier, "model_loaded", False))

    def unavailable_reason(self) -> str:
        if self.weight_modifier is None:
            return f"{self.name}: no weight modifier attached"
        if not getattr(self.weight_modifier, "model_loaded", False):
            return f"{self.name}: local model not loaded"
        return ""

    async def _invoke(self, messages: list[dict], params: CallParams) -> Completion:
        prompt = flatten_messages(messages)
        if len(prompt) > self.max_prompt_chars:
            prompt = prompt[-self.max_prompt_chars:]

        started = CLOCK.monotonic()
        # Generation is synchronous and CPU-bound; running it inline would
        # freeze the event loop for the whole decode. The executor thread may
        # outlive the timeout, but the coroutine returns and the tick continues.
        text = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None, self.weight_modifier.generate, prompt, params.max_tokens),
            timeout=params.timeout,
        )
        elapsed_ms = (CLOCK.monotonic() - started) * 1000
        text = text or ""

        checkpoint = getattr(self.weight_modifier, "current_checkpoint", None)
        self.model = checkpoint or "base"
        return Completion(
            text=text, provider=self.name, model=self.model,
            tokens_in=estimate_tokens(messages),
            tokens_out=max(1, len(text) // 4),
            latency_ms=elapsed_ms,
        )
