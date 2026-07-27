"""Model providers behind a single interface (spec M8.2)."""
from aegis.cortex.providers.anthropic import AnthropicProvider
from aegis.cortex.providers.base import (
    CallParams, Completion, NullProvider, Provider, estimate_tokens,
)
from aegis.cortex.providers.local_hf import LocalHFProvider
from aegis.cortex.providers.openai_compatible import OpenAICompatibleProvider

__all__ = [
    "AnthropicProvider", "CallParams", "Completion", "LocalHFProvider",
    "NullProvider", "OpenAICompatibleProvider", "Provider", "estimate_tokens",
]
