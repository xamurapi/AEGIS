"""The cortex: a replaceable model layer over a core that runs without it (spec M8)."""
from aegis.cortex import prompts, schemas
from aegis.cortex.breaker import CircuitBreaker
from aegis.cortex.cache import CacheEntry, ResponseCache, cache_key
from aegis.cortex.providers import (
    AnthropicProvider, CallParams, Completion, LocalHFProvider, NullProvider,
    OpenAICompatibleProvider, Provider,
)
from aegis.cortex.router import Cortex, Role

__all__ = [
    "AnthropicProvider", "CacheEntry", "CallParams", "CircuitBreaker",
    "Completion", "Cortex", "LocalHFProvider", "NullProvider",
    "OpenAICompatibleProvider", "Provider", "ResponseCache", "Role",
    "cache_key", "prompts", "schemas",
]
