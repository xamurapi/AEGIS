"""What every model provider has to look like (spec M8.2).

The router talks to providers only through this interface, which is what makes
"switch to Kimi" an environment variable rather than a code change. A provider
is responsible for one thing — turning messages into text — and for reporting
honestly how much that cost. Everything else (roles, failover, budget, caching,
schema validation) belongs to the router and is deliberately not duplicated
here.

Two invariants providers must respect:

* **Never raise.** A failed call returns a :class:`Completion` with ``ok=False``
  and the reason. The router needs to try the next provider in the chain, and
  an exception crossing that boundary would take the cognitive phase with it.
* **Never block indefinitely.** Every call is bounded by ``params.timeout``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

from aegis.clock import CLOCK

logger = logging.getLogger("aegis.cortex.provider")

# Rough characters-per-token used for pre-call cost estimates. The exact value
# is model-specific and unknowable without the tokenizer; it only has to be
# close enough for a resource lease to be sized before the call, and the actual
# usage is committed afterwards.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class CallParams:
    """Everything about *how* to call, separate from *what* to send."""

    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float | None = None
    seed: int | None = None
    timeout: float = 60.0
    stop: tuple[str, ...] = ()

    def deterministic(self) -> CallParams:
        """The same call with all sampling pinned (spec §M8.6).

        Comparative runs — A/B harnesses, the reasoning arena, evolution — must
        not have their metric differences explained by model sampling noise, so
        they pin temperature and top_p and pass a fixed seed where the provider
        supports one.
        """
        return replace(self, temperature=0.0, top_p=1.0,
                       seed=0 if self.seed is None else self.seed)

    def cache_key_part(self) -> str:
        return (f"mt={self.max_tokens};t={self.temperature};p={self.top_p};"
                f"s={self.seed};stop={','.join(self.stop)}")


@dataclass
class Completion:
    """One model response, successful or not."""

    text: str = ""
    provider: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    ok: bool = True
    error: str = ""
    cached: bool = False
    role: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @classmethod
    def failure(cls, provider: str, model: str, error: str,
                latency_ms: float = 0.0) -> Completion:
        return cls(provider=provider, model=model, ok=False,
                   error=str(error)[:500], latency_ms=latency_ms)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "ok": self.ok,
            "error": self.error,
            "cached": self.cached,
            "role": self.role,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": round(self.latency_ms, 1),
            "text_preview": self.text[:200],
        }


def estimate_tokens(messages: list[dict]) -> int:
    """Pre-call token estimate for sizing a resource lease.

    Deliberately crude and deliberately an over-estimate at the margin: a lease
    that turns out too small is a call that gets refused after the fact, which
    is worse than one that reserved slightly too much.
    """
    total = 0
    for message in messages or []:
        content = message.get("content", "") if isinstance(message, dict) else ""
        total += len(str(content)) // CHARS_PER_TOKEN + 4
    return max(1, total)


class Provider(ABC):
    """A single model endpoint."""

    #: Which wire format this speaks — used to pick the message translation.
    kind: str = "abstract"

    def __init__(self, name: str, model: str = ""):
        self.name = name
        self.model = model
        self.calls = 0
        self.errors = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.last_error = ""
        self.last_latency_ms = 0.0

    # ── contract ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this provider is configured well enough to be tried.

        A provider that is merely *down* is still available; that is the
        breaker's business. This answers "was it ever set up?" — no API key, no
        base URL, no model id.
        """

    @abstractmethod
    async def _invoke(self, messages: list[dict], params: CallParams) -> Completion:
        """Do the actual call. Subclasses implement only this."""

    # ── shared behaviour ─────────────────────────────────────────────

    async def call(self, messages: list[dict], params: CallParams) -> Completion:
        """Invoke the provider, recording cost and never raising."""
        if not self.available:
            return Completion.failure(self.name, self.model, "provider not configured")
        started = CLOCK.monotonic()
        try:
            completion = await self._invoke(messages, params)
        except Exception as exc:
            elapsed = (CLOCK.monotonic() - started) * 1000
            self.errors += 1
            self.last_error = str(exc)[:300]
            self.last_latency_ms = elapsed
            logger.warning("Provider %s failed: %s", self.name, exc)
            return Completion.failure(self.name, self.model, exc, elapsed)

        if not completion.provider:
            completion.provider = self.name
        if not completion.model:
            completion.model = self.model
        if not completion.latency_ms:
            completion.latency_ms = (CLOCK.monotonic() - started) * 1000

        self.last_latency_ms = completion.latency_ms
        if completion.ok:
            self.calls += 1
            self.tokens_in += completion.tokens_in
            self.tokens_out += completion.tokens_out
        else:
            self.errors += 1
            self.last_error = completion.error
        return completion

    def status(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "model": self.model,
            "available": self.available,
            "calls": self.calls,
            "errors": self.errors,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_in + self.tokens_out,
            "last_error": self.last_error if self.errors else "",
            "last_latency_ms": round(self.last_latency_ms, 1),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} model={self.model!r}>"


class NullProvider(Provider):
    """A named provider that is configured out of existence.

    Used where a route mentions a provider that has no key or endpoint: keeping
    a placeholder means the route can still be reported and explained, instead
    of the name silently disappearing from the status page.
    """

    kind = "null"

    def __init__(self, name: str, reason: str = "not configured"):
        super().__init__(name, model="")
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    async def _invoke(self, messages: list[dict], params: CallParams) -> Completion:
        return Completion.failure(self.name, self.model, self.reason)

    def status(self) -> dict:
        return {**super().status(), "reason": self.reason}
