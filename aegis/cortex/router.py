"""The cortex router: roles, failover, budget, cache, schemas (spec M8).

This is the "brain bark" the development text asks for — a replaceable outer
layer over a core that works without it. Two properties define it:

**Roles, not providers.** Callers ask for a *kind of thinking* (fast
classification, deep planning, code, an independent judgement) and the router
decides where that goes. That is what makes "put Kimi K3 on top" a routing table
entry instead of a refactor.

**Never load-bearing.** Every contour in this system has a deterministic path
that runs without any model at all. The router returns ``None`` rather than
raising when a role is unavailable, out of budget, or answering nonsense, and
the caller carries on. A failed cortex must degrade the system's judgement, not
its liveness — which is why ``test_offline_mode`` runs 500 ticks with every
provider dead.

Order of gates on every call: role availability → resource lease → circuit
breaker → cache → provider (with failover) → schema validation → one repair.
"""
from __future__ import annotations

import logging
from enum import Enum

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.cortex.breaker import CircuitBreaker
from aegis.cortex.cache import CacheEntry, ResponseCache, cache_key
from aegis.cortex.providers import (
    AnthropicProvider, CallParams, Completion, LocalHFProvider, NullProvider,
    OpenAICompatibleProvider, Provider, estimate_tokens,
)
from aegis.cortex import schemas as S

logger = logging.getLogger("aegis.cortex")


class Role(str, Enum):
    """What kind of thinking is being asked for."""

    FAST = "fast"    # short state appraisals, classification — the hot path
    DEEP = "deep"    # planning, strategy synthesis, hypotheses
    CODE = "code"    # skills, coding solutions
    JUDGE = "judge"  # criticism, verification, an independent second opinion


#: Sampling defaults per role. DEEP and CODE get more room to answer; JUDGE is
#: kept cold because a critic that improvises is not a critic.
ROLE_PARAMS: dict[Role, dict] = {
    Role.FAST: {"max_tokens": 1200, "temperature": 0.3},
    Role.DEEP: {"max_tokens": 4000, "temperature": 0.7},
    Role.CODE: {"max_tokens": 3000, "temperature": 0.2},
    Role.JUDGE: {"max_tokens": 2000, "temperature": 0.0},
}


class Cortex:
    """Role-based router over a set of providers."""

    def __init__(self, *, routes: dict | None = None, providers: dict | None = None,
                 cache: ResponseCache | None = None, telemetry=None,
                 resources=None, weight_modifier=None,
                 deterministic: bool | None = None):
        self.telemetry = telemetry
        # Set from stage 2 on. While it is None the outer LLM_MAX_CALLS_PER_RUN
        # fuse is the only budget; once it is set, no call happens without a
        # lease (§M4.3) — that is the difference between accounting and hoping.
        self.resources = resources
        self.deterministic = (cfg.CORTEX_DETERMINISTIC if deterministic is None
                              else bool(deterministic))

        self.providers: dict[str, Provider] = (
            dict(providers) if providers is not None
            else self.build_default_providers(weight_modifier))
        self.breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker(name, threshold=cfg.CORTEX_BREAKER_ERRORS,
                                 cooldown=cfg.CORTEX_BREAKER_COOLDOWN)
            for name in self.providers
        }
        self.cache = cache if cache is not None else ResponseCache(
            cfg.CORTEX_DIR / "cache.json",
            ttl=cfg.CORTEX_CACHE_TTL, max_entries=cfg.CORTEX_CACHE_MAX)

        self.routes: dict[Role, list[str]] = {}
        self.route_warnings: list[str] = []
        self.configure_routes(routes if routes is not None else cfg.CORTEX_ROUTES)

        # Counters published to telemetry and the dashboard.
        self.calls_by_role: dict[str, int] = {r.value: 0 for r in Role}
        self.tokens_by_provider: dict[str, int] = {name: 0 for name in self.providers}
        self.schema_failures = 0
        self.repairs = 0
        self.repairs_succeeded = 0
        self.lease_denials = 0
        self.role_unavailable = 0
        self.total_calls = 0
        self.total_errors = 0
        self.last_error = ""
        self.history: list[dict] = []

    # ── construction ─────────────────────────────────────────────────

    @staticmethod
    def build_default_providers(weight_modifier=None) -> dict[str, Provider]:
        """The providers named by the default routing table (§M8.4).

        All four hosted endpoints are the same class: they differ only in base
        URL, key and model id, which is exactly why swapping model families
        needs no code.
        """
        timeout = cfg.LLM_TIMEOUT_SECONDS
        return {
            "kimi": OpenAICompatibleProvider(
                "kimi", cfg.KIMI_MODEL, cfg.KIMI_API_KEY, cfg.KIMI_BASE_URL,
                timeout=timeout),
            "deepseek": OpenAICompatibleProvider(
                "deepseek", cfg.DEEPSEEK_MODEL, cfg.DEEPSEEK_API_KEY,
                cfg.DEEPSEEK_BASE_URL + "/v1", timeout=timeout),
            "openai": OpenAICompatibleProvider(
                "openai", cfg.OPENAI_MODEL, cfg.OPENAI_API_KEY,
                cfg.OPENAI_BASE_URL, timeout=timeout),
            "local_openai": OpenAICompatibleProvider(
                "local_openai", cfg.LOCAL_OPENAI_MODEL, cfg.LOCAL_OPENAI_API_KEY,
                cfg.LOCAL_OPENAI_BASE_URL, requires_key=False, timeout=timeout),
            "claude": AnthropicProvider(
                "claude", cfg.CLAUDE_MODEL, cfg.CLAUDE_API_KEY, timeout=timeout),
            "local_hf": LocalHFProvider("local_hf", weight_modifier),
        }

    def configure_routes(self, routes: dict) -> None:
        """Install the routing table, dropping links that cannot work.

        A route naming a provider with no key is a misconfiguration, not a
        crash: it is dropped with a warning and the chain continues to the next
        entry. A role with nothing left is marked unavailable, and the contours
        that would have used it take their deterministic path (§M8.4).
        """
        self.routes = {}
        self.route_warnings = []
        for role in Role:
            chain = routes.get(role.value, []) if isinstance(routes, dict) else []
            if isinstance(chain, str):
                chain = [chain]
            usable: list[str] = []
            for name in chain:
                provider = self.providers.get(name)
                if provider is None:
                    self.providers[name] = NullProvider(name, "not a known provider")
                    self.breakers.setdefault(name, CircuitBreaker(
                        name, threshold=cfg.CORTEX_BREAKER_ERRORS,
                        cooldown=cfg.CORTEX_BREAKER_COOLDOWN))
                    self.route_warnings.append(
                        f"role {role.value!r}: unknown provider {name!r} — dropped")
                    continue
                if not provider.available:
                    reason = getattr(provider, "unavailable_reason", lambda: "")()
                    self.route_warnings.append(
                        f"role {role.value!r}: {reason or f'{name} unavailable'} — dropped")
                    continue
                usable.append(name)
            self.routes[role] = usable
            if not usable:
                self.route_warnings.append(
                    f"role {role.value!r}: no provider available — this role will "
                    f"fall back to the deterministic path")
        for warning in self.route_warnings:
            logger.warning("Cortex route: %s", warning)

    # ── availability ─────────────────────────────────────────────────

    def available_roles(self) -> list[str]:
        return sorted(r.value for r in Role if self.chain_for(r))

    def role_available(self, role: Role | str) -> bool:
        return bool(self.chain_for(role))

    def chain_for(self, role: Role | str, *, exclude: tuple[str, ...] = ()) -> list[str]:
        """Providers to try for this role, in order, skipping open circuits.

        Uses the breaker's NON-mutating check: this method serves read paths
        too (``status()``, ``available_roles()``, every dashboard poll), and
        calling ``allows()`` here consumed half-open probes on every status
        read. The actual claim happens in ``_call_chain``, immediately before
        a provider is really called.
        """
        role = Role(role) if not isinstance(role, Role) else role
        return [name for name in self.routes.get(role, [])
                if name not in exclude
                and self.providers[name].available
                and self.breakers[name].would_allow()]

    @property
    def enabled(self) -> bool:
        return any(self.routes.get(role) for role in Role)

    # ── parameters ───────────────────────────────────────────────────

    def params_for(self, role: Role, **overrides) -> CallParams:
        base = dict(ROLE_PARAMS[role])
        base["timeout"] = cfg.LLM_TIMEOUT_SECONDS
        base.update({k: v for k, v in overrides.items() if v is not None})
        params = CallParams(**base)
        return params.deterministic() if self.deterministic else params

    # ── the call ─────────────────────────────────────────────────────

    async def call(self, role: Role | str, messages: list[dict], *,
                   lease=None, exclude_providers: tuple[str, ...] = (),
                   **param_overrides) -> Completion | None:
        """Run one request for ``role``. Returns None when it could not happen.

        None is a normal, expected outcome — no configured provider, no budget,
        every circuit open — and every caller must handle it by continuing
        deterministically rather than by failing.
        """
        role = Role(role) if not isinstance(role, Role) else role
        chain = self.chain_for(role, exclude=exclude_providers)
        if not chain:
            self.role_unavailable += 1
            logger.debug("Cortex role %s has no live provider", role.value)
            return None

        params = self.params_for(role, **param_overrides)
        estimated = estimate_tokens(messages) + params.max_tokens
        if not self._reserve(lease, estimated, role):
            self.lease_denials += 1
            return None

        completion = await self._call_chain(role, chain, messages, params)
        self._commit(lease, completion)
        self._record(role, completion)
        return completion

    async def _call_chain(self, role: Role, chain: list[str], messages: list[dict],
                          params: CallParams) -> Completion:
        """Try each provider in turn; the first success wins."""
        last: Completion | None = None
        for name in chain:
            provider = self.providers[name]
            breaker = self.breakers[name]
            key = cache_key(name, provider.model, messages, params.cache_key_part())

            cached = self.cache.get(key)
            # An EMPTY cached text is never served. A server that answers
            # ok/null-content (length truncation, a content filter) used to be
            # cached and then replayed for the whole TTL — persisted across
            # restarts — so the role was silently dead for an hour while
            # failover never ran and the breaker recorded nothing (audit:
            # cache poisoning). Skipping the entry re-runs the real chain.
            if cached is not None and cached.text.strip():
                return Completion(
                    text=cached.text, provider=cached.provider, model=cached.model,
                    tokens_in=cached.tokens_in, tokens_out=cached.tokens_out,
                    cached=True, role=role.value,
                )

            # Claim the breaker NOW, not in chain_for: the chain is computed
            # with a non-mutating check, and the half-open probe must be
            # consumed only by the caller actually about to pay for the call.
            if not breaker.allows():
                continue

            completion = await provider.call(messages, params)
            completion.role = role.value
            if completion.ok:
                breaker.record_success()
                # Cache only completions that carry text — see above.
                if completion.text.strip():
                    self.cache.put(key, CacheEntry(
                        text=completion.text, provider=completion.provider,
                        model=completion.model, tokens_in=completion.tokens_in,
                        tokens_out=completion.tokens_out, stored_at=CLOCK.now()))
                return completion

            breaker.record_failure()
            self.total_errors += 1
            self.last_error = f"[{name}] {completion.error}"
            last = completion
            logger.info("Cortex failover: %s failed for role %s (%s)",
                        name, role.value, completion.error[:120])

        return last or Completion.failure("none", "", "all providers failed")

    # ── structured output (§M8.5) ────────────────────────────────────

    async def structured(self, role: Role | str, messages: list[dict],
                         schema_name: str, *, lease=None,
                         exclude_providers: tuple[str, ...] = (),
                         **param_overrides) -> dict | None:
        """Call and return validated data, or None.

        One repair round-trip on a schema failure, then give up. "Almost right"
        data is refused outright: a half-parsed decision entering the core is
        worse than no decision, because the deterministic fallback is correct
        and the half-parsed one merely looks like it.
        """
        schema = S.schema_for(schema_name)
        completion = await self.call(role, messages, lease=lease,
                                     exclude_providers=exclude_providers,
                                     **param_overrides)
        if completion is None or not completion.ok:
            return None

        payload, errors = S.parse_and_validate(completion.text, schema)
        if not errors:
            return payload

        self.schema_failures += 1
        self._record_metric("schema_failures", 1, tags={"schema": schema_name})
        if cfg.CORTEX_MAX_REPAIRS < 1:
            return None

        self.repairs += 1
        repair_messages = list(messages) + [
            {"role": "assistant", "content": completion.text[:4000]},
            {"role": "user", "content": S.repair_instruction(errors, schema_name)},
        ]
        repaired = await self.call(role, repair_messages, lease=lease,
                                   exclude_providers=exclude_providers,
                                   **param_overrides)
        if repaired is None or not repaired.ok:
            return None
        payload, errors = S.parse_and_validate(repaired.text, schema)
        if errors:
            self.schema_failures += 1
            logger.info("Cortex repair failed for %s: %s", schema_name, errors[:3])
            return None
        self.repairs_succeeded += 1
        return payload

    async def judge(self, messages: list[dict], schema_name: str, *,
                    authored_by: str = "", lease=None, **param_overrides) -> dict | None:
        """A second opinion from a provider that did not write the artefact.

        Independence is the entire value of a critic: asking the same model
        whether its own output is good measures its self-consistency, not the
        output's quality. When no other provider is available the same one is
        used rather than skipping the check — a weak review beats none — and
        the result says which happened.
        """
        exclude = (authored_by,) if authored_by else ()
        if exclude and not self.chain_for(Role.JUDGE, exclude=exclude):
            exclude = ()
        return await self.structured(Role.JUDGE, messages, schema_name,
                                     lease=lease, exclude_providers=exclude,
                                     **param_overrides)

    def judge_is_independent(self, authored_by: str) -> bool:
        """Whether a judgement of ``authored_by``'s work can avoid that provider."""
        return bool(authored_by) and bool(
            self.chain_for(Role.JUDGE, exclude=(authored_by,)))

    # ── resources ────────────────────────────────────────────────────

    def _reserve(self, lease, estimated_tokens: int, role: Role) -> bool:
        """Whether this call is paid for.

        With no resource manager attached the outer per-run fuse is the only
        limit and the call proceeds. With one attached, a call without a valid
        lease is refused — that is what stops the cortex from being an
        unmetered hole in the budget (§M4.3).

        The pre-call estimate is checked against what the lease actually holds:
        computing an estimate and then ignoring it would make the reservation
        decorative, and the overspend would only be noticed after the tokens
        were gone.
        """
        if self.resources is None:
            return True
        if lease is None:
            logger.debug("Cortex call for role %s refused: no lease", role.value)
            return False
        if not getattr(lease, "active", True):
            return False
        allowance = getattr(lease, "tokens", None)
        if allowance is not None and estimated_tokens > allowance:
            logger.debug("Cortex call for role %s refused: needs ~%d tokens, "
                         "lease holds %d", role.value, estimated_tokens, allowance)
            return False
        return True

    def _commit(self, lease, completion: Completion) -> None:
        # Only reachable after `_reserve` passed, which already guaranteed a
        # lease; and `_call_chain` always returns a Completion, failed or not.
        # Guarding against either being None here would be unreachable code
        # pretending to be caution.
        if self.resources is None:
            return
        try:
            self.resources.commit_tokens(lease, completion.tokens,
                                         calls=0 if completion.cached else 1)
        except Exception:
            logger.exception("Failed to commit cortex resource usage")

    # ── bookkeeping ──────────────────────────────────────────────────

    def _record(self, role: Role, completion: Completion | None) -> None:
        if completion is None:
            return
        self.total_calls += 1
        self.calls_by_role[role.value] = self.calls_by_role.get(role.value, 0) + 1
        if completion.ok:
            self.tokens_by_provider[completion.provider] = (
                self.tokens_by_provider.get(completion.provider, 0) + completion.tokens)
        self.history.append(completion.to_dict())
        if len(self.history) > 100:
            self.history = self.history[-100:]

        self._record_metric("calls", 1, tags={"role": role.value})
        if completion.ok:
            self._record_metric("tokens", completion.tokens,
                                tags={"provider": completion.provider})

    def _record_metric(self, kind: str, value, tags: dict | None = None) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        name = {
            "calls": M.CORTEX_CALLS,
            "tokens": M.CORTEX_TOKENS,
            "schema_failures": M.CORTEX_SCHEMA_FAILURES,
        }.get(kind)
        if name is None:
            return
        try:
            self.telemetry.record(name, value, tags=tags)
        except Exception:
            logger.exception("Cortex telemetry record failed")

    def publish_metrics(self, tick: int) -> None:
        """Push the aggregate counters into the time series (spec Appendix G)."""
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            self.telemetry.record(M.CORTEX_REPAIRS, self.repairs, tick)
            self.telemetry.record(M.CORTEX_BREAKER_TRIPS,
                                  sum(b.trips for b in self.breakers.values()), tick)
            self.telemetry.record(M.CORTEX_CACHE_HIT_RATE, self.cache.hit_rate(), tick)
            self.telemetry.record(M.CORTEX_SCHEMA_FAILURES, self.schema_failures, tick)
            for role in Role:
                self.telemetry.record(M.CORTEX_CALLS, self.calls_by_role.get(role.value, 0),
                                      tick, tags={"role": role.value})
            for name, tokens in sorted(self.tokens_by_provider.items()):
                self.telemetry.record(M.CORTEX_TOKENS, tokens, tick,
                                      tags={"provider": name})
        except Exception:
            logger.exception("Cortex metric publication failed")

    def save(self) -> None:
        self.cache.save()

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "deterministic": self.deterministic,
            "routes": {role.value: list(names) for role, names in self.routes.items()},
            "available_roles": self.available_roles(),
            "route_warnings": list(self.route_warnings),
            "providers": {name: provider.status()
                          for name, provider in sorted(self.providers.items())},
            "breakers": {name: breaker.status()
                         for name, breaker in sorted(self.breakers.items())},
            "cache": self.cache.status(),
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "last_error": self.last_error if self.total_errors else "",
            "calls_by_role": dict(self.calls_by_role),
            "tokens_by_provider": dict(self.tokens_by_provider),
            "schema_failures": self.schema_failures,
            "repairs": self.repairs,
            "repairs_succeeded": self.repairs_succeeded,
            "lease_denials": self.lease_denials,
            "role_unavailable": self.role_unavailable,
            "recent": self.history[-5:],
        }
