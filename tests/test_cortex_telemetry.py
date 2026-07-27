"""The cortex publishes its own metrics (spec §3.5, Appendix G).

Every contour owes the time series a set of named metrics; the cortex owes
calls-by-role, tokens-by-provider, schema failures, repairs, breaker trips and
cache hit rate. A contour that reports nothing is a contour the discovery
engine cannot reason about and the dashboard cannot show.
"""
import asyncio

import pytest

from aegis.cortex.providers import NullProvider
from aegis.cortex.providers.base import CallParams, Completion, Provider
from aegis.cortex.router import Cortex, Role
from aegis.telemetry import metrics as M
from aegis.telemetry.store import Telemetry
from tests.cortex_fakes import ScriptedProvider


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def telemetry(tmp_path):
    return Telemetry(tmp_path / "telemetry")


def _cortex(telemetry, responses=("ok",), **kwargs):
    return Cortex(providers={"a": ScriptedProvider("a", responses=list(responses))},
                  routes={"fast": ["a"], "deep": ["a"]},
                  telemetry=telemetry, **kwargs)


# ── per-call metrics ─────────────────────────────────────────────────

def test_a_call_records_itself_under_its_role(telemetry):
    cortex = _cortex(telemetry)
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    telemetry.flush()
    roles = {row["tags"].get("role")
             for row in telemetry.series(M.CORTEX_CALLS).rows()}
    assert roles == {"fast"}


def test_a_call_records_tokens_under_its_provider(telemetry):
    cortex = _cortex(telemetry)
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    telemetry.flush()
    providers = {row["tags"].get("provider")
                 for row in telemetry.series(M.CORTEX_TOKENS).rows()}
    assert providers == {"a"}


def test_a_failed_call_records_no_tokens(telemetry):
    cortex = Cortex(providers={"a": ScriptedProvider("a", fail=True)},
                    routes={"fast": ["a"]}, telemetry=telemetry)
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    telemetry.flush()
    assert len(telemetry.series(M.CORTEX_TOKENS)) == 0


def test_a_schema_failure_is_recorded_with_its_schema_name(telemetry):
    cortex = _cortex(telemetry, responses=["garbage", "garbage"])
    _run(cortex.structured(Role.FAST, [{"role": "user", "content": "q"}], "answer"))
    telemetry.flush()
    names = {row["tags"].get("schema")
             for row in telemetry.series(M.CORTEX_SCHEMA_FAILURES).rows()}
    assert "answer" in names


# ── the periodic push ────────────────────────────────────────────────

def test_publish_metrics_emits_every_required_cortex_series(telemetry):
    cortex = _cortex(telemetry)
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    cortex.publish_metrics(tick=5)
    telemetry.flush()

    for metric in (M.CORTEX_CALLS, M.CORTEX_TOKENS, M.CORTEX_SCHEMA_FAILURES,
                   M.CORTEX_REPAIRS, M.CORTEX_BREAKER_TRIPS, M.CORTEX_CACHE_HIT_RATE):
        assert len(telemetry.series(metric)) >= 1, metric


def test_publish_metrics_covers_every_role_even_the_idle_ones(telemetry):
    # A role that was never called still needs a zero in the series, or a chart
    # of it would show a gap that looks like missing data.
    cortex = _cortex(telemetry)
    cortex.publish_metrics(tick=1)
    telemetry.flush()
    roles = {row["tags"].get("role")
             for row in telemetry.series(M.CORTEX_CALLS).rows()}
    assert roles == {r.value for r in Role}


def test_breaker_trips_are_summed_across_providers(telemetry):
    cortex = Cortex(providers={"a": ScriptedProvider("a", fail=True),
                               "b": ScriptedProvider("b", fail=True)},
                    routes={"deep": ["a", "b"]}, telemetry=telemetry)
    for breaker in cortex.breakers.values():
        breaker.threshold = 1
    _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}]))
    cortex.publish_metrics(tick=1)
    telemetry.flush()
    assert telemetry.series(M.CORTEX_BREAKER_TRIPS).last() == 2


def test_publishing_without_telemetry_is_a_no_op():
    cortex = Cortex(providers={}, routes={})
    cortex.publish_metrics(tick=1)      # must not raise


def test_a_broken_telemetry_store_cannot_break_a_call():
    class Exploding:
        def record(self, *a, **k):
            raise RuntimeError("disk on fire")

    cortex = Cortex(providers={"a": ScriptedProvider("a", responses=["ok"])},
                    routes={"fast": ["a"]}, telemetry=Exploding())
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}])) is not None


def test_a_broken_telemetry_store_cannot_break_publication():
    class Exploding:
        def record(self, *a, **k):
            raise RuntimeError("disk on fire")

    cortex = Cortex(providers={}, routes={}, telemetry=Exploding())
    cortex.publish_metrics(tick=1)      # must not raise


def test_an_unknown_metric_kind_is_ignored(telemetry):
    cortex = _cortex(telemetry)
    cortex._record_metric("not_a_metric", 1)
    telemetry.flush()
    assert telemetry.metrics() == [] or "not_a_metric" not in telemetry.metrics()


# ── saving ───────────────────────────────────────────────────────────

def test_save_persists_the_cache(tmp_path):
    from aegis.cortex.cache import ResponseCache
    path = tmp_path / "cache.json"
    cortex = Cortex(providers={"a": ScriptedProvider("a", responses=["ok"])},
                    routes={"fast": ["a"]}, cache=ResponseCache(path, ttl=0))
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    cortex.save()
    assert path.exists()


# ── the null provider ────────────────────────────────────────────────

def test_a_null_provider_is_never_available():
    assert NullProvider("ghost").available is False


def test_a_null_provider_explains_itself():
    provider = NullProvider("ghost", "misspelled in CORTEX_ROUTES")
    assert provider.status()["reason"] == "misspelled in CORTEX_ROUTES"


def test_calling_a_null_provider_fails_cleanly():
    completion = _run(NullProvider("ghost").call([], CallParams()))
    assert completion.ok is False


def test_a_provider_repr_names_it():
    assert "ghost" in repr(NullProvider("ghost"))


# ── completion bookkeeping ───────────────────────────────────────────

def test_completion_tokens_are_the_sum_of_both_directions():
    assert Completion(tokens_in=3, tokens_out=4).tokens == 7


def test_a_failure_completion_carries_the_reason():
    completion = Completion.failure("p", "m", RuntimeError("nope"))
    assert completion.ok is False and "nope" in completion.error


def test_a_very_long_error_is_truncated():
    completion = Completion.failure("p", "m", "x" * 5000)
    assert len(completion.error) <= 500


def test_completion_to_dict_previews_rather_than_dumps():
    completion = Completion(text="y" * 5000, provider="p", model="m")
    assert len(completion.to_dict()["text_preview"]) == 200


def test_a_provider_fills_in_its_own_identity():
    class Anonymous(Provider):
        kind = "anon"

        @property
        def available(self):
            return True

        async def _invoke(self, messages, params):
            return Completion(text="x")      # no provider/model/latency set

    completion = _run(Anonymous("named", "model-1").call([], CallParams()))
    assert completion.provider == "named"
    assert completion.model == "model-1"


def test_a_provider_that_reports_no_latency_gets_it_measured():
    from aegis.clock import FrozenClock, set_clock

    class Slow(Provider):
        kind = "slow"

        @property
        def available(self):
            return True

        async def _invoke(self, messages, params):
            clock.advance(0.25)
            return Completion(text="x")      # latency left at 0

    clock = FrozenClock()
    previous = set_clock(clock)
    try:
        completion = _run(Slow("slow", "m").call([], CallParams()))
    finally:
        set_clock(previous)
    assert completion.latency_ms == pytest.approx(250.0)


def test_a_provider_that_reports_its_own_latency_keeps_it():
    class Precise(Provider):
        kind = "precise"

        @property
        def available(self):
            return True

        async def _invoke(self, messages, params):
            return Completion(text="x", latency_ms=42.0)

    assert _run(Precise("p", "m").call([], CallParams())).latency_ms == 42.0


def test_an_unavailable_provider_refuses_before_invoking():
    class NeverReady(Provider):
        kind = "never"

        def __init__(self):
            super().__init__("never", "m")
            self.invoked = False

        @property
        def available(self):
            return False

        async def _invoke(self, messages, params):
            self.invoked = True
            return Completion(text="x")

    provider = NeverReady()
    completion = _run(provider.call([], CallParams()))
    assert completion.ok is False
    assert provider.invoked is False


def test_a_provider_records_its_failures():
    class Broken(Provider):
        kind = "broken"

        @property
        def available(self):
            return True

        async def _invoke(self, messages, params):
            raise RuntimeError("wire cut")

    provider = Broken("broken", "m")
    _run(provider.call([], CallParams()))
    status = provider.status()
    assert status["errors"] == 1
    assert "wire cut" in status["last_error"]
