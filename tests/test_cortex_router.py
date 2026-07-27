"""Roles, routing, failover and independence of the critic (spec M8.4, M8.9).

The router's job is to make "which model" a configuration question. These tests
pin the three properties that claim rests on: a route to an unconfigured
provider is dropped rather than fatal, a failing provider hands over to the next
one, and a role with nothing left degrades to no answer instead of an error.
"""
import asyncio

import pytest

from aegis.cortex.providers import NullProvider
from aegis.cortex.router import Cortex, Role
from tests.cortex_fakes import (
    FakeLease, FakeResources, ScriptedProvider, anthropic_provider, openai_provider,
)


def _run(coro):
    return asyncio.run(coro)


def _cortex(providers, routes, **kwargs):
    kwargs.setdefault("cache", _NoCache())
    return Cortex(providers=providers, routes=routes, **kwargs)


class _NoCache:
    """Cache disabled — these tests are about routing, not reuse."""

    def get(self, key):
        return None

    def put(self, key, entry):
        return None

    def hit_rate(self):
        return 0.0

    def save(self):
        return None

    def status(self):
        return {}


# ── route validation ─────────────────────────────────────────────────

def test_a_route_to_a_provider_without_a_key_is_dropped():
    providers = {"kimi": openai_provider("kimi"),
                 "claude": anthropic_provider("claude")}
    providers["claude"].api_key = ""          # no key configured
    cortex = _cortex(providers, {"deep": ["claude", "kimi"]})
    assert cortex.routes[Role.DEEP] == ["kimi"]


def test_dropping_a_route_explains_why():
    providers = {"claude": anthropic_provider("claude")}
    providers["claude"].api_key = ""
    cortex = _cortex(providers, {"deep": ["claude"]})
    assert any("ANTHROPIC_API_KEY" in w for w in cortex.route_warnings)


def test_an_unknown_provider_name_is_dropped_not_fatal():
    cortex = _cortex({"kimi": openai_provider("kimi")},
                     {"deep": ["nonexistent", "kimi"]})
    assert cortex.routes[Role.DEEP] == ["kimi"]
    assert any("unknown provider" in w for w in cortex.route_warnings)
    assert isinstance(cortex.providers["nonexistent"], NullProvider)


def test_a_role_with_no_usable_provider_is_marked_unavailable():
    cortex = _cortex({"kimi": openai_provider("kimi")}, {"deep": ["kimi"]})
    assert cortex.role_available(Role.DEEP) is True
    assert cortex.role_available(Role.FAST) is False


def test_a_string_route_is_accepted_as_a_one_element_chain():
    cortex = _cortex({"kimi": openai_provider("kimi")}, {"deep": "kimi"})
    assert cortex.routes[Role.DEEP] == ["kimi"]


def test_a_missing_role_in_the_table_is_simply_unavailable():
    cortex = _cortex({"kimi": openai_provider("kimi")}, {"deep": ["kimi"]})
    assert cortex.routes[Role.JUDGE] == []


def test_available_roles_lists_only_live_ones():
    cortex = _cortex({"kimi": openai_provider("kimi")},
                     {"deep": ["kimi"], "code": ["kimi"], "fast": []})
    assert cortex.available_roles() == ["code", "deep"]


def test_a_cortex_with_no_routes_at_all_reports_disabled():
    assert _cortex({}, {}).enabled is False


# ── the default provider set ─────────────────────────────────────────

def test_the_deepseek_endpoint_gets_the_version_suffix_it_needs():
    # DEEPSEEK_BASE_URL is the bare host; the chat-completions path lives under
    # /v1 like every other OpenAI-compatible server.
    provider = Cortex.build_default_providers()["deepseek"]
    assert provider.base_url.endswith("/v1")


def test_the_local_server_is_the_only_one_that_needs_no_key():
    providers = Cortex.build_default_providers()
    assert providers["local_openai"].requires_key is False
    assert providers["kimi"].requires_key is True
    assert providers["openai"].requires_key is True


def test_the_default_set_covers_every_provider_the_routes_name():
    import aegis.config as cfg
    named = {name for chain in cfg.CORTEX_ROUTES_DEFAULT.values() for name in chain}
    assert named <= set(Cortex.build_default_providers())


# ── switching provider is configuration, not code (§M8.8) ────────────

def test_switching_to_kimi_is_a_routing_table_change():
    providers = {"kimi": openai_provider("kimi", responses=["from-kimi"]),
                 "claude": anthropic_provider("claude", responses=["from-claude"])}
    on_claude = _cortex(providers, {"deep": ["claude"]})
    assert _run(on_claude.call(Role.DEEP, [{"role": "user", "content": "q"}])).text \
        == "from-claude"

    on_kimi = _cortex(providers, {"deep": ["kimi"]})
    assert _run(on_kimi.call(Role.DEEP, [{"role": "user", "content": "q"}])).text \
        == "from-kimi"


# ── calling ──────────────────────────────────────────────────────────

def test_a_call_returns_the_first_providers_answer():
    cortex = _cortex({"a": ScriptedProvider("a", responses=["hello"])},
                     {"fast": ["a"]})
    completion = _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    assert completion.ok and completion.text == "hello"
    assert completion.role == "fast"


def test_a_call_for_an_unavailable_role_returns_none():
    cortex = _cortex({}, {})
    assert _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}])) is None


def test_an_unavailable_role_is_counted_not_raised():
    cortex = _cortex({}, {})
    _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}]))
    assert cortex.role_unavailable == 1


def test_failover_moves_to_the_next_provider():
    failing = ScriptedProvider("a", fail=True)
    working = ScriptedProvider("b", responses=["second"])
    cortex = _cortex({"a": failing, "b": working}, {"deep": ["a", "b"]})
    completion = _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}]))
    assert completion.text == "second"
    assert completion.provider == "b"


def test_failover_tries_providers_in_declared_order():
    first = ScriptedProvider("a", responses=["first"])
    second = ScriptedProvider("b", responses=["second"])
    cortex = _cortex({"a": first, "b": second}, {"deep": ["b", "a"]})
    assert _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}])).provider == "b"


def test_every_provider_failing_returns_a_failed_completion_not_an_exception():
    cortex = _cortex({"a": ScriptedProvider("a", fail=True),
                      "b": ScriptedProvider("b", fail=True)},
                     {"deep": ["a", "b"]})
    completion = _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}]))
    assert completion.ok is False


def test_a_provider_raising_is_contained():
    cortex = _cortex({"a": ScriptedProvider("a", responses=[RuntimeError("boom")])},
                     {"deep": ["a"]})
    completion = _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}]))
    assert completion.ok is False
    assert "boom" in completion.error


# ── role parameters ──────────────────────────────────────────────────

def test_judge_is_kept_cold():
    # A critic that improvises is not a critic.
    cortex = _cortex({}, {})
    cortex.deterministic = False
    assert cortex.params_for(Role.JUDGE).temperature == 0.0
    assert cortex.params_for(Role.DEEP).temperature > 0.0


def test_deterministic_mode_pins_every_sampling_knob():
    cortex = _cortex({}, {}, deterministic=True)
    params = cortex.params_for(Role.DEEP)
    assert params.temperature == 0.0
    assert params.top_p == 1.0
    assert params.seed is not None


def test_parameter_overrides_are_applied():
    cortex = _cortex({}, {})
    cortex.deterministic = False
    assert cortex.params_for(Role.FAST, max_tokens=99).max_tokens == 99


def test_a_none_override_does_not_erase_the_default():
    cortex = _cortex({}, {})
    assert cortex.params_for(Role.FAST, max_tokens=None).max_tokens > 0


# ── the critic must be independent (§M8.4) ───────────────────────────

def test_judge_avoids_the_provider_that_authored_the_artefact():
    author = ScriptedProvider("kimi", responses=['{"answer": "author"}'])
    other = ScriptedProvider("claude", responses=['{"answer": "independent"}'])
    cortex = _cortex({"kimi": author, "claude": other},
                     {"judge": ["kimi", "claude"]})
    assert cortex.judge_is_independent("kimi") is True

    verdict = _run(cortex.judge([{"role": "user", "content": "q"}], "answer",
                                authored_by="kimi"))
    # Asking the author whether its own work is good measures self-consistency,
    # not quality — the review has to come from the other provider.
    assert verdict == {"answer": "independent"}
    assert author.invocations == []


def test_the_judge_chain_drops_the_author():
    cortex = _cortex({"kimi": ScriptedProvider("kimi"),
                      "claude": ScriptedProvider("claude")},
                     {"judge": ["kimi", "claude"]})
    assert cortex.chain_for(Role.JUDGE, exclude=("kimi",)) == ["claude"]


def test_judge_falls_back_to_the_author_when_it_is_the_only_option():
    # A weak review beats no review; the caller can tell which it got.
    only = ScriptedProvider("kimi", responses=['{"answer": 1}'])
    cortex = _cortex({"kimi": only}, {"judge": ["kimi"]})
    assert cortex.judge_is_independent("kimi") is False
    assert _run(cortex.judge([{"role": "user", "content": "q"}], "answer",
                             authored_by="kimi")) == {"answer": 1}


def test_judge_with_no_author_named_uses_the_normal_chain():
    provider = ScriptedProvider("claude", responses=['{"answer": 2}'])
    cortex = _cortex({"claude": provider}, {"judge": ["claude"]})
    assert _run(cortex.judge([{"role": "user", "content": "q"}], "answer")) \
        == {"answer": 2}


# ── accounting ───────────────────────────────────────────────────────

def test_calls_are_counted_per_role():
    cortex = _cortex({"a": ScriptedProvider("a", responses=["1", "2"])},
                     {"fast": ["a"]})
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "r"}]))
    assert cortex.calls_by_role["fast"] == 2


def test_tokens_are_counted_per_provider():
    cortex = _cortex({"a": ScriptedProvider("a", responses=["x"])}, {"fast": ["a"]})
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    assert cortex.tokens_by_provider["a"] == 30      # 10 in + 20 out


def test_a_failed_call_adds_no_tokens():
    cortex = _cortex({"a": ScriptedProvider("a", fail=True)}, {"fast": ["a"]})
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    assert cortex.tokens_by_provider.get("a", 0) == 0


def test_history_is_bounded():
    cortex = _cortex({"a": ScriptedProvider("a", responses=["x"] * 130)},
                     {"fast": ["a"]})
    for _ in range(120):
        _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    assert len(cortex.history) <= 100


def test_status_reports_routes_providers_and_breakers():
    cortex = _cortex({"a": ScriptedProvider("a")}, {"fast": ["a"]})
    status = cortex.status()
    assert status["routes"]["fast"] == ["a"]
    assert "a" in status["providers"]
    assert "a" in status["breakers"]


# ── budget gate (§M4.3) ──────────────────────────────────────────────

def test_without_a_resource_manager_a_call_needs_no_lease():
    cortex = _cortex({"a": ScriptedProvider("a", responses=["x"])}, {"fast": ["a"]})
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}])) is not None


def test_with_a_resource_manager_a_call_without_a_lease_is_refused():
    provider = ScriptedProvider("a", responses=["x"])
    cortex = _cortex({"a": provider}, {"fast": ["a"]}, resources=FakeResources())
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}])) is None
    assert provider.invocations == []          # the provider was never reached
    assert cortex.lease_denials == 1


def test_an_inactive_lease_is_refused():
    provider = ScriptedProvider("a", responses=["x"])
    cortex = _cortex({"a": provider}, {"fast": ["a"]}, resources=FakeResources())
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}],
                            lease=FakeLease(active=False))) is None


def test_a_valid_lease_lets_the_call_through_and_is_charged():
    resources = FakeResources()
    cortex = _cortex({"a": ScriptedProvider("a", responses=["x"])},
                     {"fast": ["a"]}, resources=resources)
    lease = FakeLease()
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}],
                            lease=lease)) is not None
    assert resources.commits == [(30, 1)]


def test_a_cache_hit_is_labelled_as_one():
    # Callers use this to tell a fresh judgement from a replayed one; a hit
    # that claimed to be fresh would make an arena run look independent when
    # it was the same answer twice.
    from aegis.cortex.cache import ResponseCache
    provider = ScriptedProvider("a", responses=["first", "second"])
    cortex = _cortex({"a": provider}, {"fast": ["a"]},
                     cache=ResponseCache(None, ttl=0))
    messages = [{"role": "user", "content": "q"}]
    assert _run(cortex.call(Role.FAST, messages)).cached is False
    assert _run(cortex.call(Role.FAST, messages)).cached is True


def test_a_repair_whose_call_cannot_happen_yields_none():
    from aegis.cortex.cache import ResponseCache

    class DiesAfterFirstCall(ScriptedProvider):
        async def _invoke(self, messages, params):
            completion = await super()._invoke(messages, params)
            self._available = False      # the endpoint goes away mid-exchange
            return completion

    provider = DiesAfterFirstCall("a", responses=["not json"])
    cortex = _cortex({"a": provider}, {"fast": ["a"]}, cache=ResponseCache(None))
    assert _run(cortex.structured(Role.FAST, [{"role": "user", "content": "q"}],
                                  "answer")) is None


def test_a_broken_resource_manager_cannot_break_a_call():
    class Exploding(FakeResources):
        def commit_tokens(self, lease, tokens, calls=1):
            raise RuntimeError("accounting on fire")

    cortex = _cortex({"a": ScriptedProvider("a", responses=["x"])},
                     {"fast": ["a"]}, resources=Exploding())
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}],
                            lease=FakeLease())) is not None
