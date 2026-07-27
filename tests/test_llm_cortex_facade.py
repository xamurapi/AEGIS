"""LLMEngine delegates to the cortex (spec M8.7).

The public surface the rest of the system calls does not change; what changes is
where the answer comes from. Each method is pinned to the role it should use —
FAST for the per-tick appraisals, DEEP for planning and curiosity, CODE for
anything that writes code — because that mapping is what keeps the token budget
on the roles that need frontier quality (§M8.4).
"""
import asyncio
import sys
import types

import pytest

from aegis.cortex.cache import ResponseCache
from aegis.cortex.providers import AnthropicProvider, CallParams
from aegis.cortex.router import Cortex, Role
from aegis.llm import LLMEngine
from tests.cortex_fakes import FakeAnthropicClient, ScriptedProvider


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr("aegis.llm.TOKEN_STATS_FILE", tmp_path / "stats.json")
    return LLMEngine()


def _wire(engine, responses, roles=("fast", "deep", "code", "judge")):
    """Point every role at one scripted provider and return it."""
    provider = ScriptedProvider("scripted", responses=list(responses))
    engine.cortex = Cortex(providers={"scripted": provider},
                           routes={role: ["scripted"] for role in roles},
                           cache=ResponseCache(None))
    return provider


# ── each method reaches the cortex ───────────────────────────────────

def test_evaluate_state_goes_through_the_cortex(engine):
    _wire(engine, ['{"assessment": "fine", "insight": "i"}'])
    result = _run(engine.evaluate_state({"tick": 1}))
    assert result["via"] == "cortex"
    assert result["parsed"]["assessment"] == "fine"


def test_make_decision_goes_through_the_cortex(engine):
    _wire(engine, ['{"chosen": 2, "reasoning": "r", "confidence": 0.8}'])
    result = _run(engine.make_decision(["a", "b"], {"tick": 1}))
    assert result["parsed"]["chosen"] == 2


def test_reflect_goes_through_the_cortex(engine):
    _wire(engine, ['{"learning": "something"}'])
    assert _run(engine.reflect({"tick": 1}))["parsed"]["learning"] == "something"


def test_generate_curiosity_goes_through_the_cortex(engine):
    _wire(engine, ['{"topic": "topology"}'])
    assert _run(engine.generate_curiosity([]))["parsed"]["topic"] == "topology"


def test_propose_skill_goes_through_the_cortex(engine):
    _wire(engine, ['{"code": "def solve(p):\\n    return 1\\n"}'])
    code = _run(engine.propose_skill("calc", [{"payload": {}, "expected": 1}]))
    assert code and "def solve" in code


def test_propose_coding_solution_goes_through_the_cortex(engine):
    _wire(engine, ['{"code": "def widget(n):\\n    return n\\n"}'])
    code = _run(engine.propose_coding_solution("widget", "spec", [((1,), 1)]))
    assert code and "def widget" in code


def test_propose_code_change_goes_through_the_cortex(engine):
    _wire(engine, ['{"should_modify": false, "reason": "already fine"}'])
    result = _run(engine.propose_code_change("m.py", "x = 1", {"tick": 1}))
    assert result["parsed"]["should_modify"] is False


def test_analyze_self_performance_goes_through_the_cortex(engine):
    _wire(engine, ['{"adjustments": [], "assessment": "healthy"}'])
    result = _run(engine.analyze_self_performance({"parameters": {}}))
    assert result["parsed"]["adjustments"] == []


# ── the role each method claims ──────────────────────────────────────

@pytest.mark.parametrize("call,payload,role", [
    (lambda e: e.evaluate_state({"tick": 1}), '{"assessment": "a"}', "fast"),
    (lambda e: e.reflect({"tick": 1}), '{"learning": "l"}', "fast"),
    (lambda e: e.make_decision(["a"], {}), '{"chosen": 1}', "deep"),
    (lambda e: e.generate_curiosity([]), '{"topic": "t"}', "deep"),
    (lambda e: e.analyze_self_performance({}), '{"adjustments": []}', "deep"),
    (lambda e: e.propose_skill("k", []), '{"code": "def solve(p): pass"}', "code"),
    (lambda e: e.propose_code_change("f", "s", {}), '{"should_modify": false}', "code"),
])
def test_each_method_uses_its_declared_role(engine, call, payload, role):
    _wire(engine, [payload], roles=(role,))
    _run(call(engine))
    assert engine.cortex.calls_by_role[role] == 1


def test_the_hot_path_is_the_cheap_role(engine):
    # evaluate_state and reflect fire on nearly every LLM tick; routing them to
    # DEEP would spend the API budget on the most frequent call in the system.
    _wire(engine, ['{"assessment": "a"}'], roles=("fast",))
    _run(engine.evaluate_state({"tick": 1}))
    assert engine.cortex.calls_by_role["deep"] == 0


# ── falling back ─────────────────────────────────────────────────────

def test_no_cortex_route_falls_back_to_the_legacy_path(engine):
    engine.cortex = Cortex(providers={}, routes={})
    result = _run(engine.evaluate_state({"tick": 1}))
    assert result.get("via") != "cortex"
    assert result["success"] is False          # no legacy client either


def test_a_schema_failure_falls_back_rather_than_inventing(engine):
    _wire(engine, ["garbage", "garbage"])
    result = _run(engine.evaluate_state({"tick": 1}))
    assert result.get("via") != "cortex"


def test_a_cortex_exception_falls_back(engine, monkeypatch):
    _wire(engine, ['{"assessment": "a"}'])

    async def explode(*args, **kwargs):
        raise RuntimeError("router on fire")

    monkeypatch.setattr(engine.cortex, "structured", explode)
    result = _run(engine.evaluate_state({"tick": 1}))
    assert result.get("via") != "cortex"


def test_skill_proposal_rejects_code_without_the_expected_function(engine):
    _wire(engine, ['{"code": "x = 1"}'])
    assert _run(engine.propose_skill("calc", [])) is None


def test_coding_proposal_rejects_the_wrong_function_name(engine):
    _wire(engine, ['{"code": "def other(n): return n"}'])
    assert _run(engine.propose_coding_solution("widget", "spec", [])) is None


# ── context and envelope ─────────────────────────────────────────────

def test_the_system_prompt_is_sent_first(engine):
    provider = _wire(engine, ['{"assessment": "a"}'])
    _run(engine.evaluate_state({"tick": 1}))
    assert provider.invocations[0][0]["role"] == "system"
    assert "AEGIS" in provider.invocations[0][0]["content"]


def test_the_state_is_sent_as_context(engine):
    provider = _wire(engine, ['{"assessment": "a"}'])
    _run(engine.evaluate_state({"tick": 42, "marker": "findme"}))
    joined = " ".join(m["content"] for m in provider.invocations[0])
    assert "findme" in joined


def test_an_enormous_context_is_truncated(engine):
    provider = _wire(engine, ['{"assessment": "a"}'])
    _run(engine.evaluate_state({"blob": "x" * 20000}))
    context_message = provider.invocations[0][1]["content"]
    assert len(context_message) < 4000


def test_the_envelope_matches_what_callers_already_branch_on(engine):
    _wire(engine, ['{"assessment": "a"}'])
    result = _run(engine.evaluate_state({"tick": 1}))
    assert set(result) >= {"success", "provider", "response", "parsed",
                           "tokens_in", "tokens_out", "latency_ms"}


def test_status_reports_the_cortex_alongside_the_legacy_providers(engine):
    _wire(engine, ['{"assessment": "a"}'])
    status = engine.status()
    assert status["enabled"] is True         # via the cortex, with no legacy client
    assert status["cortex"]["available_roles"]


# ── the trainable model is shared with the cortex ────────────────────

def test_attaching_a_weight_modifier_reaches_the_local_provider(engine):
    modifier = object()
    engine.weight_modifier = modifier
    assert engine.weight_modifier is modifier
    assert engine.cortex.providers["local_hf"].weight_modifier is modifier


def test_attaching_a_weight_modifier_without_that_provider_is_safe(engine):
    engine.cortex = Cortex(providers={}, routes={})
    engine.weight_modifier = object()        # must not raise


# ── the Anthropic client is built lazily ─────────────────────────────

def test_the_anthropic_client_is_constructed_on_first_use():
    built = []

    def factory(api_key, timeout):
        built.append((api_key, timeout))
        return FakeAnthropicClient(["x"])

    provider = AnthropicProvider("claude", "m", "key", client_factory=factory)
    assert built == []
    _run(provider.call([{"role": "user", "content": "q"}], CallParams(timeout=5)))
    assert built == [("key", 60.0)]


def test_the_anthropic_sdk_is_imported_only_when_needed(monkeypatch):
    from aegis.cortex.providers import anthropic as module

    fake_sdk = types.ModuleType("anthropic")
    fake_sdk.AsyncAnthropic = lambda **kwargs: ("client", kwargs)
    monkeypatch.setitem(sys.modules, "anthropic", fake_sdk)
    client, kwargs = module._build_client("k", 5.0)
    assert client == "client"
    assert kwargs["api_key"] == "k"


def test_the_openai_sdk_is_imported_only_when_needed(monkeypatch):
    from aegis.cortex.providers import openai_compatible as module

    fake_sdk = types.ModuleType("openai")
    fake_sdk.AsyncOpenAI = lambda **kwargs: ("client", kwargs)
    monkeypatch.setitem(sys.modules, "openai", fake_sdk)
    client, kwargs = module._build_client("k", "https://x/v1", 5.0)
    assert client == "client"
    assert kwargs["base_url"] == "https://x/v1"


def test_an_openai_client_without_a_key_still_gets_a_placeholder(monkeypatch):
    # Local servers ignore the key, but the SDK refuses to construct without one.
    from aegis.cortex.providers import openai_compatible as module

    fake_sdk = types.ModuleType("openai")
    fake_sdk.AsyncOpenAI = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "openai", fake_sdk)
    assert module._build_client("", "http://local/v1", 5.0)["api_key"] == "unused"
