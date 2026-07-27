"""Kimi and the other hosted OpenAI-compatible endpoints (spec M8.3b, M8.8).

The acceptance criterion is blunt: putting Kimi on top must be configuration,
never a code change. One provider class serves Kimi, DeepSeek and GPT because
the only differences are a base URL, a key and a model id — all environment
variables.
"""
import asyncio

import pytest

import aegis.config as cfg
from aegis.cortex.providers import CallParams, OpenAICompatibleProvider
from aegis.cortex.providers.anthropic import AnthropicProvider, split_system
from aegis.cortex.router import Cortex
from tests.cortex_fakes import FakeOpenAIClient, _Usage, anthropic_provider, openai_provider


def _run(coro):
    return asyncio.run(coro)


PARAMS = CallParams(max_tokens=100, temperature=0.5, timeout=5)


# ── availability is a configuration question ─────────────────────────

def test_a_fully_configured_endpoint_is_available():
    assert openai_provider("kimi").available is True


def test_a_missing_key_makes_it_unavailable():
    provider = OpenAICompatibleProvider("kimi", "m", "", "https://x/v1")
    assert provider.available is False
    assert "API key" in provider.unavailable_reason()


def test_a_missing_model_id_makes_it_unavailable():
    provider = OpenAICompatibleProvider("kimi", "", "k", "https://x/v1")
    assert provider.available is False
    assert "model id" in provider.unavailable_reason()


def test_a_missing_base_url_makes_it_unavailable():
    provider = OpenAICompatibleProvider("kimi", "m", "k", "")
    assert provider.available is False
    assert "base URL" in provider.unavailable_reason()


def test_a_configured_provider_has_no_unavailable_reason():
    assert openai_provider("kimi").unavailable_reason() == ""


# ── the call ─────────────────────────────────────────────────────────

def test_a_call_returns_the_model_text():
    provider = openai_provider("kimi", responses=["hello from kimi"])
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.ok and completion.text == "hello from kimi"


def test_the_configured_model_id_is_what_gets_sent():
    provider = openai_provider("kimi", model="kimi-k2-0905-preview")
    _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert provider.fake_client.requests[0]["model"] == "kimi-k2-0905-preview"


def test_sampling_parameters_are_passed_through():
    provider = openai_provider("kimi")
    params = CallParams(max_tokens=42, temperature=0.0, top_p=1.0, seed=7,
                        stop=("END",), timeout=5)
    _run(provider.call([{"role": "user", "content": "q"}], params))
    request = provider.fake_client.requests[0]
    assert request["max_tokens"] == 42
    assert request["temperature"] == 0.0
    assert request["top_p"] == 1.0
    assert request["seed"] == 7
    assert request["stop"] == ["END"]


def test_optional_parameters_are_omitted_when_unset():
    provider = openai_provider("kimi")
    _run(provider.call([{"role": "user", "content": "q"}],
                       CallParams(top_p=None, seed=None, timeout=5)))
    request = provider.fake_client.requests[0]
    assert "top_p" not in request and "seed" not in request and "stop" not in request


def test_reported_usage_is_recorded():
    provider = openai_provider("kimi", usage=_Usage(11, 22))
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.tokens_in == 11 and completion.tokens_out == 22


def test_missing_usage_falls_back_to_an_estimate():
    # Many local servers report no usage at all; reading zero would make the
    # resource ledger silently free.
    provider = openai_provider("kimi", usage=_Usage(None, None),
                               responses=["a longer answer here"])
    completion = _run(provider.call([{"role": "user", "content": "q" * 100}], PARAMS))
    assert completion.tokens_in > 0 and completion.tokens_out > 0


def test_a_null_content_response_is_empty_text_not_a_crash():
    provider = openai_provider("kimi", responses=[None])
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.ok and completion.text == ""


def test_a_transport_error_becomes_a_failed_completion():
    provider = openai_provider("kimi", responses=[RuntimeError("502 bad gateway")])
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.ok is False
    assert "502" in completion.error


def test_an_unconfigured_provider_refuses_without_calling():
    provider = OpenAICompatibleProvider("kimi", "m", "", "https://x/v1")
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.ok is False
    assert "not configured" in completion.error


def test_the_client_is_built_only_when_first_used():
    built = []

    def factory(key, url, timeout):
        built.append((key, url, timeout))
        return FakeOpenAIClient(["x"])

    provider = OpenAICompatibleProvider("kimi", "m", "k", "https://x/v1",
                                        client_factory=factory)
    assert built == []          # constructing the router costs nothing
    _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert built == [("k", "https://x/v1", 60.0)]


def test_counters_accumulate_across_calls():
    provider = openai_provider("kimi", responses=["a", "b"], usage=_Usage(2, 3))
    _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    _run(provider.call([{"role": "user", "content": "r"}], PARAMS))
    status = provider.status()
    assert status["calls"] == 2 and status["tokens_total"] == 10


def test_status_names_the_endpoint():
    assert openai_provider("kimi").status()["base_url"] == "https://example/v1"


# ── the default wiring reaches Kimi ──────────────────────────────────

def test_kimi_is_a_provider_the_router_knows_about():
    assert "kimi" in Cortex.build_default_providers()


def test_the_default_route_table_prefers_kimi_for_deep_and_code():
    assert cfg.CORTEX_ROUTES_DEFAULT["deep"][0] == "kimi"
    assert cfg.CORTEX_ROUTES_DEFAULT["code"][0] == "kimi"


def test_the_default_judge_route_starts_with_a_different_provider():
    # The critic must not be the author (§M8.4).
    assert cfg.CORTEX_ROUTES_DEFAULT["judge"][0] != cfg.CORTEX_ROUTES_DEFAULT["deep"][0]


def test_kimi_model_id_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("KIMI_MODEL", "kimi-next-generation")
    import importlib
    reloaded = importlib.reload(cfg)
    try:
        assert reloaded.KIMI_MODEL == "kimi-next-generation"
    finally:
        monkeypatch.delenv("KIMI_MODEL", raising=False)
        importlib.reload(cfg)


# ── Claude keeps its own wire format ─────────────────────────────────

def test_the_system_prompt_is_lifted_out_of_the_conversation():
    system, conversation = split_system([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "q"},
    ])
    assert system == "be brief"
    assert conversation == [{"role": "user", "content": "q"}]


def test_multiple_system_messages_are_joined_not_dropped():
    system, _ = split_system([
        {"role": "system", "content": "base"},
        {"role": "system", "content": "overlay"},
        {"role": "user", "content": "q"},
    ])
    assert "base" in system and "overlay" in system


def test_a_system_only_request_still_gets_a_turn():
    _, conversation = split_system([{"role": "system", "content": "think"}])
    assert conversation and conversation[0]["role"] == "user"


def test_claude_concatenates_every_content_block():
    # A response split across blocks must not be truncated to its first one.
    provider = anthropic_provider(responses=[["part one ", "part two"]])
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.text == "part one part two"


def test_claude_reports_its_usage():
    provider = anthropic_provider(responses=["x"])
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.tokens_in == 5 and completion.tokens_out == 9


def test_claude_without_a_key_is_unavailable():
    provider = AnthropicProvider("claude", "m", "")
    assert provider.available is False
    assert "ANTHROPIC_API_KEY" in provider.unavailable_reason()


def test_claude_without_a_model_is_unavailable():
    provider = AnthropicProvider("claude", "", "k")
    assert "model id" in provider.unavailable_reason()


def test_claude_passes_stop_sequences_in_its_own_field():
    provider = anthropic_provider(responses=["x"])
    _run(provider.call([{"role": "user", "content": "q"}],
                       CallParams(stop=("HALT",), timeout=5)))
    assert provider.fake_client.requests[0]["stop_sequences"] == ["HALT"]
