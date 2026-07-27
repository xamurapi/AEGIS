"""The local OpenAI-compatible server — Ollama, llama.cpp, vLLM (spec M8.3b, M8.8).

The point of this route is economic: the ``FAST`` role fires on nearly every
LLM tick, and sending it to a local quantized model means the API token budget
is spent only on ``DEEP``, ``CODE`` and ``JUDGE``. The acceptance criterion is
that a FAST call charges no hosted provider — verified here by the counters,
not by inspection.
"""
import asyncio

import pytest

import aegis.config as cfg
from aegis.cortex.providers import CallParams, LocalHFProvider, OpenAICompatibleProvider
from aegis.cortex.providers.local_hf import flatten_messages
from aegis.cortex.router import Cortex, Role
from tests.cortex_fakes import FakeOpenAIClient, _Usage, openai_provider


def _run(coro):
    return asyncio.run(coro)


PARAMS = CallParams(max_tokens=64, temperature=0.2, timeout=5)


def local_provider(responses=("ok",), model="qwen2.5:7b-instruct-q4_K_M"):
    client = FakeOpenAIClient(list(responses), _Usage(None, None))
    provider = OpenAICompatibleProvider(
        "local_openai", model, api_key="ollama",
        base_url="http://127.0.0.1:11434/v1", requires_key=False, client=client)
    provider.fake_client = client
    return provider


# ── a local server needs no key ──────────────────────────────────────

def test_a_local_server_is_available_without_an_api_key():
    provider = OpenAICompatibleProvider(
        "local_openai", "qwen2.5:7b", api_key="",
        base_url="http://127.0.0.1:11434/v1", requires_key=False)
    assert provider.available is True


def test_a_local_server_still_needs_a_model_name():
    provider = OpenAICompatibleProvider(
        "local_openai", "", api_key="", base_url="http://127.0.0.1:11434/v1",
        requires_key=False)
    assert provider.available is False
    assert "model id" in provider.unavailable_reason()


def test_the_default_local_endpoint_is_ollamas():
    assert "11434" in cfg.LOCAL_OPENAI_BASE_URL


def test_the_local_provider_is_configured_without_requiring_a_key():
    provider = Cortex.build_default_providers()["local_openai"]
    assert provider.requires_key is False


# ── talking to it ────────────────────────────────────────────────────

def test_a_local_call_returns_text():
    provider = local_provider(["local answer"])
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.ok and completion.text == "local answer"


def test_a_local_server_reporting_no_usage_still_gets_costed():
    provider = local_provider(["some answer text"])
    completion = _run(provider.call([{"role": "user", "content": "hello"}], PARAMS))
    assert completion.tokens_in > 0
    assert completion.tokens_out > 0


def test_a_dead_local_server_fails_over_rather_than_raising():
    dead = OpenAICompatibleProvider(
        "local_openai", "m", "", "http://127.0.0.1:11434/v1", requires_key=False,
        client=FakeOpenAIClient([ConnectionError("connection refused")]))
    hosted = openai_provider("kimi", responses=["hosted answer"])
    cortex = Cortex(providers={"local_openai": dead, "kimi": hosted},
                    routes={"fast": ["local_openai", "kimi"]})
    completion = _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    assert completion.text == "hosted answer"


def test_no_local_server_leaves_fast_on_the_hosted_fallback():
    unset = OpenAICompatibleProvider("local_openai", "", "", "http://127.0.0.1:11434/v1",
                                     requires_key=False)
    hosted = openai_provider("kimi")
    cortex = Cortex(providers={"local_openai": unset, "kimi": hosted},
                    routes={"fast": ["local_openai", "kimi"]})
    assert cortex.routes[Role.FAST] == ["kimi"]


# ── the token-budget claim (§M8.8) ───────────────────────────────────

def test_a_fast_call_spends_nothing_on_the_hosted_provider():
    local = local_provider(["cheap"])
    hosted = openai_provider("kimi", responses=["expensive"])
    cortex = Cortex(providers={"local_openai": local, "kimi": hosted},
                    routes={"fast": ["local_openai", "kimi"],
                            "deep": ["kimi"]})
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    assert cortex.tokens_by_provider.get("kimi", 0) == 0
    assert cortex.tokens_by_provider["local_openai"] > 0


def test_the_expensive_roles_still_reach_the_hosted_provider():
    local = local_provider(["cheap"])
    hosted = openai_provider("kimi", responses=["expensive"])
    cortex = Cortex(providers={"local_openai": local, "kimi": hosted},
                    routes={"fast": ["local_openai"], "deep": ["kimi"]})
    completion = _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}]))
    assert completion.provider == "kimi"


# ── the in-process transformers provider ─────────────────────────────

class _FakeWeightModifier:
    def __init__(self, loaded=True, text="generated"):
        self.model_loaded = loaded
        self.current_checkpoint = None
        self._text = text
        self.prompts: list[str] = []

    def generate(self, prompt, max_tokens):
        self.prompts.append(prompt)
        return self._text


def test_the_local_hf_provider_is_unavailable_without_a_loaded_model():
    provider = LocalHFProvider(weight_modifier=None)
    assert provider.available is False
    assert "weight modifier" in provider.unavailable_reason()


def test_the_local_hf_provider_is_unavailable_until_the_model_loads():
    provider = LocalHFProvider(weight_modifier=_FakeWeightModifier(loaded=False))
    assert provider.available is False
    assert "not loaded" in provider.unavailable_reason()


def test_a_loaded_local_model_answers():
    provider = LocalHFProvider(weight_modifier=_FakeWeightModifier(text="from disk"))
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.ok and completion.text == "from disk"


def test_the_conversation_is_flattened_into_a_prompt():
    text = flatten_messages([{"role": "system", "content": "S"},
                             {"role": "user", "content": "U"}])
    assert "System: S" in text and "User: U" in text
    assert text.rstrip().endswith("Assistant:")


def test_an_over_long_prompt_keeps_its_tail():
    # The most recent turn is the request; truncating the head is the only
    # truncation that keeps the question intact.
    modifier = _FakeWeightModifier()
    provider = LocalHFProvider(weight_modifier=modifier, max_prompt_chars=50)
    _run(provider.call([{"role": "user", "content": "x" * 500},
                        {"role": "user", "content": "THE-QUESTION"}], PARAMS))
    assert "THE-QUESTION" in modifier.prompts[0]
    assert len(modifier.prompts[0]) == 50


def test_the_active_checkpoint_is_reported_as_the_model():
    modifier = _FakeWeightModifier()
    modifier.current_checkpoint = "ckpt-7"
    provider = LocalHFProvider(weight_modifier=modifier)
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.model == "ckpt-7"


def test_an_unloaded_base_model_reports_base():
    provider = LocalHFProvider(weight_modifier=_FakeWeightModifier())
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.model == "base"


def test_a_generation_failure_is_contained():
    class Exploding(_FakeWeightModifier):
        def generate(self, prompt, max_tokens):
            raise RuntimeError("decode died")

    provider = LocalHFProvider(weight_modifier=Exploding())
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.ok is False and "decode died" in completion.error
