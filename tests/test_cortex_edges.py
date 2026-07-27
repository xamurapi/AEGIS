"""Edge behaviour of the cortex pieces (spec M8).

These are the branches that only fire when something is unusual — a provider
that is configured but has no model, a server that reports no usage, a schema
whose bound is an integer, a cache entry that expires while it is being swept.
They are the ones that go wrong quietly, so they get named tests rather than
incidental coverage.
"""
import asyncio

import pytest

from aegis.clock import FrozenClock, set_clock
from aegis.cortex import schemas as S
from aegis.cortex.cache import CacheEntry, ResponseCache
from aegis.cortex.providers import CallParams, LocalHFProvider
from aegis.cortex.providers.anthropic import AnthropicProvider
from aegis.cortex.providers.base import NullProvider
from aegis.cortex.router import Cortex, Role
from tests.cortex_fakes import DEFAULT_USAGE, FakeAnthropicClient, ScriptedProvider


def _run(coro):
    return asyncio.run(coro)


PARAMS = CallParams(timeout=5)


@pytest.fixture
def frozen():
    clock = FrozenClock(1_000_000.0)
    previous = set_clock(clock)
    yield clock
    set_clock(previous)


# ── Anthropic edges ──────────────────────────────────────────────────

def _claude(responses=("ok",), usage=DEFAULT_USAGE):
    client = FakeAnthropicClient(list(responses), usage)
    provider = AnthropicProvider("claude", "m", "k", client=client)
    provider.fake_client = client
    return provider


def test_a_configured_claude_has_no_complaint():
    assert _claude().unavailable_reason() == ""


def test_claude_omits_top_p_when_it_is_unset():
    provider = _claude()
    _run(provider.call([{"role": "user", "content": "q"}],
                       CallParams(top_p=None, timeout=5)))
    assert "top_p" not in provider.fake_client.requests[0]


def test_claude_sends_top_p_when_it_is_set():
    provider = _claude()
    _run(provider.call([{"role": "user", "content": "q"}],
                       CallParams(top_p=0.9, timeout=5)))
    assert provider.fake_client.requests[0]["top_p"] == 0.9


def test_claude_omits_the_system_field_when_there_is_no_system_prompt():
    provider = _claude()
    _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert "system" not in provider.fake_client.requests[0]


def test_claude_sends_the_system_prompt_in_its_own_field():
    # It is a top-level field for Anthropic, not a message — sending it as a
    # message would make the model treat the instructions as user text.
    provider = _claude()
    _run(provider.call([{"role": "system", "content": "be terse"},
                        {"role": "user", "content": "q"}], PARAMS))
    request = provider.fake_client.requests[0]
    assert request["system"] == "be terse"
    assert request["messages"] == [{"role": "user", "content": "q"}]


def test_claude_without_reported_usage_falls_back_to_an_estimate():
    provider = _claude(responses=["a fairly long answer"], usage=None)
    completion = _run(provider.call([{"role": "user", "content": "q" * 80}], PARAMS))
    assert completion.tokens_in > 0
    assert completion.tokens_out > 0


def test_claude_with_no_content_blocks_returns_empty_text():
    provider = _claude(responses=[[]])
    completion = _run(provider.call([{"role": "user", "content": "q"}], PARAMS))
    assert completion.ok and completion.text == ""


# ── the local transformers provider ──────────────────────────────────

def test_a_loaded_local_model_has_no_complaint():
    class Loaded:
        model_loaded = True
        current_checkpoint = None

        def generate(self, prompt, max_tokens):
            return "x"

    assert LocalHFProvider(weight_modifier=Loaded()).unavailable_reason() == ""


# ── the null provider ────────────────────────────────────────────────

def test_invoking_a_null_provider_directly_still_refuses():
    # `call()` normally short-circuits on `available`; the inner method has to
    # refuse on its own too, or a future caller that skips the guard would get
    # a silent success.
    completion = _run(NullProvider("ghost", "no key")._invoke([], PARAMS))
    assert completion.ok is False
    assert completion.error == "no key"


# ── structured-output failure paths ──────────────────────────────────

def test_structured_gives_up_when_the_repair_call_itself_fails():
    class FailOnSecond(ScriptedProvider):
        async def _invoke(self, messages, params):
            self.invocations.append(list(messages))
            if len(self.invocations) == 1:
                from aegis.cortex.providers.base import Completion
                return Completion(text="garbage", provider=self.name)
            raise RuntimeError("provider died during repair")

    provider = FailOnSecond("a")
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]},
                    cache=ResponseCache(None))
    assert _run(cortex.structured(
        Role.FAST, [{"role": "user", "content": "q"}], "answer")) is None


def test_structured_gives_up_immediately_when_repairs_are_disabled(monkeypatch):
    monkeypatch.setattr("aegis.cortex.router.cfg.CORTEX_MAX_REPAIRS", 0)
    provider = ScriptedProvider("a", responses=["garbage", '{"answer": 1}'])
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]},
                    cache=ResponseCache(None))
    assert _run(cortex.structured(
        Role.FAST, [{"role": "user", "content": "q"}], "answer")) is None
    assert len(provider.invocations) == 1        # no repair round-trip
    assert cortex.repairs == 0


# ── cache sweeping ───────────────────────────────────────────────────

def test_expired_entries_are_swept_during_eviction(frozen):
    cache = ResponseCache(None, ttl=100, max_entries=2)
    cache.put("old", CacheEntry("a", "p", "m", 1, 1, stored_at=frozen.now()))
    frozen.advance(200)
    cache.put("new", CacheEntry("b", "p", "m", 1, 1, stored_at=frozen.now()))
    # The stale entry is gone without anyone asking for it.
    assert "old" not in cache._entries
    assert cache.expirations >= 1


def test_expired_entries_are_swept_on_load(tmp_path, frozen):
    path = tmp_path / "cache.json"
    first = ResponseCache(path, ttl=100)
    first.put("k", CacheEntry("a", "p", "m", 1, 1, stored_at=frozen.now()))
    first.save()

    frozen.advance(200)
    assert len(ResponseCache(path, ttl=100)) == 0


def test_saving_expires_before_writing(tmp_path, frozen):
    path = tmp_path / "cache.json"
    cache = ResponseCache(path, ttl=100)
    cache.put("k", CacheEntry("a", "p", "m", 1, 1, stored_at=frozen.now()))
    frozen.advance(200)
    cache.save()
    import json
    assert json.loads(path.read_text(encoding="utf-8"))["entries"] == {}


def test_a_cache_entry_round_trips_through_its_dict_form():
    entry = CacheEntry("text", "p", "m", 1, 2, stored_at=5.0, hits=3)
    assert CacheEntry.from_dict(entry.to_dict()) == entry


def test_a_cache_entry_with_a_bad_field_is_rejected():
    assert CacheEntry.from_dict({"text": "t", "stored_at": "not a number"}) is None


# ── schema coercion edges ────────────────────────────────────────────

def test_a_non_numeric_string_is_not_coerced():
    assert S.coerce_number("twelve") is None


def test_an_unrepresentable_number_falls_back():
    assert S.coerce_int(float("inf"), 0) == 0


def test_a_none_value_is_not_a_number():
    assert S.coerce_number(None, -1.0) == -1.0


def test_an_integer_field_below_its_minimum_is_raised():
    schema = {"type": "object",
              "properties": {"n": {"type": "integer", "minimum": 5}}}
    assert S.coerce_to_schema({"n": 1}, schema) == {"n": 5}


def test_an_integer_field_above_its_maximum_is_lowered():
    schema = {"type": "object",
              "properties": {"n": {"type": "integer", "maximum": 3}}}
    assert S.coerce_to_schema({"n": 9}, schema) == {"n": 3}


def test_an_uncoercible_integer_field_is_left_for_validation_to_reject():
    schema = {"type": "object",
              "properties": {"n": {"type": "integer"}}}
    coerced = S.coerce_to_schema({"n": "not a number"}, schema)
    assert S.validate(coerced, schema) != []


def test_a_none_value_becomes_an_empty_array_not_a_list_containing_none():
    schema = {"type": "array", "items": {"type": "string"}}
    assert S.coerce_to_schema(None, schema) == []


def test_an_array_without_an_item_schema_is_left_alone():
    assert S.coerce_to_schema([1, "two"], {"type": "array"}) == [1, "two"]


def test_a_typeless_schema_coerces_nothing():
    assert S.coerce_to_schema({"a": "1"}, {"properties": {"a": {}}}) == {"a": "1"}


def test_validate_ignores_a_non_dict_schema():
    assert S.validate({"anything": 1}, "not a schema") == []


def test_an_unknown_type_name_is_not_enforced():
    # Forward compatibility: a schema using a type this validator does not know
    # must not reject every value out of hand.
    assert S.validate("x", {"type": "date-time"}) == []
