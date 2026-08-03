"""Response cache (spec §M8.6).

The cache is a correctness mechanism before it is a cost saving: comparative
runs (A/B harnesses, the reasoning arena, evolution scoring) must not have
their metric differences explained by the model answering differently the
second time.
"""
import asyncio

import pytest

from aegis.clock import FrozenClock, set_clock
from aegis.cortex.cache import CacheEntry, ResponseCache, cache_key
from aegis.cortex.router import Cortex, Role
from tests.cortex_fakes import ScriptedProvider


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def frozen():
    clock = FrozenClock(1_000_000.0)
    previous = set_clock(clock)
    yield clock
    set_clock(previous)


def _entry(text="answer", stored_at=1_000_000.0):
    return CacheEntry(text=text, provider="a", model="m",
                      tokens_in=1, tokens_out=2, stored_at=stored_at)


# ── keys ─────────────────────────────────────────────────────────────

def test_the_same_request_produces_the_same_key():
    messages = [{"role": "user", "content": "q"}]
    assert cache_key("a", "m", messages, "p") == cache_key("a", "m", messages, "p")


def test_a_different_prompt_produces_a_different_key():
    assert cache_key("a", "m", [{"role": "user", "content": "q"}], "p") \
        != cache_key("a", "m", [{"role": "user", "content": "r"}], "p")


def test_a_different_model_produces_a_different_key():
    messages = [{"role": "user", "content": "q"}]
    assert cache_key("a", "m1", messages, "p") != cache_key("a", "m2", messages, "p")


def test_different_parameters_produce_a_different_key():
    messages = [{"role": "user", "content": "q"}]
    assert cache_key("a", "m", messages, "t=0") != cache_key("a", "m", messages, "t=1")


def test_message_key_order_does_not_change_the_key():
    a = [{"role": "user", "content": "q"}]
    b = [{"content": "q", "role": "user"}]
    assert cache_key("a", "m", a, "p") == cache_key("a", "m", b, "p")


# ── hit and miss ─────────────────────────────────────────────────────

def test_a_stored_entry_is_returned(frozen):
    cache = ResponseCache(None)
    cache.put("k", _entry())
    assert cache.get("k").text == "answer"
    assert cache.hits == 1


def test_an_absent_entry_is_a_miss():
    cache = ResponseCache(None)
    assert cache.get("nothing") is None
    assert cache.misses == 1


def test_hit_rate_reflects_both(frozen):
    cache = ResponseCache(None)
    cache.put("k", _entry())
    cache.get("k")
    cache.get("absent")
    assert cache.hit_rate() == 0.5


def test_hit_rate_with_no_traffic_is_zero():
    assert ResponseCache(None).hit_rate() == 0.0


# ── expiry ───────────────────────────────────────────────────────────

def test_an_entry_past_its_ttl_is_a_miss(frozen):
    cache = ResponseCache(None, ttl=100)
    cache.put("k", _entry(stored_at=frozen.now()))
    frozen.advance(101)
    assert cache.get("k") is None
    assert cache.expirations == 1


def test_an_entry_inside_its_ttl_survives(frozen):
    cache = ResponseCache(None, ttl=100)
    cache.put("k", _entry(stored_at=frozen.now()))
    frozen.advance(99)
    assert cache.get("k") is not None


def test_a_zero_ttl_means_no_expiry(frozen):
    cache = ResponseCache(None, ttl=0)
    cache.put("k", _entry(stored_at=frozen.now()))
    frozen.advance(10_000)
    assert cache.get("k") is not None


# ── eviction ─────────────────────────────────────────────────────────

def test_the_cache_is_bounded(frozen):
    cache = ResponseCache(None, max_entries=3)
    for i in range(10):
        cache.put(f"k{i}", _entry(text=str(i)))
    assert len(cache) == 3


def test_reused_entries_outlive_one_off_ones(frozen):
    # An arena replays the same prompt on every run; a one-off answer that
    # merely arrived later must not push it out.
    cache = ResponseCache(None, max_entries=2)
    cache.put("hot", _entry(text="hot"))
    cache.get("hot")
    cache.get("hot")
    cache.put("cold1", _entry(text="cold1"))
    cache.put("cold2", _entry(text="cold2"))
    assert cache.get("hot") is not None


def test_clear_empties_the_cache(frozen):
    cache = ResponseCache(None)
    cache.put("k", _entry())
    cache.clear()
    assert len(cache) == 0


# ── persistence ──────────────────────────────────────────────────────

def test_the_cache_survives_a_restart(tmp_path, frozen):
    path = tmp_path / "cache.json"
    first = ResponseCache(path, ttl=0)
    first.put("k", _entry(text="persisted"))
    first.save()

    assert ResponseCache(path, ttl=0).get("k").text == "persisted"


def test_a_corrupt_cache_file_is_survivable(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{ broken", encoding="utf-8")
    assert len(ResponseCache(path)) == 0


def test_a_malformed_row_does_not_discard_the_rest(tmp_path, frozen):
    path = tmp_path / "cache.json"
    first = ResponseCache(path, ttl=0)
    first.put("good", _entry(text="kept"))
    first.save()
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    data["entries"]["bad"] = {"no_text_key": True}
    path.write_text(json.dumps(data), encoding="utf-8")

    reloaded = ResponseCache(path, ttl=0)
    assert reloaded.get("good") is not None
    assert reloaded.get("bad") is None


def test_saving_without_a_path_is_a_no_op():
    ResponseCache(None).save()      # must not raise


def test_status_reports_the_essentials(frozen):
    cache = ResponseCache(None, max_entries=5, ttl=60)
    cache.put("k", _entry())
    status = cache.status()
    assert status["entries"] == 1
    assert status["max_entries"] == 5
    assert status["ttl_seconds"] == 60


# ── through the router ───────────────────────────────────────────────

def test_a_repeated_request_reuses_the_cached_answer(frozen):
    provider = ScriptedProvider("a", responses=["first", "second"])
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]},
                    cache=ResponseCache(None, ttl=0))
    messages = [{"role": "user", "content": "q"}]

    first = _run(cortex.call(Role.FAST, messages))
    second = _run(cortex.call(Role.FAST, messages))

    assert first.text == second.text == "first"
    assert second.cached is True
    assert len(provider.invocations) == 1      # the model was asked once


def test_a_different_request_is_not_served_from_cache(frozen):
    provider = ScriptedProvider("a", responses=["first", "second"])
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]},
                    cache=ResponseCache(None, ttl=0))
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    second = _run(cortex.call(Role.FAST, [{"role": "user", "content": "r"}]))
    assert second.text == "second"


def test_a_failed_call_is_not_cached(frozen):
    provider = ScriptedProvider("a", fail=True)
    cache = ResponseCache(None, ttl=0)
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]}, cache=cache)
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    assert len(cache) == 0


def test_an_empty_ok_completion_is_not_cached(frozen):
    """A server that answers ok/null-content (length truncation, a content
    filter) used to poison the cache for the whole TTL — persisted across
    restarts: the role was silently dead for an hour, failover never ran and
    the breaker recorded nothing. Empty text must never be stored."""
    provider = ScriptedProvider("a", responses=["", "real answer"])
    cache = ResponseCache(None, ttl=0)
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]}, cache=cache)
    messages = [{"role": "user", "content": "q"}]

    first = _run(cortex.call(Role.FAST, messages))
    assert first.ok and first.text == ""            # the reply itself passes through
    assert len(cache) == 0                          # but is not remembered

    second = _run(cortex.call(Role.FAST, messages))
    assert second.text == "real answer"             # the chain really re-ran
    assert second.cached is False
    assert len(provider.invocations) == 2


def test_a_whitespace_only_completion_is_not_cached(frozen):
    provider = ScriptedProvider("a", responses=["  \n ", "real"])
    cache = ResponseCache(None, ttl=0)
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]}, cache=cache)
    messages = [{"role": "user", "content": "q"}]
    _run(cortex.call(Role.FAST, messages))
    assert len(cache) == 0
    assert _run(cortex.call(Role.FAST, messages)).text == "real"


def test_a_poisoned_persisted_empty_entry_is_not_served(frozen):
    """Caches written by the buggy build survive on disk. An empty entry that
    is already there must be skipped, not replayed for the rest of its TTL."""
    from aegis.cortex.cache import cache_key

    provider = ScriptedProvider("a", responses=["real"])
    cache = ResponseCache(None, ttl=0)
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]}, cache=cache)
    messages = [{"role": "user", "content": "q"}]
    params = cortex.params_for(Role.FAST)
    key = cache_key("a", provider.model, messages, params.cache_key_part())
    cache.put(key, _entry(text="", stored_at=frozen.now()))

    completion = _run(cortex.call(Role.FAST, messages))
    assert completion.text == "real"
    assert completion.cached is False
    assert len(provider.invocations) == 1


def test_a_cached_call_costs_no_provider_call_charge(frozen):
    from tests.cortex_fakes import FakeLease, FakeResources
    resources = FakeResources()
    cortex = Cortex(providers={"a": ScriptedProvider("a", responses=["x"])},
                    routes={"fast": ["a"]}, cache=ResponseCache(None, ttl=0),
                    resources=resources)
    messages = [{"role": "user", "content": "q"}]
    _run(cortex.call(Role.FAST, messages, lease=FakeLease()))
    _run(cortex.call(Role.FAST, messages, lease=FakeLease()))
    # Tokens still count (they were spent once), but the second call is not a
    # second billable request.
    assert resources.commits[1][1] == 0
