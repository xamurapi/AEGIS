"""No lease, no call (spec §M8.6, M4.3, Appendix K stage 2).

The cortex is the single largest cost centre in the system, so it is the one
place where "we meant to account for it" is not good enough. Once a resource
manager is attached, a call without a valid lease does not happen — the
provider is never reached, and the refusal is counted.

While no manager is attached the outer per-run fuse still applies; these tests
pin both halves of that transition so stage 2 cannot silently loosen it.
"""
import asyncio

import pytest

from aegis.cortex.providers.base import CHARS_PER_TOKEN, CallParams, estimate_tokens
from aegis.cortex.router import Cortex, Role
from tests.cortex_fakes import FakeLease, FakeResources, ScriptedProvider


def _run(coro):
    return asyncio.run(coro)


def _metered(responses=("ok",)):
    provider = ScriptedProvider("a", responses=list(responses))
    resources = FakeResources()
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"], "deep": ["a"]},
                    resources=resources)
    return cortex, provider, resources


# ── the gate ─────────────────────────────────────────────────────────

def test_a_metered_call_without_a_lease_never_reaches_the_provider():
    cortex, provider, _ = _metered()
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}])) is None
    assert provider.invocations == []


def test_a_refused_call_is_counted():
    cortex, _, _ = _metered()
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}]))
    _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}]))
    assert cortex.lease_denials == 2


def test_an_expired_lease_is_refused():
    cortex, provider, _ = _metered()
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}],
                            lease=FakeLease(active=False))) is None
    assert provider.invocations == []


def test_a_valid_lease_lets_the_call_happen():
    cortex, provider, _ = _metered()
    completion = _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}],
                                  lease=FakeLease()))
    assert completion is not None and completion.ok
    assert len(provider.invocations) == 1


def test_structured_calls_are_gated_the_same_way():
    cortex, provider, _ = _metered(['{"answer": 1}'])
    assert _run(cortex.structured(Role.FAST, [{"role": "user", "content": "q"}],
                                  "answer")) is None
    assert provider.invocations == []


def test_a_repair_round_trip_is_charged_to_the_same_lease():
    provider = ScriptedProvider("a", responses=["garbage", '{"answer": 1}'])
    resources = FakeResources()
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]},
                    resources=resources)
    lease = FakeLease()
    result = _run(cortex.structured(Role.FAST, [{"role": "user", "content": "q"}],
                                    "answer", lease=lease))
    assert result == {"answer": 1}
    assert len(lease.committed) == 2      # both the attempt and its repair


# ── without a manager the old fuse still governs ─────────────────────

def test_an_unmetered_cortex_calls_without_a_lease():
    provider = ScriptedProvider("a", responses=["ok"])
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]})
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}])) is not None


def test_an_unmetered_cortex_charges_nobody():
    cortex = Cortex(providers={"a": ScriptedProvider("a", responses=["ok"])},
                    routes={"fast": ["a"]})
    assert cortex.resources is None
    lease = FakeLease()
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}], lease=lease))
    # No manager means there is no ledger to post to — and nothing is posted.
    assert lease.committed == []


# ── the estimate is load-bearing, not decorative ─────────────────────

def test_a_lease_too_small_for_the_estimate_is_refused():
    cortex, provider, _ = _metered()
    # The estimate is prompt tokens plus the role's max_tokens; a lease holding
    # a handful cannot cover it.
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q" * 400}],
                            lease=FakeLease(tokens=5))) is None
    assert provider.invocations == []
    assert cortex.lease_denials == 1


def test_a_lease_large_enough_lets_the_call_through():
    cortex, provider, _ = _metered()
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}],
                            lease=FakeLease(tokens=100_000))) is not None
    assert len(provider.invocations) == 1


def test_a_lease_that_declares_no_allowance_is_not_second_guessed():
    # Stage 2 leases carry an allowance; anything else must not be blocked for
    # failing to declare one it was never asked for.
    cortex, provider, _ = _metered()
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}],
                            lease=FakeLease(tokens=None))) is not None


def test_a_lease_object_without_an_active_flag_counts_as_active():
    class BareLease:
        """A lease from some other subsystem that never declares itself dead."""

    cortex, provider, _ = _metered()
    assert _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}],
                            lease=BareLease())) is not None


# ── cost estimation ──────────────────────────────────────────────────

def test_an_estimate_grows_with_the_prompt():
    short = estimate_tokens([{"role": "user", "content": "hi"}])
    long = estimate_tokens([{"role": "user", "content": "hi" * 500}])
    assert long > short


def test_an_empty_conversation_still_estimates_at_least_one_token():
    assert estimate_tokens([]) >= 1
    assert estimate_tokens(None) >= 1


def test_the_estimate_is_in_the_right_order_of_magnitude():
    text = "x" * (CHARS_PER_TOKEN * 100)
    estimate = estimate_tokens([{"role": "user", "content": text}])
    assert 100 <= estimate <= 120


def test_a_non_string_content_does_not_break_the_estimate():
    assert estimate_tokens([{"role": "user", "content": {"nested": "object"}}]) >= 1


def test_the_committed_amount_is_what_was_actually_used():
    cortex, _, resources = _metered()
    _run(cortex.call(Role.FAST, [{"role": "user", "content": "q"}], lease=FakeLease()))
    # ScriptedProvider reports 10 in + 20 out; the commit is the real usage,
    # not the pre-call estimate the lease was sized from.
    assert resources.commits == [(30, 1)]


# ── determinism knob ─────────────────────────────────────────────────

def test_deterministic_params_pin_sampling():
    pinned = CallParams(temperature=0.9, top_p=0.8, seed=None).deterministic()
    assert (pinned.temperature, pinned.top_p, pinned.seed) == (0.0, 1.0, 0)


def test_deterministic_params_keep_an_explicit_seed():
    assert CallParams(seed=99).deterministic().seed == 99


def test_the_cache_key_part_covers_every_sampling_knob():
    a = CallParams(max_tokens=10, temperature=0.0).cache_key_part()
    b = CallParams(max_tokens=10, temperature=1.0).cache_key_part()
    assert a != b
