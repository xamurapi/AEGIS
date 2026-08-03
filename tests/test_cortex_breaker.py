"""Per-provider circuit breaker (spec §M8.4).

A dead provider must cost a local decision, not a timeout — cognitive phases
have millisecond budgets and a chain of three dead providers is three timeouts
inside one of them.
"""
import asyncio

import pytest

from aegis.clock import FrozenClock, set_clock
from aegis.cortex.breaker import CLOSED, HALF_OPEN, OPEN, CircuitBreaker
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


# ── states ───────────────────────────────────────────────────────────

def test_a_fresh_breaker_is_closed():
    breaker = CircuitBreaker("p")
    assert breaker.state == CLOSED
    assert breaker.allows() is True


def test_failures_below_the_threshold_keep_it_closed():
    breaker = CircuitBreaker("p", threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CLOSED


def test_reaching_the_threshold_opens_it(frozen):
    breaker = CircuitBreaker("p", threshold=3)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == OPEN
    assert breaker.allows() is False


def test_a_success_resets_the_failure_run():
    breaker = CircuitBreaker("p", threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == CLOSED


def test_opening_is_counted_as_a_trip(frozen):
    breaker = CircuitBreaker("p", threshold=1)
    breaker.record_failure()
    assert breaker.trips == 1


def test_repeated_failures_while_open_do_not_re_trip(frozen):
    breaker = CircuitBreaker("p", threshold=1)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.trips == 1


# ── recovery ─────────────────────────────────────────────────────────

def test_the_cooldown_moves_it_to_half_open(frozen):
    breaker = CircuitBreaker("p", threshold=1, cooldown=60)
    breaker.record_failure()
    assert breaker.state == OPEN
    frozen.advance(61)
    assert breaker.state == HALF_OPEN


def test_half_open_allows_exactly_one_probe(frozen):
    breaker = CircuitBreaker("p", threshold=1, cooldown=60)
    breaker.record_failure()
    frozen.advance(61)
    assert breaker.allows() is True
    assert breaker.probes == 1


def test_half_open_refuses_a_second_caller_until_the_probe_reports(frozen):
    """"Exactly one probe" has to mean exactly one: every additional caller
    admitted during half-open paid a full LLM timeout against a provider
    already known to be dead."""
    breaker = CircuitBreaker("p", threshold=1, cooldown=60)
    breaker.record_failure()
    frozen.advance(61)
    assert breaker.allows() is True       # the probe is handed out
    assert breaker.allows() is False      # everyone else waits for its outcome
    assert breaker.allows() is False
    assert breaker.probes == 1


def test_a_failed_probe_frees_the_slot_for_the_next_cooldown(frozen):
    breaker = CircuitBreaker("p", threshold=1, cooldown=60)
    breaker.record_failure()
    frozen.advance(61)
    breaker.allows()
    breaker.record_failure()              # probe failed -> re-open
    frozen.advance(61)
    assert breaker.allows() is True       # a fresh probe after the cool-down
    assert breaker.probes == 2


def test_would_allow_reports_without_consuming_the_probe(frozen):
    """The read paths (status(), available_roles(), the dashboard poll) need
    to ASK, not to claim — asking used to increment the probe counters."""
    breaker = CircuitBreaker("p", threshold=1, cooldown=60)
    breaker.record_failure()
    frozen.advance(61)
    assert breaker.would_allow() is True
    assert breaker.would_allow() is True
    assert breaker.probes == 0            # nothing was consumed
    assert breaker.allows() is True       # the probe is still available
    assert breaker.would_allow() is False  # and now it is honestly in flight


def test_a_successful_probe_closes_the_circuit(frozen):
    breaker = CircuitBreaker("p", threshold=1, cooldown=60)
    breaker.record_failure()
    frozen.advance(61)
    breaker.allows()
    breaker.record_success()
    assert breaker.state == CLOSED


def test_a_failed_probe_reopens_immediately(frozen):
    # The cool-down just proved too short; serving the full threshold again
    # would be N more timeouts for no new information.
    breaker = CircuitBreaker("p", threshold=5, cooldown=60)
    for _ in range(5):
        breaker.record_failure()
    frozen.advance(61)
    breaker.allows()
    breaker.record_failure()
    assert breaker.state == OPEN


def test_recovery_is_announced_once_not_on_every_success(frozen, caplog):
    # An operator needs to be told the moment a provider came back. Saying it
    # on every successful call instead would bury that moment in noise.
    import logging
    breaker = CircuitBreaker("p", threshold=1, cooldown=60)
    breaker.record_failure()
    frozen.advance(61)
    breaker.allows()

    with caplog.at_level(logging.INFO, logger="aegis.cortex.breaker"):
        breaker.record_success()
        breaker.record_success()

    recoveries = [r for r in caplog.records if "closed" in r.getMessage()]
    assert len(recoveries) == 1
    assert "p" in recoveries[0].getMessage()


def test_an_ordinary_success_says_nothing(frozen, caplog):
    import logging
    breaker = CircuitBreaker("p", threshold=3)
    with caplog.at_level(logging.INFO, logger="aegis.cortex.breaker"):
        breaker.record_success()
    assert [r for r in caplog.records if "closed" in r.getMessage()] == []


def test_remaining_cooldown_counts_down(frozen):
    breaker = CircuitBreaker("p", threshold=1, cooldown=60)
    breaker.record_failure()
    frozen.advance(20)
    assert breaker.remaining_cooldown() == pytest.approx(40)


def test_remaining_cooldown_is_zero_when_closed():
    assert CircuitBreaker("p").remaining_cooldown() == 0.0


def test_reset_returns_it_to_service(frozen):
    breaker = CircuitBreaker("p", threshold=1)
    breaker.record_failure()
    breaker.reset()
    assert breaker.state == CLOSED and breaker.allows() is True


def test_a_zero_cooldown_reopens_the_gate_at_once(frozen):
    breaker = CircuitBreaker("p", threshold=1, cooldown=0)
    breaker.record_failure()
    # With no cool-down there is nothing to wait for, so the breaker stays open
    # rather than flapping: reopening instantly would make it a no-op.
    assert breaker.state == OPEN


def test_status_reports_state_and_counters(frozen):
    breaker = CircuitBreaker("p", threshold=2, cooldown=30)
    breaker.record_failure()
    status = breaker.status()
    assert status["name"] == "p"
    assert status["consecutive_errors"] == 1
    assert status["threshold"] == 2


# ── through the router ───────────────────────────────────────────────

def test_an_open_provider_is_skipped_without_being_called(frozen):
    dead = ScriptedProvider("dead", fail=True)
    alive = ScriptedProvider("alive", responses=["ok"] * 10)
    cortex = Cortex(providers={"dead": dead, "alive": alive},
                    routes={"deep": ["dead", "alive"]})
    cortex.breakers["dead"].threshold = 2

    messages = [{"role": "user", "content": "q"}]
    for i in range(4):
        _run(cortex.call(Role.DEEP, [{"role": "user", "content": f"q{i}"}]))

    # Two attempts to trip it, then it is skipped entirely.
    assert len(dead.invocations) == 2
    assert cortex.breakers["dead"].state == OPEN


def test_the_chain_omits_open_providers():
    cortex = Cortex(providers={"a": ScriptedProvider("a"),
                               "b": ScriptedProvider("b")},
                    routes={"deep": ["a", "b"]})
    cortex.breakers["a"].threshold = 1
    cortex.breakers["a"].record_failure()
    assert cortex.chain_for(Role.DEEP) == ["b"]


def test_a_role_whose_providers_are_all_open_returns_none(frozen):
    cortex = Cortex(providers={"a": ScriptedProvider("a", fail=True)},
                    routes={"deep": ["a"]})
    cortex.breakers["a"].threshold = 1
    _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}]))
    assert _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q2"}])) is None


def test_a_recovered_provider_is_used_again(frozen):
    provider = ScriptedProvider("a", fail=True)
    cortex = Cortex(providers={"a": provider}, routes={"deep": ["a"]})
    cortex.breakers["a"].threshold = 1
    cortex.breakers["a"].cooldown = 60
    _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}]))
    assert cortex.chain_for(Role.DEEP) == []

    frozen.advance(61)
    provider._fail = False
    provider._responses = ["back"]
    completion = _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q2"}]))
    assert completion.text == "back"
    assert cortex.breakers["a"].state == CLOSED


def test_read_paths_do_not_consume_the_half_open_probe(frozen):
    """status(), available_roles() and every dashboard poll go through
    chain_for, which used to call the MUTATING allows() inside a list
    comprehension — a status page left open overnight was silently exercising
    the breaker's probe machinery."""
    provider = ScriptedProvider("a", fail=True)
    cortex = Cortex(providers={"a": provider}, routes={"deep": ["a"]})
    cortex.breakers["a"].threshold = 1
    cortex.breakers["a"].cooldown = 60
    _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}]))   # trips it
    frozen.advance(61)                                                 # half-open

    for _ in range(5):
        cortex.available_roles()
        cortex.status()
        cortex.role_available(Role.DEEP)
    assert cortex.breakers["a"].probes == 0

    # The probe the reads did not consume is still there for the real call.
    provider._fail = False
    provider._responses = ["back"]
    completion = _run(cortex.call(Role.DEEP, [{"role": "user", "content": "q2"}]))
    assert completion.text == "back"
    assert cortex.breakers["a"].probes == 1
