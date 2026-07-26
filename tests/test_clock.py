"""Tests for the injectable clock (spec §3.6)."""
import time

import pytest

from aegis.clock import (
    CLOCK, Clock, FrozenClock, frozen, monotonic, now, reset_clock, set_clock,
)


@pytest.fixture(autouse=True)
def _restore_real_clock():
    """A leaked frozen clock would freeze time for every later test."""
    yield
    reset_clock()


def test_real_clock_tracks_wall_time():
    c = Clock()
    assert abs(c.now() - time.time()) < 1.0


def test_real_clock_monotonic_never_decreases():
    c = Clock()
    first = c.monotonic()
    second = c.monotonic()
    assert second >= first


def test_frozen_clock_does_not_move_on_its_own():
    c = FrozenClock(start=1000.0)
    assert c.now() == 1000.0
    assert c.now() == 1000.0


def test_frozen_clock_advance_moves_both_clocks():
    c = FrozenClock(start=1000.0)
    assert c.monotonic() == 0.0
    c.advance(30.0)
    assert c.now() == 1030.0
    assert c.monotonic() == 30.0


def test_frozen_clock_rejects_negative_advance():
    c = FrozenClock()
    with pytest.raises(ValueError):
        c.advance(-1.0)


def test_frozen_clock_set_jumps_wall_time_only():
    """An NTP correction moves wall time, not the monotonic source."""
    c = FrozenClock(start=1000.0)
    c.advance(10.0)
    c.set(5000.0)
    assert c.now() == 5000.0
    assert c.monotonic() == 10.0


def test_proxy_follows_swapped_clock():
    """The point of the proxy: a module that bound CLOCK at import still sees
    the swap."""
    fake = FrozenClock(start=777.0)
    set_clock(fake)
    assert CLOCK.now() == 777.0
    assert now() == 777.0
    fake.advance(3.0)
    assert CLOCK.now() == 780.0
    assert monotonic() == 3.0


def test_set_clock_returns_previous():
    original = CLOCK.source
    previous = set_clock(FrozenClock())
    assert previous is original


def test_set_clock_rejects_non_clock():
    with pytest.raises(TypeError):
        set_clock(object())


def test_reset_clock_restores_real_source():
    set_clock(FrozenClock(start=1.0))
    reset_clock()
    assert abs(CLOCK.now() - time.time()) < 1.0


def test_frozen_context_manager_restores_previous():
    before = CLOCK.source
    with frozen(start=42.0) as c:
        assert CLOCK.now() == 42.0
        c.advance(1.0)
        assert CLOCK.now() == 43.0
    assert CLOCK.source is before


def test_frozen_context_manager_restores_after_exception():
    before = CLOCK.source
    with pytest.raises(RuntimeError):
        with frozen(start=42.0):
            raise RuntimeError("boom")
    assert CLOCK.source is before


def test_proxy_repr_names_active_source():
    with frozen():
        assert "FrozenClock" in repr(CLOCK)
