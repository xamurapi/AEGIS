"""Injectable clock (spec §3.6).

Wall-clock time is read through this module instead of calling ``time.time()``
directly. Twenty-plus modules had inlined ``time.time()``, which made every
time-window behaviour — rate limits, staleness thresholds, TTLs, retention
windows — testable only by sleeping, so those paths were either untested or
flaky.

``CLOCK`` is a proxy, not the clock itself: modules bind to it once at import
(``from aegis.clock import CLOCK``) and a test can still swap the underlying
source afterwards. Binding to a concrete instance would freeze whatever was
active at import time and silently ignore the swap.
"""
import time
from contextlib import contextmanager


class Clock:
    """Real time source — the production default."""

    def now(self) -> float:
        """Wall-clock seconds since the epoch (comparable across restarts)."""
        return time.time()

    def monotonic(self) -> float:
        """Monotonic seconds — for durations, never for timestamps."""
        return time.monotonic()


class FrozenClock(Clock):
    """Deterministic clock: time moves only when the test moves it.

    ``monotonic`` advances together with ``now`` so code that mixes the two
    (measure a duration, then stamp a record) stays consistent.
    """

    def __init__(self, start: float = 1_700_000_000.0):
        self._now = float(start)
        self._mono = 0.0

    def now(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> float:
        """Move time forward. Negative steps are rejected — a clock that can go
        backwards would let a test pass on logic that cannot happen in reality."""
        if seconds < 0:
            raise ValueError(f"cannot advance a clock backwards: {seconds}")
        self._now += seconds
        self._mono += seconds
        return self._now

    def set(self, timestamp: float) -> float:
        """Jump to an absolute wall-clock time (monotonic is unaffected — that
        is exactly what an NTP correction looks like to a process)."""
        self._now = float(timestamp)
        return self._now


_active: Clock = Clock()


class _ClockProxy:
    """Stable object that always delegates to the currently active clock."""

    def now(self) -> float:
        return _active.now()

    def monotonic(self) -> float:
        return _active.monotonic()

    @property
    def source(self) -> Clock:
        return _active

    def __repr__(self) -> str:
        return f"<CLOCK -> {type(_active).__name__}>"


CLOCK = _ClockProxy()


def set_clock(clock: Clock) -> Clock:
    """Install ``clock`` as the active source; returns the previous one."""
    global _active
    if not isinstance(clock, Clock):
        raise TypeError(f"expected a Clock, got {type(clock).__name__}")
    previous, _active = _active, clock
    return previous


def reset_clock() -> Clock:
    """Restore the real clock; returns the previous one."""
    return set_clock(Clock())


@contextmanager
def frozen(start: float = 1_700_000_000.0):
    """Run a block against a FrozenClock, restoring the previous source after.

    Restoration happens even if the block raises — a leaked frozen clock would
    silently freeze time for every later test in the session.
    """
    clock = FrozenClock(start)
    previous = set_clock(clock)
    try:
        yield clock
    finally:
        set_clock(previous)


def now() -> float:
    """Shorthand for ``CLOCK.now()``."""
    return _active.now()


def monotonic() -> float:
    """Shorthand for ``CLOCK.monotonic()``."""
    return _active.monotonic()
