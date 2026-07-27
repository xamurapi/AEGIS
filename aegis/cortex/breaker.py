"""Per-provider circuit breaker (spec §M8.4).

A provider that has started failing usually keeps failing for a while — an
expired key, a regional outage, a local server that was shut down. Retrying it
on every call costs a timeout each time, and those timeouts land inside
cognitive phases that have millisecond budgets. The breaker converts a slow
repeated failure into a fast local decision: after N consecutive errors the
provider is skipped entirely until a cool-down elapses.

Three states, the standard ones, because the middle one matters: after the
cool-down the provider is *half-open* and gets exactly one probe. If that
succeeds it closes; if it fails the cool-down restarts. Without half-open, a
provider that came back would either stay excluded forever or be readmitted in
full and flood the next N calls with timeouts.
"""
from __future__ import annotations

import logging

from aegis.clock import CLOCK

logger = logging.getLogger("aegis.cortex.breaker")

CLOSED = "closed"        # normal operation
OPEN = "open"            # failing; skipped without being called
HALF_OPEN = "half_open"  # cool-down elapsed; one probe allowed


class CircuitBreaker:
    """Failure tracking for one provider."""

    def __init__(self, name: str, *, threshold: int = 5, cooldown: float = 300.0):
        self.name = name
        self.threshold = max(1, int(threshold))
        self.cooldown = max(0.0, float(cooldown))
        self.consecutive_errors = 0
        self.opened_at = 0.0
        self.trips = 0
        self.probes = 0
        self._state = CLOSED

    # ── state ────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current state, resolving an elapsed cool-down to half-open."""
        if self._state == OPEN and self.cooldown > 0 \
                and CLOCK.now() - self.opened_at >= self.cooldown:
            self._state = HALF_OPEN
        return self._state

    def allows(self) -> bool:
        """Whether a call may be attempted right now."""
        state = self.state
        if state == OPEN:
            return False
        if state == HALF_OPEN:
            self.probes += 1
        return True

    def remaining_cooldown(self) -> float:
        if self._state != OPEN:
            return 0.0
        return max(0.0, self.cooldown - (CLOCK.now() - self.opened_at))

    # ── outcomes ─────────────────────────────────────────────────────

    def record_success(self) -> None:
        if self._state != CLOSED:
            logger.info("Circuit for %s closed after a successful probe", self.name)
        self.consecutive_errors = 0
        self._state = CLOSED

    def record_failure(self) -> None:
        # A probe that fails re-opens immediately: the cool-down just proved
        # too short, so serving the full threshold again would be N more
        # timeouts for no new information.
        if self._state == HALF_OPEN:
            self._open()
            return
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.threshold:
            self._open()

    def _open(self) -> None:
        if self._state != OPEN:
            self.trips += 1
            logger.warning("Circuit for %s opened after %d consecutive errors; "
                           "skipping it for %.0fs",
                           self.name, self.consecutive_errors, self.cooldown)
        self._state = OPEN
        self.opened_at = CLOCK.now()

    def reset(self) -> None:
        self.consecutive_errors = 0
        self._state = CLOSED
        self.opened_at = 0.0

    # ── reporting ────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "consecutive_errors": self.consecutive_errors,
            "threshold": self.threshold,
            "trips": self.trips,
            "probes": self.probes,
            "cooldown_remaining": round(self.remaining_cooldown(), 1),
        }
