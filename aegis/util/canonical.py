"""Canonical form and stable digest of arbitrary state (spec M9.4).

Two runs of the same deterministic system must produce the same state. Proving
that needs a comparison that ignores what is legitimately allowed to differ —
wall-clock timestamps, measured latencies — and notices everything else.

That is the whole job here:

* timestamps are dropped by key name, not by value, so a run started a second
  later is not a false alarm;
* floats are rounded, so accumulation order noise in the last bits is not one
  either — while a real behavioural difference is far above that threshold;
* dict order is normalised (sorted), list order is not (sequence order IS
  behaviour: the order goals were created in changes which one is chosen).
"""
from __future__ import annotations

import hashlib
import json

# Rounding for floats. Deep enough that any behavioural difference survives,
# shallow enough that the last-bit noise of a different summation order does not.
FLOAT_PLACES = 9

# Keys whose values are wall-clock time or a measured duration. These are the
# only things allowed to differ between two identical runs.
TIME_KEYS = frozenset({
    "timestamp", "time", "created", "updated", "resolved", "opened",
    "last_updated", "last_build_time", "last_backup_time", "last_progress_time",
    "start_time", "uptime", "uptime_seconds", "latency_s", "latency_ms",
    "elapsed_ms", "last_latency_ms", "avg_tick_ms", "last_tick_ms",
    "tick_duration_ms", "t", "created_at", "first_seen", "last_seen",
})

_MAX_DEPTH = 40


def canonical(value, exclude_keys: frozenset[str] = TIME_KEYS, _depth: int = 0):
    """Recursively normalise ``value`` into comparable, JSON-safe data."""
    if _depth > _MAX_DEPTH:
        return "<max-depth>"

    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        # NaN/inf never compare equal to themselves; render them as tokens so a
        # digest stays defined and two NaNs still match.
        if value != value:
            return "<nan>"
        if value in (float("inf"), float("-inf")):
            return f"<{'inf' if value > 0 else '-inf'}>"
        rounded = round(value, FLOAT_PLACES)
        # -0.0 and 0.0 are the same state; JSON renders them differently.
        return rounded + 0.0

    if isinstance(value, dict):
        return {
            str(k): canonical(value[k], exclude_keys, _depth + 1)
            for k in sorted(value, key=str)
            if str(k) not in exclude_keys
        }

    if isinstance(value, (list, tuple)):
        return [canonical(v, exclude_keys, _depth + 1) for v in value]

    if isinstance(value, (set, frozenset)):
        # Sets have no inherent order; sort the CANONICAL forms so the result
        # does not depend on hash seeding.
        return sorted(
            (canonical(v, exclude_keys, _depth + 1) for v in value),
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False,
                                        default=str),
        )

    return str(value)


def canonical_json(value, exclude_keys: frozenset[str] = TIME_KEYS) -> str:
    return json.dumps(
        canonical(value, exclude_keys),
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str,
    )


def digest_of(value, exclude_keys: frozenset[str] = TIME_KEYS,
              size: int = 16) -> str:
    """Stable hex digest of the canonical form of ``value``."""
    payload = canonical_json(value, exclude_keys).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=size).hexdigest()
