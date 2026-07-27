"""Discrete system state (spec M1.3, Appendix D).

A predictive model needs something to predict *about*. The raw tick state is
continuous and unbounded — energy, error rate, latency, mood, mode, focus — and
no table can be keyed on that. So it is bucketed into a small, ordered set of
labels, and the buckets are where all the modelling assumptions live.

Two properties the encoding has to have:

* **Pure.** ``encode`` reads a plain mapping and returns a key. No clock, no
  substrate, no side effects — so a state can be constructed in a test, and two
  runs that reach the same situation produce the same key.
* **Coarse.** Appendix D's bin count gives about 13 000 possible states, of
  which a real run visits a couple of hundred. Finer buckets would give a model
  that has seen every state exactly once and can predict nothing.

The bin edges themselves are genome material (M5): how coarsely the world
should be carved is not something to decide by hand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import aegis.config as cfg
from aegis.util.stats import trend

#: Order matters: it fixes the layout of the key string, and the key string is
#: what everything downstream stores, sorts and compares.
FIELDS = ("energy", "error", "mood", "mode", "focus_kind", "perf", "load")

#: Short prefixes, so a key stays readable in a log line.
PREFIX = {"energy": "e", "error": "err", "mood": "mo", "mode": "md",
          "focus_kind": "fk", "perf": "pf", "load": "ld"}
BY_PREFIX = {short: field for field, short in PREFIX.items()}

#: Ordered labels per bucketed field. The label at index i is chosen when the
#: value has passed i of the field's cut points.
LABELS = {
    "energy": ("lo", "mid", "hi"),
    "error": ("none", "low", "high"),
    "load": ("lo", "mid", "hi"),
}

UNKNOWN = "unknown"

_SAFE = re.compile(r"[^a-z0-9]+")


def sanitize(value) -> str:
    """A label safe to put in a key string: lowercase, no separators."""
    text = _SAFE.sub("_", str(value).strip().lower()).strip("_")
    return text[:24] or UNKNOWN


@dataclass(frozen=True)
class StateKey:
    """One bucketed system state."""

    energy: str = UNKNOWN
    error: str = UNKNOWN
    mood: str = UNKNOWN
    mode: str = UNKNOWN
    focus_kind: str = UNKNOWN
    perf: str = UNKNOWN
    load: str = UNKNOWN

    def key(self) -> str:
        """The canonical string form, e.g. ``e=mid|err=low|mo=curious|...``."""
        return "|".join(f"{PREFIX[field]}={getattr(self, field)}" for field in FIELDS)

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in FIELDS}

    @classmethod
    def parse(cls, key: str) -> StateKey:
        """Rebuild a key from its string form.

        Unrecognised or missing fields become ``unknown`` rather than raising:
        keys come back off disk, and one malformed row must not stop a model
        from loading.
        """
        values = {}
        for part in str(key).split("|"):
            short, _, value = part.partition("=")
            field = BY_PREFIX.get(short.strip())
            if field:
                values[field] = sanitize(value)
        return cls(**values)

    def __str__(self) -> str:
        return self.key()


def bucket(value: float, cuts: list[float], labels: tuple[str, ...]) -> str:
    """Which label a value falls into.

    One rule for every bucketed field: count how many cut points the value has
    reached, and clamp to the last label. Having a single rule is what lets the
    bin edges be configuration (and genome) rather than a chain of hand-written
    comparisons per field.
    """
    if not labels:
        return UNKNOWN
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return UNKNOWN
    passed = sum(1 for cut in cuts if numeric >= cut)
    return labels[min(len(labels) - 1, passed)]


class StateEncoder:
    """Turns raw tick readings into a :class:`StateKey`."""

    def __init__(self, bins: dict | None = None):
        self.bins = dict(bins if bins is not None else cfg.WM_STATE_BINS)

    # ── per-field cut points ─────────────────────────────────────────

    def _cuts(self, field: str, order: tuple[str, ...]) -> list[float]:
        """Cut points for a field, in ascending order.

        Read from the configured bin map by name so that reordering the JSON
        cannot silently reorder the buckets.
        """
        spec = self.bins.get(field)
        if not isinstance(spec, dict):
            return []
        cuts = []
        for name in order:
            try:
                cuts.append(float(spec[name]))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(cuts)

    def energy_label(self, value) -> str:
        return bucket(value, self._cuts("energy", ("lo", "hi")), LABELS["energy"])

    def error_label(self, value) -> str:
        return bucket(value, self._cuts("error", ("none", "low", "high")),
                      LABELS["error"])

    def load_label(self, avg_tick_ms, threshold_ms) -> str:
        """Latency as a fraction of the health monitor's own threshold.

        Expressed as a fraction rather than in milliseconds so the bucket means
        the same thing on a fast machine and a slow one — otherwise the model
        would learn the hardware instead of the behaviour.
        """
        try:
            threshold = float(threshold_ms)
            measured = float(avg_tick_ms)
        except (TypeError, ValueError):
            return UNKNOWN
        if threshold <= 0:
            # No threshold means the fraction is undefined, not zero. Reporting
            # "lo" here would describe a state as comfortable on the strength of
            # a missing number.
            return UNKNOWN
        return bucket(measured / threshold, self._cuts("load", ("lo", "hi")),
                      LABELS["load"])

    def perf_label(self, history) -> str:
        """Benchmark direction over the configured window."""
        spec = self.bins.get("perf") if isinstance(self.bins.get("perf"), dict) else {}
        try:
            window = int(spec.get("window", 5))
        except (TypeError, ValueError):
            window = 5
        try:
            band = float(spec.get("flat_band", 0.01))
        except (TypeError, ValueError):
            band = 0.01
        values = [v for v in (history or []) if isinstance(v, (int, float))]
        if len(values) < 2:
            return "flat"
        return trend(values[-max(2, window):], flat_band=band)

    # ── the encoding ─────────────────────────────────────────────────

    def encode(self, inputs) -> StateKey:
        """Bucket one set of raw readings.

        Anything missing becomes ``unknown``, which is a real state the model
        can learn about — a tick taken before the first benchmark genuinely is
        a different situation from one taken after.
        """
        data = dict(inputs or {})
        return StateKey(
            energy=self.energy_label(data.get("energy")),
            error=self.error_label(data.get("error_rate")),
            mood=sanitize(data.get("mood", UNKNOWN)),
            mode=sanitize(data.get("mode", UNKNOWN)),
            focus_kind=sanitize(data.get("focus_kind", UNKNOWN)),
            perf=self.perf_label(data.get("bench_history")),
            load=self.load_label(data.get("avg_tick_ms"),
                                 data.get("tick_threshold_ms")),
        )

    def space_size(self, moods: int = 6, modes: int = 4, focus_kinds: int = 5) -> int:
        """Upper bound on the number of distinct states (Appendix D)."""
        return (len(LABELS["energy"]) * len(LABELS["error"]) * len(LABELS["load"])
                * 3 * moods * modes * focus_kinds)


def collect_state_inputs(substrate) -> dict:
    """Read the raw values a state is encoded from, off the live system.

    Kept separate from :meth:`StateEncoder.encode` on purpose: this half knows
    about the substrate and cannot be unit-tested without one, while the half
    that decides what the numbers *mean* is pure and can.
    """
    total_ticks = substrate.health.successful_ticks + substrate.health.failed_ticks
    error_rate = (substrate.health.error_count / total_ticks) if total_ticks else 0.0
    durations = list(substrate.health.tick_durations)
    focus = substrate.goals.get_current_focus()
    focus_name = focus["name"] if focus else "idle"

    try:
        focus_kind = substrate.goal_intelligence._classify_drive(focus_name)
    except Exception:
        focus_kind = "idle"

    return {
        "energy": substrate.emotions.energy,
        "error_rate": error_rate,
        "mood": substrate.emotions.mood,
        "mode": substrate.consciousness.mode,
        "focus_kind": focus_kind if focus else "idle",
        "bench_history": [row.get("score") for row in substrate.evaluator.history],
        "avg_tick_ms": (sum(durations) / len(durations)) if durations else 0.0,
        "tick_threshold_ms": substrate.health.thresholds.get("tick_duration_ms", 1.0),
    }
