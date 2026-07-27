"""Return on investment, and what the system does about it (spec M4.5).

Every activity consumes budget and produces something: a benchmark delta, a
reward, a confirmed rule, an accepted strategy, a replicated discovery. Dividing
one by the other gives a comparable number, and comparable numbers are what let
the system move its own money toward what works.

Two design choices carry the weight:

* **Welford, not a plain average.** ROI is noisy — one lucky generation is not
  a trend — so the running mean and variance are tracked together and a
  reallocation only trusts an activity that has been measured enough times.
* **Nothing gets zero.** An activity with no budget produces no results, so its
  ROI can never recover: the reallocation would be a one-way ratchet that
  eventually funds a single activity. ``RESOURCE_MIN_SHARE`` is the floor that
  keeps exploration alive.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.layers.motivation.resources import ResourceCost
from aegis.store.migrations import read_store, write_store

logger = logging.getLogger("aegis.roi")

#: The four drives budgets are split between (Appendix C, ``res_share_*``).
DRIVES = ("competence", "knowledge", "coherence", "stability")

#: Weights that turn a mixed ResourceCost into one comparable number.
#:
#: One unit is deliberately calibrated as "roughly a thousand tokens", and the
#: other resources are priced against that. Tokens weigh most per unit because
#: they are externally billed and capped per hour; wall time and slots are
#: renewable — ten seconds of local compute is genuinely cheaper than a
#: ten-thousand-token API call, and the weights have to say so or every ROI
#: comparison would treat them as equivalent.
COST_WEIGHTS: dict[str, float] = {
    "llm_tokens": 1.0 / 1000.0,       # 1000 tokens  -> 1.0
    "llm_calls": 0.5,
    "wall_ms": 1.0 / 10_000.0,        # 10 seconds   -> 1.0
    "subprocess_slots": 0.5,
    "training_slots": 5.0,            # the scarcest thing the system has
    "disk_bytes": 1.0 / (1024 * 1024),
    "net_calls": 0.2,
}

#: Below this many observations an activity's ROI is not trusted for
#: reallocation — it keeps its default share instead.
MIN_OBSERVATIONS = 3


def normalize_cost(cost: ResourceCost) -> float:
    """A single comparable magnitude for a mixed resource cost."""
    total = sum(getattr(cost, kind) * weight for kind, weight in COST_WEIGHTS.items())
    return max(0.0, total)


@dataclass
class ActivityROI:
    """Running ROI statistics for one activity (Welford)."""

    activity: str
    drive: str = "knowledge"
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    total_cost: float = 0.0
    total_value: float = 0.0
    updated: float = 0.0

    def observe(self, roi: float) -> None:
        self.n += 1
        delta = roi - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (roi - self.mean)
        self.updated = CLOCK.now()

    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    def trusted(self) -> bool:
        return self.n >= MIN_OBSERVATIONS

    def to_dict(self) -> dict:
        return {"activity": self.activity, "drive": self.drive, "n": self.n,
                "mean": round(self.mean, 6), "m2": round(self.m2, 6),
                "total_cost": round(self.total_cost, 4),
                "total_value": round(self.total_value, 4),
                "updated": self.updated}

    @classmethod
    def from_dict(cls, data: dict) -> ActivityROI | None:
        try:
            return cls(activity=str(data["activity"]),
                       drive=str(data.get("drive", "knowledge")),
                       n=int(data.get("n", 0)),
                       mean=float(data.get("mean", 0.0)),
                       m2=float(data.get("m2", 0.0)),
                       total_cost=float(data.get("total_cost", 0.0)),
                       total_value=float(data.get("total_value", 0.0)),
                       updated=float(data.get("updated", 0.0)))
        except (KeyError, TypeError, ValueError):
            return None


class ROITracker:
    """Measures what each activity returns, and reallocates accordingly."""

    def __init__(self, store_path: Path | None = None, telemetry=None):
        self._store_path = store_path or (cfg.MOTIVATION_DIR / "roi.json")
        self.telemetry = telemetry
        self.activities: dict[str, ActivityROI] = {}
        self.shares: dict[str, float] = self.default_shares()
        self.reallocations = 0
        self.last_reallocation_tick = 0
        self._load()

    @staticmethod
    def default_shares() -> dict[str, float]:
        return {"competence": 0.35, "knowledge": 0.30,
                "coherence": 0.20, "stability": 0.15}

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        data = read_store(self._store_path, store="roi")
        for row in (data.get("activities") or []):
            entry = ActivityROI.from_dict(row) if isinstance(row, dict) else None
            if entry is not None:
                self.activities[entry.activity] = entry
        shares = data.get("shares")
        if isinstance(shares, dict):
            restored = {}
            for drive in DRIVES:
                try:
                    restored[drive] = float(shares.get(drive, self.shares[drive]))
                except (TypeError, ValueError):
                    restored[drive] = self.shares[drive]
            # Validated, NOT renormalised. The stored values are already a
            # finished allocation; pushing them through the floor-plus-remainder
            # rule again would drag them toward the floor a little further on
            # every restart, so a long-lived system would slowly forget what it
            # had learned to fund.
            self.shares = restored if self._is_valid_allocation(restored) \
                else self._normalized(restored)
        try:
            self.reallocations = int(data.get("reallocations", 0))
            self.last_reallocation_tick = int(data.get("last_reallocation_tick", 0))
        except (TypeError, ValueError):
            self.reallocations, self.last_reallocation_tick = 0, 0

    def save(self) -> None:
        write_store(self._store_path, {
            "activities": [self.activities[name].to_dict()
                           for name in sorted(self.activities)],
            "shares": {drive: round(self.shares[drive], 6) for drive in DRIVES},
            "reallocations": self.reallocations,
            "last_reallocation_tick": self.last_reallocation_tick,
        })

    # ── measurement ──────────────────────────────────────────────────

    def record(self, activity: str, cost: ResourceCost, value: float,
               drive: str = "knowledge") -> float:
        """Record one completed activity and return the ROI it scored.

        A free activity that produced value is not infinitely profitable — it
        is simply cheap, and reporting infinity would let one zero-cost action
        capture the entire budget. Its cost is floored at the smallest unit the
        weighting can express.
        """
        entry = self.activities.get(activity)
        if entry is None:
            entry = ActivityROI(activity=activity, drive=drive)
            self.activities[activity] = entry
        entry.drive = drive or entry.drive

        normalized = max(normalize_cost(cost), 1e-3)
        roi = float(value) / normalized
        entry.total_cost += normalized
        entry.total_value += float(value)
        entry.observe(roi)
        self._record_metric(activity, entry.mean)
        return roi

    def roi(self, activity: str) -> float:
        entry = self.activities.get(activity)
        return entry.mean if entry else 0.0

    def drive_roi(self, drive: str) -> float:
        """Mean ROI across the trusted activities serving one drive."""
        entries = [e for e in self.activities.values()
                   if e.drive == drive and e.trusted()]
        if not entries:
            return 0.0
        return sum(e.mean for e in entries) / len(entries)

    # ── reallocation ─────────────────────────────────────────────────

    def should_reallocate(self, tick: int) -> bool:
        interval = max(1, cfg.RESOURCE_REALLOC_EVERY_N_TICKS)
        return tick > 0 and tick - self.last_reallocation_tick >= interval

    def reallocate(self, tick: int = 0) -> dict:
        """Move budget share toward what is paying off.

        Negative ROI is clamped to zero rather than allowed to invert the
        proportion: an activity that is currently losing should be funded less,
        not funded backwards.
        """
        scores = {drive: max(0.0, self.drive_roi(drive)) for drive in DRIVES}
        total = sum(scores.values())
        before = dict(self.shares)

        if total > 0:
            self.shares = self._normalized(scores)
        # Nothing has demonstrated a return yet, so the current split stands
        # untouched. Re-running the allocation rule over an allocation would
        # move it toward the floor for no reason at all.
        self.reallocations += 1
        self.last_reallocation_tick = int(tick)
        return {"before": before, "after": dict(self.shares),
                "roi": {d: round(scores[d], 6) for d in DRIVES},
                "tick": int(tick)}

    @staticmethod
    def floor() -> float:
        """The guaranteed minimum share per drive.

        Capped at an equal split: floors that together exceed the whole budget
        would leave nothing to distribute and make the arithmetic impossible.
        Defined once so the allocation rule and its validity check can never
        disagree about what the floor is.
        """
        return max(0.0, min(1.0 / len(DRIVES), cfg.RESOURCE_MIN_SHARE))

    @classmethod
    def _is_valid_allocation(cls, shares: dict[str, float]) -> bool:
        """Whether these values are already a finished allocation."""
        floor = cls.floor()
        return (abs(sum(shares.values()) - 1.0) < 1e-6
                and all(value >= floor - 1e-9 for value in shares.values()))

    @classmethod
    def _normalized(cls, shares: dict[str, float]) -> dict[str, float]:
        """Turn arbitrary weights into shares that sum to 1, none below the floor.

        Hand out the floor to everyone first, then split what is left in
        proportion to the weights. Doing it in that order makes both properties
        true by construction — no drive can fall under the floor, and the total
        is exactly one — where clamping *after* normalising needs a corrective
        pass that is hard to read and easy to get subtly wrong.
        """
        count = len(DRIVES)
        floor = cls.floor()
        remainder = 1.0 - floor * count

        weights = {drive: max(0.0, float(shares.get(drive, 0.0))) for drive in DRIVES}
        total = sum(weights.values())
        if total <= 0:
            return {drive: 1.0 / count for drive in DRIVES}
        return {drive: floor + remainder * (weight / total)
                for drive, weight in weights.items()}

    def share(self, drive: str) -> float:
        return self.shares.get(drive, 0.0)

    def budget_for(self, drive: str, total: int) -> int:
        """This drive's slice of a total allowance."""
        return int(total * self.share(drive))

    # ── reporting ────────────────────────────────────────────────────

    def _record_metric(self, activity: str, value: float) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            self.telemetry.record(M.RES_ROI, value, tags={"activity": activity})
        except Exception:
            logger.exception("ROI telemetry record failed")

    def publish_metrics(self, tick: int) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            for drive in DRIVES:
                self.telemetry.record(M.RES_SHARE, self.shares[drive], tick,
                                      tags={"drive": drive})
            for name in sorted(self.activities):
                self.telemetry.record(M.RES_ROI, self.activities[name].mean, tick,
                                      tags={"activity": name})
        except Exception:
            logger.exception("ROI metric publication failed")

    def status(self) -> dict:
        ranked = sorted(self.activities.values(), key=lambda e: e.mean, reverse=True)
        return {
            "shares": {drive: round(self.shares[drive], 4) for drive in DRIVES},
            "tracked_activities": len(self.activities),
            "reallocations": self.reallocations,
            "last_reallocation_tick": self.last_reallocation_tick,
            "min_share": cfg.RESOURCE_MIN_SHARE,
            "top": [{"activity": e.activity, "drive": e.drive,
                     "roi": round(e.mean, 5), "n": e.n,
                     "variance": round(e.variance(), 6)}
                    for e in ranked[:8]],
        }
