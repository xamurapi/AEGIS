"""Resources: the link that makes motivation real (spec M4.3).

The development text asks for ``goal → value → priority → resource → action``.
The last arrow is the one that was missing, and it is the one that matters:
without it, "motivation" is a number the system computes and then ignores,
because every action runs on schedule regardless of what it wanted.

Here an action that cannot get a lease does not execute. That single rule is
what turns a preference into a decision.

Three kinds of resource, because they run out differently:

* **windowed** — llm tokens, calls, network requests. A budget per hour that
  refills as the window slides.
* **per-tick** — wall-clock milliseconds. Refilled every tick; unused time is
  not bankable, because it was not saved, it was just not spent.
* **concurrent** — subprocess and training slots. Held while in use and handed
  back on commit or release, never "spent".

Two guarantees on top: safety-critical work keeps a floor of every budget that
nothing can take (Appendix B, category 7), and a task that keeps losing races
gains priority as it waits, so a low-priority activity cannot be starved
forever.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.store.migrations import read_store, write_store

logger = logging.getLogger("aegis.resources")

#: How each resource runs out.
WINDOWED = "windowed"      # a budget per hour
PER_TICK = "per_tick"      # a budget per tick
CONCURRENT = "concurrent"  # slots held, then returned
CUMULATIVE = "cumulative"  # a budget for the whole run

HOUR_SECONDS = 3600.0


@dataclass(frozen=True)
class ResourceCost:
    """What one action is expected to consume."""

    llm_tokens: int = 0
    llm_calls: int = 0
    wall_ms: int = 0
    subprocess_slots: int = 0
    training_slots: int = 0
    disk_bytes: int = 0
    net_calls: int = 0

    KINDS = ("llm_tokens", "llm_calls", "wall_ms", "subprocess_slots",
             "training_slots", "disk_bytes", "net_calls")

    def as_dict(self) -> dict[str, int]:
        return {kind: getattr(self, kind) for kind in self.KINDS}

    def is_free(self) -> bool:
        return all(getattr(self, kind) <= 0 for kind in self.KINDS)

    def __add__(self, other: ResourceCost) -> ResourceCost:
        return ResourceCost(**{kind: getattr(self, kind) + getattr(other, kind)
                               for kind in self.KINDS})

    def scaled(self, factor: float) -> ResourceCost:
        return ResourceCost(**{kind: int(getattr(self, kind) * factor)
                               for kind in self.KINDS})

    @classmethod
    def from_dict(cls, data: dict | None) -> ResourceCost:
        data = data or {}
        values = {}
        for kind in cls.KINDS:
            try:
                values[kind] = int(data.get(kind, 0) or 0)
            except (TypeError, ValueError):
                values[kind] = 0
        return cls(**values)


#: kind -> (how it runs out, config limit)
RESOURCE_KINDS: dict[str, str] = {
    "llm_tokens": WINDOWED,
    "llm_calls": WINDOWED,
    "net_calls": WINDOWED,
    "wall_ms": PER_TICK,
    "subprocess_slots": CONCURRENT,
    "training_slots": CONCURRENT,
    "disk_bytes": CUMULATIVE,
}


def default_limits() -> dict[str, int]:
    return {
        "llm_tokens": cfg.RES_TOKENS_PER_HOUR,
        "llm_calls": cfg.RES_CALLS_PER_HOUR,
        "net_calls": cfg.RES_NET_CALLS_PER_HOUR,
        "wall_ms": cfg.RES_WALL_MS_PER_TICK,
        "subprocess_slots": cfg.RES_SUBPROC_SLOTS,
        "training_slots": cfg.RES_TRAINING_SLOTS,
        "disk_bytes": cfg.RES_DISK_MB * 1024 * 1024,
    }


@dataclass
class ResourceBudget:
    """One resource's allowance and what has been taken from it."""

    kind: str
    mode: str
    limit: int
    used: int = 0
    held: int = 0                  # concurrent slots currently out on lease
    window_started: float = 0.0
    denials: int = 0

    def available(self) -> int:
        if self.mode == CONCURRENT:
            return max(0, self.limit - self.held)
        return max(0, self.limit - self.used)

    def status(self) -> dict:
        return {
            "kind": self.kind,
            "mode": self.mode,
            "limit": self.limit,
            "used": self.used,
            "held": self.held,
            "available": self.available(),
            "denials": self.denials,
        }


@dataclass
class Lease:
    """A granted claim on resources. Holding one is what permits an action."""

    id: str
    purpose: str
    cost: ResourceCost
    priority: float
    granted_tick: int
    granted_at: float
    # No default: whether a lease may spend the safety floor is never something
    # to fall back on quietly. The manager always states it, and a caller
    # constructing a Lease by hand has to state it too.
    safety_critical: bool
    active: bool = True
    committed: ResourceCost | None = None

    @property
    def tokens(self) -> int:
        """Token allowance — read by the cortex before it spends anything."""
        return self.cost.llm_tokens

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "purpose": self.purpose,
            "priority": round(self.priority, 4),
            "granted_tick": self.granted_tick,
            "safety_critical": self.safety_critical,
            "active": self.active,
            "cost": self.cost.as_dict(),
            "committed": self.committed.as_dict() if self.committed else None,
        }


class ResourceManager:
    """Grants, tracks and reclaims the system's own operating budget."""

    def __init__(self, store_path: Path | None = None, telemetry=None,
                 limits: dict[str, int] | None = None):
        self._store_path = store_path or (cfg.MOTIVATION_DIR / "budgets.json")
        self.telemetry = telemetry
        self.tick = 0
        self.budgets: dict[str, ResourceBudget] = {}
        base = dict(default_limits())
        base.update(limits or {})
        now = CLOCK.now()
        for kind, mode in RESOURCE_KINDS.items():
            self.budgets[kind] = ResourceBudget(
                kind=kind, mode=mode, limit=int(base.get(kind, 0)),
                window_started=now)

        self._leases: dict[str, Lease] = {}
        self._seq = 0
        self.granted = 0
        self.denied = 0
        self.released = 0
        self.committed = 0
        # Ticks a purpose has spent waiting for a lease it was denied. This is
        # what the aging bonus is computed from, and what proves starvation is
        # bounded rather than merely unlikely.
        self.waiting: dict[str, int] = {}
        self.starvation_ticks = 0
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        data = read_store(self._store_path, store="resources")
        for kind, row in (data.get("budgets") or {}).items():
            budget = self.budgets.get(kind)
            if budget is None or not isinstance(row, dict):
                continue
            # Limits come from configuration, never from disk: a persisted
            # limit would silently outrank an operator's env change.
            try:
                budget.used = max(0, int(row.get("used", 0)))
                budget.window_started = float(row.get("window_started", CLOCK.now()))
                budget.denials = max(0, int(row.get("denials", 0)))
            except (TypeError, ValueError):
                logger.debug("Ignoring malformed budget row for %s", kind)
        # A stored value of the wrong SHAPE (a string where a mapping belongs)
        # raises AttributeError, not ValueError — catching only the numeric
        # errors would let a corrupt file take the boot down.
        waiting = data.get("waiting")
        try:
            self.waiting = ({str(k): int(v) for k, v in waiting.items()}
                            if isinstance(waiting, dict) else {})
            self.starvation_ticks = int(data.get("starvation_ticks", 0))
        except (AttributeError, TypeError, ValueError):
            self.waiting, self.starvation_ticks = {}, 0

    def save(self) -> None:
        write_store(self._store_path, {
            "budgets": {kind: {"used": b.used, "window_started": b.window_started,
                               "denials": b.denials}
                        for kind, b in sorted(self.budgets.items())},
            "waiting": dict(sorted(self.waiting.items())),
            "starvation_ticks": self.starvation_ticks,
        })

    # ── the tick boundary ────────────────────────────────────────────

    def begin_tick(self, tick: int) -> None:
        """Refill per-tick budgets and slide the hourly window.

        Per-tick allowances do not accumulate: unused milliseconds were not
        saved, they simply were not needed, and banking them would let one tick
        spend a minute of wall clock it never earned.
        """
        self.tick = int(tick)
        now = CLOCK.now()
        for budget in self.budgets.values():
            if budget.mode == PER_TICK:
                budget.used = 0
            elif budget.mode == WINDOWED and now - budget.window_started >= HOUR_SECONDS:
                budget.used = 0
                budget.window_started = now
        for purpose in list(self.waiting):
            self.waiting[purpose] += 1
            if self.waiting[purpose] > cfg.PRIORITY_AGING_MAX_TICKS:
                self.starvation_ticks += 1

    # ── granting ─────────────────────────────────────────────────────

    def can_afford(self, cost: ResourceCost, *, safety_critical: bool = False) -> bool:
        return not self._shortfalls(cost, safety_critical)

    def _shortfalls(self, cost: ResourceCost, safety_critical: bool) -> list[str]:
        """Which budgets this request would overdraw."""
        short = []
        for kind in ResourceCost.KINDS:
            want = getattr(cost, kind)
            if want <= 0:
                continue
            budget = self.budgets.get(kind)
            if budget is None:
                continue
            available = budget.available()
            if not safety_critical:
                # Ordinary work may not eat into the slice reserved for health
                # checks, checkpoints and the ethics gate. Safety-critical work
                # may spend that floor — that is what it is for.
                available -= self._floor(budget)
            if want > available:
                short.append(kind)
        return short

    @staticmethod
    def _floor(budget: ResourceBudget) -> int:
        return int(budget.limit * cfg.RESOURCE_SAFETY_FLOOR)

    def reserve(self, cost: ResourceCost, purpose: str, priority: float = 0.0,
                *, safety_critical: bool = False) -> Lease | None:
        """Grant a lease, or refuse. A refusal is a normal outcome to record.

        There is no pre-emption: a lease already granted is inviolable, because
        an action halfway through its work cannot be un-started. Contention is
        resolved by ordering (the caller asks in priority order) and by aging,
        which lifts whoever has been waiting longest.
        """
        shortfalls = self._shortfalls(cost, safety_critical)
        if shortfalls:
            self.denied += 1
            for kind in shortfalls:
                self.budgets[kind].denials += 1
            self.waiting.setdefault(purpose, 0)
            self._record("denied", 1)
            logger.debug("Lease refused for %s: short on %s", purpose, shortfalls)
            return None

        self._seq += 1
        lease = Lease(
            id=f"lease_{self._seq:08d}",
            purpose=purpose,
            cost=cost,
            priority=float(priority),
            granted_tick=self.tick,
            granted_at=CLOCK.now(),
            safety_critical=safety_critical,
        )
        for kind in ResourceCost.KINDS:
            want = getattr(cost, kind)
            if want <= 0:
                continue
            budget = self.budgets[kind]
            if budget.mode == CONCURRENT:
                budget.held += want
            else:
                budget.used += want
        self._leases[lease.id] = lease
        self.granted += 1
        self.waiting.pop(purpose, None)
        return lease

    # ── settling ─────────────────────────────────────────────────────

    def commit(self, lease: Lease, actual: ResourceCost | None = None) -> None:
        """Close a lease with what was really spent.

        The reservation was an estimate; the difference is returned to the
        budget. Without this, an action that reserved generously and used
        little would permanently shrink the system's own allowance.
        """
        if lease is None or not lease.active:
            return
        actual = actual if actual is not None else lease.cost
        for kind in ResourceCost.KINDS:
            reserved = getattr(lease.cost, kind)
            spent = max(0, getattr(actual, kind))
            budget = self.budgets[kind]
            if budget.mode == CONCURRENT:
                budget.held = max(0, budget.held - reserved)
            else:
                budget.used = max(0, budget.used - reserved + spent)
        lease.committed = actual
        lease.active = False
        self._leases.pop(lease.id, None)
        self.committed += 1
        for kind in ResourceCost.KINDS:
            spent = getattr(actual, kind)
            if spent:
                self._record("spent", spent, tags={"kind": kind})

    def commit_tokens(self, lease: Lease, tokens: int, calls: int = 1) -> None:
        """Charge model usage against a still-open lease (used by the cortex).

        A single lease can cover several calls — a schema repair round-trip is
        the obvious case — so this accumulates rather than closing the lease.
        """
        if lease is None or not getattr(lease, "active", False):
            return
        used = ResourceCost(llm_tokens=max(0, int(tokens)),
                            llm_calls=max(0, int(calls)))
        lease.committed = (lease.committed or ResourceCost()) + used
        self._record("spent", used.llm_tokens, tags={"kind": "llm_tokens"})

    def release(self, lease: Lease) -> None:
        """Hand back a lease that was never used — the action did not run."""
        if lease is None or not lease.active:
            return
        for kind in ResourceCost.KINDS:
            reserved = getattr(lease.cost, kind)
            if reserved <= 0:
                continue
            budget = self.budgets[kind]
            if budget.mode == CONCURRENT:
                budget.held = max(0, budget.held - reserved)
            else:
                budget.used = max(0, budget.used - reserved)
        lease.active = False
        self._leases.pop(lease.id, None)
        self.released += 1

    def finalize_tick(self) -> None:
        """Settle leases still open at the end of a tick.

        An action that took a lease and then failed would otherwise hold the
        reservation forever, and the budget would leak away one crash at a time.
        A still-open lease is charged what it asked for — pessimistic on
        purpose, since nobody reported otherwise.
        """
        for lease in list(self._leases.values()):
            self.commit(lease, lease.cost)

    # ── aging (anti-starvation) ──────────────────────────────────────

    def aging_bonus(self, purpose: str) -> float:
        """Priority a purpose has earned by waiting.

        Capped, so a long-ignored trivial task cannot eventually outrank a
        health check; unbounded aging turns anti-starvation into a different
        starvation.
        """
        waited = self.waiting.get(purpose, 0)
        return min(1.0, waited * cfg.PRIORITY_AGING)

    def waiting_ticks(self, purpose: str) -> int:
        return self.waiting.get(purpose, 0)

    # ── reporting ────────────────────────────────────────────────────

    def spent(self, kind: str) -> int:
        budget = self.budgets.get(kind)
        return budget.used if budget else 0

    def open_leases(self) -> list[Lease]:
        return [self._leases[key] for key in sorted(self._leases)]

    def _record(self, metric: str, value, tags: dict | None = None) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        name = {"spent": M.RES_SPENT, "denied": M.RES_DENIED}.get(metric)
        if name is None:
            return
        try:
            self.telemetry.record(name, value, self.tick, tags=tags)
        except Exception:
            logger.exception("Resource telemetry record failed")

    def publish_metrics(self, tick: int) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            self.telemetry.record(M.RES_DENIED, self.denied, tick)
            self.telemetry.record(M.RES_STARVATION_TICKS, self.starvation_ticks, tick)
            for kind, budget in sorted(self.budgets.items()):
                self.telemetry.record(M.RES_SPENT, budget.used, tick,
                                      tags={"kind": kind})
        except Exception:
            logger.exception("Resource metric publication failed")

    def status(self) -> dict:
        return {
            "tick": self.tick,
            "budgets": {kind: b.status() for kind, b in sorted(self.budgets.items())},
            "granted": self.granted,
            "denied": self.denied,
            "released": self.released,
            "committed": self.committed,
            "open_leases": [lease.to_dict() for lease in self.open_leases()],
            "waiting": dict(sorted(self.waiting.items())),
            "starvation_ticks": self.starvation_ticks,
            "safety_floor": cfg.RESOURCE_SAFETY_FLOOR,
        }
