"""The register of what the system has come to know (spec M7.8).

A discovery moves ``proposed → supported → replicated → law``, and a refutation
is kept **forever**. That last rule is the one worth defending: a refuted
hypothesis is not a failure to be tidied away, it is knowledge — it says a
relationship that looked real is not — and it is also the anti-rediscovery
mechanism. Without a permanent record of refutations the scan would propose the
same appealing pattern every thousand ticks, spend an experiment on it every
time, and never be able to tell that it had already answered the question.

Replication is required to be in a **different time window** from the original
support. Re-analysing the same rows twice is not replication; it is arithmetic
performed twice. The ledger enforces this by remembering which window supported
each discovery and refusing a replication that overlaps it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.store.migrations import read_store, write_store

logger = logging.getLogger("aegis.discovery")

STATUSES = ("proposed", "supported", "replicated", "law", "refuted", "invalid")

#: Statuses a discovery may still be worked on from.
OPEN = ("proposed", "supported", "replicated")

#: Entries kept. Refutations are never dropped for age — see the module note —
#: so the cap applies to the rest.
MAX_ENTRIES = 2_000


@dataclass
class Discovery:
    """One claim, its evidence, and where it has been applied."""

    id: str
    hypothesis: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    prereg: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    status: str = "proposed"
    effect_size: float = 0.0
    p_value: float = 1.0
    ci: tuple[float, float] = (0.0, 0.0)
    replications: int = 0
    first_tick: int = 0
    last_tick: int = 0
    windows: list = field(default_factory=list)
    applications: list = field(default_factory=list)
    history: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"id": self.id, "hypothesis": dict(self.hypothesis),
                "model": dict(self.model), "prereg": dict(self.prereg),
                "result": dict(self.result), "status": self.status,
                "effect_size": round(float(self.effect_size), 6),
                "p_value": round(float(self.p_value), 8),
                "ci": [round(float(self.ci[0]), 6), round(float(self.ci[1]), 6)],
                "replications": int(self.replications),
                "first_tick": int(self.first_tick),
                "last_tick": int(self.last_tick),
                "windows": [list(window) for window in self.windows],
                "applications": list(self.applications),
                "history": list(self.history[-20:])}

    @classmethod
    def from_dict(cls, data: dict) -> "Discovery | None":
        if not isinstance(data, dict) or not data.get("id"):
            return None
        ci = data.get("ci") or [0.0, 0.0]
        try:
            return cls(
                id=str(data["id"]),
                hypothesis=dict(data.get("hypothesis") or {}),
                model=dict(data.get("model") or {}),
                prereg=dict(data.get("prereg") or {}),
                result=dict(data.get("result") or {}),
                status=(str(data.get("status", "proposed"))
                        if str(data.get("status")) in STATUSES else "proposed"),
                effect_size=float(data.get("effect_size", 0.0)),
                p_value=float(data.get("p_value", 1.0)),
                ci=(float(ci[0]), float(ci[1])),
                replications=int(data.get("replications", 0)),
                first_tick=int(data.get("first_tick", 0)),
                last_tick=int(data.get("last_tick", 0)),
                windows=[list(window) for window in data.get("windows") or []
                         if isinstance(window, (list, tuple)) and len(window) == 2],
                applications=[str(item) for item in data.get("applications") or []],
                history=[item for item in data.get("history") or []])
        except (TypeError, ValueError):
            return None

    @property
    def formula(self) -> str:
        return str(self.model.get("expr", ""))

    def overlaps(self, window) -> bool:
        low, high = int(window[0]), int(window[1])
        for start, end in self.windows:
            if low <= int(end) and int(start) <= high:
                return True
        return False


class Ledger:
    """Persistent register of discoveries, and the only place status changes."""

    def __init__(self, path: Path | None = None, *,
                 law_reps: int | None = None, max_entries: int = MAX_ENTRIES):
        self.path = Path(path) if path else \
            Path(cfg.DATA_DIR) / "discovery" / "ledger.json"
        self.law_reps = int(cfg.DISC_LAW_REPS if law_reps is None else law_reps)
        self.max_entries = int(max_entries)
        self.entries: dict[str, Discovery] = {}
        self.rejected = 0
        self._load()

    # ── registration ─────────────────────────────────────────────────

    def propose(self, hypothesis, model=None, prereg=None,
                tick: int = 0) -> Discovery | None:
        """Open a record for a hypothesis about to be tested.

        Refuses anything already refuted. This is the anti-rediscovery gate, and
        it is the reason refutations are kept forever.
        """
        identifier = str(getattr(hypothesis, "id", "") or
                         (hypothesis or {}).get("id", ""))
        if not identifier:
            self.rejected += 1
            return None
        existing = self.entries.get(identifier)
        if existing is not None:
            if existing.status == "refuted":
                self.rejected += 1
                logger.debug("Refusing to re-open refuted discovery %s", identifier)
                return None
            return existing

        record = Discovery(
            id=identifier,
            hypothesis=(hypothesis.as_dict()
                        if hasattr(hypothesis, "as_dict") else dict(hypothesis)),
            model=(model.as_dict() if hasattr(model, "as_dict")
                   else dict(model or {})),
            prereg=(prereg.as_dict() if hasattr(prereg, "as_dict")
                    else dict(prereg or {})),
            status="proposed", first_tick=int(tick), last_tick=int(tick))
        record.history.append({"tick": int(tick), "to": "proposed"})
        self.entries[identifier] = record
        self._trim()
        return record

    def record_result(self, identifier: str, result: dict, *, tick: int = 0,
                      window=None) -> Discovery | None:
        """Fold an experiment's outcome in and move the status accordingly."""
        record = self.entries.get(str(identifier))
        if record is None or not isinstance(result, dict):
            self.rejected += 1
            return None

        outcome = str(result.get("status", "pending"))
        if outcome == "pending":
            return record
        record.result = dict(result)
        record.last_tick = int(tick)
        record.effect_size = float(result.get("effect_size", record.effect_size))
        record.p_value = float(result.get("p_value", record.p_value))
        if isinstance(result.get("ci"), (list, tuple)) and len(result["ci"]) == 2:
            record.ci = (float(result["ci"][0]), float(result["ci"][1]))

        if outcome == "invalid":
            return self._move(record, "invalid", tick,
                              str(result.get("reason", "")))
        if outcome == "refuted":
            return self._move(record, "refuted", tick,
                              str(result.get("reason", "the experiment refuted it")))
        if outcome != "supported":
            self.rejected += 1
            return record

        if window is not None:
            if record.status in ("supported", "replicated") and \
                    record.overlaps(window):
                # Not replication — the same evidence read twice.
                record.history.append({"tick": int(tick),
                                       "note": "overlapping window ignored"})
                return record
            record.windows.append([int(window[0]), int(window[1])])

        if record.status == "proposed":
            return self._move(record, "supported", tick, "the experiment supported it")
        record.replications += 1
        if record.replications + 1 >= self.law_reps and self._stable(record):
            return self._move(record, "law", tick,
                              f"{record.replications + 1} confirmations, stable effect")
        return self._move(record, "replicated", tick,
                          f"replication {record.replications}")

    @staticmethod
    def _stable(record: Discovery) -> bool:
        """Whether the effect has held its size across confirmations.

        A relationship that is significant every time but whose size swings
        wildly is not a law — it is several different effects sharing a name.
        """
        sizes = [float(entry["effect"]) for entry in record.history
                 if isinstance(entry, dict) and "effect" in entry]
        sizes.append(float(record.effect_size))
        if len(sizes) < 2:
            return True
        largest, smallest = max(abs(v) for v in sizes), min(abs(v) for v in sizes)
        if largest <= 0:
            return False
        return smallest / largest >= 0.5

    def _move(self, record: Discovery, status: str, tick: int,
              reason: str = "") -> Discovery:
        previous = record.status
        record.status = status
        record.history.append({"tick": int(tick), "from": previous, "to": status,
                               "reason": reason,
                               "effect": round(float(record.effect_size), 6)})
        logger.info("Discovery %s: %s → %s (%s)", record.id, previous, status, reason)
        return record

    # ── application (M7.9) ───────────────────────────────────────────

    def note_application(self, identifier: str, where: str,
                         tick: int = 0) -> bool:
        """Record that a discovery has been *used*.

        A discovery nobody applies is a report. The list is also what makes the
        reverse check possible: if a metric fell after an application, the
        discovery goes back for re-testing.
        """
        record = self.entries.get(str(identifier))
        if record is None or record.status not in ("supported", "replicated", "law"):
            return False
        entry = str(where)
        if entry not in record.applications:
            record.applications.append(entry)
            record.history.append({"tick": int(tick), "applied": entry})
        return True

    def retest(self, identifier: str, reason: str, tick: int = 0) -> bool:
        """Send an applied discovery back for re-testing after a regression."""
        record = self.entries.get(str(identifier))
        if record is None or record.status not in ("supported", "replicated", "law"):
            return False
        self._move(record, "proposed", tick, f"re-testing: {reason}")
        return True

    # ── reading ──────────────────────────────────────────────────────

    def get(self, identifier: str) -> Discovery | None:
        return self.entries.get(str(identifier))

    def by_status(self, status: str) -> list[Discovery]:
        return sorted((record for record in self.entries.values()
                       if record.status == str(status)),
                      key=lambda record: record.id)

    def is_refuted(self, identifier: str) -> bool:
        record = self.entries.get(str(identifier))
        return record is not None and record.status == "refuted"

    def laws(self) -> list[Discovery]:
        return self.by_status("law")

    def counts(self) -> dict:
        out = {status: 0 for status in STATUSES}
        for record in self.entries.values():
            out[record.status] = out.get(record.status, 0) + 1
        return out

    # ── persistence ──────────────────────────────────────────────────

    def _trim(self) -> None:
        if len(self.entries) <= self.max_entries:
            return
        droppable = sorted(
            (record for record in self.entries.values()
             if record.status not in ("refuted", "law")),
            key=lambda record: (record.last_tick, record.id))
        for record in droppable[:len(self.entries) - self.max_entries]:
            self.entries.pop(record.id, None)

    def save(self) -> bool:
        return write_store(self.path, {
            "saved": CLOCK.now(),
            "entries": [record.as_dict() for record in
                        sorted(self.entries.values(), key=lambda r: r.id)]})

    def _load(self) -> None:
        payload = read_store(self.path, store="discovery_ledger")
        for raw in payload.get("entries") or []:
            record = Discovery.from_dict(raw)
            if record is not None:
                self.entries[record.id] = record

    def status(self) -> dict:
        counts = self.counts()
        return {"total": len(self.entries), "rejected": self.rejected,
                "law_reps": self.law_reps, **counts,
                "applied": sum(1 for record in self.entries.values()
                               if record.applications)}
