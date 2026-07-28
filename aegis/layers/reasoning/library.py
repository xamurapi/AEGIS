"""The strategy library: what the system knows how to think with (spec M6.4).

Eight strategies ship built in (Appendix E). They are not examples — they are
the baseline the arena measures every synthesised strategy against, and the
fallback when nothing has been learned yet. A synthesised strategy that cannot
beat ``direct`` on held-out tasks has earned nothing.

Every strategy carries its own record: how often it was used, how often it
solved, what it cost, and per family. Aggregate statistics would hide the only
thing worth knowing — a strategy is rarely good or bad in general, it is good
at ``constraint_puzzle`` and wasteful on ``magnitude``, and the selector needs
that split to choose well.

Admission is total. A strategy enters only through :meth:`Library.admit`, which
validates against the DSL and refuses anything the interpreter could not run.
That is the security boundary: after admission, running a strategy is safe by
construction, so nothing downstream needs to re-check it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import aegis.config as cfg
from aegis.layers.reasoning.dsl import DSLError, cost_of, digest, normalise, validate
from aegis.store.migrations import read_store, write_store
from aegis.util.stats import wilson_interval, wilson_lower

logger = logging.getLogger("aegis.reasoning")

#: Where a strategy came from. Built-ins are never retired — a library that
#: could retire its own baseline would lose the ability to say what "better"
#: means.
ORIGINS = ("builtin", "synth", "cortex", "mutation")


@dataclass
class Strategy:
    """One named way of thinking, plus its record."""

    name: str
    steps: list = field(default_factory=list)
    origin: str = "synth"
    parent: str = ""
    created_tick: int = 0
    retired: bool = False
    #: family -> {"used", "solved", "abstained", "cost_ms", "steps"}
    record: dict[str, dict] = field(default_factory=dict)

    # ── identity ─────────────────────────────────────────────────────

    @property
    def digest(self) -> str:
        return digest(self.steps)

    @property
    def builtin(self) -> bool:
        return self.origin == "builtin"

    # ── record keeping ───────────────────────────────────────────────

    def note(self, family: str, *, solved: bool, abstained: bool = False,
             cost_ms: float = 0.0, steps: int = 0) -> None:
        row = self.record.setdefault(str(family or "?"), {
            "used": 0, "solved": 0, "abstained": 0, "cost_ms": 0.0, "steps": 0})
        row["used"] += 1
        row["solved"] += 1 if solved else 0
        row["abstained"] += 1 if abstained else 0
        row["cost_ms"] += float(cost_ms)
        row["steps"] += int(steps)

    def used(self, family: str = "") -> int:
        if family:
            return int(self.record.get(family, {}).get("used", 0))
        return sum(int(row.get("used", 0)) for row in self.record.values())

    def solved(self, family: str = "") -> int:
        if family:
            return int(self.record.get(family, {}).get("solved", 0))
        return sum(int(row.get("solved", 0)) for row in self.record.values())

    def accuracy(self, family: str = "") -> float:
        used = self.used(family)
        return self.solved(family) / used if used else 0.0

    def interval(self, family: str = "") -> tuple[float, float]:
        """Accuracy with its uncertainty attached."""
        return wilson_interval(self.solved(family), self.used(family))

    def lower(self, family: str = "") -> float:
        """Pessimistic accuracy — what the selector compares on.

        A strategy that solved 1 of 1 is not better than one that solved 40 of
        50, and only the interval says so.
        """
        return wilson_lower(self.solved(family), self.used(family))

    def mean_cost_ms(self, family: str = "") -> float:
        if family:
            row = self.record.get(family, {})
            used = int(row.get("used", 0))
            return float(row.get("cost_ms", 0.0)) / used if used else 0.0
        used = self.used()
        if not used:
            return 0.0
        return sum(float(row.get("cost_ms", 0.0))
                   for row in self.record.values()) / used

    def families(self) -> list[str]:
        return sorted(self.record)

    # ── serialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"name": self.name, "steps": normalise(self.steps),
                "origin": self.origin, "parent": self.parent,
                "created_tick": self.created_tick, "retired": self.retired,
                "record": {family: dict(row)
                           for family, row in sorted(self.record.items())}}

    @classmethod
    def from_dict(cls, data: dict) -> "Strategy":
        record = {}
        for family, row in (data.get("record") or {}).items():
            if isinstance(row, dict):
                record[str(family)] = {
                    "used": int(row.get("used", 0)),
                    "solved": int(row.get("solved", 0)),
                    "abstained": int(row.get("abstained", 0)),
                    "cost_ms": float(row.get("cost_ms", 0.0)),
                    "steps": int(row.get("steps", 0)),
                }
        origin = str(data.get("origin", "synth"))
        return cls(
            name=str(data.get("name", "?")),
            steps=list(data.get("steps") or []),
            origin=origin if origin in ORIGINS else "synth",
            parent=str(data.get("parent", "")),
            created_tick=int(data.get("created_tick", 0)),
            retired=bool(data.get("retired", False)),
            record=record,
        )


def _step(op: str, **fields) -> dict:
    return {"op": op, **fields}


#: The eight built-in strategies of M6.4, under the spec's names. Each is a
#: different bet about where the difficulty lives — going straight at it,
#: breaking it up, checking, asking twice, knowing when not to answer.
#:
#: Deliberately, **no built-in combines them all**. Decomposition and abstention
#: live in different strategies, and each alone tops out well short of the
#: benchmark. That is not a gap to be patched by adding a ninth built-in: it is
#: the room the synthesiser needs. A library that already contained the best
#: strategy would make M6 unfalsifiable — every measured gain would be the
#: selector finding what someone had written by hand.
BUILTIN_STRATEGIES: dict[str, list] = {
    # Answer at once. The baseline every synthesised strategy must beat, and
    # frequently the right choice: most of the cost of thinking is wasted on
    # problems that were not hard.
    "direct": [
        _step("SOLVE"),
    ],
    # Answer, then check the answer holds together before standing behind it.
    "verify_then_answer": [
        _step("SOLVE"),
        _step("VERIFY", checker="consistency"),
        _step("BRANCH", cond="verify_failed",
              then=[_step("REFLECT"), _step("SOLVE")]),
    ],
    # Break it up, work the parts, keep the result. Pays off when the difficulty
    # is in the size of the problem rather than in any one part.
    "decompose_solve_combine": [
        _step("DECOMPOSE", max_parts="$gene:reason_decompose_parts"),
        _step("SOLVE"),
        _step("VERIFY", checker="type"),
    ],
    # Write the calculation down and run it instead of reasoning about it. Needs a model
    # to write the expression; without one it falls back, which is the honest
    # behaviour rather than a zero.
    "program_of_thought": [
        _step("LLM_STEP",
              template="Write one Python expression that answers the question.",
              role="fast"),
        _step("COMPUTE", expr="$last"),
        _step("VERIFY", checker="type"),
        _step("BRANCH", cond="insufficient", then=[_step("SOLVE")]),
    ],
    # Ask several times and take the majority. Buys nothing when the underlying
    # step is deterministic — which the arena should discover rather than be told.
    "self_consistency_k": [
        _step("VOTE", n="$gene:reason_vote_n", agg="majority",
              body=[_step("SOLVE")]),
        _step("VERIFY", checker="type"),
    ],
    # Look for something already solved that this resembles, then work.
    "analogy_from_graph": [
        _step("RETRIEVE", source="graph", k=5),
        _step("BRANCH", cond="nothing_retrieved",
              then=[_step("RETRIEVE", source="memory", k=5)]),
        _step("SOLVE"),
        _step("VERIFY", checker="type"),
    ],
    # Ask whether this is winnable before spending anything on it.
    "predictive_check": [
        _step("PREDICT", horizon=1),
        _step("BRANCH", cond="p_success_below:0.25",
              then=[_step("ABSTAIN", reason="predicted failure")],
              else_=[_step("SOLVE"), _step("VERIFY", checker="type")]),
    ],
    # Refuse when the answer would be a guess. Scored as a success on tasks with
    # missing data, which is the point: without it the benchmark rewards
    # guessing, and a confident wrong answer is worse than an admission.
    "abstain_on_low_confidence": [
        _step("SOLVE"),
        _step("VERIFY", checker="confidence"),
        _step("BRANCH", cond="insufficient",
              then=[_step("ABSTAIN", reason="the answer would be a guess")]),
    ],
}


def _fix_keywords(steps):
    """``else`` and ``while`` are Python keywords; the DSL fields are not.

    Written as ``else_``/``while_`` above so the table stays readable as Python,
    renamed here so the stored form matches the grammar.
    """
    out = []
    for step in steps:
        rendered = {}
        for key, value in step.items():
            key = {"else_": "else", "while_": "while"}.get(key, key)
            rendered[key] = _fix_keywords(value) if isinstance(value, list) else value
        out.append(rendered)
    return out


BUILTIN_STRATEGIES = {name: _fix_keywords(steps)
                      for name, steps in BUILTIN_STRATEGIES.items()}


class Library:
    """Every strategy the system has, with admission and persistence."""

    def __init__(self, store_path: Path | None = None, *, max_strategies: int | None = None):
        self._store_path = Path(store_path or (cfg.REASONING_DIR / "strategies.json"))
        self.max_strategies = int(max_strategies or cfg.REASON_MAX_STRATEGIES)
        self.strategies: dict[str, Strategy] = {}
        self.refused = 0
        self._load()
        self.seed()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        data = read_store(self._store_path, store="reasoning_strategies")
        for row in data.get("strategies") or []:
            if not isinstance(row, dict):
                continue
            try:
                strategy = Strategy.from_dict(row)
            except (TypeError, ValueError):
                logger.debug("Ignoring a malformed stored strategy")
                continue
            if strategy.builtin or not validate(strategy.steps):
                self.strategies[strategy.name] = strategy
            else:
                # A stored strategy that no longer validates is one the grammar
                # has moved past. Dropping it is right: keeping it would mean
                # the interpreter meets an operation it does not have.
                logger.info("Dropping stored strategy %r: no longer admissible",
                            strategy.name)
        try:
            self.refused = max(0, int(data.get("refused", 0)))
        except (TypeError, ValueError):
            self.refused = 0

    def save(self) -> None:
        write_store(self._store_path, {
            "strategies": [strategy.to_dict()
                           for _, strategy in sorted(self.strategies.items())],
            "refused": self.refused,
        })

    def seed(self) -> None:
        """Install the built-ins, and repair them if a stored copy drifted.

        The steps are overwritten from the table rather than trusted from disk:
        the baseline has to be the same baseline across runs, or a comparison
        against it means nothing.
        """
        for name, steps in sorted(BUILTIN_STRATEGIES.items()):
            problems = validate(steps)
            if problems:                # a built-in that does not validate is a bug here
                raise DSLError(f"built-in strategy {name!r}: {problems}")
            existing = self.strategies.get(name)
            if existing is None:
                self.strategies[name] = Strategy(name=name, steps=list(steps),
                                                 origin="builtin")
            else:
                existing.steps = list(steps)
                existing.origin = "builtin"
                existing.retired = False

    # ── admission ────────────────────────────────────────────────────

    def admit(self, name: str, steps, *, origin: str = "synth",
              parent: str = "", tick: int = 0) -> Strategy:
        """Add a strategy, or refuse it with a reason.

        Raises :class:`DSLError` on anything the interpreter could not run,
        including a duplicate: two names for one strategy would split its record
        in half and make both halves look inconclusive.
        """
        name = str(name or "").strip()
        if not name:
            self.refused += 1
            raise DSLError("a strategy needs a name")
        if name in self.strategies:
            self.refused += 1
            raise DSLError(f"a strategy named {name!r} already exists")
        problems = validate(steps)
        if problems:
            self.refused += 1
            raise DSLError("; ".join(problems))

        incoming = digest(steps)
        for other in self.strategies.values():
            if other.digest == incoming:
                self.refused += 1
                raise DSLError(f"identical to {other.name!r}")

        strategy = Strategy(name=name, steps=normalise(steps),
                            origin=origin if origin in ORIGINS else "synth",
                            parent=str(parent), created_tick=int(tick))
        self.strategies[name] = strategy
        self._evict()
        return strategy

    def retire(self, name: str, *, reason: str = "") -> bool:
        """Stop using a strategy without forgetting it happened.

        Built-ins are exempt. They are the measuring stick, and a measuring
        stick that can be discarded when it reads badly is not one.
        """
        strategy = self.strategies.get(str(name))
        if strategy is None or strategy.builtin:
            return False
        strategy.retired = True
        logger.info("Retired strategy %r%s", name, f": {reason}" if reason else "")
        return True

    def _evict(self) -> None:
        """Keep the library bounded. Retired and unproven go first."""
        removable = [s for s in self.strategies.values() if not s.builtin]
        while len(self.strategies) > self.max_strategies and removable:
            worst = min(removable, key=lambda s: (
                not s.retired, s.lower(), s.used(), s.name))
            removable.remove(worst)
            self.strategies.pop(worst.name, None)
            logger.info("Evicted strategy %r to stay within the cap", worst.name)

    # ── lookup ───────────────────────────────────────────────────────

    def get(self, name: str) -> Strategy | None:
        return self.strategies.get(str(name))

    def active(self) -> list[Strategy]:
        return [s for _, s in sorted(self.strategies.items()) if not s.retired]

    def builtins(self) -> list[Strategy]:
        return [s for _, s in sorted(self.strategies.items()) if s.builtin]

    def best_for(self, family: str, *, min_used: int = 3) -> Strategy | None:
        """The strategy with the best evidence on this family, if any has any.

        On the lower bound of the interval, not the point estimate — the whole
        reason to keep the interval is to stop one lucky trial from taking the
        family.
        """
        candidates = [s for s in self.active() if s.used(family) >= min_used]
        if not candidates:
            return None
        return max(candidates, key=lambda s: (s.lower(family),
                                              -s.mean_cost_ms(family), s.name))

    def note_result(self, name: str, family: str, *, solved: bool,
                    abstained: bool = False, cost_ms: float = 0.0,
                    steps: int = 0) -> None:
        strategy = self.strategies.get(str(name))
        if strategy is not None:
            strategy.note(family, solved=solved, abstained=abstained,
                          cost_ms=cost_ms, steps=steps)

    # ── reporting ────────────────────────────────────────────────────

    def status(self) -> dict:
        active = self.active()
        return {
            "total": len(self.strategies),
            "active": len(active),
            "builtin": len(self.builtins()),
            "synthesised": len([s for s in self.strategies.values() if not s.builtin]),
            "retired": len([s for s in self.strategies.values() if s.retired]),
            "refused": self.refused,
            "used": sum(s.used() for s in self.strategies.values()),
        }

    def table(self) -> list[dict]:
        """One row per strategy, for the dashboard and the operator."""
        rows = []
        for strategy in sorted(self.strategies.values(), key=lambda s: s.name):
            rows.append({
                "name": strategy.name,
                "origin": strategy.origin,
                "retired": strategy.retired,
                "used": strategy.used(),
                "accuracy": round(strategy.accuracy(), 4),
                "lower": round(strategy.lower(), 4),
                "cost_ms": round(strategy.mean_cost_ms(), 2),
                "est_tokens": cost_of(strategy.steps).llm_tokens,
                "families": strategy.families(),
            })
        return rows
