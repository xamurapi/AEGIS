"""The reasoning contour (spec M6).

One object the rest of the system talks to. It owns the strategy library, the
interpreter, a queue of problems and the record of what happened, and it exposes
exactly the five things the action registry asks about — whether there is work
queued, how many results exist, where the system is weakest, what strategies are
awaiting judgement, and how to solve one task.

Two decisions shape it.

**Every attempt is recorded, not just the score.** A trace says which strategy
ran, which steps it took, what each returned and what it cost. That record is
the dataset stage 9 mines for weaknesses and the evidence the arena judges a
synthesised strategy on. A contour that stored only accuracy would be able to
say it was getting worse and never say where.

**Selection is evidence-led and pessimistic.** Every active strategy gets a
minimum number of attempts on a family before any of them is preferred, and the
comparison is on the lower bound of the interval. Otherwise one lucky first
attempt takes a family permanently and no later evidence can dislodge it,
because nothing else ever runs there again.
"""
from __future__ import annotations

import logging
from pathlib import Path

import aegis.config as cfg
from aegis.eval import reasoning_bench
from aegis.layers.reasoning.dsl import DSLError, cost_of, digest, validate
from aegis.layers.reasoning.interpreter import Interpreter, Trace
from aegis.layers.reasoning.library import BUILTIN_STRATEGIES, Library, Strategy
from aegis.layers.reasoning.reasoner import REASONER, DeterministicReasoner
from aegis.util.quasirandom import hash_index
from aegis.util.stats import wilson_interval, wilson_lower

logger = logging.getLogger("aegis.reasoning")

__all__ = ["ReasoningEngine", "Interpreter", "Trace", "Library", "Strategy",
           "DeterministicReasoner", "REASONER", "BUILTIN_STRATEGIES",
           "DSLError", "validate", "cost_of", "digest"]

#: How many attempts every active strategy gets on a family before the engine
#: starts preferring one. Below this the intervals overlap so completely that
#: "best" is a coin toss dressed up as a measurement.
MIN_ATTEMPTS_PER_FAMILY = 3

#: How many results a family needs before it can be called a weakness. A family
#: that has been tried twice is not weak, it is unmeasured.
MIN_RESULTS_FOR_WEAKNESS = 8

#: How many tasks are pulled into the queue at a time. Small: the queue is
#: worked one task per tick, and a large refill would pin the same block of
#: problems in memory for hundreds of ticks.
REFILL_SIZE = 16


class ReasoningEngine:
    """The reasoning contour: strategies, a queue, and the record of both."""

    def __init__(self, *, cortex=None, world_model=None, memory=None,
                 graph=None, solver=None, sandbox=None, telemetry=None,
                 store_path: Path | None = None, genome: dict | None = None,
                 task_source=None):
        self.telemetry = telemetry
        self.library = Library(store_path=store_path)
        self.genome = dict(genome or {})
        self.interpreter = Interpreter(
            cortex=cortex, world_model=world_model, memory=memory, graph=graph,
            solver=solver, sandbox=sandbox, genome=self.genome)

        #: Problems waiting to be worked.
        self.queue: list = []
        #: Where the next block of problems comes from. Injected so a rollout
        #: can hand in its own set without the engine reaching for a benchmark.
        self.task_source = task_source or reasoning_bench.build
        self._cursor = 0

        #: One row per attempt — the evidence everything downstream reads.
        self.results: list[dict] = []
        #: Full traces, bounded. The dataset for stage 9's synthesiser.
        self.traces: list[dict] = []
        #: Strategies proposed but not yet judged (filled by stage 9).
        self.candidates: list[dict] = []

        self.attempts = 0
        self.solved_count = 0
        self.abstentions = 0
        #: Answered confidently and wrongly. The number that matters most: a
        #: system that is wrong while sure is worse than one that is silent.
        self.confident_errors = 0

    # ── configuration ────────────────────────────────────────────────

    def set_genome(self, genome: dict) -> None:
        """Adopt an evolved configuration (M5 → M6)."""
        self.genome = dict(genome or {})
        self.interpreter.genome = dict(self.genome)

    def _budget(self) -> int:
        try:
            return int(self.genome.get("reason_budget", cfg.REASON_MAX_STEPS))
        except (TypeError, ValueError):
            return cfg.REASON_MAX_STEPS

    # ── the queue ────────────────────────────────────────────────────

    def enqueue(self, task) -> None:
        self.queue.append(task)

    def refill(self, count: int = REFILL_SIZE) -> int:
        """Pull the next block of problems. Deterministic and non-repeating.

        The cursor walks forward rather than sampling, so two runs of the same
        length meet the same problems in the same order — without which nothing
        in this contour could be compared across runs (§3.1).
        """
        added = 0
        for _ in range(max(0, int(count))):
            try:
                self.queue.append(self.task_source(self._cursor))
            except Exception:
                logger.exception("Building a reasoning task failed")
                break
            self._cursor += 1
            added += 1
        return added

    def has_queued_task(self) -> bool:
        """Whether there is reasoning work to do.

        Refills when empty rather than reporting "nothing to do": the supply of
        problems is unbounded, and an empty queue is a fact about the queue, not
        about whether the system has anything to think about.
        """
        if not self.queue:
            self.refill()
        return bool(self.queue)

    def result_count(self) -> int:
        return len(self.results)

    def pending_candidates(self) -> list[dict]:
        return list(self.candidates)

    # ── selection ────────────────────────────────────────────────────

    def select(self, family: str, key: str = "") -> Strategy:
        """Which strategy to try on this family.

        Explore first, then exploit. While a family is unmeasured the choice is
        spread across the unproven strategies by a hash of the task — not by
        "whichever is least used", which inside one scoring pass records nothing
        and therefore returns the same strategy every time. Measured: that made
        the baseline the alphabetically first strategy, which happened to be the
        best one, and the whole learning curve came out flat at the ceiling.

        The hash is a stable function of the task, so the spread is the same on
        every run without an RNG anywhere (§3.1).
        """
        active = self.library.active()
        if not active:                  # cannot happen: built-ins never retire
            self.library.seed()
            active = self.library.active()
        unproven = [s for s in active if s.used(family) < MIN_ATTEMPTS_PER_FAMILY]
        if unproven:
            return unproven[hash_index(len(unproven), "reason_explore",
                                       family, key)]
        best = self.library.best_for(family, min_used=MIN_ATTEMPTS_PER_FAMILY)
        return best or active[0]

    # ── working a task ───────────────────────────────────────────────

    def solve(self, count: int = 1) -> dict:
        """Work up to ``count`` queued problems. The ``reason_task`` executor.

        Returns a summary rather than the traces: this is called from inside a
        tick, and handing the planner a few hundred step records per action
        would put the whole reasoning history into the event bus.
        """
        worked, solved, abstained = 0, 0, 0
        for _ in range(max(1, int(count))):
            if not self.has_queued_task():
                break
            task = self.queue.pop(0)
            row = self.attempt(task)
            worked += 1
            solved += 1 if row["solved"] else 0
            abstained += 1 if row["abstained"] else 0
        return {"worked": worked, "solved": solved, "abstained": abstained,
                "queued": len(self.queue), "results": len(self.results)}

    def attempt(self, task, strategy: Strategy | None = None) -> dict:
        """Run one problem under one strategy and record what happened."""
        family = str(getattr(task, "family", "?"))
        strategy = strategy or self.select(family, str(getattr(task, "id", "")))
        trace = self.interpreter.run(strategy, task, budget=self._budget())
        return self.record(task, strategy, trace)

    def record(self, task, strategy: Strategy, trace: Trace) -> dict:
        """Fold one trace into the evidence."""
        family = str(getattr(task, "family", "?"))
        solved = bool(trace.solved)
        confident_error = bool(not solved and not trace.abstained
                               and trace.answer is not None)
        row = {
            "task": str(getattr(task, "id", "")),
            "family": family,
            "strategy": strategy.name,
            "solved": solved,
            "abstained": bool(trace.abstained),
            "confident_error": confident_error,
            "steps": trace.step_count,
            "elapsed_ms": round(trace.elapsed_ms, 3),
            "budget_exhausted": trace.budget_exhausted,
        }
        self.results.append(row)
        self.traces.append(trace.as_dict())
        self._trim()

        self.library.note_result(strategy.name, family, solved=solved,
                                 abstained=trace.abstained,
                                 cost_ms=trace.elapsed_ms, steps=trace.step_count)
        self.attempts += 1
        self.solved_count += 1 if solved else 0
        self.abstentions += 1 if trace.abstained else 0
        self.confident_errors += 1 if confident_error else 0
        return row

    def _trim(self) -> None:
        limit = max(100, int(cfg.REASON_MAX_TRACES))
        if len(self.results) > limit:
            del self.results[:len(self.results) - limit]
        if len(self.traces) > limit:
            del self.traces[:len(self.traces) - limit]

    # ── what the evidence says ───────────────────────────────────────

    def per_family(self) -> dict[str, dict]:
        """Attempts, successes and an interval, per family."""
        table: dict[str, dict] = {}
        for row in self.results:
            entry = table.setdefault(row["family"], {
                "used": 0, "solved": 0, "abstained": 0, "confident_errors": 0})
            entry["used"] += 1
            entry["solved"] += 1 if row["solved"] else 0
            entry["abstained"] += 1 if row["abstained"] else 0
            entry["confident_errors"] += 1 if row["confident_error"] else 0
        for family, entry in table.items():
            low, high = wilson_interval(entry["solved"], entry["used"])
            entry["accuracy"] = entry["solved"] / entry["used"] if entry["used"] else 0.0
            entry["lower"], entry["upper"] = low, high
            entry["family"] = family
        return table

    def weaknesses(self) -> list[dict]:
        """Families with enough evidence to be called weak, worst first.

        The *upper* bound is the test, not the point estimate: a family is a
        weakness when even the optimistic reading of its record is poor. That
        keeps the synthesiser off families that merely had a bad afternoon.
        """
        found = []
        for family, entry in sorted(self.per_family().items()):
            if entry["used"] < MIN_RESULTS_FOR_WEAKNESS:
                continue
            if entry["upper"] >= 1.0:
                continue
            found.append({
                "family": family,
                "used": entry["used"],
                "solved": entry["solved"],
                "accuracy": round(entry["accuracy"], 4),
                "lower": round(entry["lower"], 4),
                "upper": round(entry["upper"], 4),
                "confident_errors": entry["confident_errors"],
                "best_strategy": getattr(self.library.best_for(family), "name", ""),
            })
        found.sort(key=lambda entry: (entry["upper"], entry["family"]))
        return found

    def top_weakness(self) -> dict | None:
        found = self.weaknesses()
        return found[0] if found else None

    def dataset(self) -> list[dict]:
        """Traces paired with their outcome — the training set of M6.5.

        Kept as data rather than written out on every attempt: whoever consumes
        it decides the format, and a contour that wrote a file per trace would
        spend more time on disk than on thinking.
        """
        rows = []
        for trace, row in zip(self.traces, self.results[-len(self.traces):]):
            rows.append({"task": row["task"], "family": row["family"],
                         "strategy": row["strategy"], "solved": row["solved"],
                         "trace": trace})
        return rows

    # ── reporting ────────────────────────────────────────────────────

    def accuracy(self) -> float:
        return self.solved_count / self.attempts if self.attempts else 0.0

    def holdout_score(self, count: int = 32) -> float:
        """Accuracy on problems the engine has never queued.

        Held out by construction rather than by bookkeeping: the queue walks
        forward from zero and this walks backward from a far index, so the two
        cannot meet within any run length the system will ever reach.
        """
        solved = 0
        total = max(1, int(count))
        for offset in range(total):
            task = reasoning_bench.build(10_000_000 - offset)
            strategy = self.select(str(task.family), task.id)
            trace = self.interpreter.run(strategy, task, budget=self._budget())
            solved += 1 if trace.solved else 0
        return solved / total

    def status(self) -> dict:
        return {
            "attempts": self.attempts,
            "solved": self.solved_count,
            "accuracy": round(self.accuracy(), 4),
            "abstentions": self.abstentions,
            "confident_errors": self.confident_errors,
            "queued": len(self.queue),
            "results": len(self.results),
            "library": self.library.status(),
            "weakness": self.top_weakness(),
        }

    def publish_metrics(self, tick: int) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            self.telemetry.record(M.REASON_STRATEGIES_ACTIVE,
                                  len(self.library.active()), tick)
            self.telemetry.record(M.REASON_ABSTAIN_RATE,
                                  self.abstentions / self.attempts
                                  if self.attempts else 0.0, tick)
            self.telemetry.record(M.REASON_CONFIDENT_ERROR,
                                  self.confident_errors / self.attempts
                                  if self.attempts else 0.0, tick)
            for strategy in self.library.active():
                if strategy.used():
                    self.telemetry.record(M.REASON_WIN_RATE,
                                          wilson_lower(strategy.solved(),
                                                       strategy.used()),
                                          tick, tags={"strategy": strategy.name})
        except Exception:
            logger.exception("Reasoning metric publication failed")

    def save(self) -> None:
        self.library.save()
