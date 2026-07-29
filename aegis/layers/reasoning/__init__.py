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
import math
from pathlib import Path

import aegis.config as cfg
from aegis.eval import reasoning_bench
from aegis.layers.reasoning.arena import Arena, Verdict, conclude_trial
from aegis.layers.reasoning.dsl import DSLError, cost_of, digest, validate
from aegis.layers.reasoning.interpreter import Interpreter, Trace
from aegis.layers.reasoning.library import BUILTIN_STRATEGIES, Library, Strategy
from aegis.layers.reasoning.reasoner import REASONER, DeterministicReasoner
from aegis.layers.reasoning.synthesis import (
    Candidate, Synthesiser, traffic_share,
)
from aegis.layers.reasoning.weakness import Weakness, WeaknessDetector
from aegis.util.quasirandom import hash_index
from aegis.util.stats import wilson_interval, wilson_lower

logger = logging.getLogger("aegis.reasoning")

__all__ = ["ReasoningEngine", "Interpreter", "Trace", "Library", "Strategy",
           "DeterministicReasoner", "REASONER", "BUILTIN_STRATEGIES",
           "DSLError", "validate", "cost_of", "digest", "Arena", "Verdict",
           "Synthesiser", "Candidate", "WeaknessDetector", "Weakness"]

#: How often a trial gets a request of its class: one in this many. Small enough
#: that a trial accumulates its REASON_TRIAL_N applications in a reasonable
#: number of ticks, large enough that a bad trial cannot spoil the whole class
#: while it is being judged.
TRIAL_EVERY = 4

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


class _WeaknessRef:
    """What the arena needs to know about a weakness: its name and its class.

    A candidate outlives the scan that produced it — it can sit in the queue
    while later scans replace the detector's findings — so the arena is handed
    the label the candidate was written against rather than whatever the
    detector happens to be reporting now.
    """

    __slots__ = ("label", "family")

    def __init__(self, label: str, family: str):
        self.label = str(label)
        self.family = str(family)


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
        #: The three halves of improvement (M6.6-M6.8): find where it is bad,
        #: write something better, and refuse to believe it until it is shown.
        self.detector = WeaknessDetector()
        self.synthesiser = Synthesiser(cortex=cortex)
        self.arena = Arena(self.interpreter)
        self.trial_every = TRIAL_EVERY

        #: Strategies proposed but not yet judged by the arena.
        self.candidates: list[Candidate] = []
        #: Every verdict the arena has reached, newest last.
        self.verdicts: list[dict] = []
        #: The detector's last findings, in rank order.
        self.found: list[Weakness] = []
        self.promotions = 0
        self.demotions = 0
        self.last_scan_tick = 0

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

    @property
    def _ucb_c(self) -> float:
        """The exploration constant, as a gene (Appendix C).

        Read on every use rather than cached: a genome applied mid-run has to
        take effect, and a cached copy is how a gene becomes decorative.
        """
        try:
            return float(self.genome.get("reason_ucb_c", cfg.REASON_UCB_C))
        except (TypeError, ValueError):
            return cfg.REASON_UCB_C

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

    def pending_candidates(self) -> list[Candidate]:
        return list(self.candidates)

    # ── selection ────────────────────────────────────────────────────

    def select(self, family: str, key: str = "") -> Strategy:
        """Which strategy to try on this family (M6.8).

        Three layers, in order.

        **A trial's share.** A strategy the arena accepted takes every k-th
        request of its class, decided by hashing the request rather than by a
        counter — a counter would make the split depend on how many other
        strategies happened to run first, and two runs of one experiment would
        divide the traffic differently.

        **Exploration while the class is unmeasured.** Spread across the
        untried strategies by a hash of the task, not by "whichever is least
        used": inside one scoring pass nothing is recorded, so least-used
        returns the same strategy every time. Measured — that made the baseline
        the alphabetically first strategy, which happened to be the best one,
        and the whole learning curve came out flat at the ceiling.

        **UCB once there is evidence.** ``win_rate + c·sqrt(ln N / n)``,
        deterministic, with ``c`` the ``reason_ucb_c`` gene. No RNG anywhere
        (§3.1).
        """
        active = self.library.active()
        if not active:                  # cannot happen: built-ins never retire
            self.library.seed()
            active = self.library.active()

        for trial in self.library.trials(family):
            if traffic_share(trial.name, key, self.trial_every):
                return trial

        unproven = [s for s in active if s.used(family) < MIN_ATTEMPTS_PER_FAMILY]
        if unproven:
            return unproven[hash_index(len(unproven), "reason_explore",
                                       family, key)]
        return max(active, key=lambda s: (self._ucb(s, family, active), s.name))

    def best_known(self, family: str, key: str = "") -> Strategy:
        """What the system would answer with if it had to answer now.

        Not the same question as :meth:`select`, and the difference matters when
        measuring. ``select`` runs an exploration schedule: it deliberately
        spends traffic on strategies it does not believe in, and it hands a
        whole class to one of them at a time. Scoring a held-out set through it
        measured the schedule rather than the system, and the curve swung
        twenty points depending on which cycle it was read at.

        With no evidence yet there is nothing to be greedy about, so this falls
        back to the exploring choice — which is the honest baseline: a system
        that knows nothing does have to guess.
        """
        best = self.library.best_for(family, min_used=MIN_ATTEMPTS_PER_FAMILY)
        return best or self.select(family, key)

    def _ucb(self, strategy: Strategy, family: str, active) -> float:
        """Upper confidence bound on this strategy for this family.

        The exploration term is what keeps a strategy that lost its first few
        attempts from being written off: as everything else accumulates uses,
        its bound rises again and it gets another look. Without it the first
        strategy to reach a decent rate takes the family for good.
        """
        used = strategy.used(family)
        if used <= 0:
            return float("inf")
        total = sum(other.used(family) for other in active) or 1
        exploration = self._ucb_c * math.sqrt(math.log(total + 1) / used)
        return strategy.accuracy(family) + exploration

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
            # The task's own feature labels ride along, because the weakness
            # detector groups along them and a record that only said which
            # family failed could not describe a weakness narrower than a family.
            "features": dict(getattr(task, "features", {}) or {}),
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

    # ── improvement: find, write, judge (M6.6-M6.8) ──────────────────

    def scan_weakness(self, window: int | None = None) -> list[dict]:
        """The contour's periodic review. The ``scan_weakness`` action.

        Two jobs, and they are one action on purpose: Appendix A is the
        registry's contract, and adding an action to it is a change to the
        spec rather than a convenience. Concluding finished trials belongs to
        the same "look at how this is going" moment as scanning, and a trial
        nobody concluded would hold its share of a class for ever.

        The scan itself is finer than :meth:`weaknesses`: that one groups by
        family and is what the planner's precondition consults, because it is
        cheap and always available. This groups by *feature combination* with a
        significance test behind it, which is what a synthesiser can actually
        aim at — "arithmetic chains whose data is incomplete" is a target;
        "reasoning is weak" is not.
        """
        self.review_trials(tick=self.attempts)
        self.found = self.detector.scan(self.results, window=window)
        self.last_scan_tick = self.attempts
        return [weakness.as_dict() for weakness in self.found]

    def _target(self):
        """The weakness to work on, scanning first if nothing is on hand."""
        if not getattr(self, "found", None):
            self.scan_weakness()
        return self.found[0] if self.found else None

    def propose_strategy(self, tick: int = 0) -> list[dict]:
        """The template path of M6.7. Synchronous, model-free, always available."""
        weakness = self._target()
        if weakness is None:
            return []
        fresh = self.synthesiser.propose(weakness, self.library, tick)
        self.candidates.extend(fresh)
        return [candidate.as_dict() for candidate in fresh]

    async def propose_strategy_async(self, tick: int = 0) -> list[dict]:
        """The templates, plus a model's proposal when one is reachable.

        The two paths are kept apart so the half that must work cannot fail
        with the half that might. A cortex that is down, slow or wrong costs
        this call one candidate, not all of them.
        """
        weakness = self._target()
        if weakness is None:
            return []
        fresh = self.synthesiser.propose(weakness, self.library, tick)
        try:
            fresh += await self.synthesiser.propose_with_cortex(
                weakness, self.library, tick, existing=fresh)
        except Exception:
            logger.exception("The cortex synthesis path failed")
        self.candidates.extend(fresh)
        return [candidate.as_dict() for candidate in fresh]

    def evaluate_candidate(self, tick: int = 0) -> dict | None:
        """Judge the oldest pending candidate. The arena action (M6.8).

        One per call. An arena run is six scored passes over three task sets,
        and doing the whole queue in one action would be an action that takes
        minutes — the same mistake ``evolve_generation`` made before it was
        detached.
        """
        if not self.candidates:
            return None
        candidate = self.candidates.pop(0)
        family = self._family_of(candidate)
        incumbent = self.library.best_for(family) or self.library.get("direct")
        weakness = _WeaknessRef(candidate.weakness, family)

        verdict = self.arena.evaluate(candidate, weakness, incumbent)
        candidate.verdict = verdict.as_dict()
        record = {"candidate": candidate.name, "family": family,
                  "incumbent": getattr(incumbent, "name", ""),
                  "weakness": candidate.weakness, "tick": int(tick),
                  **verdict.as_dict()}
        self.verdicts.append(record)
        if len(self.verdicts) > 200:
            del self.verdicts[:len(self.verdicts) - 200]

        if not verdict.accepted:
            return record
        try:
            self.library.admit(
                candidate.name, candidate.steps, origin=candidate.origin,
                parent=candidate.parent, tick=tick, status="trial",
                weakness=candidate.weakness, family=family,
                incumbent=getattr(incumbent, "name", ""))
        except DSLError as error:
            # Accepted by the arena and refused by the library is a real
            # disagreement, not a formality: the arena runs steps, the library
            # admits strategies, and only one of them checks for duplicates.
            record["reasons"] = [f"admission refused: {error}"]
            record["accepted"] = False
            logger.info("A strategy the arena accepted was refused: %s", error)
        return record

    def review_trials(self, tick: int = 0) -> list[dict]:
        """Conclude trials that have had their run (M6.8).

        The second judgement, on live traffic. An arena run says a strategy is
        better on problems chosen for it; this says whether it was better on the
        problems that actually arrived.
        """
        concluded = []
        for trial in self.library.trials():
            family = trial.family or ""
            incumbent = self.library.get(trial.incumbent)
            outcome, reason = conclude_trial(trial, incumbent, family)
            if outcome == "trial":
                continue
            if outcome == "active":
                self.library.promote(trial.name)
                self.promotions += 1
            else:
                self.library.retire(trial.name, reason=reason)
                self.demotions += 1
            concluded.append({"strategy": trial.name, "family": family,
                              "outcome": outcome, "reason": reason,
                              "tick": int(tick)})
        return concluded

    @staticmethod
    def _family_of(candidate) -> str:
        """Which class a candidate was written for.

        Read off the weakness label rather than carried separately: the label is
        what the detector produced and what the operator sees, and a second copy
        of the same fact is a second copy that can disagree.
        """
        for part in str(getattr(candidate, "weakness", "")).split(" AND "):
            if part.startswith("family="):
                return part.split("=", 1)[1]
        return ""

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

    def strategies_report(self) -> list[dict]:
        """Every strategy with its record, per family — the panel of §M10.1."""
        return [strategy.to_dict() for strategy in
                sorted(self.library.strategies.values(),
                       key=lambda item: item.name)]

    def weaknesses_report(self) -> list[dict]:
        """What the last scan found, ranked. Empty is a real answer.

        A contour that reported nothing when it had found nothing looks broken;
        one that always reports something is worse. The rank and the support
        are carried so an operator can see which of these is happening.
        """
        return [weakness.as_dict() for weakness in self.detector.last]

    def trace(self, trace_id: str) -> dict | None:
        """One reasoning trace, for ``/api/reasoning/trace/{id}`` (§M10.1).

        Keyed on the task id rather than an identifier of its own. A trace has
        no independent existence — it is what happened when one strategy met
        one problem — and the task id is what an operator actually has in hand,
        because it is the column the attempt table shows. Searched newest
        first, so a re-attempt of the same problem returns the latest trace.
        """
        wanted = str(trace_id)
        for record in reversed(self.traces):
            if str(record.get("task_id", "")) == wanted:
                return dict(record)
        return None

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
            "detector": self.detector.status(),
            "synthesiser": self.synthesiser.status(),
            "arena": self.arena.status(),
            "candidates": len(self.candidates),
            "promotions": self.promotions,
            "demotions": self.demotions,
        }

    def publish_metrics(self, tick: int) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            self.telemetry.record(M.REASON_STRATEGIES_ACTIVE,
                                  len(self.library.active()), tick)
            self.telemetry.record(M.REASON_PASS_HOLDOUT, self.accuracy(), tick)
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
