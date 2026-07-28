"""Running a strategy, under limits that cannot be argued with (spec M6.3).

The interpreter is the reason the DSL is safe to synthesise. It can do exactly
the twelve operations the grammar names, it counts every step it takes against
one budget, and it stops when the budget is gone — including inside a ``LOOP``
whose condition is always true, which is the failure a synthesiser will
eventually write.

Everything it can reach is injected: the cortex, the world model, the memory,
the sandbox. A strategy therefore cannot reach anything the caller did not hand
it, and a test can run a full strategy with nothing attached at all — which is
also how the deterministic path works when no model is available.

The output is a :class:`Trace`. It is not a log: it is the record the arena
scores, the training data the dataset builder reads, and the explanation an
operator gets. A step that happened and left no trace did not happen.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.layers.motivation.resources import ResourceCost
from aegis.layers.reasoning.dsl import (
    MAX_LOOP_ITERATIONS, OPS, cost_of,
)
from aegis.layers.reasoning.reasoner import REASONER

logger = logging.getLogger("aegis.reasoning")

#: What a step returns when nothing is attached to do it. Distinct from a
#: failure: "there was no model" and "the model was wrong" are different facts
#: and the arena scores them differently.
UNAVAILABLE = "__unavailable__"

#: Hard ceiling on how finely a problem may be cut up. The gene asks; this
#: decides, for the same reason every other gene is clamped at the reader.
MAX_DECOMPOSE_PARTS = 8


def _clauses(text: str) -> list[str]:
    """Sentences, plus the ``, then …`` steps inside one sentence.

    A question mark stays attached to its clause: the chain reasoner uses it to
    tell "the problem ended here" from "the part cap cut it off here", and those
    are different situations.
    """
    rough = re.sub(r",\s+(then|and then)\s+", ". ", str(text or ""))
    rough = rough.replace(";", ".")
    parts = []
    for piece in re.split(r"(?<=[.?!])\s+", rough):
        piece = piece.strip()
        if piece.endswith("."):
            piece = piece[:-1].strip()
        if piece:
            parts.append(piece)
    return parts


#: Operations that produce an answer. Everything else — retrieving, predicting,
#: verifying, branching — informs the run without becoming its result. Letting
#: ``VERIFY`` set the answer was the obvious version of this and turned every
#: verified answer into the boolean ``True``.
ANSWERING_OPS = frozenset({"SOLVE", "COMPUTE", "LLM_STEP", "VOTE"})


@dataclass
class Step:
    """One executed operation."""

    index: int
    op: str
    result: object = None
    ok: bool = True
    note: str = ""
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict:
        return {"index": self.index, "op": self.op, "ok": self.ok,
                "note": self.note, "elapsed_ms": round(self.elapsed_ms, 3),
                "result": _renderable(self.result)}


def _renderable(value):
    """Trace entries are persisted and shipped, so they hold data, not objects."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_renderable(item) for item in value][:10]
    if isinstance(value, dict):
        return {str(k): _renderable(v) for k, v in list(value.items())[:10]}
    return str(value)[:200]


@dataclass
class Trace:
    """What happened, in order, with what it cost."""

    strategy: str = ""
    task_id: str = ""
    answer: object = None
    abstained: bool = False
    solved: bool | None = None
    steps: list[Step] = field(default_factory=list)
    budget_exhausted: bool = False
    cost: ResourceCost = field(default_factory=ResourceCost)
    elapsed_ms: float = 0.0

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "task_id": self.task_id,
            "answer": _renderable(self.answer),
            "abstained": self.abstained,
            "solved": self.solved,
            "budget_exhausted": self.budget_exhausted,
            "steps": [step.as_dict() for step in self.steps],
            "cost": self.cost.as_dict(),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


class Interpreter:
    """Executes a strategy. Knows nothing it was not given."""

    def __init__(self, *, cortex=None, world_model=None, memory=None,
                 graph=None, solver=None, genome=None, sandbox=None,
                 max_steps: int | None = None):
        self.cortex = cortex
        self.world_model = world_model
        self.memory = memory
        self.graph = graph
        self.solver = solver
        self.genome = dict(genome or {})
        self.sandbox = sandbox
        self.max_steps = int(cfg.REASON_MAX_STEPS if max_steps is None else max_steps)

    # ── entry point ──────────────────────────────────────────────────

    def run(self, strategy, task, *, budget: int | None = None) -> Trace:
        """Run ``strategy`` against ``task`` and return the trace.

        ``budget`` overrides the step ceiling for this run — it is the
        ``reason_budget`` gene (Appendix C), so evolution can buy more thinking
        or less. It can never exceed the configured maximum: a gene may spend
        within a limit, never past it.
        """
        steps = getattr(strategy, "steps", strategy)
        name = getattr(strategy, "name", "anonymous")
        ceiling = self.max_steps
        if budget is not None:
            ceiling = max(1, min(self.max_steps, int(budget)))

        trace = Trace(strategy=name, task_id=getattr(task, "id", ""))
        started = CLOCK.monotonic()
        state = {"remaining": ceiling, "index": 0, "last": None,
                 "verified": None, "parts": [], "abstain": False,
                 "confident": None, "prediction": None, "retrieved": []}
        try:
            self._block(steps, task, trace, state)
        except Exception:
            logger.exception("A reasoning strategy failed")
            trace.add(Step(index=state["index"], op="ERROR", ok=False,
                           note="the strategy raised"))
        trace.elapsed_ms = (CLOCK.monotonic() - started) * 1000
        trace.answer = state["last"]
        trace.abstained = bool(state["abstain"])
        trace.budget_exhausted = state["remaining"] <= 0
        try:
            trace.cost = cost_of(steps)
        except Exception:
            # Pricing a malformed strategy must not take down the tick either.
            logger.debug("Could not price a strategy", exc_info=True)
        if hasattr(task, "verify"):
            trace.solved = bool(task.verify(trace.answer))
        return trace

    # ── the block walker ─────────────────────────────────────────────

    def _block(self, steps, task, trace: Trace, state: dict) -> None:
        for step in steps if isinstance(steps, list) else []:
            if state["remaining"] <= 0:
                trace.add(Step(index=state["index"], op="BUDGET", ok=False,
                               note="step budget exhausted"))
                return
            if state["abstain"]:
                return                  # an abstention ends the strategy
            if not isinstance(step, dict) or str(step.get("op")) not in OPS:
                trace.add(Step(index=state["index"], op=str(step),
                               ok=False, note="unknown operation"))
                state["index"] += 1
                state["remaining"] -= 1
                continue
            self._execute(step, task, trace, state)

    def _execute(self, step: dict, task, trace: Trace, state: dict) -> None:
        operation = str(step["op"])
        state["remaining"] -= 1
        record = trace.add(Step(index=state["index"], op=operation))
        state["index"] += 1
        started = CLOCK.monotonic()

        # The step is recorded *before* its handler runs, so a trace reads in
        # the order things happened. Recording it afterwards put every nested
        # step ahead of the BRANCH or VOTE that caused it, which reads as though
        # the branch was decided after its own body had already run.
        handler = getattr(self, f"_op_{operation.lower()}")
        try:
            result, ok, note = handler(step, task, trace, state)
        except Exception as exc:
            result, ok, note = None, False, f"{type(exc).__name__}: {exc}"
            logger.debug("Reasoning step %s failed", operation, exc_info=True)

        if ok and result is not None and operation in ANSWERING_OPS:
            state["last"] = result
        record.result, record.ok, record.note = result, ok, note
        record.elapsed_ms = (CLOCK.monotonic() - started) * 1000

    # ── operations ───────────────────────────────────────────────────

    def _op_decompose(self, step, task, trace, state):
        """Split the problem into clauses, up to a cap.

        Sentences *and* ``, then`` sequences, because a chain of instructions is
        written as one sentence and splitting only on the full stop would leave
        it undivided — which is the case decomposition exists for.

        The cap is a real limit, not a formality: a chain longer than the cap
        comes back truncated and the reasoner answers from a partial problem.
        That is what makes ``reason_decompose_parts`` a gene worth selecting on.
        """
        # A strategy that names no cap gets the system's, which is a gene. So
        # granularity is evolvable even for strategies that never mention it —
        # otherwise only strategies written to ask for it could be tuned.
        limit = self._resolve_int(
            step.get("max_parts", "$gene:reason_decompose_parts"), 4, 2,
            MAX_DECOMPOSE_PARTS)
        text = str(getattr(task, "prompt", ""))
        parts = _clauses(text)[:limit]
        state["parts"] = parts
        return parts, bool(parts), f"{len(parts)} part(s)"

    def _op_retrieve(self, step, task, trace, state):
        """Look something up. What is found informs the run; it is not the answer."""
        source = str(step.get("source"))
        limit = self._resolve_int(step.get("k"), 5, 1, 20)
        query = str(getattr(task, "prompt", ""))[:120]
        hits: list = []
        note = f"no {source} attached"
        if source == "memory" and hasattr(self.memory, "retrieve"):
            hits = list(self.memory.retrieve(query, limit=limit) or [])
            note = f"{len(hits)} from memory"
        elif source == "graph" and hasattr(self.graph, "related"):
            hits = list(self.graph.related(query) or [])
            note = f"{len(hits)} from the graph"
        elif source == "skills" and self.solver is not None:
            skills = self.solver.library.for_kind(str(getattr(task, "family", "")))
            hits = [skill.name for skill in skills]
            note = f"{len(hits)} skill(s)"
        hits = hits[:limit]
        state["retrieved"] = hits
        return hits, True, note

    def _op_predict(self, step, task, trace, state):
        horizon = self._resolve_int(step.get("horizon"), 1, 1, 5)
        if not hasattr(self.world_model, "predict_outcome"):
            state["prediction"] = None
            return UNAVAILABLE, True, "no world model attached"
        outcome = self.world_model.predict_outcome(
            getattr(task, "state", None), str(getattr(task, "family", "")))
        prediction = {"p_success": getattr(outcome, "p_success", None),
                      "horizon": horizon}
        state["prediction"] = prediction
        return prediction, True, "predicted"

    def _op_solve(self, step, task, trace, state):
        """Attempt the task.

        The reference reasoner is used unconditionally rather than only as a
        fallback: it reports whether it parsed the problem or guessed at it, and
        that flag is what abstention branches on. An external solver that
        answered without saying which it did would take that signal away.
        """
        answer = REASONER.solve(task, clauses=state.get("parts") or None)
        state["confident"] = answer.confident
        if answer.value is None:
            return None, False, f"no answer ({answer.method})"
        return answer.value, True, answer.method

    def _op_compute(self, step, task, trace, state):
        """Arithmetic, in the sandbox. Never ``eval``.

        A synthesised expression evaluated in this process is arbitrary code
        execution; the sandbox is the same one skills run in, with the same
        static gate (Appendix B, category 3).
        """
        expression = step.get("expr")
        if expression == "$last":
            expression = state.get("last")
        if not isinstance(expression, str) or not expression.strip():
            return None, False, "nothing to compute"
        if self.sandbox is None:
            return UNAVAILABLE, True, "no sandbox attached"
        out = self.sandbox(f"def solve(p):\n    return {expression}\n",
                           "solve", {})
        if out.get("ok"):
            return out.get("result"), True, "computed"
        return None, False, str(out.get("error", "compute failed"))[:80]

    def _op_llm_step(self, step, task, trace, state):
        if self.cortex is None:
            return UNAVAILABLE, True, "no cortex attached"
        role = str(step.get("role", "fast"))
        if not self.cortex.role_available(role):
            return UNAVAILABLE, True, f"role {role} unavailable"
        return UNAVAILABLE, True, "cortex step deferred to the engine"

    def _op_verify(self, step, task, trace, state):
        """Check the working answer — never against the right answer.

        ``checker="task"`` deliberately does **not** call ``task.verify``. That
        method is the benchmark's grader, and a strategy allowed to call it
        could answer every question by trying values until the grader agreed.
        Measured: a strategy that abstained whenever the grader disagreed scored
        100% on a benchmark it had not solved at all. A task may expose a
        *solver-facing* check — a coding task's unit tests, say — and that is
        what ``task`` means here; a task without one is honestly unverifiable.
        """
        checker = str(step.get("checker", "type"))
        answer = state.get("last")
        has_value = answer is not None and answer != UNAVAILABLE
        if checker == "task":
            check = getattr(task, "self_check", None)
            if not callable(check):
                state["verified"] = None
                return None, True, "the task carries no solver-facing check"
            ok = bool(check(answer))
            state["verified"] = ok
            return ok, True, "checked" if ok else "check failed"
        if checker == "confidence":
            ok = has_value and state.get("confident") is not False
            state["verified"] = ok
            return ok, True, "reasoned" if ok else "guessed"
        if checker == "type":
            state["verified"] = has_value
            return has_value, True, "has a value" if has_value else "no value"
        if checker == "consistency":
            # Solve again and see whether the same answer comes back. Worth
            # nothing against a deterministic reasoner, which is exactly what a
            # strategy using it should be scored as discovering.
            again = REASONER.solve(task, clauses=state.get("parts") or None)
            ok = has_value and again.value == answer
            state["verified"] = ok
            return ok, True, "stable" if ok else "unstable"
        state["verified"] = None
        return None, True, f"no checker {checker!r}"

    def _op_vote(self, step, task, trace, state):
        """Run a branch several times and aggregate.

        Deterministically, so the votes differ only if the branch itself does.
        With a deterministic branch every vote agrees, which is the honest
        outcome — self-consistency buys nothing when there is no variance to
        consult.
        """
        rounds = self._resolve_int(step.get("n", "$gene:reason_vote_n"), 1, 1, 5)
        aggregator = str(step.get("agg", "majority"))
        body = step.get("body") or []
        if not body:
            # Voting on nothing produced N copies of whatever the previous step
            # left behind and reported a unanimous majority for it.
            return None, False, "no votes"
        answers = []
        for _ in range(rounds):
            if state["remaining"] <= 0:
                break
            self._block(body, task, trace, state)
            answers.append(state.get("last"))
        if not answers:
            return None, False, "no votes"
        if aggregator == "first":
            return answers[0], True, "first"
        if aggregator == "unanimous":
            agreed = all(answer == answers[0] for answer in answers)
            return (answers[0] if agreed else None), agreed, \
                "unanimous" if agreed else "split"
        counts: dict[str, int] = {}
        for answer in answers:
            counts[str(answer)] = counts.get(str(answer), 0) + 1
        best = max(sorted(counts), key=lambda key: counts[key])
        winner = next(a for a in answers if str(a) == best)
        return winner, True, f"majority {counts[best]}/{len(answers)}"

    def _op_branch(self, step, task, trace, state):
        taken = self._condition(str(step.get("cond", "")), state)
        body = step.get("then" if taken else "else") or []
        self._block(body, task, trace, state)
        return taken, True, "then" if taken else "else"

    def _op_loop(self, step, task, trace, state):
        limit = self._resolve_int(step.get("max_iter"), MAX_LOOP_ITERATIONS,
                                  1, MAX_LOOP_ITERATIONS)
        condition = str(step.get("while", ""))
        iterations = 0
        while iterations < limit:
            if state["remaining"] <= 0:
                break
            if condition and not self._condition(condition, state):
                break
            self._block(step.get("body") or [], task, trace, state)
            iterations += 1
        return iterations, True, f"{iterations} iteration(s)"

    def _op_reflect(self, step, task, trace, state):
        return {"verified": state.get("verified"),
                "parts": len(state.get("parts") or [])}, True, "reflected"

    def _op_abstain(self, step, task, trace, state):
        """Refuse to answer. Better than a confident mistake, and scored so."""
        state["abstain"] = True
        state["last"] = None
        return None, True, str(step.get("reason", "insufficient data"))[:80]

    # ── helpers ──────────────────────────────────────────────────────

    def _condition(self, name: str, state: dict) -> bool:
        """The closed set of predicates a BRANCH or LOOP may test.

        Closed on purpose: a condition that could be an arbitrary expression
        would need an evaluator, and an evaluator is the thing this DSL exists
        to avoid.
        """
        if name == "verify_failed":
            return state.get("verified") is False
        if name == "insufficient":
            # Not merely "no answer": an answer the reasoner admits it guessed
            # at counts as insufficient too. That is the whole content of
            # knowing when not to answer — silence is easy to detect, an
            # unfounded number is not.
            return (state.get("last") in (None, UNAVAILABLE)
                    or state.get("confident") is False)
        if name == "parts_remaining":
            return bool(state.get("parts"))
        if name == "nothing_retrieved":
            return not state.get("retrieved")
        if name.startswith("p_success_below:"):
            try:
                threshold = float(name.split(":", 1)[1])
            except (TypeError, ValueError):
                return False
            prediction = state.get("prediction")
            probability = prediction.get("p_success") if isinstance(prediction, dict) else None
            return probability is not None and probability < threshold
        return False

    def _resolve_int(self, value, default: int, low: int, high: int) -> int:
        """Resolve ``$gene:<name>`` and clamp. The tie between M5 and M6.

        Clamped here rather than trusted: a gene is a number evolution chose,
        and the interpreter's limits are not evolution's to widen.
        """
        if isinstance(value, str) and value.startswith("$gene:"):
            value = self.genome.get(value.split(":", 1)[1], default)
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(low, min(high, number))
