"""The reference reasoner: what ``SOLVE`` does when no model is attached.

With a cortex available, reasoning steps go to a language model. Without one —
in the test suite, in an offline rollout, in every deterministic run the spec
requires (§3.1) — ``SOLVE`` still has to do something, and returning "no model"
would make the whole reasoning contour unmeasurable exactly where it needs to
be measured.

So this is a small, real reasoner: eight parsers, each of which genuinely solves
one shape of problem from the prompt text. It never sees ``task.expected``.
What it cannot solve, it guesses at — by answering the last number in the text,
which is the single most common shallow heuristic and is wrong in the one place
the benchmark cares most about.

Two properties make it useful rather than decorative:

* **Confidence is reported.** A parse that matched is confident; the numeric
  fallback is not. That flag is what ``ABSTAIN`` strategies branch on, so
  "know when not to answer" is a real capability here rather than a slogan.
* **Some parsers need clauses.** The chain solver applies one instruction per
  clause and cannot read a chain out of an undivided blob. That is why
  ``DECOMPOSE`` earns its cost on ``arithmetic_chain`` and why the
  ``reason_decompose_parts`` gene changes accuracy: a chain cut short by the
  part cap gives the wrong answer. The gradient evolution climbs there is real,
  not arranged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from aegis.eval.reasoning_bench import ABSTAIN

#: Words that mark a question about a *remainder* — the part of a whole that
#: was not mentioned. Only used to describe the fallback, never to answer.
RESIDUAL_MARKERS = ("left", "remaining", "remain", "rest", "leftover")

_UNIT_SCALE = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0}
_UNIT_RE = "mm|cm|km|m"          # longest first: "m" must not eat "mm"


@dataclass(frozen=True)
class Answer:
    """A value, and whether it was reasoned or guessed."""

    value: object
    confident: bool = True
    method: str = ""


def _numbers(text: str) -> list[float]:
    return [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _tidy(value: float):
    """A whole number comes back as an int, so ``7`` does not print as ``7.0``."""
    return int(value) if float(value).is_integer() else float(value)


class DeterministicReasoner:
    """Solves what it can parse; says so when it is guessing."""

    def solve(self, task, clauses=None) -> Answer:
        try:
            prompt = str(getattr(task, "prompt", "") or "")
        except Exception:
            # A task whose own prompt cannot be read is a task with nothing to
            # reason about. Returning "no answer" keeps the failure where it
            # belongs — in the trace — instead of aborting the tick.
            return Answer(None, confident=False, method="none")
        clauses = [str(part).strip() for part in (clauses or []) if str(part).strip()]
        text = " ".join(clauses) if clauses else prompt

        for parser in (self._chain, self._units, self._grid, self._digits,
                       self._consistency, self._implication, self._elimination):
            try:
                answer = parser(text, clauses)
            except Exception:
                answer = None       # a parser that trips is a parser that abstains
            if answer is not None:
                return answer
        return self._fallback(text)

    # ── parsers ──────────────────────────────────────────────────────

    def _chain(self, text: str, clauses: list[str]) -> Answer | None:
        """One instruction per clause. Needs the problem broken up first.

        Deliberately clause-local rather than a scan over the whole text: a
        reasoner that can pick operations out of an undivided paragraph would
        make ``DECOMPOSE`` free, and the cost of decomposition is precisely what
        the arena is meant to weigh.
        """
        if not clauses:
            return None
        start = None
        index = 0
        for position, clause in enumerate(clauses):
            match = re.fullmatch(r"start with (-?\d+)", clause.strip().lower())
            if match:
                start = int(match.group(1))
                index = position + 1
                break
        if start is None:
            return None

        value = start
        complete = False
        applied = 0
        stated = True
        for clause in clauses[index:]:
            body = clause.strip().lower().rstrip(".")
            if "?" in clause:
                complete = True
                break
            match = re.fullmatch(r"(add|subtract|multiply by) (-?\d+)", body)
            if not match:
                # An instruction whose operand is not a number is a hole in the
                # chain, not a clause to skip past. Skipping it silently is how
                # a reasoner produces an answer to a problem it was never given
                # enough to solve.
                if re.match(r"(add|subtract|multiply by)\b", body):
                    stated = False
                continue
            operand = int(match.group(2))
            operation = match.group(1)
            if operation == "add":
                value += operand
            elif operation == "subtract":
                value -= operand
            else:
                value *= operand
            applied += 1
        if not applied:
            return None
        # Not reaching the question clause means the chain was cut short by the
        # part cap. The arithmetic is still reported — it may be right — but it
        # is reported as a guess, because nothing here saw the end of the chain.
        return Answer(value, confident=complete and stated, method="chain")

    def _units(self, text: str, clauses: list[str]) -> Answer | None:
        match = re.search(
            rf"(-?\d+(?:\.\d+)?)\s+({_UNIT_RE})\b.*?how many\s+({_UNIT_RE})\b",
            text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        amount = float(match.group(1))
        source = _UNIT_SCALE[match.group(2).lower()]
        target = _UNIT_SCALE[match.group(3).lower()]
        return Answer(round(amount * source / target, 6), method="units")

    def _grid(self, text: str, clauses: list[str]) -> Answer | None:
        match = re.search(r"how many moves.*?(\d+)\s+right and\s+(\d+)\s+up",
                          text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        return Answer(int(match.group(1)) + int(match.group(2)), method="grid")

    def _digits(self, text: str, clauses: list[str]) -> Answer | None:
        match = re.search(r"how many digits does the number (\d+)",
                          text, re.IGNORECASE)
        if not match:
            return None
        return Answer(len(match.group(1)), method="digits")

    def _consistency(self, text: str, clauses: list[str]) -> Answer | None:
        """Two stated quantities for one thing: can both hold?"""
        if "can both statements be true" not in text.lower():
            return None
        stated = re.findall(r"exactly (-?\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        if len(stated) < 2:
            return None
        return Answer(len({float(value) for value in stated}) == 1,
                      method="consistency")

    def _implication(self, text: str, clauses: list[str]) -> Answer | None:
        """Forward-chain a set of implications to a yes/no.

        Deriving the goal is a confident *yes*. Failing to derive it is **not** a
        confident no: from implications that do not reach the goal, nothing
        follows about the goal. Reporting "false" there would be a closed-world
        assumption nobody stated, and it is exactly how a missing rule turns
        into a confident wrong answer.
        """
        goal = re.search(r"is (\w+) true\?", text, re.IGNORECASE)
        if not goal:
            return None
        rules = re.findall(r"if (\w+) then (\w+)", text, re.IGNORECASE)
        facts = {name.upper() for name in
                 re.findall(r"(\w+) is true(?!\?)", text, re.IGNORECASE)}
        facts.discard(goal.group(1).upper())
        if not rules or not facts:
            return None
        pairs = [(a.upper(), b.upper()) for a, b in rules]
        changed = True
        while changed:                  # terminates: facts only grows, and it is finite
            changed = False
            for premise, conclusion in pairs:
                if premise in facts and conclusion not in facts:
                    facts.add(conclusion)
                    changed = True
        derived = goal.group(1).upper() in facts
        return Answer(derived, confident=derived, method="implication")

    def _elimination(self, text: str, clauses: list[str]) -> Answer | None:
        """Assign items to people from negative clues, by elimination."""
        setup = re.search(
            r"(\w+), (\w+) and (\w+) each carry one of a (\w+), a (\w+) and a (\w+)",
            text, re.IGNORECASE)
        asked = re.search(r"what does (\w+) carry", text, re.IGNORECASE)
        if not setup or not asked:
            return None
        people = [setup.group(i) for i in (1, 2, 3)]
        items = [setup.group(i) for i in (4, 5, 6)]
        if len(set(items)) != 3:
            return None            # not a puzzle with one solution
        domains = {person: set(items) for person in people}
        for clue in re.finditer(
                r"(\w+) does not carry the (\w+)(?: or the (\w+))?",
                text, re.IGNORECASE):
            person = clue.group(1)
            if person not in domains:
                continue
            for denied in (clue.group(2), clue.group(3)):
                if denied in items:
                    domains[person].discard(denied)

        for _ in range(len(people)):    # bounded: one pass can fix at most one person
            settled = {next(iter(values)) for values in domains.values()
                       if len(values) == 1}
            for person, values in domains.items():
                if len(values) > 1:
                    values -= settled
        answer = domains.get(asked.group(1))
        if not answer or len(answer) != 1:
            return None
        return Answer(next(iter(answer)), method="elimination")

    # ── the guess ────────────────────────────────────────────────────

    def _fallback(self, text: str) -> Answer:
        """Answer the last number in the text, and admit it is a guess.

        This is what shallow reading looks like, and it is included on purpose:
        on a question about a quantity nobody stated, it produces a confident-
        sounding wrong number. That is the failure abstention exists to prevent,
        and a benchmark where the reasoner simply fell silent instead would
        never measure whether abstention was worth anything.
        """
        numbers = _numbers(text)
        if not numbers:
            return Answer(None, confident=False, method="none")
        return Answer(_tidy(numbers[-1]), confident=False, method="guess")


#: One shared instance — it holds no state, and building a fresh one per step
#: would recompile the same patterns on every reasoning step.
REASONER = DeterministicReasoner()


def abstention() -> object:
    """The value that counts as declining to answer."""
    return ABSTAIN
