"""A benchmark for *reasoning*, built from generators (spec M6.5).

The skill benchmark asks whether the system can compute something. This one
asks whether it can work something out — and the difference matters, because a
skill that memorises twenty-eight answers scores full marks on the first and
nothing on the second.

Generators rather than a list, for three reasons:

* **Unbounded held-out.** ``build(i)`` is a pure function of the index, so a
  held-out set can be as large as it needs to be and can be *different* next
  time without anybody curating one.
* **No leak.** There is nothing to leak: the tasks do not exist until they are
  asked for, and the synthesiser is never shown the held-out indices.
* **Labelled.** Every task carries the features it exercises — how many steps,
  which operations, whether it needs arithmetic, whether the data is
  deliberately incomplete. Those labels are the axes the weakness detector
  (M6.6) groups failures along; without them "the system reasons badly" is not
  a statement anything can act on.

One family deserves calling out. ``missing_data`` tasks have no correct answer,
and the correct behaviour is to **abstain**. A system that always answers
scores zero on them however clever it is, which is the point: a confident wrong
answer is worse than an admission, and nothing else in the suite measures that.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aegis.util.quasirandom import hash_index

#: The answer a task with insufficient data expects.
ABSTAIN = "__abstain__"

#: Families, in a fixed order. The index picks a family and then a case within
#: it, so consecutive indices spread across families rather than walking one.
FAMILIES: tuple[str, ...] = (
    "arithmetic_chain",
    "unit_conversion",
    "constraint_puzzle",
    "grid_planning",
    "rule_chain",
    "contradiction",
    "magnitude",
    "missing_data",
)

_NAMES = ("Ada", "Bo", "Cy", "Di", "Eli", "Fay", "Gil", "Hana")
_ITEMS = ("crate", "barrel", "sack", "tin", "spool", "crate", "jar", "bale")


@dataclass(frozen=True)
class ReasoningTask:
    """One reasoning problem, its answer, and what it exercises."""

    id: str
    family: str
    prompt: str
    expected: object
    #: What this task demands — the axes a weakness is described along (M6.6).
    features: dict = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return int(self.features.get("steps", 1))

    @property
    def needs_arithmetic(self) -> bool:
        return bool(self.features.get("numeric"))

    @property
    def answerable(self) -> bool:
        """False when abstaining is the correct answer."""
        return self.expected != ABSTAIN

    def verify(self, answer) -> bool:
        """Programmatic ground truth. Never a judgement call.

        Numbers compare numerically with a tolerance, because a reasoner that
        answers ``7.0`` has not made a mistake. Everything else compares as a
        trimmed, case-folded string, so ``"Yes"`` and ``"yes"`` are the same
        answer and ``"probably"`` is not.
        """
        try:
            if self.expected == ABSTAIN:
                return _is_abstention(answer)
            if _is_abstention(answer):
                return False
            if isinstance(self.expected, bool):
                return isinstance(answer, bool) and answer == self.expected
            if isinstance(self.expected, (int, float)):
                number = _as_number(answer)
                if number is None:
                    return False
                # RELATIVE tolerance with an absolute floor. A flat 1e-6 was
                # fine for answers near 1 but let a mm->km conversion (expected
                # values down to 2e-6) verify answers wrong by up to 50% — the
                # tolerance has to scale with the answer it is judging. The
                # floor keeps an expected value of exactly 0 comparable.
                tolerance = max(1e-9, abs(float(self.expected)) * 1e-6)
                return abs(number - self.expected) <= tolerance
            return str(answer).strip().casefold() == str(self.expected).strip().casefold()
        except Exception:
            return False


def _is_abstention(answer) -> bool:
    if answer is None:
        return True
    text = str(answer).strip().casefold()
    return text in {ABSTAIN, "abstain", "unknown", "insufficient data",
                    "not enough information", "cannot be determined"}


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _pick(options, *material):
    return options[hash_index(len(options), *material)]


def _number(index, low, high, salt=""):
    span = max(1, high - low + 1)
    return low + hash_index(span, "reason_num", salt, index)


# ── the families ─────────────────────────────────────────────────────

def _arithmetic_chain(index: int) -> ReasoningTask:
    """Several operations in sequence — the simplest thing that has a *chain*.

    One case in three leaves an operand unstated, and the correct answer to
    those is to abstain. That mixture is deliberate and is the hardest class in
    the benchmark: getting it right needs the chain broken up *and* the gap
    noticed, and no single built-in strategy does both (M6.4). A family where
    every case yielded to one technique would let selection alone reach the
    ceiling, and there would be nothing left for synthesis to find.
    """
    steps = _number(index, 2, 5, "steps")
    value = _number(index, 2, 40, "start")
    # Which step, if any, is unstated. 0 means the chain is complete.
    hole = hash_index(3 * steps, "reason_hole", index)
    hole = hole if hole <= steps else 0
    parts = [f"start with {value}"]
    for step in range(steps):
        operation = _pick(("add", "multiply by", "subtract"), "reason_op", index, step)
        operand = _number(index * 13 + step, 2, 12, "operand")
        if step + 1 == hole:
            parts.append(f"{operation} some amount")
            continue
        if operation == "add":
            value += operand
        elif operation == "subtract":
            value -= operand
        else:
            value *= operand
        parts.append(f"{operation} {operand}")
    return ReasoningTask(
        id=f"reason_arith_{index}",
        family="arithmetic_chain",
        prompt=", then ".join(parts) + ". What is the result?",
        expected=ABSTAIN if hole else value,
        features={"steps": steps + 1, "numeric": True, "ops": ["compute"],
                  "incomplete": bool(hole)},
    )


_UNITS = {"m": 1.0, "cm": 0.01, "km": 1000.0, "mm": 0.001}


def _unit_conversion(index: int) -> ReasoningTask:
    """A word problem where the units are the trap."""
    units = tuple(_UNITS)
    source = _pick(units, "reason_unit_a", index)
    # A different unit, always: "how many m is 9 m" is not a conversion, and a
    # family whose cases are sometimes trivial measures something else on those.
    others = tuple(unit for unit in units if unit != source)
    target = _pick(others, "reason_unit_b", index)
    amount = _number(index, 2, 500, "amount")
    result = amount * _UNITS[source] / _UNITS[target]
    return ReasoningTask(
        id=f"reason_units_{index}",
        family="unit_conversion",
        prompt=f"A rope is {amount} {source} long. How many {target} is that?",
        expected=round(result, 6),
        features={"steps": 2, "numeric": True, "units": True,
                  "ops": ["compute"]},
    )


def _constraint_puzzle(index: int) -> ReasoningTask:
    """Three people, three items, two clues — the third follows."""
    people = [_NAMES[(index + offset) % len(_NAMES)] for offset in range(3)]
    items = [_ITEMS[(index * 3 + offset) % len(_ITEMS)] for offset in range(3)]
    # Distinct items, or the puzzle has no unique answer.
    items = list(dict.fromkeys(items))
    while len(items) < 3:
        items.append(_pick(_ITEMS, "reason_item_fill", index, len(items)))
    return ReasoningTask(
        id=f"reason_constraint_{index}",
        family="constraint_puzzle",
        prompt=(f"{people[0]}, {people[1]} and {people[2]} each carry one of a "
                f"{items[0]}, a {items[1]} and a {items[2]}. "
                f"{people[0]} does not carry the {items[1]} or the {items[2]}. "
                f"{people[1]} does not carry the {items[2]}. "
                f"What does {people[2]} carry?"),
        expected=items[2],
        features={"steps": 3, "numeric": False, "constraints": 2,
                  "ops": ["decompose", "verify"]},
    )


def _grid_planning(index: int) -> ReasoningTask:
    """Shortest path on an open grid — planning with a checkable answer."""
    width = _number(index, 2, 9, "w")
    height = _number(index, 2, 9, "h")
    return ReasoningTask(
        id=f"reason_grid_{index}",
        family="grid_planning",
        prompt=(f"On a grid you may move only right or up. How many moves does "
                f"it take to go from the bottom-left corner to a point "
                f"{width} right and {height} up?"),
        expected=width + height,
        features={"steps": 2, "numeric": True, "planning": True,
                  "ops": ["compute"]},
    )


def _rule_chain(index: int) -> ReasoningTask:
    """Follow a chain of implications to its end.

    One case in three has a link missing. The right answer there is to abstain,
    not to say "no": from a set of implications that does not reach the goal,
    nothing follows about the goal. A reasoner that reports "false" whenever it
    failed to derive something is reasoning under a closed-world assumption
    nobody stated.
    """
    length = _number(index, 2, 4, "len")
    letters = [chr(ord("A") + (index + offset) % 26) for offset in range(length + 1)]
    # One case in three, the same way _arithmetic_chain engineers it: the draw
    # ranges over 3*length values of which only 1..length name a removable
    # link, so P(broken) = 1/3 and the gap can fall on ANY link. The old
    # ``hash_index(3, ...)`` drew from {0, 1, 2} where BOTH 1 and 2 remove a
    # link — 2/3 of chains were broken, an always-abstain strategy scored ~2/3
    # on this family, and strategy selection was skewed toward abstention
    # (audit: rule_chain mix).
    gap = hash_index(3 * length, "reason_gap", index)
    missing = gap if gap <= length else 0             # 0 = the chain is whole
    rules = " ".join(f"If {letters[i]} then {letters[i + 1]}."
                     for i in range(length) if i + 1 != missing)
    return ReasoningTask(
        id=f"reason_rules_{index}",
        family="rule_chain",
        prompt=f"{rules} {letters[0]} is true. Is {letters[-1]} true?",
        expected=ABSTAIN if missing else True,
        features={"steps": length + 1, "numeric": False, "logic": True,
                  "ops": ["decompose"], "incomplete": bool(missing)},
    )


def _contradiction(index: int) -> ReasoningTask:
    """Two statements that cannot both hold. Half the cases are consistent."""
    name = _pick(_NAMES, "reason_person", index)
    contradictory = hash_index(2, "reason_contra", index) == 1
    first = _number(index, 10, 60, "first")
    second = first + _number(index, 1, 20, "delta") if contradictory else first
    prompt = (f"{name} is exactly {first} years old. "
              f"{name} is exactly {second} years old. "
              "Can both statements be true?")
    return ReasoningTask(
        id=f"reason_contra_{index}",
        family="contradiction",
        prompt=prompt,
        expected=not contradictory,
        features={"steps": 2, "numeric": True, "logic": True,
                  "ops": ["verify"]},
    )


def _magnitude(index: int) -> ReasoningTask:
    """Order of magnitude — right ballpark, not right to the digit."""
    exponent = _number(index, 1, 9, "exp")
    value = 10 ** exponent
    return ReasoningTask(
        id=f"reason_magnitude_{index}",
        family="magnitude",
        prompt=(f"Roughly how many digits does the number {value} have when "
                "written out in full?"),
        expected=exponent + 1,
        features={"steps": 2, "numeric": True, "estimation": True,
                  "ops": ["compute"]},
    )


def _missing_data(index: int) -> ReasoningTask:
    """A question the text does not contain the answer to.

    The correct response is to abstain. A system that always answers scores
    zero here however clever it is — which is the whole point: a confident
    wrong answer is worse than an admission, and nothing else measures that.
    """
    name = _pick(_NAMES, "reason_missing_person", index)
    item = _pick(_ITEMS, "reason_missing_item", index)
    count = _number(index, 3, 40, "count")
    return ReasoningTask(
        id=f"reason_missing_{index}",
        family="missing_data",
        prompt=(f"{name} loaded {count} {item}s onto a cart and drove away. "
                f"How many {item}s were left behind?"),
        expected=ABSTAIN,
        features={"steps": 1, "numeric": True, "incomplete": True,
                  "ops": ["abstain"]},
    )


BUILDERS = {
    "arithmetic_chain": _arithmetic_chain,
    "unit_conversion": _unit_conversion,
    "constraint_puzzle": _constraint_puzzle,
    "grid_planning": _grid_planning,
    "rule_chain": _rule_chain,
    "contradiction": _contradiction,
    "magnitude": _magnitude,
    "missing_data": _missing_data,
}


# ── the public surface ───────────────────────────────────────────────

def build(index: int) -> ReasoningTask:
    """The ``index``-th task. Pure, deterministic, unbounded.

    The family is chosen by position rather than by hash, so a slice of
    consecutive indices covers every family evenly — a held-out set that
    happened to be nine-tenths arithmetic would measure arithmetic.
    """
    index = int(index)
    family = FAMILIES[index % len(FAMILIES)]
    return BUILDERS[family](index // len(FAMILIES))


def build_family(family: str, count: int, start: int = 0) -> list[ReasoningTask]:
    """``count`` tasks of one family — what a weakness is probed with."""
    builder = BUILDERS.get(str(family))
    if builder is None:
        raise KeyError(f"no generator for family {family!r}")
    return [builder(start + offset) for offset in range(max(0, int(count)))]


def benchmark(count: int = 64, start: int = 0) -> list[ReasoningTask]:
    """A run of consecutive tasks, covering every family."""
    return [build(start + offset) for offset in range(max(0, int(count)))]


def split(count: int = 64, holdout: float = 0.5) -> tuple[list, list]:
    """``(train, held_out)`` — disjoint index ranges, not a shuffle.

    Disjoint *ranges* rather than a partition of one set: the held-out half is
    then made of indices the trainer has never been given, and can be extended
    without touching the training half.
    """
    count = max(0, int(count))
    train_size = max(0, int(count * (1.0 - max(0.0, min(1.0, holdout)))))
    return benchmark(train_size, start=0), benchmark(count - train_size,
                                                     start=100_000)


def features_of(tasks) -> dict[str, int]:
    """How often each feature appears — the axes available to group failures."""
    counts: dict[str, int] = {}
    for task in tasks or []:
        for name, value in task.features.items():
            if name == "ops":
                for operation in value:
                    counts[f"op:{operation}"] = counts.get(f"op:{operation}", 0) + 1
            elif isinstance(value, bool):
                if value:
                    counts[name] = counts.get(name, 0) + 1
            else:
                counts[f"{name}={value}"] = counts.get(f"{name}={value}", 0) + 1
        counts[f"family={task.family}"] = counts.get(f"family={task.family}", 0) + 1
    return dict(sorted(counts.items()))


def reference_answer(task: ReasoningTask) -> object:
    """What a perfect reasoner would say. Used to sanity-check a strategy.

    Deliberately *not* a solver: it is the generator's own answer, so a test
    can prove that a strategy which follows the DSL faithfully can reach full
    marks, and that a benchmark nobody can pass is a broken benchmark rather
    than a hard one.
    """
    return task.expected
