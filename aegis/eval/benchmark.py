"""Point 1 — a benchmark domain with VERIFIABLE feedback.

Each task carries a JSON-serializable ``payload`` and an ``expected`` answer.
``verify`` checks a candidate answer against ground truth with type-aware
normalization, so success is decided externally (not by self-report). The set
spans several "kinds"; a skill claims one or more kinds and is judged purely on
whether its output passes ``verify``.
"""
from dataclasses import dataclass
from typing import Any


def _norm(v: Any) -> Any:
    """Normalize for comparison: ints/floats numerically, strings trimmed.

    Booleans are tagged so they only ever compare equal to other booleans —
    without this, Python's ``1.0 == True`` would let an integer/float ``1``
    satisfy a task whose expected answer is boolean ``True``.
    """
    if isinstance(v, bool):
        return ("__bool__", v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        # Allow a numeric string answer to match a numeric expected value.
        try:
            return float(s)
        except ValueError:
            return s
    return v


def _as_number(v: Any) -> float | None:
    """Return v as a float if it *is* a number or a numeric string, else None.

    Booleans are explicitly rejected — ``True`` is not the number 1 here."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def values_match(answer: Any, expected: Any) -> bool:
    """Type-aware equality for verifiable feedback.

    Unlike a symmetric ``_norm(a) == _norm(b)``, this is asymmetric on the
    *expected* type so a candidate cannot cheat by returning the wrong Python
    type:

    * ``bool`` compares only to ``bool`` (avoids Python's ``1 == True`` trap).
    * a ``str`` expected requires a ``str`` answer — a number must NOT satisfy a
      string spec (e.g. fizzbuzz ``7`` must be ``"7"``, to_binary ``"1101"``).
      Previously ``_norm("7") == _norm(7)`` let a numeric answer pass a string
      task.
    * a numeric expected accepts a number or a numeric string, with float/int
      tolerance preserved (so ``"42"`` still verifies against ``42``).
    """
    if isinstance(expected, bool) or isinstance(answer, bool):
        return (isinstance(expected, bool) and isinstance(answer, bool)
                and expected == answer)
    if isinstance(expected, str):
        return isinstance(answer, str) and answer.strip() == expected.strip()
    if isinstance(expected, (int, float)):
        num = _as_number(answer)
        return num is not None and num == float(expected)
    return answer == expected


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    prompt: str
    payload: dict
    expected: Any

    def verify(self, answer: Any) -> bool:
        try:
            return values_match(answer, self.expected)
        except Exception:
            return False


# The default benchmark. Kept small so a full run is a few seconds of
# sandboxed subprocess work. Some kinds (is_prime, sort_csv) are intentionally
# left without a seeded skill so the synthesis loop has something to improve.
DEFAULT_BENCHMARK: list[Task] = [
    # calc
    Task("calc_add", "calc", "12 + 30", {"a": 12, "b": 30, "op": "add"}, 42),
    Task("calc_mul", "calc", "6 * 9", {"a": 6, "b": 9, "op": "mul"}, 54),
    Task("calc_sub", "calc", "100 - 37", {"a": 100, "b": 37, "op": "sub"}, 63),
    # reverse
    Task("rev_hello", "reverse", "reverse 'hello'", {"s": "hello"}, "olleh"),
    Task("rev_aegis", "reverse", "reverse 'AEGIS'", {"s": "AEGIS"}, "SIGEA"),
    # count_vowels
    Task("vow_consc", "count_vowels", "vowels in 'consciousness'", {"s": "consciousness"}, 5),
    Task("vow_rhythm", "count_vowels", "vowels in 'rhythm'", {"s": "rhythm"}, 0),
    # fib (0-indexed: fib(0)=0, fib(1)=1)
    Task("fib_10", "fib", "fib(10)", {"n": 10}, 55),
    Task("fib_15", "fib", "fib(15)", {"n": 15}, 610),
    # palindrome
    Task("pal_yes", "palindrome", "is 'racecar' a palindrome?", {"s": "racecar"}, True),
    Task("pal_no", "palindrome", "is 'hello' a palindrome?", {"s": "hello"}, False),
    # gcd
    Task("gcd_a", "gcd", "gcd(48, 36)", {"a": 48, "b": 36}, 12),
    Task("gcd_b", "gcd", "gcd(17, 5)", {"a": 17, "b": 5}, 1),
    # factorial
    Task("fact_6", "factorial", "6!", {"n": 6}, 720),
    Task("fact_0", "factorial", "0!", {"n": 0}, 1),
    # word_count
    Task("wc_a", "word_count", "words in 'the quick brown fox'", {"s": "the quick brown fox"}, 4),
    Task("wc_b", "word_count", "words in '  spaced   out  words '", {"s": "  spaced   out  words "}, 3),
    # sum_digits
    Task("sd_a", "sum_digits", "digit sum of 9875", {"n": 9875}, 29),
    Task("sd_b", "sum_digits", "digit sum of 100", {"n": 100}, 1),
    # upper (a string->string primitive, also used by auto-composition)
    Task("up_a", "upper", "uppercase 'abc'", {"s": "abc"}, "ABC"),
    Task("up_b", "upper", "uppercase 'AeGiS'", {"s": "AeGiS"}, "AEGIS"),
    # is_prime  (initially unsolved — synthesis target)
    Task("prime_97", "is_prime", "is 97 prime?", {"n": 97}, True),
    Task("prime_100", "is_prime", "is 100 prime?", {"n": 100}, False),
    # sort_csv  (initially unsolved — synthesis target)
    Task("sort_a", "sort_csv", "sort '3,1,2'", {"s": "3,1,2"}, "1,2,3"),
    Task("sort_b", "sort_csv", "sort '10,2,33,4'", {"s": "10,2,33,4"}, "2,4,10,33"),
    # roman  (initially unsolved — synthesis target)
    Task("roman_14", "roman", "14 as Roman numeral", {"n": 14}, "XIV"),
    Task("roman_2024", "roman", "2024 as Roman numeral", {"n": 2024}, "MMXXIV"),
    # to_binary  (initially unsolved — synthesis target)
    Task("bin_13", "to_binary", "13 in binary", {"n": 13}, "1101"),
    Task("bin_255", "to_binary", "255 in binary", {"n": 255}, "11111111"),
]


def all_kinds(tasks: list[Task] | None = None) -> list[str]:
    tasks = tasks or DEFAULT_BENCHMARK
    seen: list[str] = []
    for t in tasks:
        if t.kind not in seen:
            seen.append(t.kind)
    return seen


def tasks_for_kind(kind: str, tasks: list[Task] | None = None) -> list[Task]:
    tasks = tasks or DEFAULT_BENCHMARK
    return [t for t in tasks if t.kind == kind]


def split_tasks(kind: str, tasks: list[Task] | None = None) -> tuple[list[Task], list[Task]]:
    """Split a kind's tasks into (train, holdout).

    Skill synthesis only sees the *train* examples; acceptance is judged on the
    *holdout* tasks the proposed skill never saw — so a skill that merely
    memorizes the shown cases will not pass the gate. The last task of each kind
    is held out (with a single task, train == holdout as a degenerate fallback).

    Kept as it was. The three-way split below is what evolution and the arena
    select on; this two-way one is what skill synthesis has always used, and
    changing it would change the meaning of every stored acceptance decision.
    """
    ts = tasks_for_kind(kind, tasks)
    if len(ts) <= 1:
        return ts, ts
    return ts[:-1], ts[-1:]


# ── the three-way split (spec M9.3) ──────────────────────────────────

#: Which split each position in a kind's hash-ordering falls into.
#:
#: Dealing round-robin over a *hash ordering of the task ids* rather than over
#: list positions is what makes the split stable: adding a task changes where
#: that task lands and nothing else. Assigning by "position in the list" would
#: re-deal every existing task the moment one was inserted, and every recorded
#: valid/test score before that point would silently start describing a
#: different set.
#:
#: The cycle gives 50/25/25 over four tasks, and degrades sensibly below that:
#: two tasks give train+valid, three give train+valid+train. A kind cannot have
#: three splits out of two items, and pretending otherwise would produce an
#: empty split that silently scores 0.
SPLIT_CYCLE: tuple[str, ...] = ("train", "valid", "train", "test")
SPLITS: tuple[str, ...] = ("train", "valid", "test")


def _split_rank(task: Task) -> tuple[float, str]:
    from aegis.util.quasirandom import hash_unit

    return (hash_unit("eval_split", task.id), task.id)


def _dedup_key(task: Task) -> tuple[str, str]:
    """What makes two tasks *the same problem* for split purposes.

    Canonical JSON of the payload plus ``repr`` of the expected answer —
    ``repr`` rather than the value itself so ``True`` and ``1`` stay distinct,
    the same way ``values_match`` keeps them distinct when verifying.
    """
    import json

    return (json.dumps(task.payload, sort_keys=True, ensure_ascii=False),
            repr(task.expected))


def assign_splits(tasks: list[Task] | None = None) -> dict[str, str]:
    """``task id -> split``, computed from the ids and payloads.

    Per kind, so every kind is represented in every split it can support. A
    global deal would leave whole kinds out of `valid`, and a score on a valid
    set missing half the kinds is not a measurement of the same thing.

    Tasks with an IDENTICAL payload and answer are dealt to the SAME split.
    The generators draw from finite pools (24 words, factorial's 0..12), so
    two different indices regularly produce the very same problem — and the
    hash deal used to put ``gen_factorial_1 {'n': 0}`` in `test` while an
    identical case sat in `train`, which made the held-out score partly a
    memorisation score (audit: split leakage). Coercing duplicates to the
    split of their first occurrence in the hash order keeps the assignment a
    pure, stable function of the task set while guaranteeing a held-out
    problem was never seen during training.
    """
    tasks = list(tasks if tasks is not None else DEFAULT_BENCHMARK)
    assignment: dict[str, str] = {}
    for kind in all_kinds(tasks):
        ordered = sorted(tasks_for_kind(kind, tasks), key=_split_rank)
        first_split: dict[tuple[str, str], str] = {}
        for position, task in enumerate(ordered):
            dealt = SPLIT_CYCLE[position % len(SPLIT_CYCLE)]
            key = _dedup_key(task)
            assignment[task.id] = first_split.setdefault(key, dealt)
    return assignment


def split_of(task: Task, tasks: list[Task] | None = None) -> str:
    """Which split one task belongs to."""
    return assign_splits(tasks).get(task.id, "train")


def three_way_split(tasks: list[Task] | None = None) -> dict[str, list[Task]]:
    """``{"train": [...], "valid": [...], "test": [...]}``, in a fixed order."""
    tasks = list(tasks if tasks is not None else DEFAULT_BENCHMARK)
    assignment = assign_splits(tasks)
    out: dict[str, list[Task]] = {name: [] for name in SPLITS}
    for task in tasks:
        out[assignment.get(task.id, "train")].append(task)
    return out


def tasks_in_split(split: str, tasks: list[Task] | None = None) -> list[Task]:
    return three_way_split(tasks).get(str(split), [])
