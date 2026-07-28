"""Generated tasks and the three-way split (spec M9.3).

The default benchmark is two or three tasks per kind — enough to notice a skill
that does nothing, not enough to notice a skill that memorised two answers. And
memorising is what a synthesiser under pressure will do, because it is the
cheapest thing that passes.

So the generators have to satisfy something stronger than "they produce tasks":
their expected answers must be *right*. A generator that agrees with a wrong
reference implementation is a benchmark that measures agreement with a bug, so
every reference here is checked against brute force or against the stdlib, and
the seeded skills — written long before the generators — are required to solve
generated cases of the kinds they claim.
"""
import math

import pytest

from aegis.eval.benchmark import (
    DEFAULT_BENCHMARK, SPLITS, SPLIT_CYCLE, all_kinds, assign_splits,
    split_of, tasks_for_kind, tasks_in_split, three_way_split,
)
from aegis.eval.generators import (
    BUILDERS, KINDS, build, count_vowels, fib, generated_benchmark, is_prime,
    sort_csv, sum_digits, to_roman, variations,
)
from aegis.eval.skill_library import SkillLibrary
from aegis.eval.solver import MultiAgentSolver


@pytest.fixture(scope="module")
def solver():
    return MultiAgentSolver(SkillLibrary(store_path=None), timeout=10.0)


# ── the reference implementations ────────────────────────────────────

def test_primality_agrees_with_brute_force():
    for n in range(0, 400):
        assert is_prime(n) == (n > 1 and all(n % d for d in range(2, n)))


def test_fibonacci_uses_the_benchmark_convention():
    """fib(0) = 0 — the hand-written tasks say so, and a generator with the
    other convention would mark every correct answer wrong."""
    assert [fib(i) for i in range(11)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert fib(15) == 610


def test_roman_numerals_use_subtractive_notation():
    assert to_roman(14) == "XIV"
    assert to_roman(2024) == "MMXXIV"
    assert to_roman(3999) == "MMMCMXCIX"
    assert to_roman(4) == "IV" and to_roman(9) == "IX" and to_roman(40) == "XL"


def test_roman_round_trips_for_every_value_it_can_generate():
    values = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}

    def from_roman(text):
        total, previous = 0, 0
        for symbol in reversed(text):
            value = values[symbol]
            total += value if value >= previous else -value
            previous = max(previous, value)
        return total

    for number in range(1, 4000):
        assert from_roman(to_roman(number)) == number


def test_csv_sorting_is_numeric_not_lexicographic():
    assert sort_csv("10,2,33,4") == "2,4,10,33"
    assert sort_csv("3,1,2") == "1,2,3"


def test_the_small_helpers_agree_with_the_obvious_definition():
    assert sum_digits(9875) == 29 and sum_digits(100) == 1
    assert count_vowels("consciousness") == 5 and count_vowels("rhythm") == 0


# ── the generators themselves ────────────────────────────────────────

def test_every_declared_kind_has_a_generator():
    assert set(KINDS) == set(BUILDERS)


def test_every_generated_task_is_self_consistent():
    """The expected answer is recomputed from the payload by a second route."""
    checks = {
        "calc": lambda p: {"add": p["a"] + p["b"], "mul": p["a"] * p["b"],
                           "sub": p["a"] - p["b"]}[p["op"]],
        "reverse": lambda p: p["s"][::-1],
        "count_vowels": lambda p: count_vowels(p["s"]),
        "fib": lambda p: fib(p["n"]),
        "palindrome": lambda p: p["s"] == p["s"][::-1],
        "gcd": lambda p: math.gcd(p["a"], p["b"]),
        "factorial": lambda p: math.factorial(p["n"]),
        "word_count": lambda p: len(p["s"].split()),
        "sum_digits": lambda p: sum_digits(p["n"]),
        "upper": lambda p: p["s"].upper(),
        "is_prime": lambda p: is_prime(p["n"]),
        "sort_csv": lambda p: sort_csv(p["s"]),
        "roman": lambda p: to_roman(p["n"]),
        "to_binary": lambda p: format(p["n"], "b"),
    }
    for kind in KINDS:
        for index in range(25):
            task = build(kind, index)
            assert task.expected == checks[kind](task.payload), (kind, index)


def test_generation_is_deterministic():
    first = [(t.id, t.payload, t.expected) for t in generated_benchmark(5)]
    second = [(t.id, t.payload, t.expected) for t in generated_benchmark(5)]
    assert first == second


def test_ids_do_not_collide():
    ids = [task.id for task in generated_benchmark(per_kind=40)]
    assert len(ids) == len(set(ids))


def test_consecutive_indices_give_unrelated_cases():
    """A generator whose index walked a range would produce a held-out split
    that was simply "the larger numbers", and a skill could pass it by
    extrapolating rather than by being right."""
    numbers = [build("roman", index).payload["n"] for index in range(30)]
    assert numbers != sorted(numbers)
    assert len(set(numbers)) > 20


def test_both_answers_occur_for_the_boolean_kinds():
    """A kind whose generated answer is always True is a kind where `return
    True` scores 100%."""
    for kind in ("palindrome", "is_prime"):
        answers = {build(kind, index).expected for index in range(60)}
        assert answers == {True, False}, kind


def test_every_arithmetic_operation_is_generated():
    ops = {build("calc", index).payload["op"] for index in range(60)}
    assert ops == {"add", "mul", "sub"}


def test_a_generated_task_verifies_its_own_answer():
    for kind in KINDS:
        task = build(kind, 3)
        assert task.verify(task.expected)


def test_an_unknown_kind_is_refused():
    with pytest.raises(KeyError):
        build("no_such_kind", 0)


def test_asking_for_nothing_gives_nothing():
    assert variations("calc", 0) == []
    assert variations("calc", -3) == []


def test_a_start_offset_shifts_the_window():
    assert [t.id for t in variations("calc", 2, start=10)] == \
        ["gen_calc_10", "gen_calc_11"]


@pytest.mark.parametrize("kind", ["calc", "reverse", "count_vowels", "fib",
                                  "palindrome", "gcd", "factorial",
                                  "word_count", "sum_digits", "upper"])
def test_the_seeded_skills_solve_generated_cases(solver, kind):
    """The generators produce *the same kind of problem*, not new ones.

    The seed skills were written years before these generators and know nothing
    about them; if they solve generated cases, the generator is producing that
    kind. If they did not, the generator would be quietly inventing a harder
    task and the pass-rate would fall for a reason nobody chose.
    """
    solved = sum(1 for index in range(6)
                 if solver.solve(build(kind, index)).solved)
    assert solved == 6, f"{kind}: only {solved}/6 generated cases solved"


# ── the split ────────────────────────────────────────────────────────

def test_the_split_is_a_partition():
    tasks = list(DEFAULT_BENCHMARK) + generated_benchmark(per_kind=6)
    split = three_way_split(tasks)
    assert set(split) == set(SPLITS)
    assert sorted(t.id for group in split.values() for t in group) == \
        sorted(task.id for task in tasks)


def test_the_split_follows_the_ids_not_the_order():
    """Stability is the point: adding a task must not re-deal the others.

    Assigning by position would re-deal everything the moment a task was
    inserted, and every recorded valid/test score before that point would
    silently start describing a different set.
    """
    tasks = list(DEFAULT_BENCHMARK) + generated_benchmark(per_kind=6)
    before = assign_splits(tasks)
    after = assign_splits(tasks + [build("calc", 999)])
    assert all(after[task_id] == split for task_id, split in before.items())


def test_shuffling_the_input_does_not_change_the_split():
    tasks = list(DEFAULT_BENCHMARK)
    assert assign_splits(tasks) == assign_splits(list(reversed(tasks)))


def test_every_kind_reaches_every_split_it_can_support():
    tasks = list(DEFAULT_BENCHMARK) + generated_benchmark(per_kind=8)
    split = three_way_split(tasks)
    kinds = set(all_kinds(tasks))
    for name in SPLITS:
        assert {task.kind for task in split[name]} == kinds, name


def test_a_kind_with_two_tasks_has_no_test_split():
    """Three splits cannot be made out of two items, and pretending otherwise
    would produce an empty split that silently scores zero."""
    small = tasks_for_kind("is_prime")
    assert len(small) == 2
    split = three_way_split(small)
    assert len(split["train"]) == 1 and len(split["valid"]) == 1
    assert split["test"] == []


def test_the_cycle_gives_the_intended_proportions():
    assert SPLIT_CYCLE.count("train") == 2
    assert SPLIT_CYCLE.count("valid") == 1
    assert SPLIT_CYCLE.count("test") == 1


def test_a_large_kind_lands_near_fifty_twentyfive_twentyfive():
    tasks = variations("calc", 400)
    split = three_way_split(tasks)
    assert len(split["train"]) == 200
    assert len(split["valid"]) == 100
    assert len(split["test"]) == 100


def test_split_of_names_the_group_a_task_is_in():
    tasks = list(DEFAULT_BENCHMARK)
    for task in tasks:
        assert task in tasks_in_split(split_of(task, tasks), tasks)


def test_an_unknown_task_defaults_to_train():
    assert split_of(build("calc", 5), list(DEFAULT_BENCHMARK)) == "train"


def test_an_unknown_split_name_is_empty():
    assert tasks_in_split("holdout", list(DEFAULT_BENCHMARK)) == []


# ── the benchmark is defined by these values ─────────────────────────

#: What ``build(kind, 0)`` and ``build(kind, 1)`` produce today.
#:
#: An approval test, and deliberately so. The generators *are* the benchmark:
#: change how an index picks its material and every held-out score in the
#: repository starts describing a different set of tasks, with nothing failing
#: to say so. Mutants proved the point — shifting the word-selection arithmetic
#: produced perfectly self-consistent tasks that were simply different ones, and
#: every other test still passed.
#:
#: Updating these values is allowed. Updating them silently is not: a diff here
#: means the benchmark moved, and the commit has to say why.
GOLDEN = {
    "calc": [
        ({'a': 352, 'b': 269, 'op': 'sub'}, 83),
        ({'a': 148, 'b': 167, 'op': 'sub'}, -19),
    ],
    "reverse": [
        ({'s': 'syzygy'}, 'ygyzys'),
        ({'s': 'consciousness'}, 'ssensuoicsnoc'),
    ],
    "count_vowels": [
        ({'s': 'syzygy'}, 0),
        ({'s': 'consciousness'}, 5),
    ],
    "fib": [
        ({'n': 25}, 75025),
        ({'n': 3}, 2),
    ],
    "palindrome": [
        ({'s': 'syzygyygyzys'}, True),
        ({'s': 'consciousness'}, False),
    ],
    "gcd": [
        ({'a': 36, 'b': 360}, 36),
        ({'a': 239, 'b': 433}, 1),
    ],
    "factorial": [
        ({'n': 2}, 2),
        ({'n': 0}, 1),
    ],
    "word_count": [
        ({'s': 'syzygy'}, 1),
        ({'s': 'myth  banana'}, 2),
    ],
    "sum_digits": [
        ({'n': 386626}, 31),
        ({'n': 715665}, 30),
    ],
    "upper": [
        ({'s': 'syzygy'}, 'SYZYGY'),
        ({'s': 'consciousness'}, 'CONSCIOUSNESS'),
    ],
    "is_prime": [
        ({'n': 396}, False),
        ({'n': 55}, False),
    ],
    "sort_csv": [
        ({'s': '73,38,7'}, '7,38,73'),
        ({'s': '34,87'}, '34,87'),
    ],
    "roman": [
        ({'n': 3436}, 'MMMCDXXXVI'),
        ({'n': 1802}, 'MDCCCII'),
    ],
    "to_binary": [
        ({'n': 2178}, '100010000010'),
        ({'n': 2577}, '101000010001'),
    ],
}


def test_the_generators_still_produce_the_benchmark_they_defined():
    for kind, expectations in GOLDEN.items():
        for index, (payload, expected) in enumerate(expectations):
            task = build(kind, index)
            assert task.payload == payload, f"{kind}[{index}] payload moved"
            assert task.expected == expected, f"{kind}[{index}] answer moved"


def test_every_kind_is_pinned():
    assert set(GOLDEN) == set(KINDS)


def test_a_generated_number_stays_inside_its_range():
    """`_number(index, low, high)` is inclusive on both ends, and nothing may
    fall outside: a factorial index above 12 or a roman numeral above 3999 is a
    task the reference implementation was never meant to answer."""
    from aegis.eval.generators import _number

    assert {_number(index, 5, 9) for index in range(400)} == {5, 6, 7, 8, 9}
    assert {_number(index, 3, 3) for index in range(50)} == {3}


def test_the_bounded_kinds_respect_their_own_limits():
    for index in range(300):
        assert 0 <= build("factorial", index).payload["n"] <= 12
        assert 1 <= build("roman", index).payload["n"] <= 3999
        assert 0 <= build("fib", index).payload["n"] <= 30


# ── the Task contract ────────────────────────────────────────────────

def test_a_task_is_immutable():
    """Frozen on purpose. A task is the ground truth an answer is judged
    against; anything that could edit `expected` in flight could make a wrong
    answer right, and the whole benchmark rests on that not being possible."""
    task = build("calc", 0)
    with pytest.raises(Exception):
        task.expected = 999


def test_a_task_can_be_put_in_a_set():
    """Frozen also makes it hashable, which is what lets splits and reports
    key on tasks rather than on copies of their ids."""
    tasks = [build("calc", 0), build("calc", 0), build("calc", 1)]
    assert len({(t.id, t.kind) for t in tasks}) == 2


def test_a_non_scalar_answer_compares_by_equality():
    """The fallback arm of `values_match`: neither bool, nor str, nor number."""
    from aegis.eval.benchmark import values_match

    assert values_match([1, 2, 3], [1, 2, 3])
    assert not values_match([1, 2, 3], [3, 2, 1])
    assert values_match({"a": 1}, {"a": 1})
    assert not values_match({"a": 1}, {"a": 2})
    assert values_match(None, None)


def test_a_comparison_that_raises_is_a_failure_not_a_crash():
    """A verifier is fed whatever a skill returned, including objects that
    misbehave on comparison. That has to score zero, not take the run down."""
    from aegis.eval.benchmark import Task

    class _Hostile:
        def __eq__(self, other):
            raise RuntimeError("I refuse to be compared")

        __hash__ = None

    task = Task("hostile", "odd", "?", {}, _Hostile())
    assert task.verify("anything") is False
