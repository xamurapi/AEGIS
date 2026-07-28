"""Task generators: an unbounded benchmark from a fixed set of kinds (spec M9.3).

The default benchmark is twenty-eight hand-written tasks, two or three per kind.
That is enough to notice a skill that does nothing and not enough to notice a
skill that memorised two answers — and memorising is exactly what a synthesiser
under pressure will do, because it is the cheapest thing that passes.

So each kind gets a generator: ``build(kind, index) -> Task``, a pure function of
the index. Three properties make it usable as a benchmark rather than as a toy:

* **Deterministic.** No RNG anywhere (§3.1). The index picks the case through a
  hash, so consecutive indices give unrelated cases and the same index always
  gives the same one.
* **Unbounded.** A held-out split can be as large as it needs to be, and it can
  be *different* held-out tasks next time without anybody curating them.
* **Self-verifying.** Every generated task carries its expected answer computed
  by a reference implementation right here — so the benchmark cannot drift away
  from what it claims to measure.

The reference implementations are deliberately the obvious ones. They are the
ground truth, not a solution to be admired: a clever one would be a second thing
that can be wrong.
"""
from __future__ import annotations

import math

from aegis.eval.benchmark import Task
from aegis.util.quasirandom import hash_index

#: Words drawn on for the string kinds. Chosen to vary in length, vowel count
#: and case so the generated cases are not all the same shape.
WORDS: tuple[str, ...] = (
    "consciousness", "rhythm", "aegis", "substrate", "entropy", "queue",
    "syzygy", "banana", "level", "deified", "abstraction", "myth",
    "oxygen", "quixotic", "rotator", "planner", "policy", "gradient",
    "noon", "kayak", "strength", "aurora", "eleven", "crypt",
)

#: Every kind the generators can produce. Matches the hand-written benchmark, so
#: a generated task is a drop-in extra case for an existing skill.
KINDS: tuple[str, ...] = (
    "calc", "reverse", "count_vowels", "fib", "palindrome", "gcd",
    "factorial", "word_count", "sum_digits", "upper", "is_prime",
    "sort_csv", "roman", "to_binary",
)

_OPS = ("add", "mul", "sub")

_ROMAN = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
          (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
          (5, "V"), (4, "IV"), (1, "I"))


# ── reference implementations ────────────────────────────────────────

def to_roman(number: int) -> str:
    """Classic subtractive notation, for 1..3999."""
    number = int(number)
    out = []
    for value, symbol in _ROMAN:
        count, number = divmod(number, value)
        out.append(symbol * count)
    return "".join(out)


def is_prime(number: int) -> bool:
    number = int(number)
    if number < 2:
        return False
    if number < 4:
        return True
    if number % 2 == 0:
        return False
    limit = int(math.isqrt(number))
    return all(number % factor for factor in range(3, limit + 1, 2))


def fib(n: int) -> int:
    """0-indexed: fib(0) = 0, fib(1) = 1 — the benchmark's convention."""
    a, b = 0, 1
    for _ in range(int(n)):
        a, b = b, a + b
    return a


def count_vowels(text: str) -> int:
    return sum(1 for character in str(text).lower() if character in "aeiou")


def sum_digits(number: int) -> int:
    return sum(int(digit) for digit in str(abs(int(number))))


def sort_csv(text: str) -> str:
    """Numeric sort, not lexicographic — 10 comes after 2."""
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    return ",".join(str(value) for value in sorted(int(part) for part in parts))


# ── the generators ───────────────────────────────────────────────────

def _word(index: int, salt: str = "") -> str:
    return WORDS[hash_index(len(WORDS), "gen_word", salt, index)]


def _number(index: int, low: int, high: int, salt: str = "") -> int:
    """A number in [low, high], picked deterministically from the index."""
    span = max(1, high - low + 1)
    return low + hash_index(span, "gen_num", salt, index)


def _calc(index: int) -> tuple[dict, object, str]:
    op = _OPS[hash_index(len(_OPS), "gen_op", index)]
    a = _number(index, 2, 400, "a")
    b = _number(index, 2, 400, "b")
    expected = {"add": a + b, "mul": a * b, "sub": a - b}[op]
    symbol = {"add": "+", "mul": "*", "sub": "-"}[op]
    return {"a": a, "b": b, "op": op}, expected, f"{a} {symbol} {b}"


def _reverse(index: int) -> tuple[dict, object, str]:
    word = _word(index)
    return {"s": word}, word[::-1], f"reverse {word!r}"


def _count_vowels(index: int) -> tuple[dict, object, str]:
    word = _word(index)
    return {"s": word}, count_vowels(word), f"vowels in {word!r}"


def _fib(index: int) -> tuple[dict, object, str]:
    n = _number(index, 0, 30)
    return {"n": n}, fib(n), f"fib({n})"


def _palindrome(index: int) -> tuple[dict, object, str]:
    word = _word(index)
    # Half the cases are made palindromic, so the answer is not always the same.
    if hash_index(2, "gen_pal", index):
        word = word + word[::-1]
    return {"s": word}, word == word[::-1], f"is {word!r} a palindrome?"


def _gcd(index: int) -> tuple[dict, object, str]:
    a = _number(index, 1, 500, "a")
    b = _number(index, 1, 500, "b")
    return {"a": a, "b": b}, math.gcd(a, b), f"gcd({a}, {b})"


def _factorial(index: int) -> tuple[dict, object, str]:
    n = _number(index, 0, 12)
    return {"n": n}, math.factorial(n), f"{n}!"


def _word_count(index: int) -> tuple[dict, object, str]:
    count = _number(index, 1, 6)
    words = [_word(index * 7 + offset) for offset in range(count)]
    # Irregular spacing, because `len(s.split())` and `s.count(" ") + 1` differ
    # exactly there and only one of them is right.
    text = "  ".join(words) if hash_index(2, "gen_space", index) else " ".join(words)
    return {"s": text}, len(text.split()), f"words in {text!r}"


def _sum_digits(index: int) -> tuple[dict, object, str]:
    n = _number(index, 0, 999_999)
    return {"n": n}, sum_digits(n), f"digit sum of {n}"


def _upper(index: int) -> tuple[dict, object, str]:
    word = _word(index)
    return {"s": word}, word.upper(), f"uppercase {word!r}"


def _is_prime(index: int) -> tuple[dict, object, str]:
    n = _number(index, 0, 500)
    return {"n": n}, is_prime(n), f"is {n} prime?"


def _sort_csv(index: int) -> tuple[dict, object, str]:
    count = _number(index, 2, 6)
    numbers = [_number(index * 11 + offset, 1, 99) for offset in range(count)]
    text = ",".join(str(value) for value in numbers)
    return {"s": text}, sort_csv(text), f"sort {text!r}"


def _roman(index: int) -> tuple[dict, object, str]:
    n = _number(index, 1, 3999)
    return {"n": n}, to_roman(n), f"{n} as Roman numeral"


def _to_binary(index: int) -> tuple[dict, object, str]:
    n = _number(index, 0, 4095)
    return {"n": n}, format(n, "b"), f"{n} in binary"


BUILDERS = {
    "calc": _calc,
    "reverse": _reverse,
    "count_vowels": _count_vowels,
    "fib": _fib,
    "palindrome": _palindrome,
    "gcd": _gcd,
    "factorial": _factorial,
    "word_count": _word_count,
    "sum_digits": _sum_digits,
    "upper": _upper,
    "is_prime": _is_prime,
    "sort_csv": _sort_csv,
    "roman": _roman,
    "to_binary": _to_binary,
}


# ── the public surface ───────────────────────────────────────────────

def build(kind: str, index: int) -> Task:
    """One generated task. Pure in ``(kind, index)``."""
    builder = BUILDERS.get(str(kind))
    if builder is None:
        raise KeyError(f"no generator for kind {kind!r}")
    payload, expected, prompt = builder(int(index))
    return Task(id=f"gen_{kind}_{int(index)}", kind=str(kind), prompt=prompt,
                payload=payload, expected=expected)


def variations(kind: str, count: int, start: int = 0) -> list[Task]:
    """``count`` generated tasks of one kind, from ``start``."""
    return [build(kind, start + offset) for offset in range(max(0, int(count)))]


def generated_benchmark(per_kind: int = 8, start: int = 0,
                        kinds: tuple[str, ...] = KINDS) -> list[Task]:
    """A whole benchmark built from generators.

    Used for the held-out splits and for the evolution harness, where the point
    is that the tasks were never seen before — including by whoever wrote the
    seed skills.
    """
    tasks: list[Task] = []
    for kind in kinds:
        tasks.extend(variations(kind, per_kind, start))
    return tasks
