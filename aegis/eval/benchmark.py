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
    """Normalize for comparison: ints/floats numerically, strings trimmed."""
    if isinstance(v, bool):
        return v
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


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    prompt: str
    payload: dict
    expected: Any

    def verify(self, answer: Any) -> bool:
        try:
            return _norm(answer) == _norm(self.expected)
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
    # is_prime  (initially unsolved)
    Task("prime_97", "is_prime", "is 97 prime?", {"n": 97}, True),
    Task("prime_100", "is_prime", "is 100 prime?", {"n": 100}, False),
    # sort_csv  (initially unsolved)
    Task("sort_a", "sort_csv", "sort '3,1,2'", {"s": "3,1,2"}, "1,2,3"),
    Task("sort_b", "sort_csv", "sort '10,2,33,4'", {"s": "10,2,33,4"}, "2,4,10,33"),
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
