"""Step 1 — a real CODING benchmark: write a function, verified by HIDDEN tests.

Unlike the payload->answer tasks, a coding task asks for a whole function. The
agent (LLM) is shown a spec plus a few *visible* tests and must produce code;
the verifier then runs it against *hidden* tests in the sandbox. A solution that
hardcodes the visible cases fails the hidden ones — the closest thing here to a
genuine "can it actually program this" signal.
"""
from dataclasses import dataclass

from aegis.eval.benchmark import values_match
from aegis.eval.sandbox import run_tests


@dataclass(frozen=True)
class CodingTask:
    id: str
    func_name: str
    spec: str
    visible_tests: tuple   # ((args, expected), ...) shown to the solver
    hidden_tests: tuple    # ((args, expected), ...) used to grade

    def kind_key(self) -> str:
        return f"code:{self.id}"


# Hidden tests deliberately include cases the visible ones don't cover
# (boundaries, wrap-around), so memorizing the visible set is not enough.
CODING_BENCHMARK: list[CodingTask] = [
    CodingTask(
        "is_even", "is_even",
        "Return True if integer n is even, else False.",
        visible_tests=(([2], True), ([3], False)),
        hidden_tests=(([0], True), ([7], False), ([-4], True), ([101], False)),
    ),
    CodingTask(
        "clamp", "clamp",
        "clamp(x, lo, hi): return x bounded to the inclusive range [lo, hi].",
        visible_tests=(([5, 0, 10], 5), ([-3, 0, 10], 0)),
        hidden_tests=(([15, 0, 10], 10), ([0, 0, 10], 0), ([10, 0, 10], 10), ([7, 1, 5], 5)),
    ),
    CodingTask(
        "fizzbuzz", "fizzbuzz",
        "fizzbuzz(n): 'FizzBuzz' if divisible by 15, 'Fizz' if by 3, 'Buzz' if by 5, else str(n).",
        visible_tests=(([3], "Fizz"), ([5], "Buzz")),
        hidden_tests=(([15], "FizzBuzz"), ([7], "7"), ([9], "Fizz"), ([20], "Buzz")),
    ),
    CodingTask(
        "reverse_words", "reverse_words",
        "reverse_words(s): reverse the ORDER of space-separated words.",
        visible_tests=((["a b c"], "c b a"), (["hello world"], "world hello")),
        hidden_tests=((["one"], "one"), (["x y z w"], "w z y x"), (["the quick fox"], "fox quick the")),
    ),
    CodingTask(
        "sum_list", "sum_list",
        "sum_list(nums): sum of a list of integers (0 for empty).",
        visible_tests=(([[1, 2, 3]], 6), ([[]], 0)),
        hidden_tests=(([[5]], 5), ([[-1, 1]], 0), ([[10, 20, 30]], 60)),
    ),
    CodingTask(
        "is_anagram", "is_anagram",
        "is_anagram(a, b): True if a and b are anagrams of each other.",
        visible_tests=((["listen", "silent"], True), (["a", "b"], False)),
        hidden_tests=((["", ""], True), (["aab", "aba"], True), (["abc", "abd"], False)),
    ),
    CodingTask(
        "digit_count", "digit_count",
        "digit_count(n): number of decimal digits of non-negative integer n (0 has 1).",
        visible_tests=(([100], 3), ([7], 1)),
        hidden_tests=(([0], 1), ([9999], 4), ([12345], 5)),
    ),
    CodingTask(
        "max_of", "max_of",
        "max_of(nums): the maximum value in a non-empty list of integers.",
        visible_tests=(([[3, 1, 2]], 3), ([[5]], 5)),
        hidden_tests=(([[-5, -1, -9]], -1), ([[0, 0, 0]], 0), ([[8, 3, 8, 1]], 8)),
    ),
]


def verify_solution(code: str, task: CodingTask, timeout: float = 3.0) -> dict:
    """Run candidate ``code`` against the task's HIDDEN tests.

    Returns {"solved": bool, "passed": int, "total": int, "error": str|None}.
    All hidden tests must pass for ``solved`` to be True.
    """
    arg_lists = [list(args) for args, _ in task.hidden_tests]
    out = run_tests(code, task.func_name, arg_lists, timeout=timeout)
    if not out.get("ok"):
        return {"solved": False, "passed": 0, "total": len(task.hidden_tests),
                "error": out.get("error", "sandbox failure")}
    passed = 0
    for res, (_, expected) in zip(out["results"], task.hidden_tests):
        if res.get("ok") and values_match(res.get("result"), expected):
            passed += 1
    total = len(task.hidden_tests)
    # total > 0 guards against a vacuous solved=True when hidden_tests is empty.
    return {"solved": total > 0 and passed == total,
            "passed": passed, "total": total, "error": None}
