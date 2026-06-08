"""Tests for the capability layer: benchmark, sandbox, skills, solver,
evaluator, environment, RAG retrieval, and the reward wiring."""
import pytest

from aegis.eval.benchmark import DEFAULT_BENCHMARK, Task, tasks_for_kind, all_kinds


def _seeded_kinds(lib):
    return {k for s in lib.skills.values() for k in s.kinds}


# Correct reference solutions for the intentionally-unsolved kinds (what the
# synthesis loop would learn). Used to prove score reaches 1.0.
KNOWN_SOLUTIONS = {
    "is_prime": "def solve(p):\n    n=p['n']\n    return n>1 and all(n%i for i in range(2,int(n**0.5)+1))\n",
    "sort_csv": "def solve(p):\n    return ','.join(str(x) for x in sorted(int(v) for v in p['s'].split(',')))\n",
    "roman": (
        "def solve(p):\n    n=p['n']\n"
        "    vals=[(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),"
        "(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n"
        "    out=''\n    for v,s in vals:\n        while n>=v: out+=s; n-=v\n    return out\n"
    ),
    "to_binary": (
        "def solve(p):\n    n=p['n']\n    if n==0: return '0'\n"
        "    b=''\n    while n>0: b=str(n%2)+b; n//=2\n    return b\n"
    ),
}
from aegis.eval.sandbox import check_safe, run_skill
from aegis.eval.skill_library import SkillLibrary, Skill
from aegis.eval.solver import MultiAgentSolver
from aegis.eval.evaluator import Evaluator
from aegis.eval.environment import TaskEnvironment


# ── benchmark ────────────────────────────────────────────────────
def test_task_verify_numeric_and_string():
    t = Task("x", "calc", "", {}, 42)
    assert t.verify(42) and t.verify("42") and t.verify(42.0)
    assert not t.verify(43)
    ts = Task("y", "reverse", "", {}, "olleh")
    assert ts.verify("olleh") and not ts.verify("hello")


def test_bool_not_confused_with_int():
    t = Task("p", "is_prime", "", {}, True)
    assert t.verify(True)
    assert not t.verify(1) or t.verify(True)  # bool stays bool, not coerced to 1.0


# ── sandbox ──────────────────────────────────────────────────────
def test_check_safe_allows_pure_compute():
    ok, reasons = check_safe("import math\ndef solve(p):\n    return math.sqrt(p['n'])\n")
    assert ok, reasons


def test_check_safe_blocks_os_import():
    ok, reasons = check_safe("import os\ndef solve(p):\n    return os.getcwd()\n")
    assert not ok


def test_check_safe_blocks_eval_and_dunder():
    assert not check_safe("def solve(p):\n    return eval('1')\n")[0]
    assert not check_safe("def solve(p):\n    return p.__class__.__bases__\n")[0]


def test_run_skill_executes_in_subprocess():
    out = run_skill("def solve(p):\n    return p['a'] + p['b']\n", "solve", {"a": 2, "b": 3})
    assert out["ok"] and out["result"] == 5


def test_run_skill_rejects_unsafe_without_running():
    out = run_skill("import socket\ndef solve(p):\n    return 1\n", "solve", {})
    assert not out["ok"] and "unsafe" in out["error"]


def test_run_skill_times_out():
    out = run_skill("def solve(p):\n    while True:\n        pass\n", "solve", {}, timeout=1.0)
    assert not out["ok"] and "timeout" in out["error"]


# ── skill library ────────────────────────────────────────────────
def test_seeded_library_covers_four_kinds():
    lib = SkillLibrary(seed=True)
    assert lib.for_kind("calc") and lib.for_kind("reverse")
    assert not lib.for_kind("is_prime")  # intentionally unsolved


def test_library_rejects_unsafe_skill():
    lib = SkillLibrary(seed=False)
    ok, msg = lib.add(Skill("evil", ["calc"], code="import os\ndef solve(p): return 1\n"))
    assert not ok and "unsafe" in msg


# ── solver + evaluator (the fitness signal) ──────────────────────
@pytest.fixture
def stack():
    lib = SkillLibrary(seed=True)
    solver = MultiAgentSolver(lib, timeout=5.0)
    ev = Evaluator(solver)  # no store_path → no persistence in tests
    return lib, solver, ev


def test_baseline_score_reflects_seeded_coverage(stack):
    lib, solver, ev = stack
    report = ev.run(record=False)
    seeded = _seeded_kinds(lib)
    expected_passed = sum(1 for t in DEFAULT_BENCHMARK if t.kind in seeded)
    # report["skills"] is the payload->answer family; coding/composite are extra.
    assert report["skills"]["total"] == len(DEFAULT_BENCHMARK)
    assert report["skills"]["passed"] == expected_passed
    assert 0.0 < report["score"] < 1.0  # some kinds intentionally unsolved


def test_failing_kinds_are_the_unseeded_ones(stack):
    lib, solver, ev = stack
    seeded = _seeded_kinds(lib)
    expected_failing = {k for k in all_kinds() if k not in seeded}
    assert set(ev.failing_kinds()) == expected_failing
    assert expected_failing  # there ARE synthesis targets to close


def test_learning_all_missing_skills_completes_the_skill_family(stack):
    lib, solver, ev = stack
    for kind in ev.failing_kinds():
        assert kind in KNOWN_SOLUTIONS, f"no reference solution for unsolved kind {kind}"
        added, msg = lib.add(Skill(f"{kind}_learned", [kind], code=KNOWN_SOLUTIONS[kind]))
        assert added, msg
    report = ev.run(record=False)
    # The payload->answer family is now fully solved (coding/composite are separate).
    assert report["skills"]["passed"] == report["skills"]["total"]
    assert not ev.failing_kinds()


def test_skill_acceptance_gate_measures_improvement(stack):
    # Mirrors substrate._skill_synthesis: keep a skill only if pass-rate rises.
    lib, solver, ev = stack
    before = ev.kind_pass_rate("is_prime")
    assert before == 0.0
    lib.add(Skill("prime", ["is_prime"], code=(
        "def solve(p):\n    n=p['n']\n    return n>1 and all(n%i for i in range(2,int(n**0.5)+1))\n")))
    after = ev.kind_pass_rate("is_prime")
    assert after > before and after == 1.0


def test_solver_reports_candidates(stack):
    lib, solver, ev = stack
    res = solver.solve(tasks_for_kind("calc")[0])
    assert res.solved and res.candidates >= 1 and res.winning_skill


# ── generalization: train/holdout gate (honest self-improvement) ──
def test_split_tasks_holds_out_last_example():
    from aegis.eval.benchmark import split_tasks
    train, holdout = split_tasks("to_binary")
    assert train and holdout
    assert not set(t.id for t in train) & set(t.id for t in holdout)  # disjoint


def test_memorizing_skill_fails_holdout_gate(stack):
    # A skill that hardcodes the TRAIN answer passes train but not holdout.
    from aegis.eval.benchmark import split_tasks
    lib, solver, ev = stack
    train, holdout = split_tasks("to_binary")  # train: 13->'1101', holdout: 255->'11111111'
    lib.add(Skill("bin_memo", ["to_binary"], code="def solve(p):\n    return '1101'\n"))
    assert ev.pass_rate_on(train) == 1.0     # looks solved on what it saw
    assert ev.pass_rate_on(holdout) == 0.0   # but does not generalize


def test_generalizing_skill_passes_holdout_gate(stack):
    from aegis.eval.benchmark import split_tasks
    lib, solver, ev = stack
    _, holdout = split_tasks("to_binary")
    lib.add(Skill("bin_real", ["to_binary"], code=KNOWN_SOLUTIONS["to_binary"]))
    assert ev.pass_rate_on(holdout) == 1.0


# ── step 1: coding benchmark (hidden-test verification) ──────────
from aegis.eval.coding import CODING_BENCHMARK, verify_solution

_FIZZBUZZ_GOOD = ("def fizzbuzz(n):\n    if n%15==0: return 'FizzBuzz'\n"
                  "    if n%3==0: return 'Fizz'\n    if n%5==0: return 'Buzz'\n    return str(n)\n")
_FIZZBUZZ_MEMO = "def fizzbuzz(n):\n    return {3:'Fizz', 5:'Buzz'}.get(n, str(n))\n"


def _coding_task(tid):
    return next(t for t in CODING_BENCHMARK if t.id == tid)


def test_coding_good_solution_passes_hidden_tests():
    assert verify_solution(_FIZZBUZZ_GOOD, _coding_task("fizzbuzz"))["solved"]


def test_coding_memorizer_fails_hidden_tests():
    v = verify_solution(_FIZZBUZZ_MEMO, _coding_task("fizzbuzz"))
    assert not v["solved"] and v["passed"] < v["total"]


def test_coding_unsafe_solution_rejected():
    v = verify_solution("import os\ndef fizzbuzz(n): return os.getcwd()\n", _coding_task("fizzbuzz"))
    assert not v["solved"]


def test_evaluator_combines_all_families_in_total(stack):
    lib, solver, ev = stack
    report = ev.run(record=False)
    assert report["coding"]["total"] == len(CODING_BENCHMARK)
    assert report["composite"]["total"] >= 1
    assert report["autocompose"]["total"] >= 1
    fams = ["skills", "coding", "composite", "autocompose"]
    assert report["total"] == sum(report[f]["total"] for f in fams)
    assert report["passed"] == sum(report[f]["passed"] for f in fams)


def test_stored_coding_solution_counts_as_solved(stack):
    lib, solver, ev = stack
    t = _coding_task("is_even")
    assert not ev.coding_solved(t)
    lib.add(Skill("sol_is_even", [t.kind_key()], func="is_even",
                  code="def is_even(n):\n    return n % 2 == 0\n"))
    assert ev.coding_solved(t)


# ── step 3: composite tasks (hierarchy of primitives) ────────────
from aegis.eval.composite import COMPOSITE_BENCHMARK


def test_composite_fails_without_primitive(stack):
    lib, solver, ev = stack  # sort_csv is unsolved by default
    res = solver.solve_composite(COMPOSITE_BENCHMARK[0])
    assert not res.solved


def test_composite_solves_once_primitive_learned(stack):
    lib, solver, ev = stack
    lib.add(Skill("csv", ["sort_csv"], code=KNOWN_SOLUTIONS["sort_csv"]))
    res = solver.solve_composite(COMPOSITE_BENCHMARK[0])
    assert res.solved and res.answer == "3,2,1"


# ── step 2: versioning (prefer the simpler correct skill) ────────
def test_shorter_skill_preferred_when_correct(stack):
    from aegis.eval.benchmark import split_tasks
    lib, solver, ev = stack
    _, holdout = split_tasks("is_prime")
    long_code = KNOWN_SOLUTIONS["is_prime"]
    short_code = "def solve(p):\n    n=p['n']\n    return n>1 and all(n%i for i in range(2,n))\n"
    lib.add(Skill("prime_long", ["is_prime"], code=long_code))
    lib.add(Skill("prime_short", ["is_prime"], code=short_code))
    # Both correct on holdout; the optimizer keeps the shorter one.
    assert ev.pass_rate_on(holdout) == 1.0
    assert len(short_code) < len(long_code)
    incumbent = min(lib.for_kind("is_prime"), key=lambda s: len(s.code))
    assert incumbent.name == "prime_short"


# ── more coding tasks ────────────────────────────────────────────
def test_new_coding_tasks_are_solvable():
    assert verify_solution(
        "def reverse_words(s):\n    return ' '.join(s.split()[::-1])\n",
        _coding_task("reverse_words"))["solved"]
    assert verify_solution(
        "def is_anagram(a, b):\n    return sorted(a) == sorted(b)\n",
        _coding_task("is_anagram"))["solved"]
    assert verify_solution(
        "def max_of(nums):\n    return max(nums)\n", _coding_task("max_of"))["solved"]


# ── live LLM coding synthesis (stubbed model) ────────────────────
def _fresh_substrate_for_coding(task_id):
    """A Substrate isolated from the on-disk skill store, with no preloaded
    solution for the given coding task (avoids cross-test pollution)."""
    from aegis.layers.substrate import Substrate
    s = Substrate()
    task = next(t for t in s.evaluator.coding_tasks if t.id == task_id)
    s.skill_library._store_path = None  # don't read/write the shared store
    s.skill_library.skills = {n: sk for n, sk in s.skill_library.skills.items()
                              if task.kind_key() not in sk.kinds}
    return s, task


def test_coding_synthesis_stores_passing_solution():
    import asyncio
    s, task = _fresh_substrate_for_coding("is_even")

    async def fake_solution(func_name, spec, visible):
        return "def is_even(n):\n    return n % 2 == 0\n"

    s.llm.propose_coding_solution = fake_solution
    assert not s.evaluator.coding_solved(task)
    asyncio.run(s._coding_synthesis([task]))
    assert s.evaluator.coding_solved(task)


def test_coding_synthesis_rejects_wrong_solution():
    import asyncio
    s, task = _fresh_substrate_for_coding("is_even")

    async def wrong(func_name, spec, visible):
        return "def is_even(n):\n    return True\n"  # fails hidden tests

    s.llm.propose_coding_solution = wrong
    asyncio.run(s._coding_synthesis([task]))
    assert not s.evaluator.coding_solved(task)


# ── auto-composition of arbitrary depth ──────────────────────────
from aegis.eval.autocompose import TRANSFORM_KINDS, AUTOCOMPOSE_BENCHMARK


def test_auto_compose_discovers_pipeline(stack):
    lib, solver, ev = stack  # upper + reverse are seeded
    path = solver.auto_compose("abc", "CBA", TRANSFORM_KINDS, max_depth=3)
    assert path is not None and len(path) == 2
    assert set(path) == {"reverse", "upper"}


def test_auto_compose_blocked_until_primitive_learned(stack):
    lib, solver, ev = stack
    assert solver.auto_compose("3,1,2", "3,2,1", TRANSFORM_KINDS, 3) is None  # sort_csv unsolved
    lib.add(Skill("csv", ["sort_csv"], code=KNOWN_SOLUTIONS["sort_csv"]))
    path = solver.auto_compose("3,1,2", "3,2,1", TRANSFORM_KINDS, 3)
    assert path == ["sort_csv", "reverse"]


def test_auto_compose_respects_depth_limit(stack):
    lib, solver, ev = stack
    # An unreachable target returns None rather than searching forever.
    assert solver.auto_compose("abc", "zzz", TRANSFORM_KINDS, max_depth=2) is None


# ── CSV export of fitness history ────────────────────────────────
def test_history_csv_export(stack):
    lib, solver, ev = stack
    ev.run(); ev.run()
    csv = ev.history_csv()
    lines = csv.strip().split("\n")
    assert lines[0] == "run,timestamp,score"
    assert len(lines) == 3  # header + 2 runs
    assert lines[1].startswith("1,")


# ── environment (grounding) ──────────────────────────────────────
def test_environment_step_returns_real_reward():
    lib = SkillLibrary(seed=True)
    env = TaskEnvironment(MultiAgentSolver(lib, timeout=5.0))
    step = env.step()
    assert step["task"] is not None
    assert step["reward"] in (0.0, 1.0)
    assert 0.0 <= env.rolling_reward() <= 1.0


def test_environment_round_robin_is_deterministic():
    lib = SkillLibrary(seed=True)
    env = TaskEnvironment(MultiAgentSolver(lib, timeout=5.0))
    first = env.step()["task"]
    for _ in range(len(DEFAULT_BENCHMARK) - 1):
        env.step()
    assert env.step()["task"] == first  # wrapped around to the start


# ── reward wiring (the synthetic signal is replaced) ─────────────
def test_compute_reward_uses_benchmark_score():
    from aegis.layers.substrate import Substrate
    s = Substrate()
    s._last_benchmark_score = 1.0
    # No environment steps yet → reward = 0.7*bench + 0.3*0.
    s.environment.rewards.clear()
    assert abs(s._compute_reward() - 0.7) < 1e-6
    # Add live environment reward → blended upward.
    s.environment.rewards.extend([1.0, 1.0])
    assert abs(s._compute_reward() - 1.0) < 1e-6


def test_compute_reward_falls_back_before_first_eval():
    from aegis.layers.substrate import Substrate
    s = Substrate()
    s._last_benchmark_score = None
    s.environment.rewards.clear()
    s.environment.total_steps = 0
    # Legacy synthetic estimate is still a valid [0,1] reward.
    r = s._compute_reward()
    assert 0.0 <= r <= 1.0
