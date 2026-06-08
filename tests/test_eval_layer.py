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
    assert report["total"] == len(DEFAULT_BENCHMARK)
    assert report["passed"] == expected_passed
    assert 0.0 < report["score"] < 1.0  # some kinds intentionally unsolved


def test_failing_kinds_are_the_unseeded_ones(stack):
    lib, solver, ev = stack
    seeded = _seeded_kinds(lib)
    expected_failing = {k for k in all_kinds() if k not in seeded}
    assert set(ev.failing_kinds()) == expected_failing
    assert expected_failing  # there ARE synthesis targets to close


def test_learning_all_missing_skills_reaches_full_score(stack):
    lib, solver, ev = stack
    for kind in ev.failing_kinds():
        assert kind in KNOWN_SOLUTIONS, f"no reference solution for unsolved kind {kind}"
        added, msg = lib.add(Skill(f"{kind}_learned", [kind], code=KNOWN_SOLUTIONS[kind]))
        assert added, msg
    report = ev.run(record=False)
    assert report["score"] == 1.0


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
