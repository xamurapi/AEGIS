"""Evaluating a configuration must not train the live one (spec M5.5).

The obvious implementation of "score this variant" — set the parameter, run the
benchmark, set it back — is wrong in a way that is hard to see. The benchmark
*records* which skills succeeded, so measuring a variant changes the success
counters the live solver ranks by. Ten variants later the library has been
trained by an experiment nobody meant to be training, and the ordering gene has
been tuned by an evaluation rather than by evolution.

So an evaluation is data: a request in, a report out, nothing live reachable
from inside. That is also what lets it cross into a pool worker.
"""
import pytest

from aegis.eval.generators import build, generated_benchmark
from aegis.eval.isolated import (
    export_skills, library_from_export, make_request, run_request,
    task_from_dict, task_to_dict,
)
from aegis.eval.pool import EvaluationPool
from aegis.eval.skill_library import Skill, SkillLibrary
from aegis.eval.solver import SOLVER_ORDERS, MultiAgentSolver, order_skills


@pytest.fixture
def library():
    return SkillLibrary(store_path=None)


# ── the request is data ──────────────────────────────────────────────

def test_a_task_survives_the_round_trip():
    task = build("roman", 7)
    restored = task_from_dict(task_to_dict(task))
    assert (restored.id, restored.kind, restored.payload, restored.expected) == \
        (task.id, task.kind, task.payload, task.expected)


def test_an_export_carries_the_code_not_a_hash(library):
    """`snapshot()` hashes the code because it is for digests. A worker needs
    to run the skill, so the export carries the source."""
    rows = export_skills(library)
    assert rows and all(row["code"].startswith("def ") for row in rows)
    assert all("code_hash" not in row for row in rows)


def test_an_export_is_sorted_so_two_runs_agree(library):
    names = [row["name"] for row in export_skills(library)]
    assert names == sorted(names)


def test_a_library_rebuilt_from_an_export_matches_it(library):
    rebuilt = library_from_export(export_skills(library))
    assert export_skills(rebuilt) == export_skills(library)


def test_a_rebuilt_library_is_not_secretly_seeded():
    """Seeding would add the built-in skills to a variant that was meant to be
    evaluated without them, and every ablation would measure the same library."""
    assert library_from_export([]).skills == {}


def test_an_unusable_exported_row_is_skipped():
    rebuilt = library_from_export([
        {"name": "good", "kinds": ["calc"], "code": "def solve(p):\n    return 1\n"},
        {"nonsense": True},
        "not a row",
    ])
    assert list(rebuilt.skills) == ["good"]


def test_a_request_is_picklable():
    import pickle

    request = make_request(SkillLibrary(store_path=None),
                           generated_benchmark(per_kind=1))
    assert pickle.loads(pickle.dumps(request)) == request


# ── the isolation itself ─────────────────────────────────────────────

def test_running_a_request_does_not_touch_the_live_library(library):
    before = export_skills(library)
    run_request(make_request(library, generated_benchmark(per_kind=2), timeout=10.0))
    assert export_skills(library) == before


def test_the_live_success_counters_are_not_moved_by_an_evaluation(library):
    """The specific damage: `solve` records attempts, so scoring a variant would
    otherwise train the ranking the live solver uses."""
    attempts_before = {row["name"]: row["attempts"] for row in export_skills(library)}
    run_request(make_request(library, generated_benchmark(per_kind=3), timeout=10.0))
    attempts_after = {row["name"]: row["attempts"] for row in export_skills(library)}
    assert attempts_after == attempts_before


def test_the_report_hands_back_the_counters_it_did_produce(library):
    """Isolation is not amnesia. The learning is returned, so a caller that
    wants it can fold it in deliberately instead of getting it by accident."""
    report = run_request(make_request(library, generated_benchmark(per_kind=2),
                                      timeout=10.0))
    assert sum(row["attempts"] for row in report["skills"]) > 0


def test_a_report_scores_what_it_ran(library):
    tasks = generated_benchmark(per_kind=2, kinds=("calc", "reverse"))
    report = run_request(make_request(library, tasks, timeout=10.0))
    assert report["total"] == len(tasks)
    assert report["passed"] == len(report["solved"])
    assert report["score"] == pytest.approx(report["passed"] / report["total"])
    assert set(report["per_kind"]) == {"calc", "reverse"}


def test_a_library_without_the_skill_scores_zero():
    empty = SkillLibrary(store_path=None, seed=False)
    report = run_request(make_request(empty, generated_benchmark(per_kind=2,
                                                                kinds=("calc",))))
    assert report["score"] == 0.0
    assert report["solved"] == []


def test_an_empty_request_is_a_zero_not_a_crash():
    report = run_request({})
    assert report["total"] == 0 and report["score"] == 0.0


def test_the_label_comes_back_for_matching_up_results(library):
    report = run_request(make_request(library, [], label="variant_7"))
    assert report["label"] == "variant_7"


def test_a_request_runs_in_a_pool_worker(library):
    """The reason the request is data at all."""
    pool = EvaluationPool(workers=2, task_timeout=120.0)
    try:
        requests = [make_request(library, generated_benchmark(per_kind=1,
                                                              kinds=("calc",)),
                                 timeout=10.0, label=f"v{index}")
                    for index in range(2)]
        results = pool.map(run_request, requests)
        assert all(result.ok for result in results), [r.error for r in results]
        assert [r.value["label"] for r in results] == ["v0", "v1"]
    finally:
        pool.shutdown()


# ── the solver_order gene ────────────────────────────────────────────

def _skill(name, code="def solve(p):\n    return 1\n", **kw):
    return Skill(name=name, kinds=["calc"], code=code, **kw)


def test_ordering_by_success_puts_the_best_first():
    skills = [_skill("weak", attempts=10, successes=1),
              _skill("strong", attempts=10, successes=9),
              _skill("untried")]
    assert [s.name for s in order_skills(skills, "by_success")][0] == "strong"


def test_ordering_by_length_puts_the_shortest_first():
    skills = [_skill("long", code="def solve(p):\n" + "    x = 1\n" * 20),
              _skill("short", code="def solve(p):\n    return 1\n")]
    assert [s.name for s in order_skills(skills, "by_length")][0] == "short"


def test_ordering_by_recency_puts_the_newest_first():
    # Names chosen so alphabetical order DISAGREES with recency: with equal
    # success counts every other ordering falls back to the name, and a test
    # whose newest skill also sorted first alphabetically would pass under any
    # of them.
    skills = [_skill("alpha", created=100.0), _skill("beta", created=200.0)]
    assert [s.name for s in order_skills(skills, "by_recency")] == ["beta", "alpha"]
    assert [s.name for s in order_skills(skills, "by_success")] == ["alpha", "beta"]
    assert [s.name for s in order_skills(skills, "by_length")] == ["alpha", "beta"]


def test_ties_break_on_the_name_so_the_order_is_total():
    """A partial order would leave the choice to whatever `for_kind` returned,
    and two identical runs could try skills in different sequences."""
    skills = [_skill("zebra"), _skill("alpha"), _skill("middle")]
    for order in SOLVER_ORDERS:
        assert [s.name for s in order_skills(skills, order)] == \
            ["alpha", "middle", "zebra"]


def test_an_unknown_order_falls_back_to_success():
    skills = [_skill("weak", attempts=10, successes=1),
              _skill("strong", attempts=10, successes=9)]
    assert order_skills(skills, "by_astrology")[0].name == "strong"
    assert MultiAgentSolver(SkillLibrary(store_path=None),
                            order="by_astrology").order == "by_success"


def test_the_genome_retunes_the_solver(library):
    solver = MultiAgentSolver(library)
    solver.set_genome({"solver_order": "by_length", "solver_timeout": 7.5})
    assert solver.order == "by_length" and solver.timeout == 7.5


def test_an_unusable_solver_gene_is_ignored(library):
    solver = MultiAgentSolver(library, timeout=3.0)
    solver.set_genome({"solver_order": "sideways", "solver_timeout": "quick"})
    assert solver.order == "by_success" and solver.timeout == 3.0


def test_the_solver_timeout_gene_is_clamped_to_its_range(library):
    solver = MultiAgentSolver(library)
    solver.set_genome({"solver_timeout": 900.0})
    assert solver.timeout == 10.0
    solver.set_genome({"solver_timeout": 0.001})
    assert solver.timeout == 0.5


def test_the_order_reaches_an_isolated_run(library):
    request = make_request(library, generated_benchmark(per_kind=1, kinds=("calc",)),
                           solver_order="by_length")
    assert request["solver_order"] == "by_length"
    assert run_request(request)["total"] == 1


# ── what the solver reports about itself ─────────────────────────────
# `elapsed_ms` is read by the phase-budget checks and by the dashboard, and
# nothing asserted it: every arithmetic mutant in the timing survived. A solver
# that reports microseconds as milliseconds makes §3.4's budgets meaningless
# while every functional test still passes.

class _Clock:
    """A clock that advances half a second per reading."""

    def __init__(self, step=0.5):
        self.now_value = 1.0
        self.step = step

    def now(self):
        value = self.now_value
        self.now_value += self.step
        return value

    def monotonic(self):
        return self.now()


@pytest.fixture
def timed(monkeypatch):
    import aegis.eval.solver as module

    clock = _Clock()
    monkeypatch.setattr(module, "CLOCK", clock)
    return clock


def test_a_solved_task_reports_its_elapsed_time_in_milliseconds(timed, library):
    from aegis.eval.generators import build

    solver = MultiAgentSolver(library, timeout=10.0)
    result = solver.solve(build("calc", 0))
    assert result.solved
    assert result.elapsed_ms == pytest.approx(500.0)


def test_an_unsolved_task_reports_its_elapsed_time_too(timed):
    from aegis.eval.generators import build

    solver = MultiAgentSolver(SkillLibrary(store_path=None, seed=False))
    result = solver.solve(build("calc", 0))
    assert not result.solved
    assert result.elapsed_ms == pytest.approx(500.0)


def test_a_solved_composite_reports_its_elapsed_time(timed, library):
    """A composite the seeded library can actually finish, so the timing on the
    *success* path is the one being measured."""
    from aegis.eval.composite import CompositeTask

    task = CompositeTask("up_then_rev", ("upper", "reverse"), {"s": "abc"}, "CBA")
    solver = MultiAgentSolver(library, timeout=10.0)
    result = solver.solve_composite(task)
    assert result.solved
    assert result.winning_skill == "upper+reverse"
    assert result.elapsed_ms == pytest.approx(500.0)


def test_a_composite_with_no_skill_for_a_step_still_reports_its_time(timed):
    from aegis.eval.composite import COMPOSITE_BENCHMARK

    solver = MultiAgentSolver(SkillLibrary(store_path=None, seed=False))
    result = solver.solve_composite(COMPOSITE_BENCHMARK[0])
    assert not result.solved
    assert result.elapsed_ms == pytest.approx(500.0)


def test_a_composite_whose_step_produces_nothing_reports_its_time(timed):
    """The middle failure: a skill exists for the kind but never returns."""
    from aegis.eval.composite import COMPOSITE_BENCHMARK
    from aegis.eval.skill_library import Skill

    task = COMPOSITE_BENCHMARK[0]
    library = SkillLibrary(store_path=None, seed=False)
    for kind in task.pipeline:
        library.skills[f"broken_{kind}"] = Skill(
            name=f"broken_{kind}", kinds=[kind],
            code="def solve(p):\n    raise ValueError('no')\n")

    solver = MultiAgentSolver(library, timeout=10.0)
    result = solver.solve_composite(task)
    assert not result.solved
    assert result.elapsed_ms == pytest.approx(500.0)


def test_auto_compose_of_a_target_already_reached_is_an_empty_pipeline(library):
    """`start == target` — no transformation is needed, and the empty pipeline
    is the right answer. Inverting the test would send it searching for a way
    to turn a string into itself."""
    solver = MultiAgentSolver(library, timeout=10.0)
    assert solver.auto_compose("abc", "abc", ["upper"]) == []


def test_auto_compose_finds_a_one_step_pipeline(library):
    solver = MultiAgentSolver(library, timeout=10.0)
    assert solver.auto_compose("abc", "ABC", ["upper", "reverse"]) == ["upper"]


def test_auto_compose_gives_up_within_its_depth(library):
    solver = MultiAgentSolver(library, timeout=10.0)
    assert solver.auto_compose("abc", "unreachable", ["upper"], max_depth=2) is None
