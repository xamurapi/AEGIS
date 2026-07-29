"""Judging a candidate, and concluding a trial (spec M6.8, M6.11).

Three gates, all of which must pass at once. Each has its own test for
acceptance and its own test for the rejection it exists to cause — a gate that
is never observed to refuse anything is a gate nobody has checked.
"""
import pytest

from aegis.eval import reasoning_bench as bench
from aegis.layers.reasoning import ReasoningEngine
from aegis.layers.reasoning.arena import (
    ARENA_TASKS, HOLDOUT_BASE, REGRESSION_BASE, TRAIN_BASE, Arena, Verdict,
    conclude_trial,
)
from aegis.layers.reasoning.interpreter import Interpreter
from aegis.layers.reasoning.library import Library, Strategy
from aegis.layers.reasoning.weakness import Weakness

#: The combination no built-in has: break it up, and know when not to answer.
COMBINED = [
    {"op": "DECOMPOSE", "max_parts": 10},
    {"op": "SOLVE"},
    {"op": "VERIFY", "checker": "confidence"},
    {"op": "BRANCH", "cond": "insufficient", "then": [{"op": "ABSTAIN"}]},
]


@pytest.fixture
def arena():
    return Arena(Interpreter(genome={"reason_decompose_parts": 10}))


@pytest.fixture
def library(tmp_path):
    return Library(store_path=tmp_path / "strategies.json")


def _weakness(family="arithmetic_chain"):
    return Weakness(combo=(f"family={family}",), fail_rate=0.8, base_rate=0.2,
                    support=40, fails=32, lower=0.6, excess=0.6, p_value=1e-6,
                    rank=24.0, family=family)


# ── the task sets ────────────────────────────────────────────────────

def test_the_three_sets_are_disjoint(arena):
    sets = arena.sets_for("arithmetic_chain")
    ids = [{task.id for task in tasks} for tasks in sets.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_the_arena_never_meets_the_working_queue_or_the_holdout_score(arena):
    """A candidate judged on problems it was tuned on has not been judged."""
    working = {bench.build(index).id for index in range(2000)}
    reported = {bench.build(10_000_000 - offset).id for offset in range(200)}
    for tasks in arena.sets_for("arithmetic_chain").values():
        arena_ids = {task.id for task in tasks}
        assert not (arena_ids & working) and not (arena_ids & reported)


def test_the_weak_class_is_what_the_class_sets_are_drawn_from(arena):
    sets = arena.sets_for("missing_data")
    assert all(task.family == "missing_data" for task in sets["train"])
    assert all(task.family == "missing_data" for task in sets["holdout"])
    assert len({task.family for task in sets["regression"]}) > 1


def test_a_weakness_spanning_classes_is_judged_on_everything(arena):
    sets = arena.sets_for("")
    assert len({task.family for task in sets["holdout"]}) > 1


def test_an_unknown_class_falls_back_to_everything(arena):
    assert arena.sets_for("telepathy")["train"]


def test_the_bases_are_far_apart():
    assert TRAIN_BASE != HOLDOUT_BASE != REGRESSION_BASE
    assert min(HOLDOUT_BASE - TRAIN_BASE,
               REGRESSION_BASE - HOLDOUT_BASE) > ARENA_TASKS * 100


# ── acceptance ───────────────────────────────────────────────────────

def test_a_genuinely_better_strategy_is_accepted(arena, library):
    verdict = arena.evaluate(_Candidate(COMBINED), _weakness(),
                             library.get("direct"))
    assert verdict.accepted and not verdict.reasons
    assert verdict.holdout_gain >= arena.min_gain


def test_the_verdict_carries_every_number_behind_it(arena, library):
    verdict = arena.evaluate(_Candidate(COMBINED), _weakness(),
                             library.get("direct"))
    assert verdict.candidate_holdout > verdict.incumbent_holdout
    assert 0.0 <= verdict.p_value <= 1.0
    assert verdict.cost_ratio > 0


# ── each gate refuses something ──────────────────────────────────────

def test_a_strategy_that_does_not_help_is_refused(arena, library):
    """The gain gate. A copy of the incumbent gains nothing and must not pass
    merely because it breaks nothing."""
    verdict = arena.evaluate(_Candidate([{"op": "SOLVE"}]), _weakness(),
                             library.get("direct"))
    assert not verdict.accepted
    assert any("gain" in reason for reason in verdict.reasons)


def test_a_strategy_that_wins_its_class_by_losing_elsewhere_is_refused(library):
    """The regression gate. Without it the system accumulates these one
    weakness at a time and gets worse while every step looked like progress."""
    arena = Arena(Interpreter(genome={"reason_decompose_parts": 10}))
    # Abstaining always wins `missing_data` outright and loses everything else.
    always_abstain = [{"op": "ABSTAIN", "reason": "no"}]
    verdict = arena.evaluate(_Candidate(always_abstain),
                             _weakness("missing_data"), library.get("direct"))
    assert verdict.holdout_gain > 0
    assert not verdict.accepted
    assert any("general benchmark" in reason for reason in verdict.reasons)


def test_a_strategy_that_buys_its_gain_with_cost_is_refused(library):
    """The cost gate. Wrapping everything in a five-way vote will usually buy a
    point of accuracy; without this that is a free win."""
    arena = Arena(Interpreter(genome={"reason_decompose_parts": 10}),
                  min_gain=-1.0, regression_limit=1.0, cost_tolerance=1.2)
    expensive = [{"op": "VOTE", "n": 5, "agg": "majority", "body": COMBINED}]
    verdict = arena.evaluate(_Candidate(expensive), _weakness(),
                             library.get("direct"))
    assert not verdict.accepted
    assert any("costs" in reason for reason in verdict.reasons)


def test_all_three_reasons_can_be_reported_at_once(library):
    arena = Arena(Interpreter(), min_gain=0.9, regression_limit=0.0,
                  cost_tolerance=0.01)
    verdict = arena.evaluate(
        _Candidate([{"op": "VOTE", "n": 5, "body": [{"op": "ABSTAIN"}]}]),
        _weakness(), library.get("decompose_solve_combine"))
    assert len(verdict.reasons) == 3


def test_cost_is_priced_from_the_steps_not_from_the_clock(arena, library):
    """Wall time is the honest measure of what a strategy costs and also the
    one that changes with machine load; a candidate judged on a busy minute
    would be rejected for the machine's reasons."""
    first = arena._cost_ratio(COMBINED, library.get("direct"))
    second = arena._cost_ratio(COMBINED, library.get("direct"))
    assert first == second


def test_a_free_incumbent_does_not_divide_by_zero(arena):
    assert arena._cost_ratio([{"op": "SOLVE"}], []) == float("inf")
    assert arena._cost_ratio([], []) == 1.0


def test_the_arena_counts_what_it_did(arena, library):
    arena.evaluate(_Candidate(COMBINED), _weakness(), library.get("direct"))
    arena.evaluate(_Candidate([{"op": "SOLVE"}]), _weakness(),
                   library.get("direct"))
    assert arena.status() == {**arena.status(), "runs": 2, "accepted": 1}


def test_a_verdict_renders_as_data(arena, library):
    import json

    verdict = arena.evaluate(_Candidate(COMBINED), _weakness(),
                             library.get("direct"))
    assert json.loads(json.dumps(verdict.as_dict()))["accepted"] is True


def test_an_empty_verdict_is_a_rejection():
    assert Verdict().accepted is False


# ── what a score counts ──────────────────────────────────────────────

def test_a_score_is_a_share_of_what_was_attempted(arena, library):
    """Half of a class solved is 0.5, not 24. Every gate compares two of these,
    so a count wearing a rate's name would make every comparison meaningless."""
    tasks = bench.build_family("missing_data", 8)
    always = arena.score([{"op": "ABSTAIN"}], tasks)
    never = arena.score([{"op": "SOLVE"}], tasks)
    assert always.total == 8 and always.solved == 8 and always.rate == 1.0
    assert never.rate == 0.0
    assert never.confident_error_rate == 1.0
    assert always.confident_error_rate == 0.0


def test_an_answer_that_is_merely_absent_is_not_a_confident_error(arena):
    """Silence and a wrong number are different failures, and the arena reports
    the second because it is the one abstention exists to prevent."""
    class Bare:
        id = "bare"
        family = "bare"
        prompt = "Consider the matter."

        def verify(self, answer):
            return False

    score = arena.score([{"op": "SOLVE"}], [Bare()])
    assert score.solved == 0 and score.confident_errors == 0


def test_an_empty_set_scores_zero_rather_than_dividing_by_it(arena):
    score = arena.score([{"op": "SOLVE"}], [])
    assert score.rate == 0.0 and score.confident_error_rate == 0.0


def test_a_gain_is_the_candidate_minus_the_incumbent(arena, library):
    """Both ways round, because a sum would look like a gain whenever both arms
    did well and would be largest exactly when the candidate added nothing."""
    better = arena.evaluate(_Candidate(COMBINED), _weakness(),
                            library.get("direct"))
    worse = arena.evaluate(_Candidate([{"op": "SOLVE"}]), _weakness(),
                           library.get("decompose_solve_combine"))
    assert better.train_gain > 0 and better.holdout_gain > 0
    assert worse.train_gain < 0 and worse.holdout_gain < 0
    assert better.holdout_gain == pytest.approx(
        better.candidate_holdout - better.incumbent_holdout)


def test_abstention_is_reported_as_a_fall_in_confident_errors(arena, library):
    verdict = arena.evaluate(_Candidate(COMBINED), _weakness("missing_data"),
                             library.get("direct"))
    assert verdict.confident_error_delta < 0


# ── concluding a trial ───────────────────────────────────────────────

def _strategy(name, family, solved, used, status="trial"):
    strategy = Strategy(name=name, status=status)
    for index in range(used):
        strategy.note(family, solved=index < solved)
    return strategy


def test_a_trial_that_has_not_had_its_run_keeps_going():
    trial = _strategy("t", "alpha", 5, 5)
    outcome, reason = conclude_trial(trial, None, "alpha", min_uses=50)
    assert outcome == "trial" and "5/50" in reason


def test_a_trial_that_beats_the_incumbent_takes_over():
    trial = _strategy("t", "alpha", 48, 50)
    incumbent = _strategy("i", "alpha", 30, 50, status="active")
    outcome, _ = conclude_trial(trial, incumbent, "alpha", min_uses=50)
    assert outcome == "active"


def test_a_trial_that_does_not_beat_the_incumbent_goes():
    trial = _strategy("t", "alpha", 30, 50)
    incumbent = _strategy("i", "alpha", 48, 50, status="active")
    outcome, _ = conclude_trial(trial, incumbent, "alpha", min_uses=50)
    assert outcome == "retired"


def test_a_lucky_run_does_not_displace_a_long_record():
    """Compared on lower bounds. A trial with a favourable handful of problems
    has a wide interval; the incumbent, with hundreds behind it, does not."""
    trial = _strategy("t", "alpha", 20, 20)
    incumbent = _strategy("i", "alpha", 470, 500, status="active")
    outcome, _ = conclude_trial(trial, incumbent, "alpha", min_uses=20)
    assert outcome == "retired"


def test_a_trial_with_no_incumbent_is_promoted_only_on_its_own_evidence():
    good = _strategy("t", "alpha", 45, 50)
    poor = _strategy("t", "alpha", 10, 50)
    assert conclude_trial(good, None, "alpha", min_uses=50)[0] == "active"
    assert conclude_trial(poor, None, "alpha", min_uses=50)[0] == "retired"


def test_an_incumbent_that_never_ran_here_is_no_incumbent():
    trial = _strategy("t", "alpha", 45, 50)
    stranger = _strategy("i", "beta", 50, 50, status="active")
    assert conclude_trial(trial, stranger, "alpha", min_uses=50)[0] == "active"


# ── end to end ───────────────────────────────────────────────────────

def test_the_loop_finds_what_no_built_in_has(tmp_path):
    """The claim of M6 as a whole: the system writes a strategy nobody gave it,
    and the strategy is better."""
    engine = ReasoningEngine(store_path=tmp_path / "strategies.json")
    engine.set_genome({"reason_decompose_parts": 10})
    accepted = []
    for cycle in range(1, 4):
        engine.solve(64)
        engine.scan_weakness()
        engine.propose_strategy(tick=cycle)
        while engine.pending_candidates():
            verdict = engine.evaluate_candidate(tick=cycle)
            if verdict and verdict["accepted"]:
                accepted.append(verdict)
    assert accepted, "the arena accepted nothing in three cycles"
    trials = engine.library.trials()
    assert trials and all(strategy.on_trial for strategy in trials)


def test_an_accepted_strategy_enters_on_trial_not_in_service(tmp_path):
    """An arena run says a strategy is better on problems chosen for it. Live
    traffic is the only place it can be wrong in a way that cannot see."""
    engine = ReasoningEngine(store_path=tmp_path / "strategies.json")
    engine.set_genome({"reason_decompose_parts": 10})
    engine.solve(64)
    engine.scan_weakness()
    engine.propose_strategy(tick=1)
    while engine.pending_candidates():
        engine.evaluate_candidate(tick=1)
    for strategy in engine.library.trials():
        assert strategy not in engine.library.active()
        assert strategy in engine.library.in_use()


def test_a_trial_gets_traffic_and_is_eventually_concluded(tmp_path):
    """Measured: a trial accepted for a weakness that spanned classes got no
    traffic at all and sat at zero applications for a whole thirty-cycle run."""
    engine = ReasoningEngine(store_path=tmp_path / "strategies.json")
    engine.set_genome({"reason_decompose_parts": 10})
    for cycle in range(1, 12):
        engine.solve(64)
        engine.scan_weakness()
        engine.propose_strategy(tick=cycle)
        while engine.pending_candidates():
            engine.evaluate_candidate(tick=cycle)
        engine.review_trials(tick=cycle)
    assert engine.promotions + engine.demotions > 0
    promoted = [s for s in engine.library.active() if not s.builtin]
    assert promoted and all(strategy.used() > 0 for strategy in promoted)


class _Candidate:
    """A bare candidate: the arena only ever needs the steps."""

    def __init__(self, steps):
        self.steps = steps
        self.name = "candidate"
