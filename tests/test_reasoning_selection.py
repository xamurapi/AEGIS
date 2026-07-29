"""Choosing a strategy, and the bookkeeping around a trial (spec M6.8, M6.9).

Selection is where every other part of this contour cashes out: a weakness
found, a strategy written and an arena verdict all amount to nothing if the
selector never picks the result. The three layers — a trial's share, then
exploration, then UCB — are tested separately because each answers a different
question and each has been got wrong in a different way.
"""
import math

import pytest

from aegis.eval import reasoning_bench as bench
from aegis.layers.reasoning import (
    MIN_ATTEMPTS_PER_FAMILY, TRIAL_EVERY, ReasoningEngine,
)


@pytest.fixture
def engine(tmp_path):
    return ReasoningEngine(store_path=tmp_path / "strategies.json")


def _measure(engine, family, *, solved, used, name="direct"):
    strategy = engine.library.get(name)
    for index in range(used):
        strategy.note(family, solved=index < solved)
    return strategy


# ── UCB ──────────────────────────────────────────────────────────────

def test_a_strategy_nobody_has_tried_here_is_tried(engine):
    """An unbounded score for zero attempts is what "explore" means. Any finite
    default would let a strategy with one lucky success outrank one with none
    for ever."""
    active = engine.library.active()
    assert engine._ucb(active[0], "grid_planning", active) == float("inf")


def test_the_better_strategy_wins_when_both_have_been_tried(engine):
    good = _measure(engine, "grid_planning", solved=40, used=40, name="direct")
    poor = _measure(engine, "grid_planning", solved=4, used=40,
                    name="predictive_check")
    active = engine.library.active()
    assert engine._ucb(good, "grid_planning", active) > \
        engine._ucb(poor, "grid_planning", active)


def test_a_strategy_that_has_been_tried_less_gets_a_bigger_benefit_of_doubt(engine):
    """Without the exploration term the first strategy to reach a decent rate
    takes the family for good and no later evidence can dislodge it."""
    much = _measure(engine, "grid_planning", solved=80, used=100, name="direct")
    little = _measure(engine, "grid_planning", solved=8, used=10,
                      name="predictive_check")
    active = engine.library.active()
    assert much.accuracy("grid_planning") == little.accuracy("grid_planning")
    assert engine._ucb(little, "grid_planning", active) > \
        engine._ucb(much, "grid_planning", active)


def test_the_exploration_term_is_the_formula_the_spec_names(engine):
    strategy = _measure(engine, "grid_planning", solved=6, used=10)
    active = [strategy]
    expected = 0.6 + engine._ucb_c * math.sqrt(math.log(11) / 10)
    assert engine._ucb(strategy, "grid_planning", active) == pytest.approx(expected)


def test_the_exploration_constant_is_a_gene(engine):
    strategy = _measure(engine, "grid_planning", solved=6, used=10)
    active = [strategy]
    engine.set_genome({"reason_ucb_c": 0.0})
    assert engine._ucb(strategy, "grid_planning", active) == pytest.approx(0.6)
    engine.set_genome({"reason_ucb_c": 3.0})
    assert engine._ucb(strategy, "grid_planning", active) > 1.0


def test_a_nonsense_exploration_gene_falls_back_to_the_configuration(engine):
    import aegis.config as cfg

    engine.set_genome({"reason_ucb_c": "lots"})
    assert engine._ucb_c == cfg.REASON_UCB_C


def test_a_family_nobody_has_touched_does_not_divide_by_zero(engine):
    strategy = _measure(engine, "grid_planning", solved=1, used=1)
    assert engine._ucb(strategy, "magnitude", [strategy]) == float("inf")


def test_selection_prefers_the_strategy_with_the_best_bound(engine):
    """Once every strategy has been given a fair look. Before that UCB is
    supposed to prefer the one it knows least about, and does."""
    for strategy in engine.library.active():
        for _ in range(20):
            strategy.note("grid_planning", solved=False)
    winner = engine.library.get("direct")
    for _ in range(60):
        winner.note("grid_planning", solved=True)
    assert engine.select("grid_planning", "k").name == "direct"


def test_before_that_it_prefers_what_it_knows_least_about(engine):
    for strategy in engine.library.active():
        for _ in range(MIN_ATTEMPTS_PER_FAMILY):
            strategy.note("grid_planning", solved=False)
    for _ in range(40):
        engine.library.get("direct").note("grid_planning", solved=True)
    assert engine.select("grid_planning", "k").name != "direct"


# ── what the system knows, versus what it is trying ──────────────────

def test_what_it_knows_is_not_the_same_question_as_what_to_try(engine):
    """Reading a held-out set through the exploration schedule measured the
    schedule: the curve swung twenty points depending on the cycle it was read
    at."""
    best = _measure(engine, "grid_planning", solved=40, used=40, name="direct")
    for name in ("predictive_check", "self_consistency_k"):
        _measure(engine, "grid_planning", solved=0, used=40, name=name)
    for strategy in engine.library.active():
        for _ in range(MIN_ATTEMPTS_PER_FAMILY):
            strategy.note("grid_planning", solved=False)
    assert engine.best_known("grid_planning", "k") is best


def test_with_nothing_measured_there_is_nothing_to_be_greedy_about(engine):
    """A system that knows nothing does have to guess, and reporting that guess
    is the honest baseline."""
    chosen = {engine.best_known("grid_planning", f"k{index}").name
              for index in range(40)}
    assert len(chosen) > 1


# ── a trial's share ──────────────────────────────────────────────────

def _promote_to_trial(engine, family=""):
    return engine.library.admit(
        "candidate", [{"op": "REFLECT"}, {"op": "SOLVE"}], status="trial",
        family=family, incumbent="direct", weakness="family=grid_planning")


def test_a_trial_gets_some_of_its_class_and_not_all_of_it(engine):
    _promote_to_trial(engine, "grid_planning")
    for strategy in engine.library.active():
        for _ in range(MIN_ATTEMPTS_PER_FAMILY):
            strategy.note("grid_planning", solved=True)
    chosen = [engine.select("grid_planning", f"k{index}").name
              for index in range(200)]
    share = chosen.count("candidate") / len(chosen)
    assert 0.1 < share < 0.5
    assert TRIAL_EVERY > 1


def test_a_trial_for_one_class_does_not_take_another(engine):
    _promote_to_trial(engine, "grid_planning")
    for strategy in engine.library.active():
        for _ in range(MIN_ATTEMPTS_PER_FAMILY):
            strategy.note("magnitude", solved=True)
    chosen = {engine.select("magnitude", f"k{index}").name for index in range(60)}
    assert "candidate" not in chosen


def test_a_trial_with_no_class_applies_everywhere(engine):
    """Measured: filtering these out meant a trial accepted for a weakness that
    spanned classes got no traffic at all and was never concluded."""
    _promote_to_trial(engine, "")
    for strategy in engine.library.active():
        for _ in range(MIN_ATTEMPTS_PER_FAMILY):
            strategy.note("magnitude", solved=True)
    chosen = {engine.select("magnitude", f"k{index}").name for index in range(60)}
    assert "candidate" in chosen


def test_a_trial_is_never_what_the_system_would_answer_with(engine):
    """It is on trial. Reporting it as the system's answer would make the trial
    period a deployment."""
    trial = _promote_to_trial(engine, "grid_planning")
    for _ in range(50):
        trial.note("grid_planning", solved=True)
    for strategy in engine.library.active():
        for _ in range(MIN_ATTEMPTS_PER_FAMILY):
            strategy.note("grid_planning", solved=True)
    assert engine.best_known("grid_planning", "k") is not trial


# ── concluding, promoting, retiring ──────────────────────────────────

def test_a_promoted_trial_becomes_ordinary(engine):
    trial = _promote_to_trial(engine, "grid_planning")
    for _ in range(200):
        trial.note("grid_planning", solved=True)
    concluded = engine.review_trials(tick=7)
    assert concluded and concluded[0]["outcome"] == "active"
    assert engine.promotions == 1 and trial in engine.library.active()


def test_a_trial_that_did_not_earn_it_is_retired(engine):
    trial = _promote_to_trial(engine, "grid_planning")
    incumbent = _measure(engine, "grid_planning", solved=190, used=200)
    for index in range(200):
        trial.note("grid_planning", solved=index < 100)
    concluded = engine.review_trials(tick=7)
    assert concluded[0]["outcome"] == "retired"
    assert engine.demotions == 1 and trial.retired
    assert incumbent in engine.library.active()


def test_a_trial_is_judged_on_the_class_it_was_written_for(engine):
    """Not on its whole record. A trial admitted for one class can be run
    elsewhere by an earlier scan's traffic, and judging it on the mixture would
    retire a strategy that did exactly what it was accepted to do."""
    trial = _promote_to_trial(engine, "grid_planning")
    for _ in range(200):
        trial.note("grid_planning", solved=True)
        trial.note("magnitude", solved=False)
    assert engine.review_trials(tick=7)[0]["outcome"] == "active"


def test_a_trial_that_has_not_had_its_run_is_left_alone(engine):
    trial = _promote_to_trial(engine, "grid_planning")
    trial.note("grid_planning", solved=True)
    assert engine.review_trials(tick=7) == []
    assert trial.on_trial


def test_the_periodic_scan_concludes_trials_too(engine):
    """One action rather than two: Appendix A is the registry's contract, and
    adding an action to it is a change to the spec."""
    trial = _promote_to_trial(engine, "grid_planning")
    for _ in range(200):
        trial.note("grid_planning", solved=True)
    engine.scan_weakness()
    assert engine.promotions == 1


# ── the verdict log ──────────────────────────────────────────────────

def test_a_candidate_the_library_refuses_is_recorded_as_refused(engine):
    """Accepted by the arena and refused by the library is a real disagreement:
    the arena runs steps, the library admits strategies, and only one of them
    checks for duplicates."""
    from aegis.layers.reasoning.synthesis import Candidate

    engine.candidates.append(Candidate(name="direct", steps=[{"op": "REFLECT"}],
                                       weakness="family=grid_planning"))
    engine.arena.min_gain = -1.0
    engine.arena.regression_limit = 1.0
    engine.arena.cost_tolerance = 1e6
    record = engine.evaluate_candidate(tick=1)
    assert record["accepted"] is False
    assert any("admission refused" in reason for reason in record["reasons"])


def test_with_nothing_proposed_there_is_nothing_to_judge(engine):
    assert engine.evaluate_candidate(tick=1) is None


def test_the_verdict_log_is_bounded(engine):
    from aegis.layers.reasoning.synthesis import Candidate

    for index in range(230):
        engine.candidates.append(
            Candidate(name=f"c{index}", steps=[{"op": "REFLECT"}] * (index + 1),
                      weakness="family=grid_planning"))
        engine.evaluate_candidate(tick=index)
    assert len(engine.verdicts) == 200


def test_the_class_a_candidate_was_written_for_is_read_off_its_weakness(engine):
    assert engine._family_of(
        _Named("family=arithmetic_chain AND incomplete")) == "arithmetic_chain"
    assert engine._family_of(_Named("incomplete")) == ""
    assert engine._family_of(_Named("")) == ""


class _Named:
    def __init__(self, weakness):
        self.weakness = weakness


# ── the loop as a whole ──────────────────────────────────────────────

def test_working_and_improving_leaves_the_system_better(engine):
    """The claim of M6 end to end, on problems it never worked."""
    engine.set_genome({"reason_decompose_parts": 10})
    holdout = [bench.build(9_000_000 - offset) for offset in range(80)]

    def score():
        return sum(1 for task in holdout
                   if engine.interpreter.run(
                       engine.best_known(task.family, task.id), task,
                       budget=engine._budget()).solved) / len(holdout)

    before = score()
    for cycle in range(1, 9):
        engine.solve(64)
        engine.scan_weakness()
        engine.propose_strategy(tick=cycle)
        while engine.pending_candidates():
            engine.evaluate_candidate(tick=cycle)
    assert score() > before
