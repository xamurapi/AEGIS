"""The guard conditions of the predictive contour (spec M1).

Each of these is a branch that decides *whether* the model does something —
whether to back off, whether to prune, whether to trust the leftover mass. They
never affect a happy-path result, so they are invisible to every other test,
and getting one backwards produces a model that is quietly wrong rather than
one that fails.
"""
import pytest

from aegis.clock import FrozenClock, set_clock
from aegis.layers.world.outcome import OutcomeModel
from aegis.layers.world.prediction import Prediction, PredictionScorer
from aegis.layers.world.simulate import Simulator
from aegis.layers.world.state import StateKey
from aegis.layers.world.transition import TransitionModel
from aegis.util.stats import calibration_curve, expected_calibration_error


def state(name: str) -> StateKey:
    return StateKey(energy=name)


# ── bin indexing ─────────────────────────────────────────────────────

def test_the_top_of_the_range_does_not_overflow_its_bin():
    # int(1.0 * bins) == bins, one past the last index. Getting this clamp
    # wrong turns a perfect forecast into an IndexError.
    assert calibration_curve([(1.0, True)], bins=5)[4]["n"] == 1
    assert expected_calibration_error([(1.0, True)], bins=5) == pytest.approx(0.0)


def test_a_value_just_below_the_top_shares_the_last_bin():
    curve = calibration_curve([(1.0, True), (0.99, True)], bins=5)
    assert curve[4]["n"] == 2


# ── transition: the marginal is counted, not shared ──────────────────

def test_the_action_marginal_counts_every_observation():
    model = TransitionModel(smoothing=0.0, half_life=0)
    for _ in range(3):
        model.observe(state("a"), "go", state("x"))
    # If the marginal decremented rather than accumulated, an unseen state
    # would report a negative or zero-weighted estimate.
    assert model.by_action["go"][state("x").key()] == pytest.approx(3.0)
    assert model.probability(state("unseen"), "go", state("x")) == pytest.approx(1.0)


def test_ageing_measures_elapsed_activity_not_accumulated_evidence():
    model = TransitionModel(half_life=4, smoothing=1.0)
    model.observe(state("a"), "go", state("b"))
    before = model.support(state("a"), "go")
    # Nothing else happened, so nothing has been forgotten.
    model.observe(state("a"), "go", state("b"))
    assert model.support(state("a"), "go") > before


def test_a_successor_is_pruned_only_once_it_has_faded():
    model = TransitionModel(half_life=1, smoothing=1.0)
    model.observe(state("a"), "go", state("ancient"))
    entry = model.pairs[model.pair_key(state("a"), "go")]
    assert state("ancient").key() in entry.next        # still fresh

    for i in range(40):
        model.observe(state(f"o{i}"), "go", state("recent"))
    model.observe(state("a"), "go", state("recent"))
    assert state("ancient").key() not in entry.next    # long since faded


def test_ageing_measures_the_gap_since_this_entry_was_last_touched():
    """The elapsed interval is a difference, not a sum.

    Adding the two counters instead of subtracting them would make an entry
    age faster the longer the model had been running — evidence would decay
    for no reason at all once the observation counter grew large.
    """
    model = TransitionModel(half_life=10, smoothing=1.0)
    for i in range(10):
        model.observe(state(f"warm{i}"), "go", state("b"))
    model.observe(state("a"), "go", state("b"))       # stamped at 10
    for i in range(9):
        model.observe(state(f"more{i}"), "go", state("b"))
    model.observe(state("a"), "go", state("b"))       # 10 elapsed -> halve
    assert model.support(state("a"), "go") == pytest.approx(1.5)


def test_the_action_marginal_is_preferred_over_the_global_prior():
    model = TransitionModel(smoothing=0.0, half_life=0)
    for _ in range(4):
        model.observe(state("a"), "left", state("x"))
    for _ in range(4):
        model.observe(state("a"), "right", state("y"))
    # Globally x and y are equally common, but "left" has only ever led to x.
    assert model.probability(state("unseen"), "left", state("x")) == pytest.approx(1.0)
    assert model.probability(state("unseen"), "left", state("y")) == pytest.approx(0.0)


def test_a_seen_pair_is_preferred_over_the_action_marginal():
    model = TransitionModel(smoothing=0.0, half_life=0)
    for _ in range(6):
        model.observe(state("elsewhere"), "go", state("common"))
    for _ in range(3):
        model.observe(state("here"), "go", state("local"))
    # From "here" the local record answers, not what the action does elsewhere.
    assert [key for key, _ in model.top_next(state("here"), "go", 1)] == \
        [state("local").key()]


def test_a_seen_pair_does_not_offer_successors_it_never_reached():
    # With smoothing on, a state the pair never led to still has a non-zero
    # probability through the back-off — but it is not a *candidate*, and
    # listing it would let the planner branch into a place this action has
    # never once gone.
    model = TransitionModel(smoothing=1.0, half_life=0)
    for _ in range(20):
        model.observe(state("elsewhere"), "go", state("far"))
    for _ in range(5):
        model.observe(state("here"), "go", state("near"))
    listed = {key for key, _ in model.top_next(state("here"), "go", 5)}
    assert listed == {state("near").key()}
    assert model.probability(state("here"), "go", state("far")) > 0


def test_an_empty_action_marginal_falls_through_to_the_prior():
    model = TransitionModel(smoothing=0.0, half_life=0)
    for _ in range(4):
        model.observe(state("a"), "seen", state("x"))
    # "never_tried" has no marginal of its own, so the global prior answers.
    assert model.probability(state("a"), "never_tried", state("x")) == pytest.approx(1.0)


def test_an_empty_model_has_no_prior_to_offer():
    assert TransitionModel(half_life=0).probability(state("a"), "go", state("x")) == 0.0


def test_a_pair_with_no_weight_left_uses_the_marginal_candidates():
    model = TransitionModel(half_life=0, smoothing=1.0)
    for _ in range(5):
        model.observe(state("a"), "go", state("x"))
    entry = model.pairs[model.pair_key(state("a"), "go")]
    entry.n = 0.0                                  # fully decayed away
    assert [key for key, _ in model.top_next(state("a"), "go", 1)] == \
        [state("x").key()]


def test_an_action_with_no_marginal_of_its_own_uses_the_prior():
    # Restored from a store whose action marginals did not survive: the global
    # prior is the only thing left to name candidates with.
    model = TransitionModel(half_life=0, smoothing=1.0)
    model.load({"prior": {state("x").key(): 5.0}})
    assert model.by_action == {}
    assert [key for key, _ in model.top_next(state("a"), "go", 1)] == \
        [state("x").key()]


def test_an_entirely_empty_model_names_no_candidates():
    assert TransitionModel(half_life=0).top_next(state("a"), "go", 3) == []


def test_capacity_is_unbounded_when_the_ceiling_is_zero():
    model = TransitionModel(max_states=0, half_life=0)
    for i in range(50):
        model.observe(state(f"s{i}"), "go", state("b"))
    assert len(model.pairs) == 50
    assert model.collapsed == 0


def test_capacity_leaves_a_table_at_exactly_its_ceiling_alone():
    model = TransitionModel(max_states=5, half_life=0)
    for i in range(5):
        model.observe(state(f"s{i}"), "go", state("b"))
    assert len(model.pairs) == 5
    assert model.collapsed == 0


# ── outcome: when the local record answers ───────────────────────────

def test_a_thin_local_record_answers_when_there_is_no_marginal():
    """Recovering from a store whose action marginals did not survive.

    Normally every observation writes both the pair and its marginal, so this
    only happens after a partial or hand-edited load. A thin local record is
    not much, but it beats having nothing to say — falling back to the neutral
    prior would discard the only evidence there is.
    """
    model = OutcomeModel(min_n=10, half_life=0)
    model.load({"pairs": {f"{state('s').key()}#go": {"n": 1, "successes": 1,
                                                     "reward": {"n": 1, "mean": 0.9}}}})
    assert model.by_action == {}
    prediction = model.predict(state("s"), "go")
    assert prediction.backed_off is False
    assert prediction.expected_reward == pytest.approx(0.9)


def test_a_thin_local_record_defers_once_a_marginal_exists():
    model = OutcomeModel(min_n=10, half_life=0)
    for name in "abcdef":
        for _ in range(10):
            model.observe(state(name), "go", success=True, reward=0.9)
    model.observe(state("z"), "go", success=False, reward=0.0)
    assert model.predict(state("z"), "go").backed_off is True


def test_an_empty_local_record_defers():
    model = OutcomeModel(min_n=3, half_life=0)
    for _ in range(5):
        model.observe(state("a"), "go", success=True, reward=0.9)
    assert model.predict(state("never_here"), "go").backed_off is True


def test_knowledge_is_the_ratio_the_spec_declares():
    model = OutcomeModel(min_n=4, half_life=0)
    for _ in range(3):
        model.observe(state("s"), "go", success=True, reward=0.5)
    assert model.knows(state("s"), "go") == pytest.approx(0.75)


# ── prediction: the leftover mass ────────────────────────────────────

def test_leftover_mass_needs_both_a_share_and_somewhere_to_put_it():
    base = dict(id="p", tick=1, state="s", action="a", p_success=0.5,
                expected_reward=0.5, reward_sd=0.0, predicted_next=[("a", 0.6)])
    # Mass but no states to spread it over, or states but no mass: neither can
    # produce a probability, and inventing one would understate the surprise.
    assert Prediction(**base, other_mass=0.4, other_states=0).probability_of("z") == 0.0
    assert Prediction(**base, other_mass=0.0, other_states=4).probability_of("z") == 0.0
    assert Prediction(**base, other_mass=0.4, other_states=4).probability_of("z") > 0


def test_a_forecast_with_no_effects_recorded_still_restores():
    restored = Prediction.from_dict({"id": "p", "predicted_next": None,
                                     "predicted_effects": None})
    assert restored is not None
    assert restored.predicted_effects == []


# ── prediction: the baseline claim ───────────────────────────────────

def test_beating_the_baselines_requires_every_number_to_exist():
    scorer = PredictionScorer(store_path=None)
    scorer.brier = 0.01
    assert scorer.beats_baselines() is False          # no baselines recorded yet
    scorer.baseline_brier_mean = 0.2
    assert scorer.beats_baselines() is False          # still one missing
    scorer.baseline_brier_half = 0.25
    assert scorer.beats_baselines() is True


def test_losing_to_either_baseline_is_not_beating_them():
    scorer = PredictionScorer(store_path=None)
    scorer.brier, scorer.baseline_brier_half = 0.1, 0.25
    scorer.baseline_brier_mean = 0.05
    assert scorer.beats_baselines() is False


# ── prediction: the log ──────────────────────────────────────────────

def test_the_log_directory_is_created_on_demand(tmp_path):
    path = tmp_path / "deep" / "nested" / "predictions.jsonl"
    scorer = PredictionScorer(store_path=path)
    scorer.open(Prediction(id="p", tick=1, state="s", action="a", p_success=0.5,
                           expected_reward=0.5, reward_sd=0.0,
                           predicted_next=[("n", 1.0)]))
    scorer.score("p", True, 0.5, "n")
    assert path.exists()


def test_the_log_is_left_alone_until_it_is_twice_its_budget(tmp_path):
    path = tmp_path / "predictions.jsonl"
    scorer = PredictionScorer(store_path=path, max_predictions=10)
    for i in range(15):
        scorer.open(Prediction(id=f"p{i}", tick=i, state="s", action="a",
                               p_success=0.5, expected_reward=0.5, reward_sd=0.0,
                               predicted_next=[("n", 1.0)]))
        scorer.score(f"p{i}", True, 0.5, "n")
    # Compacting on every append would rewrite the whole file each tick.
    assert len(path.read_text(encoding="utf-8").splitlines()) == 15


def test_reading_a_log_that_was_never_written_is_empty(tmp_path):
    scorer = PredictionScorer(store_path=tmp_path / "never.jsonl")
    assert scorer.recent() == []


def test_reading_with_no_log_configured_is_empty():
    assert PredictionScorer(store_path=None).recent() == []


# ── simulate: timing and traversal ───────────────────────────────────

@pytest.fixture
def frozen():
    clock = FrozenClock(1_000_000.0)
    previous = set_clock(clock)
    yield clock
    set_clock(previous)


def _teach(transitions, outcomes, source, action, target, reward=0.5, times=10):
    for _ in range(times):
        transitions.observe(source, action, target)
        outcomes.observe(source, action, success=True, reward=reward)


def test_the_elapsed_time_is_the_interval_not_the_endpoint(frozen):
    # The clock is advanced before the measurement starts, so a rollout that
    # reported the absolute time rather than the elapsed interval — or added
    # the two — would be visibly wrong instead of accidentally right.
    frozen.advance(100.0)
    transitions, outcomes = TransitionModel(half_life=0), OutcomeModel(half_life=0)
    _teach(transitions, outcomes, state("s"), "a", state("s"))

    class Timed(Simulator):
        def immediate_value(self, state_key, action):
            frozen.advance(0.002)          # 2 ms of "work"
            return super().immediate_value(state_key, action)

    simulator = Timed(transitions, outcomes)
    result = simulator.rollout(state("s"), ["a"], depth=1)
    assert result.elapsed_ms == pytest.approx(2.0, abs=0.001)
    assert simulator.last_elapsed_ms == pytest.approx(result.elapsed_ms)


def test_a_search_with_nothing_to_do_still_times_itself(frozen):
    frozen.advance(100.0)
    transitions, outcomes = TransitionModel(half_life=0), OutcomeModel(half_life=0)
    simulator = Simulator(transitions, outcomes)
    result = simulator.rollout(state("s"), [], depth=3)
    # No work was done, so the interval is zero — not the 100 seconds that had
    # elapsed before it was asked.
    assert result.elapsed_ms == pytest.approx(0.0, abs=0.001)
    assert simulator.last_elapsed_ms == pytest.approx(0.0, abs=0.001)


def test_a_search_with_nothing_to_do_is_not_truncated(frozen):
    transitions, outcomes = TransitionModel(half_life=0), OutcomeModel(half_life=0)
    # Nothing was cut short — there was nothing to cut.
    assert Simulator(transitions, outcomes).rollout(
        state("s"), [], depth=3).truncated is False


def test_a_fresh_result_is_not_marked_truncated():
    transitions, outcomes = TransitionModel(half_life=0), OutcomeModel(half_life=0)
    _teach(transitions, outcomes, state("s"), "a", state("s"))
    assert Simulator(transitions, outcomes).rollout(
        state("s"), ["a"], depth=2).truncated is False


def test_the_expanded_branch_is_renormalised_to_its_own_mass():
    """The top-k probabilities do not sum to one.

    Treating the unexpanded remainder as worth zero would make every deep plan
    look worse than a shallow one for no reason; multiplying by the mass
    instead of dividing would shrink the future the same way.
    """
    transitions, outcomes = TransitionModel(half_life=0, smoothing=0.0), \
        OutcomeModel(half_life=0)
    for _ in range(5):
        transitions.observe(state("s"), "a", state("rich"))
    for _ in range(5):
        transitions.observe(state("s"), "a", state("poor"))
    for _ in range(10):
        outcomes.observe(state("s"), "a", success=True, reward=0.0)
        outcomes.observe(state("rich"), "a", success=True, reward=1.0)
        outcomes.observe(state("poor"), "a", success=True, reward=1.0)

    # Only one successor is expanded, so the branch carries half the mass and
    # has to be renormalised back to one.
    simulator = Simulator(transitions, outcomes, branch=1, discount=1.0,
                          explore_bonus=0.0)
    here = simulator.immediate_value(state("s").key(), "a")
    there = simulator.immediate_value(state("rich").key(), "a")
    assert simulator.rollout(state("s"), ["a"], depth=2).value == \
        pytest.approx(here + there)


def test_the_trace_stops_when_the_depth_runs_out():
    transitions, outcomes = TransitionModel(half_life=0), OutcomeModel(half_life=0)
    _teach(transitions, outcomes, state("s"), "a", state("s"))
    simulator = Simulator(transitions, outcomes)
    assert len(simulator.rollout(state("s"), ["a"], depth=2).steps) == 2


def test_the_trace_stops_when_the_model_runs_out():
    transitions, outcomes = TransitionModel(half_life=0), OutcomeModel(half_life=0)
    outcomes.observe(state("s"), "a", success=True, reward=0.5)
    simulator = Simulator(transitions, outcomes)
    # No recorded successor: the plan is one step long even at depth 4.
    assert len(simulator.rollout(state("s"), ["a"], depth=4).steps) == 1


def test_the_discounted_future_uses_the_step_after_this_one():
    transitions, outcomes = TransitionModel(half_life=0), OutcomeModel(half_life=0)
    _teach(transitions, outcomes, state("s"), "a", state("t"), reward=0.0)
    _teach(transitions, outcomes, state("t"), "a", state("t"), reward=1.0)
    simulator = Simulator(transitions, outcomes, explore_bonus=0.0, discount=1.0)
    # Depth 3 sees two rewarded steps beyond the barren first one; depth 2 sees
    # one. If the recursion did not decrement, both would be identical.
    assert simulator.rollout(state("s"), ["a"], depth=3).value > \
        simulator.rollout(state("s"), ["a"], depth=2).value
