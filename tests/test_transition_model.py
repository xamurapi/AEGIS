"""Where actions lead: P(s' | s, a) (spec M1.3, M1.5).

Counting is the easy part. The two things worth testing are the ones that make
the counts usable: backing off to a coarser estimate when the evidence is thin,
and forgetting old evidence so the model follows a system that changes under it.
"""
import pytest

from aegis.layers.world.state import StateKey
from aegis.layers.world.transition import TransitionEntry, TransitionModel


def state(name: str) -> StateKey:
    return StateKey(energy=name)


@pytest.fixture
def model():
    return TransitionModel(smoothing=1.0, min_n=3, half_life=0, max_states=1000)


# ── counting ─────────────────────────────────────────────────────────

def test_an_observed_transition_becomes_the_likely_one(model):
    for _ in range(10):
        model.observe(state("a"), "go", state("b"))
    assert model.probability(state("a"), "go", state("b")) > 0.8


def test_the_top_successor_is_the_one_seen_most(model):
    for _ in range(9):
        model.observe(state("a"), "go", state("b"))
    model.observe(state("a"), "go", state("c"))
    assert model.top_next(state("a"), "go", 1)[0][0] == state("b").key()


def test_successors_come_back_most_likely_first(model):
    for _ in range(5):
        model.observe(state("a"), "go", state("b"))
    for _ in range(2):
        model.observe(state("a"), "go", state("c"))
    ranked = [key for key, _ in model.top_next(state("a"), "go", 3)]
    assert ranked[0] == state("b").key()


def test_only_k_successors_are_returned(model):
    for name in "bcdef":
        model.observe(state("a"), "go", state(name))
    assert len(model.top_next(state("a"), "go", 2)) == 2


def test_ties_break_on_the_state_key_not_on_dict_order(model):
    # Two identical runs must expand the same branches (§3.1).
    for name in ("z", "a", "m"):
        model.observe(state("s"), "go", state(name))
    first = model.top_next(state("s"), "go", 3)
    second = TransitionModel(half_life=0)
    for name in ("m", "a", "z"):
        second.observe(state("s"), "go", state(name))
    assert [k for k, _ in first] == [k for k, _ in second.top_next(state("s"), "go", 3)]


def test_observations_are_counted(model):
    model.observe(state("a"), "go", state("b"))
    model.observe(state("a"), "go", state("b"))
    assert model.observations == 2


def test_states_are_enumerable(model):
    model.observe(state("a"), "go", state("b"))
    model.observe(state("c"), "go", state("b"))
    assert model.states() == {state("a").key(), state("c").key()}


# ── back-off ─────────────────────────────────────────────────────────

def test_an_unseen_pair_falls_back_to_what_the_action_usually_does(model):
    # The action has always led to "b", just never from "z". Reporting zero
    # would be a confident claim built on no evidence at all.
    for name in "cdefg":
        model.observe(state(name), "go", state("b"))
    assert model.probability(state("z"), "go", state("b")) > 0.5


def test_an_entirely_unseen_action_falls_back_to_the_global_prior(model):
    for _ in range(10):
        model.observe(state("a"), "go", state("b"))
    assert model.probability(state("z"), "never_tried", state("b")) > 0.5


def test_an_empty_model_predicts_nothing_rather_than_guessing(model):
    assert model.probability(state("a"), "go", state("b")) == 0.0
    assert model.top_next(state("a"), "go", 3) == []


def test_thin_evidence_is_pulled_toward_the_back_off(model):
    # One observation from a new state should not outrank what the action does
    # everywhere else.
    for name in "cdefghij":
        model.observe(state(name), "go", state("common"))
    model.observe(state("z"), "go", state("rare"))
    assert model.probability(state("z"), "go", state("common")) > 0.3


def test_more_evidence_overcomes_the_back_off(model):
    for name in "cdefg":
        model.observe(state(name), "go", state("common"))
    for _ in range(50):
        model.observe(state("z"), "go", state("rare"))
    assert model.probability(state("z"), "go", state("rare")) > \
        model.probability(state("z"), "go", state("common"))


def test_an_unseen_pair_still_offers_candidates(model):
    for _ in range(5):
        model.observe(state("a"), "go", state("b"))
    assert model.top_next(state("z"), "go", 3)      # from the action marginal


# ── how much is known ────────────────────────────────────────────────

def test_an_unseen_pair_is_not_known(model):
    assert model.knows(state("a"), "go") == 0.0


def test_knowledge_saturates_at_the_minimum_sample(model):
    for _ in range(3):
        model.observe(state("a"), "go", state("b"))
    assert model.knows(state("a"), "go") == 1.0


def test_knowledge_does_not_exceed_one(model):
    for _ in range(100):
        model.observe(state("a"), "go", state("b"))
    assert model.knows(state("a"), "go") == 1.0


def test_partial_evidence_is_partially_known(model):
    model.observe(state("a"), "go", state("b"))
    assert 0 < model.knows(state("a"), "go") < 1


def test_a_zero_minimum_treats_everything_as_known():
    assert TransitionModel(min_n=0).knows(state("a"), "go") == 1.0


# ── surprise ─────────────────────────────────────────────────────────

def test_an_expected_outcome_is_unsurprising(model):
    for _ in range(20):
        model.observe(state("a"), "go", state("b"))
    assert model.surprise(state("a"), "go", state("b")) < 0.5


def test_an_unexpected_outcome_is_surprising(model):
    for _ in range(20):
        model.observe(state("a"), "go", state("b"))
    assert model.surprise(state("a"), "go", state("never_seen")) > 2.0


def test_surprise_is_finite_even_for_an_impossible_outcome(model):
    # An infinite surprise would swamp every average it entered; a probability
    # of exactly zero always means "not in the table", not "cannot happen".
    assert model.surprise(state("a"), "go", state("b")) < 100


# ── forgetting ───────────────────────────────────────────────────────

def test_old_evidence_loses_weight():
    # After a genome is promoted the world the model learned is not the world
    # any more; evidence has to age or the model averages over a system that
    # no longer exists.
    model = TransitionModel(half_life=10, min_n=3)
    for _ in range(10):
        model.observe(state("a"), "go", state("old"))
    weight_before = model.support(state("a"), "go")
    for _ in range(40):
        model.observe(state("a"), "go", state("new"))
    assert model.probability(state("a"), "go", state("new")) > \
        model.probability(state("a"), "go", state("old"))
    assert model.support(state("a"), "go") < weight_before + 40


def test_forgetting_can_be_switched_off():
    model = TransitionModel(half_life=0)
    for _ in range(50):
        model.observe(state("a"), "go", state("b"))
    assert model.support(state("a"), "go") == 50


def test_a_faded_successor_is_eventually_dropped():
    model = TransitionModel(half_life=2)
    model.observe(state("a"), "go", state("ancient"))
    for _ in range(80):
        model.observe(state("a"), "go", state("recent"))
    entry = model.pairs[model.pair_key(state("a"), "go")]
    assert state("ancient").key() not in entry.next


# ── capacity ─────────────────────────────────────────────────────────

def test_the_table_is_bounded():
    model = TransitionModel(max_states=5, half_life=0)
    for i in range(50):
        model.observe(state(f"s{i}"), "go", state("b"))
    assert len(model.pairs) <= 5


def test_capacity_drops_the_least_informative_pair_not_the_oldest():
    # A rare pair observed forty times is exactly what is worth keeping; a pair
    # observed once tells almost nothing.
    model = TransitionModel(max_states=3, half_life=0)
    for _ in range(40):
        model.observe(state("valuable"), "go", state("b"))
    for i in range(10):
        model.observe(state(f"thin{i}"), "go", state("b"))
    assert model.pair_key(state("valuable"), "go") in model.pairs


def test_collapsed_pairs_are_counted():
    model = TransitionModel(max_states=2, half_life=0)
    for i in range(10):
        model.observe(state(f"s{i}"), "go", state("b"))
    assert model.collapsed > 0


def test_a_collapsed_pair_still_informs_the_back_off():
    # Losing resolution is acceptable; losing the knowledge is not.
    model = TransitionModel(max_states=1, half_life=0)
    for i in range(10):
        model.observe(state(f"s{i}"), "go", state("common"))
    assert model.probability(state("brand_new"), "go", state("common")) > 0.5


# ── persistence ──────────────────────────────────────────────────────

def test_the_table_round_trips(model):
    for _ in range(5):
        model.observe(state("a"), "go", state("b"))
    revived = TransitionModel(half_life=0)
    revived.load(model.to_dict())
    assert revived.probability(state("a"), "go", state("b")) == \
        pytest.approx(model.probability(state("a"), "go", state("b")))


def test_loading_junk_is_survivable(model):
    model.load("not a dict")
    model.load({"pairs": "not a dict"})
    assert model.status()["pairs"] == 0


def test_a_malformed_pair_row_is_skipped(model):
    model.load({"pairs": {"good#go": {"n": 2, "next": {"x": 2}},
                          "bad#go": {"n": "many"}}})
    assert "good#go" in model.pairs and "bad#go" not in model.pairs


def test_counters_survive_a_round_trip(model):
    for _ in range(4):
        model.observe(state("a"), "go", state("b"))
    revived = TransitionModel()
    revived.load(model.to_dict())
    assert revived.observations == 4


def test_a_malformed_counter_resets_rather_than_raising(model):
    model.load({"observations": "lots"})
    assert model.observations == 0


def test_an_entry_row_rejects_a_bad_shape():
    assert TransitionEntry.from_dict({"next": "not a mapping"}) is None


def test_status_reports_the_shape_of_the_table(model):
    model.observe(state("a"), "go", state("b"))
    status = model.status()
    assert status["pairs"] == 1 and status["states"] == 1 and status["actions"] == 1
