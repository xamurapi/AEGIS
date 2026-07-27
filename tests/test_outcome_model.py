"""What an action is worth (spec M1.3).

The asymmetry is the point: choosing reads the pessimistic bound so a lucky
sample cannot outrank a proven method, while reporting reads the point estimate
so a human gets the honest number. Everything else here is Welford and Wilson
doing what they are supposed to.
"""
import pytest

from aegis.layers.world.outcome import OutcomeEntry, OutcomeModel
from aegis.layers.world.state import StateKey


def state(name: str) -> StateKey:
    return StateKey(energy=name)


@pytest.fixture
def model():
    return OutcomeModel(min_n=3, half_life=0, smoothing=1.0)


# ── success rate ─────────────────────────────────────────────────────

def test_an_unseen_pair_is_a_coin_flip(model):
    # 0.5, not 0: no evidence is not evidence of failure, and a zero prior
    # would make the planner refuse to try anything new.
    assert model.p_success(state("a"), "go") == 0.5


def test_a_reliable_action_reports_a_high_rate(model):
    for _ in range(20):
        model.observe(state("a"), "go", success=True, reward=1.0)
    assert model.p_success(state("a"), "go") > 0.9


def test_a_failing_action_reports_a_low_rate(model):
    for _ in range(20):
        model.observe(state("a"), "go", success=False, reward=0.0)
    assert model.p_success(state("a"), "go") < 0.1


def test_a_mixed_record_lands_in_between(model):
    for _ in range(10):
        model.observe(state("a"), "go", success=True, reward=1.0)
        model.observe(state("a"), "go", success=False, reward=0.0)
    assert 0.4 < model.p_success(state("a"), "go") < 0.6


# ── the pessimistic bound ────────────────────────────────────────────

def test_one_lucky_success_does_not_look_like_certainty(model):
    model.observe(state("a"), "go", success=True, reward=1.0)
    assert model.p_success(state("a"), "go", pessimistic=True) < 0.5


def test_a_long_record_narrows_the_gap(model):
    for _ in range(200):
        model.observe(state("a"), "go", success=True, reward=1.0)
    optimistic = model.p_success(state("a"), "go")
    pessimistic = model.p_success(state("a"), "go", pessimistic=True)
    assert optimistic - pessimistic < 0.05


def test_the_pessimistic_estimate_never_exceeds_the_point_estimate(model):
    for i in range(30):
        model.observe(state("a"), "go", success=(i % 3 != 0), reward=0.5)
    assert model.p_success(state("a"), "go", pessimistic=True) <= \
        model.p_success(state("a"), "go")


def test_a_proven_action_outranks_a_lucky_one(model):
    # This is what the asymmetry is for: choosing must not be swayed by a
    # sample of one.
    model.observe(state("a"), "lucky", success=True, reward=1.0)
    for i in range(40):
        model.observe(state("a"), "proven", success=(i % 10 != 0), reward=0.9)
    assert model.p_success(state("a"), "proven", pessimistic=True) > \
        model.p_success(state("a"), "lucky", pessimistic=True)


# ── reward and cost ──────────────────────────────────────────────────

def test_expected_reward_is_the_mean_observed(model):
    for value in (0.2, 0.4, 0.6):
        model.observe(state("a"), "go", success=True, reward=value)
    assert model.expected_reward(state("a"), "go") == pytest.approx(0.4)


def test_an_unseen_pair_returns_the_neutral_reward(model):
    assert model.expected_reward(state("a"), "go") == OutcomeModel.NEUTRAL_REWARD


def test_reward_spread_is_tracked(model):
    for value in (0.0, 1.0, 0.0, 1.0):
        model.observe(state("a"), "swingy", success=True, reward=value)
    for _ in range(4):
        model.observe(state("a"), "steady", success=True, reward=0.5)
    assert model.reward_sd(state("a"), "swingy") > model.reward_sd(state("a"), "steady")


def test_a_single_observation_has_no_spread(model):
    model.observe(state("a"), "go", success=True, reward=0.5)
    assert model.reward_sd(state("a"), "go") == 0.0


def test_cost_is_averaged(model):
    for value in (10.0, 20.0):
        model.observe(state("a"), "go", success=True, reward=0.5, cost=value)
    assert model.expected_cost(state("a"), "go") == pytest.approx(15.0)


def test_an_unseen_pair_costs_nothing_known(model):
    assert model.expected_cost(state("a"), "go") == 0.0


# ── back-off ─────────────────────────────────────────────────────────

def test_a_new_state_borrows_what_the_action_does_elsewhere(model):
    for name in "bcdefg":
        model.observe(state(name), "go", success=True, reward=0.9)
    assert model.p_success(state("brand_new"), "go") > 0.7
    assert model.expected_reward(state("brand_new"), "go") == pytest.approx(0.9)


def test_a_back_off_answer_says_so(model):
    for name in "bcdefg":
        model.observe(state(name), "go", success=True, reward=0.9)
    assert model.predict(state("brand_new"), "go").backed_off is True


def test_a_well_evidenced_pair_is_not_a_back_off(model):
    for _ in range(5):
        model.observe(state("a"), "go", success=True, reward=0.9)
    assert model.predict(state("a"), "go").backed_off is False


def test_thin_local_evidence_prefers_the_action_marginal(model):
    # Two observations of a pair is not enough to overrule what the action does
    # across forty others.
    for name in "bcdefghij":
        for _ in range(5):
            model.observe(state(name), "go", success=True, reward=0.9)
    model.observe(state("z"), "go", success=False, reward=0.0)
    model.observe(state("z"), "go", success=False, reward=0.0)
    assert model.p_success(state("z"), "go") > 0.5


def test_enough_local_evidence_overrules_the_marginal(model):
    for name in "bcdefghij":
        model.observe(state(name), "go", success=True, reward=0.9)
    for _ in range(10):
        model.observe(state("z"), "go", success=False, reward=0.0)
    assert model.p_success(state("z"), "go") < 0.4


# ── how much is known ────────────────────────────────────────────────

def test_an_unseen_pair_is_unknown(model):
    assert model.knows(state("a"), "go") == 0.0


def test_knowledge_saturates(model):
    for _ in range(3):
        model.observe(state("a"), "go", success=True, reward=0.5)
    assert model.knows(state("a"), "go") == 1.0


def test_a_zero_minimum_treats_everything_as_known():
    assert OutcomeModel(min_n=0).knows(state("a"), "go") == 1.0


# ── the full prediction ──────────────────────────────────────────────

def test_a_prediction_is_produced_for_an_entirely_unseen_pair(model):
    # A planner that could not price a new action would never try one.
    prediction = model.predict(state("a"), "never_done")
    assert prediction.p_success == 0.5
    assert prediction.backed_off is True


def test_a_prediction_reports_every_field(model):
    for _ in range(5):
        model.observe(state("a"), "go", success=True, reward=0.7, cost=2.0)
    fields = model.predict(state("a"), "go").as_dict()
    assert set(fields) >= {"p_success", "p_success_pessimistic", "expected_reward",
                           "reward_sd", "expected_cost", "support", "known",
                           "backed_off"}


def test_a_prediction_accepts_a_bare_state_key(model):
    for _ in range(4):
        model.observe(state("a"), "go", success=True, reward=0.7)
    assert model.predict(state("a").key(), "go").support == 4


def test_actions_seen_are_enumerable(model):
    model.observe(state("a"), "go", success=True)
    model.observe(state("a"), "stop", success=True)
    assert model.actions_seen() == ["go", "stop"]


# ── forgetting ───────────────────────────────────────────────────────

def test_an_action_that_stopped_working_is_believed():
    # The world genuinely changes underneath the model; an estimate that
    # averaged over all of history would keep recommending what used to work.
    model = OutcomeModel(min_n=3, half_life=10)
    for _ in range(20):
        model.observe(state("a"), "go", success=True, reward=1.0)
    for _ in range(60):
        model.observe(state("a"), "go", success=False, reward=0.0)
    assert model.p_success(state("a"), "go") < 0.3


def test_forgetting_can_be_switched_off():
    model = OutcomeModel(half_life=0)
    for _ in range(50):
        model.observe(state("a"), "go", success=True, reward=1.0)
    assert model.support(state("a"), "go") == 50


# ── persistence ──────────────────────────────────────────────────────

def test_the_model_round_trips(model):
    for _ in range(6):
        model.observe(state("a"), "go", success=True, reward=0.8, cost=1.0)
    revived = OutcomeModel(half_life=0)
    revived.load(model.to_dict())
    assert revived.p_success(state("a"), "go") == \
        pytest.approx(model.p_success(state("a"), "go"))
    assert revived.expected_reward(state("a"), "go") == pytest.approx(0.8)


def test_loading_junk_is_survivable(model):
    model.load("not a dict")
    model.load({"pairs": "not a dict"})
    assert model.status()["pairs"] == 0


def test_a_malformed_row_is_skipped(model):
    model.load({"pairs": {"good#go": {"n": 3, "successes": 3},
                          "bad#go": {"n": "many"}}})
    assert "good#go" in model.pairs and "bad#go" not in model.pairs


def test_a_malformed_counter_resets(model):
    model.load({"observations": "lots"})
    assert model.observations == 0


def test_an_entry_rejects_a_bad_shape():
    assert OutcomeEntry.from_dict({"n": "many"}) is None


def test_a_corrupt_sub_record_does_not_discard_the_whole_entry():
    # Losing an action's success counts because its reward statistics were
    # malformed would throw away the useful half along with the broken one.
    entry = OutcomeEntry.from_dict({"n": 5, "successes": 4, "reward": "corrupt"})
    assert entry is not None
    assert entry.n == 5 and entry.successes == 4
    assert entry.reward.n == 0


def test_status_reports_the_shape(model):
    model.observe(state("a"), "go", success=True)
    status = model.status()
    assert status["pairs"] == 1 and status["actions"] == 1
