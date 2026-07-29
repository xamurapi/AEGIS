"""pytest-bdd step definitions for tests/features/prediction.feature.

Executable Gherkin over the real predictive world model (M1). The feature file
describes the contour the spec asks for — forecast recorded before the action,
scored after it, error feeding surprise — and these steps are what make the
description a test rather than a claim.
"""
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aegis.layers.world.state import StateKey
from aegis.layers.world_model import PredictiveWorldModel

scenarios("features/prediction.feature")

STATE = StateKey(energy="mid", error="low", mood="curious", mode="focused",
                 focus_kind="knowledge", perf="flat", load="lo")
NEXT_STATE = StateKey(energy="hi", error="none", mood="curious", mode="focused",
                      focus_kind="knowledge", perf="up", load="lo")
#: Somewhere the model has never seen this action lead. Surprise is measured
#: over *successors* — how unexpected the world's next state was — not over
#: whether the action worked, so an unexpected outcome with the usual successor
#: is correctly no surprise at all.
UNEXPECTED = StateKey(energy="lo", error="high", mood="anxious", mode="reactive",
                      focus_kind="stability", perf="down", load="hi")
ACTION = "run_benchmark"


@given("a predictive world model", target_fixture="ctx")
def _model(tmp_path):
    return {"wm": PredictiveWorldModel(store_path=tmp_path / "world_model" / "model.json")}


@given(parsers.parse("thirty observations where an action always succeeds"))
def _observations(ctx):
    for _ in range(30):
        ctx["wm"].observe_outcome(STATE.key(), ACTION, True, reward=1.0)
        ctx["wm"].observe_transition(STATE.key(), ACTION, NEXT_STATE.key())


@when("a forecast is made for an action in a state")
def _forecast(ctx):
    ctx["prediction"] = ctx["wm"].make_prediction(STATE, ACTION, tick=1)


@when(parsers.parse("the action succeeds with a reward of {reward:f}"))
def _succeeds(ctx, reward):
    ctx["score"] = ctx["wm"].score_prediction(ctx["prediction"].id, True,
                                              reward, NEXT_STATE)


@when("the world goes somewhere the model did not expect")
def _unexpected(ctx):
    ctx["score"] = ctx["wm"].score_prediction(ctx["prediction"].id, False, 0.0,
                                              UNEXPECTED)


@when("the same forecast is scored again")
def _score_again(ctx):
    ctx["second"] = ctx["wm"].score_prediction(ctx["prediction"].id, True, 1.0,
                                               NEXT_STATE)


@when("an outcome is predicted for a pair nobody has seen")
def _unseen(ctx):
    ctx["outcome"] = ctx["wm"].predict_outcome(
        StateKey(energy="lo", error="high").key(), "never_taken")


@when("an outcome is predicted for that pair")
def _known_pair(ctx):
    ctx["outcome"] = ctx["wm"].predict_outcome(STATE.key(), ACTION)


@then("the forecast should carry a probability of success")
def _has_probability(ctx):
    assert 0.0 <= ctx["prediction"].p_success <= 1.0


@then("the forecast should carry an expected reward")
def _has_reward(ctx):
    assert isinstance(ctx["prediction"].expected_reward, float)


@then("the forecast should name the state it was made in")
def _names_state(ctx):
    assert ctx["prediction"].state == STATE.key()
    assert ctx["prediction"].action == ACTION


@then("the forecast should be closed")
def _closed(ctx):
    assert ctx["score"] is not None


@then("the Brier score should be recorded")
def _brier(ctx):
    assert 0.0 <= ctx["wm"].calibration()["brier"] <= 1.0


@then("the second scoring should be refused")
def _refused(ctx):
    """A forecast scored twice would count one outcome as two pieces of
    evidence, which is how a calibration curve is made to look better than the
    model is."""
    assert ctx["second"] is None


@then("surprise should be above zero")
def _surprised(ctx):
    assert ctx["wm"].surprise() > 0.0


@then("it should return a backoff estimate rather than an error")
def _backoff(ctx):
    assert ctx["outcome"] is not None
    assert 0.0 <= ctx["outcome"].p_success <= 1.0


@then("the model should report that it knows little there")
def _knows_little(ctx):
    assert ctx["wm"].knows(StateKey(energy="lo", error="high").key(),
                           "never_taken") < 0.5


@then(parsers.parse("the predicted probability of success should be above {bar:f}"))
def _above(ctx, bar):
    assert ctx["outcome"].p_success > bar


@then("the model should report that it knows a lot there")
def _knows_a_lot(ctx):
    assert ctx["wm"].knows(STATE.key(), ACTION) >= 0.5


@then("the lower bound should be below the point estimate")
def _pessimistic(ctx):
    """Choosing on the pessimistic bound is what stops three lucky attempts
    from outranking three hundred solid ones."""
    assert ctx["outcome"].p_success_pessimistic <= ctx["outcome"].p_success
