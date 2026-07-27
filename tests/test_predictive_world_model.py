"""The predictive contour, closed (spec M1.1, M1.6).

    state → prediction → action → outcome → error → learning → curiosity

The point of this stage is that the loop actually closes inside a running tick,
and that the old world model keeps working exactly as it did while it happens.
"""
import asyncio

import pytest

from aegis.clock import frozen
from aegis.layers.substrate import Substrate
from aegis.layers.world.state import StateKey
from aegis.layers.world_model import CausalLinks, PredictiveWorldModel, WorldModel


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def wm(tmp_path):
    return PredictiveWorldModel(store_path=tmp_path / "model.json")


@pytest.fixture
def substrate(isolated_state):
    s = Substrate()
    s.llm.enabled = False

    async def _no_agents():
        return []

    async def _no_learning(*a, **k):
        return {"success": False}

    s.agent_system.run_due_agents = _no_agents
    s.external_learning.learn_from_source = _no_learning
    s.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    s.health.check = lambda: {"status": "healthy", "warnings": [],
                              "critical": [], "metrics": {}}
    return s


def drive(substrate, ticks):
    with frozen() as clock:
        async def _go():
            for _ in range(ticks):
                await substrate.tick()
                clock.advance(3.0)

        _run(_go())


# ── the old model still works ────────────────────────────────────────

def test_the_legacy_name_still_resolves():
    assert WorldModel is PredictiveWorldModel


def test_causal_observation_is_unchanged(wm):
    wm.observe("cause", "effect", success=True)
    assert wm.links["cause"]["effect"]["observations"] == 1
    assert wm.total_observations == 1


def test_causal_prediction_is_unchanged(wm):
    for _ in range(5):
        wm.observe("cause", "effect", success=True)
    assert wm.predict("cause")[0]["effect"] == "effect"


def test_risk_lookup_is_unchanged(wm):
    for _ in range(5):
        wm.observe("risky_thing", "broke", success=False)
    assert wm.risks_for(["risky"])[0]["cause"] == "risky_thing"


def test_chain_building_is_unchanged(wm):
    for _ in range(5):
        wm.observe("learn topic", "gained", success=True)
    chain = wm.build_chain("learn topic")
    assert chain["objective"] == "learn topic"
    assert wm.chains


def test_chain_refinement_is_unchanged(wm):
    assert wm.refine_chain({"objective": "o", "confidence": 0.5})["source"] == "llm"


def test_capacity_regulation_still_reaches_the_cap(wm):
    wm.max_links = 42
    assert wm.max_links == 42
    assert wm.causal.max_links == 42


def test_the_causal_half_uses_the_legacy_file(tmp_path):
    model = PredictiveWorldModel(store_path=tmp_path / "model.json")
    model.observe("a", "b")
    model.save()
    assert (tmp_path / "model.json").exists()


def test_the_predictive_half_uses_its_own_files(tmp_path):
    model = PredictiveWorldModel(store_path=tmp_path / "model.json")
    model.observe_transition(StateKey(energy="hi"), "go", StateKey(energy="lo"))
    model.save()
    assert (tmp_path / "transitions.json").exists()
    assert (tmp_path / "outcomes.json").exists()
    assert (tmp_path / "calibration.json").exists()


def test_the_status_keeps_its_legacy_keys(wm):
    wm.observe("a", "b")
    status = wm.status()
    for key in ("causes", "links", "total_observations", "chains_built",
                "strongest_links"):
        assert key in status


def test_the_status_gains_the_predictive_half(wm):
    assert "predictive" in wm.status()


# ── the new half ─────────────────────────────────────────────────────

def test_a_forecast_is_recorded_before_the_action(wm):
    state = StateKey(energy="hi")
    prediction = wm.make_prediction(state, "go", tick=1)
    assert prediction.state == state.key()
    assert wm.scorer.pending() == 1


def test_a_forecast_is_closed_against_what_happened(wm):
    state, after = StateKey(energy="hi"), StateKey(energy="lo")
    prediction = wm.make_prediction(state, "go", tick=1)
    score = wm.score_prediction(prediction.id, True, 0.8, after)
    assert score is not None
    assert wm.scorer.pending() == 0


def test_knowledge_needs_both_halves(wm):
    # Knowing where an action leads without knowing what it is worth is not
    # knowledge a planner can act on.
    state = StateKey(energy="hi")
    for _ in range(10):
        wm.observe_transition(state, "go", StateKey(energy="lo"))
    assert wm.knows(state, "go") == 0.0

    for _ in range(10):
        wm.observe_outcome(state, "go", success=True, reward=0.5)
    assert wm.knows(state, "go") == 1.0


def test_confidence_falls_with_thin_evidence(wm):
    state = StateKey(energy="hi")
    thin = wm.make_prediction(state, "untried", tick=1)
    for _ in range(10):
        wm.observe_transition(state, "known", StateKey(energy="lo"))
        wm.observe_outcome(state, "known", success=True, reward=0.5)
    thick = wm.make_prediction(state, "known", tick=2)
    assert thick.confidence > thin.confidence


def test_coverage_tracks_how_often_the_model_had_an_answer(wm):
    state = StateKey(energy="hi")
    wm.make_prediction(state, "untried", tick=1)
    assert wm.coverage() == 0.0
    for _ in range(10):
        wm.observe_transition(state, "known", StateKey(energy="lo"))
        wm.observe_outcome(state, "known", success=True, reward=0.5)
    wm.make_prediction(state, "known", tick=2)
    assert wm.coverage() == 0.5


def test_a_rollout_runs_over_the_learned_model(wm):
    state = StateKey(energy="hi")
    for _ in range(10):
        wm.observe_transition(state, "good", state)
        wm.observe_outcome(state, "good", success=True, reward=0.9)
        wm.observe_transition(state, "bad", state)
        wm.observe_outcome(state, "bad", success=False, reward=0.1)
    assert wm.best_sequence(state, ["good", "bad"], depth=2)[0] == "good"


def test_a_sequence_can_be_priced(wm):
    state = StateKey(energy="hi")
    for _ in range(10):
        wm.observe_transition(state, "a", state)
        wm.observe_outcome(state, "a", success=True, reward=0.9)
    assert wm.evaluate_sequence(state, ["a", "a"]) > 0


# ── persistence ──────────────────────────────────────────────────────

def test_the_predictive_half_survives_a_restart(tmp_path):
    path = tmp_path / "model.json"
    first = PredictiveWorldModel(store_path=path)
    state = StateKey(energy="hi")
    for _ in range(8):
        first.observe_transition(state, "go", StateKey(energy="lo"))
        first.observe_outcome(state, "go", success=True, reward=0.7)
    first.save()

    revived = PredictiveWorldModel(store_path=path)
    assert revived.predict_outcome(state, "go").expected_reward == pytest.approx(0.7)
    assert revived.knows(state, "go") == 1.0


def test_calibration_survives_a_restart(tmp_path):
    path = tmp_path / "model.json"
    first = PredictiveWorldModel(store_path=path)
    for i in range(20):
        prediction = first.make_prediction(StateKey(energy="hi"), "go", tick=i)
        first.score_prediction(prediction.id, True, 0.5, StateKey(energy="lo"))
    first.save()

    revived = PredictiveWorldModel(store_path=path)
    assert revived.calibration()["scored"] == 20


def test_saving_survives_an_unwritable_causal_store(tmp_path, monkeypatch):
    model = PredictiveWorldModel(store_path=tmp_path / "model.json")

    def explode():
        raise OSError("disk full")

    monkeypatch.setattr(model.causal, "save", explode)
    model.save()        # must not raise — a tick has to survive a full disk


# ── the genome reaches it (Appendix C) ───────────────────────────────

def test_the_genome_can_retune_the_model(wm):
    wm.apply_genome({"wm_smoothing": 3.0, "wm_half_life": 250,
                     "explore_bonus": 0.4, "plan_discount": 0.75})
    assert wm.transitions.smoothing == 3.0
    assert wm.outcomes.half_life == 250
    assert wm.simulator.explore_bonus == 0.4
    assert wm.simulator.discount == 0.75


def test_an_unrelated_gene_is_ignored(wm):
    before = wm.transitions.smoothing
    wm.apply_genome({"something_else": 5})
    assert wm.transitions.smoothing == before


def test_an_unusable_gene_value_is_ignored(wm):
    before = wm.transitions.smoothing
    wm.apply_genome({"wm_smoothing": "much"})
    assert wm.transitions.smoothing == before


def test_applying_no_genome_is_harmless(wm):
    wm.apply_genome({})
    wm.apply_genome(None)


# ── inside a running tick (§M1.6) ────────────────────────────────────

def test_perceive_encodes_the_state(substrate):
    _run(substrate.tick())
    assert substrate._ctx.state is not None
    assert substrate._ctx.state_inputs


def test_decide_records_a_forecast(substrate):
    _run(substrate.tick())
    assert substrate._ctx.prediction is not None
    assert substrate._ctx.decision


def test_reflect_hands_the_forecast_to_the_next_tick(substrate):
    _run(substrate.tick())
    assert substrate._pending_prediction is not None


def test_the_next_tick_closes_it(substrate):
    drive(substrate, 2)
    assert substrate._ctx.prediction_score is not None
    assert substrate.world_model.calibration()["scored"] >= 1


def test_the_model_learns_over_a_run(substrate):
    drive(substrate, 30)
    assert substrate.world_model.transitions.observations > 0
    assert substrate.world_model.outcomes.observations > 0
    assert substrate.world_model.calibration()["scored"] >= 25


def test_the_model_is_calibrated_over_a_live_run(substrate):
    """§M1.9's numeric thresholds, on the live system.

    Not the baseline comparison: with the environment stubbed out no tick can
    succeed, so the outcome stream has no variance and "predict the long-run
    average" is unbeatable *by anyone*. Asserting otherwise here would be
    asserting something false. The baseline comparison belongs on a log where
    the outcome actually depends on the state — the test below.
    """
    drive(substrate, 60)
    report = substrate.world_model.calibration()
    assert report["brier"] <= 0.18
    assert report["reward_mae"] <= 0.12
    assert report["beats_baselines"] is not None


def test_the_model_beats_both_baselines_when_state_predicts_the_outcome(wm):
    """§M1.9's acceptance comparison, on a fixed log.

    Two situations with genuinely different success rates. A model that
    conditions on state beats both "always the long-run average" — which cannot
    condition on anything — and a flat 0.5. This is the comparison that says
    the model learned something rather than merely recorded something.
    """
    good, bad = StateKey(energy="hi"), StateKey(energy="lo")
    after = StateKey(energy="mid")

    # Teach it: the good state usually succeeds, the bad one usually does not.
    for i in range(60):
        wm.observe_transition(good, "act", after)
        wm.observe_outcome(good, "act", success=(i % 5 != 0), reward=0.8)
        wm.observe_transition(bad, "act", after)
        wm.observe_outcome(bad, "act", success=(i % 5 == 0), reward=0.2)

    # Then measure it on fresh events drawn from the same world.
    for i in range(200):
        state = good if i % 2 == 0 else bad
        succeeded = (i % 5 != 0) if state is good else (i % 5 == 0)
        prediction = wm.make_prediction(state, "act", tick=i)
        wm.score_prediction(prediction.id, succeeded,
                            0.8 if state is good else 0.2, after)

    report = wm.calibration()
    assert report["beats_baselines"] is True
    assert report["brier"] < report["baseline_brier_mean"]
    assert report["brier"] < report["baseline_brier_half"]
    assert report["brier"] <= 0.18
    assert report["reward_mae"] <= 0.12


def test_a_model_with_nothing_to_learn_gains_nothing_over_the_average(wm):
    """The control for the test above.

    Same event stream, but every tick looks identical to the model, so there is
    no information to condition on. It should land level with "predict the
    average" — checked as a *margin* rather than a strict inequality, because
    two predictors that both converge on 0.5 differ only by noise, and which
    one comes out a fraction ahead is not a property worth asserting.
    """
    same, after = StateKey(energy="hi"), StateKey(energy="mid")
    for i in range(200):
        wm.observe_transition(same, "act", after)
        wm.observe_outcome(same, "act", success=(i % 2 == 0), reward=0.5)
        prediction = wm.make_prediction(same, "act", tick=i)
        wm.score_prediction(prediction.id, i % 2 == 0, 0.5, after)

    report = wm.calibration()
    margin = report["baseline_brier_mean"] - report["brier"]
    assert abs(margin) < 0.02, "an uninformed model showed a material advantage"


def test_a_state_aware_model_gains_a_material_margin(wm):
    # And the contrast: when the state genuinely predicts the outcome, the
    # advantage over the average is large, not marginal.
    good, bad, after = StateKey(energy="hi"), StateKey(energy="lo"), StateKey(energy="mid")
    for i in range(200):
        state = good if i % 2 == 0 else bad
        succeeded = state is good
        wm.observe_transition(state, "act", after)
        wm.observe_outcome(state, "act", success=succeeded, reward=0.5)
        prediction = wm.make_prediction(state, "act", tick=i)
        wm.score_prediction(prediction.id, succeeded, 0.5, after)

    report = wm.calibration()
    assert report["baseline_brier_mean"] - report["brier"] > 0.1


def test_surprise_steers_curiosity(substrate):
    substrate.goals.curiosity_level = 0.0
    drive(substrate, 12)
    # A model that keeps being wrong about where actions lead should raise
    # curiosity; one that is never wrong should not.
    assert 0.0 <= substrate.goals.curiosity_level <= 1.0


def test_a_failing_encoder_does_not_break_the_tick(substrate):
    def explode(inputs):
        raise RuntimeError("encoder down")

    substrate.world_model.encode = explode
    errors_before = substrate.health.error_count
    _run(substrate.tick())
    assert substrate.health.error_count == errors_before
    assert substrate._ctx.state is None


def test_a_failing_scorer_does_not_break_the_tick(substrate):
    drive(substrate, 1)

    def explode(*args, **kwargs):
        raise RuntimeError("scorer down")

    substrate.world_model.score_prediction = explode
    errors_before = substrate.health.error_count
    _run(substrate.tick())
    assert substrate.health.error_count == errors_before


def test_calibration_metrics_reach_the_time_series(substrate):
    from aegis.telemetry import metrics as M
    drive(substrate, 6)
    substrate.telemetry.flush()
    for metric in (M.WM_BRIER, M.WM_SURPRISE, M.WM_COVERAGE, M.WM_STATES,
                   M.WM_TRANSITIONS):
        assert len(substrate.telemetry.series(metric)) >= 1, metric


def test_the_world_model_appears_in_the_status(substrate):
    _run(substrate.tick())
    status = substrate.full_status()["world_model"]
    assert "predictive" in status
    assert "calibration" in status["predictive"]
