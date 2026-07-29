"""The whole loop, end to end (spec M7.10).

The two acceptance tests of the contour face in opposite directions and both
have to hold, because either alone is trivial to pass. An engine that registers
nothing passes the noise test. An engine that registers everything passes the
recovery test. Only the pair says the loop works:

* **A planted law is recovered.** ``reward = 2.5·surprise − brier²`` is written
  into the telemetry; the engine has to generate the hypothesis, recover the
  formula to ``R²_valid ≥ 0.9``, run the experiment and register the discovery.
* **Noise registers nothing.** Over a thousand comparisons of unrelated series,
  not one ``supported`` discovery.

The last section is the one the spec cares about most after those: knowledge
that changes nothing is a report. A confirmed discovery has to reach the systems
it is about, and the application has to be recorded well enough that a later
regression can send exactly the responsible claim back for re-testing.
"""
import pytest

from aegis.layers.discovery import DiscoveryEngine
from aegis.telemetry.store import Telemetry
from aegis.util.quasirandom import hash_unit


def _planted(telemetry, count=400, start=0):
    """``reward = 2.5·surprise − brier²`` plus a little noise."""
    for tick in range(start, start + count):
        surprise = hash_unit("s", tick)
        brier = hash_unit("b", tick)
        telemetry.record("aegis.wm.surprise", surprise, tick=tick)
        telemetry.record("aegis.wm.brier", brier, tick=tick)
        telemetry.record("aegis.reward.value",
                         2.5 * surprise - brier * brier
                         + 0.02 * (hash_unit("n", tick) - 0.5), tick=tick)
    telemetry.flush()


def _noise(telemetry, count=500, series=6):
    for tick in range(count):
        for index in range(series):
            telemetry.record(f"aegis.noise.v{index}",
                             hash_unit("noise", index, tick), tick=tick)
        telemetry.record("aegis.reward.value", hash_unit("reward", tick),
                         tick=tick)
    telemetry.flush()


@pytest.fixture
def telemetry(tmp_path):
    return Telemetry(tmp_path / "telemetry")


@pytest.fixture
def engine(tmp_path, telemetry):
    return DiscoveryEngine(directory=tmp_path / "discovery", telemetry=telemetry,
                           watched=("aegis.wm.surprise", "aegis.wm.brier"))


# ── the recovery test ────────────────────────────────────────────────

def test_a_planted_law_is_recovered_and_registered(engine, telemetry):
    """The spec's key end-to-end test (M7.10). It proves the contour is wired
    together, not merely that five modules each work alone."""
    _planted(telemetry, count=400)

    found = engine.scan(tick=400)
    assert found, "no hypothesis was generated from a planted law"

    model = engine.fit_next(tick=400)
    assert model is not None
    assert model.r2_valid >= 0.9, f"R²_valid was {model.r2_valid}"
    assert set(model.terms) == {"aegis.wm.surprise", "aegis.wm.brier^2"}

    prereg = engine.preregister_next(tick=400)
    assert prereg is not None and prereg.intact()

    _planted(telemetry, count=300, start=400)
    engine.ingest()
    result = engine.run_observational(prereg.hypothesis_id, tick=700)
    assert result["status"] == "supported"
    assert engine.ledger.get(prereg.hypothesis_id).status == "supported"


def test_the_recovered_formula_is_written_down(engine, telemetry):
    _planted(telemetry)
    engine.scan(tick=400)
    model = engine.fit_next(tick=400)
    assert "aegis.wm.surprise" in model.expr and "brier^2" in model.expr


def test_the_coefficients_are_the_planted_ones(engine, telemetry):
    _planted(telemetry)
    engine.scan(tick=400)
    model = engine.fit_next(tick=400)
    by_term = dict(zip(model.terms, model.params))
    assert by_term["aegis.wm.surprise"] == pytest.approx(2.5, abs=0.05)
    assert by_term["aegis.wm.brier^2"] == pytest.approx(-1.0, abs=0.05)


# ── the noise test ───────────────────────────────────────────────────

def test_noise_registers_no_discovery(tmp_path, telemetry):
    """The other half of the acceptance (M7.10). Over a thousand comparisons of
    unrelated series, nothing may reach ``supported``."""
    _noise(telemetry, count=500)
    engine = DiscoveryEngine(
        directory=tmp_path / "discovery", telemetry=telemetry,
        watched=tuple(f"aegis.noise.v{index}" for index in range(6)))

    for round_number in range(10):
        engine.scan(tick=500 + round_number)
        while engine.fit_next(tick=500 + round_number) is not None:
            pass
        while engine.preregister_next(tick=500 + round_number) is not None:
            pass

    assert engine.scanner.tested >= 1000, \
        f"only {engine.scanner.tested} comparisons — the test proves little"
    assert engine.ledger.counts()["supported"] == 0
    assert engine.ledger.counts()["law"] == 0


def test_noise_does_not_even_produce_hypotheses(tmp_path, telemetry):
    _noise(telemetry, count=500)
    engine = DiscoveryEngine(
        directory=tmp_path / "discovery", telemetry=telemetry,
        watched=tuple(f"aegis.noise.v{index}" for index in range(6)))
    assert engine.scan(tick=500) == []


# ── the loop's own guards ────────────────────────────────────────────

def test_too_little_data_produces_no_hypotheses(engine, telemetry):
    _planted(telemetry, count=20)
    assert engine.scan(tick=20) == []


def test_an_engine_with_no_telemetry_ingests_nothing(tmp_path):
    engine = DiscoveryEngine(directory=tmp_path / "d")
    assert engine.ingest() == 0
    assert engine.scan(tick=1) == []


def test_a_refuted_hypothesis_is_not_proposed_again(engine, telemetry):
    """The anti-rediscovery guard, at the level of the loop rather than the
    ledger: a rescan must not queue what has already been answered."""
    _planted(telemetry)
    found = engine.scan(tick=400)
    engine.fit_next(tick=400)
    prereg = engine.preregister_next(tick=400)
    engine.ledger.record_result(prereg.hypothesis_id, {"status": "refuted"},
                                tick=401)

    engine.pending = []
    again = engine.scan(tick=402)
    assert all(item.id != prereg.hypothesis_id for item in again)


def test_the_same_hypothesis_is_not_queued_twice(engine, telemetry):
    """Rescanning unchanged data must add nothing.

    This is also the test that catches re-ingestion: if the pool re-appended
    the telemetry it already held, the second scan would run on doubled rows,
    reach different lags and queue a different set of hypotheses — which is
    what it did before the watermark went in.
    """
    _planted(telemetry)
    engine.scan(tick=400)
    before = [item.id for item in engine.pending]
    rows_before = engine.pool.row_count("telemetry")

    engine.scan(tick=401)
    assert [item.id for item in engine.pending] == before
    assert engine.pool.row_count("telemetry") == rows_before


def test_fitting_with_nothing_pending_yields_nothing(engine):
    assert engine.fit_next(tick=1) is None


def test_preregistering_with_no_model_yields_nothing(engine, telemetry):
    _planted(telemetry)
    engine.scan(tick=400)
    assert engine.preregister_next(tick=400) is None


def test_running_an_experiment_for_an_unknown_plan_is_invalid(engine):
    assert engine.run_observational("nobody", tick=1)["status"] == "invalid"


def test_a_hypothesis_that_cannot_be_fitted_leaves_the_queue(engine, telemetry):
    """Otherwise ``fit_next`` returns to the same unfittable hypothesis every
    time it is called and the queue never advances."""
    _planted(telemetry, count=400)
    engine.scan(tick=400)
    engine.pool.data["telemetry"] = engine.pool.data["telemetry"][:5]
    before = len(engine.pending)
    engine.fit_next(tick=400)
    assert len(engine.pending) < before


# ── the cortex path ──────────────────────────────────────────────────

def test_a_formal_hypothesis_can_be_accepted(engine, telemetry):
    _planted(telemetry)
    engine.ingest()
    item = engine.accept_formal("aegis.reward.value ~ f(aegis.wm.surprise@lag1)",
                                tick=5)
    assert item is not None and item.origin == "cortex"
    assert item in engine.pending


def test_a_statement_outside_the_grammar_is_refused(engine, telemetry):
    _planted(telemetry)
    engine.ingest()
    assert engine.accept_formal("reward := anything I like") is None
    assert engine.accept_formal("aegis.reward.value ~ f(invented)") is None


def test_a_refuted_formal_hypothesis_is_refused(engine, telemetry):
    _planted(telemetry)
    engine.ingest()
    item = engine.accept_formal("aegis.reward.value ~ f(aegis.wm.surprise@lag1)")
    engine.ledger.propose(item)
    engine.ledger.record_result(item.id, {"status": "refuted"})
    engine.pending = []
    assert engine.accept_formal(
        "aegis.reward.value ~ f(aegis.wm.surprise@lag1)") is None


# ── interventions through the engine ─────────────────────────────────

class _Knob:
    def __init__(self, value=0.15):
        self.value = value

    def apply(self, name, value):
        self.value = value

    def read(self):
        return self.value


def test_an_intervention_on_an_uncontrolled_variable_never_starts(engine):
    knob = _Knob()
    assert engine.start_intervention("hyp", "ETHICAL_THRESHOLD_AUTO",
                                     (0.1, 0.2), tick=1,
                                     apply=knob.apply, read=knob.read) is False
    assert engine.intervention is None
    assert knob.value == 0.15


def test_only_one_intervention_runs_at_a_time(engine):
    knob = _Knob()
    assert engine.start_intervention("hyp_a", "explore_bonus", (0.1, 0.2),
                                     tick=1, apply=knob.apply,
                                     read=knob.read) is True
    assert engine.start_intervention("hyp_b", "plan_beam", (3, 5), tick=2,
                                     apply=knob.apply, read=knob.read) is False


def test_stepping_with_no_intervention_is_inactive(engine):
    assert engine.step_intervention(1, reward=1.0)["state"] == "inactive"


def test_a_critical_health_reading_ends_the_series_and_restores(engine):
    knob = _Knob(0.15)
    engine.start_intervention("hyp_a", "explore_bonus", (0.10, 0.20), tick=0,
                              apply=knob.apply, read=knob.read)
    engine.step_intervention(0, reward=1.0)
    outcome = engine.step_intervention(1, reward=1.0, health="critical")
    assert outcome["state"] == "aborted"
    assert knob.value == 0.15
    assert engine.intervention is None


# ── applying what was learned (M7.9) ─────────────────────────────────

class _Recorder:
    def __init__(self):
        self.calls = []

    def note_prior(self, formula, effect):
        self.calls.append(("prior", formula, effect))
        return True

    def note_discovery(self, *args):
        self.calls.append(("discovery", args))
        return True

    def narrow_gene(self, name, effect):
        self.calls.append(("gene", name, effect))
        return True


def _confirm(engine, identifier="hyp_applied"):
    hypothesis = {"id": identifier, "target": "aegis.reward.value",
                  "predictors": ["explore_bonus"]}
    engine.ledger.propose(hypothesis, {"expr": "0.5 + 2*explore_bonus"})
    for window in [(0, 100), (200, 300)]:
        engine.ledger.record_result(identifier,
                                    {"status": "supported", "effect_size": 1.0,
                                     "p_value": 0.001}, window=window)
    return identifier


def test_a_confirmed_discovery_reaches_the_systems_it_is_about(engine):
    """Knowledge that changes nothing is a claim the world never has to answer
    for (M7.9)."""
    identifier = _confirm(engine)
    world_model, policy, evolution = _Recorder(), _Recorder(), _Recorder()
    applied = engine.applications(tick=5, world_model=world_model, policy=policy,
                                  evolution=evolution)
    assert applied
    assert world_model.calls and policy.calls and evolution.calls
    assert engine.ledger.get(identifier).applications


def test_an_unconfirmed_discovery_is_not_applied(engine):
    engine.ledger.propose({"id": "hyp_open", "target": "y",
                           "predictors": ["x"]}, {"expr": "x"})
    assert engine.applications(tick=1, world_model=_Recorder()) == []


def test_an_application_that_raises_does_not_take_the_tick_with_it(engine):
    class _Broken:
        def note_prior(self, formula, effect):
            raise RuntimeError("the world model is busy")

    _confirm(engine)
    assert engine.applications(tick=1, world_model=_Broken()) == []


def test_a_system_that_offers_no_hook_is_skipped(engine):
    _confirm(engine)
    assert engine.applications(tick=1, world_model=object()) == []


def test_a_regression_after_an_application_sends_it_back_for_re_testing(engine):
    """The reverse check of M7.9: a discovery whose application made things
    worse is not knowledge yet, whatever the experiment said."""
    identifier = _confirm(engine)
    engine.applications(tick=5, world_model=_Recorder())
    sent = engine.review_applications(metric_before=1.0, metric_after=0.5, tick=9)
    assert identifier in sent
    assert engine.ledger.get(identifier).status == "proposed"


def test_no_regression_leaves_applied_discoveries_alone(engine):
    identifier = _confirm(engine)
    engine.applications(tick=5, world_model=_Recorder())
    assert engine.review_applications(1.0, 1.5, tick=9) == []
    assert engine.ledger.get(identifier).status == "replicated"


# ── observability and persistence ────────────────────────────────────

def test_the_engine_records_its_metrics(engine, telemetry):
    _planted(telemetry)
    engine.scan(tick=400)
    assert telemetry.series("aegis.disc.hypotheses_tested").last() is not None
    assert telemetry.series("aegis.disc.fdr_rejections").last() is not None


def test_the_status_describes_the_whole_loop(engine, telemetry):
    _planted(telemetry)
    engine.scan(tick=400)
    status = engine.status()
    for key in ("scans", "fits", "experiments", "hypotheses_tested",
                "fdr_rejections", "pending", "discoveries", "supported",
                "replicated", "laws", "refuted", "pool"):
        assert key in status, key


def test_the_engine_saves_and_reloads(tmp_path, telemetry):
    _planted(telemetry)
    engine = DiscoveryEngine(directory=tmp_path / "d", telemetry=telemetry,
                             watched=("aegis.wm.surprise", "aegis.wm.brier"))
    engine.scan(tick=400)
    engine.ledger.propose({"id": "hyp_saved", "target": "y",
                           "predictors": ["x"]}, {"expr": "x"})
    assert engine.save() is True

    reloaded = DiscoveryEngine(directory=tmp_path / "d", telemetry=telemetry)
    assert reloaded.ledger.get("hyp_saved") is not None
    assert reloaded.pool.row_count("telemetry") > 0


# ── which plans are still worth stepping ─────────────────────────────

def test_a_plan_whose_discovery_is_closed_is_no_longer_active(engine, telemetry):
    """`run_experiment` is offered whenever a plan is open. Leaving refuted and
    invalid plans in that list would make the action permanently available and
    permanently a no-op — the system would keep choosing to do nothing."""
    _planted(telemetry)
    engine.scan(tick=400)
    engine.fit_next(tick=400)
    prereg = engine.preregister_next(tick=400)
    assert prereg in engine.active_preregistrations()

    engine.ledger.record_result(prereg.hypothesis_id, {"status": "refuted"})
    assert prereg not in engine.active_preregistrations()


def test_a_plan_with_no_ledger_entry_is_still_active(engine, telemetry):
    from aegis.layers.discovery import preregister

    _planted(telemetry)
    engine.preregs["hyp_orphan"] = preregister({"id": "hyp_orphan"}, None, tick=1)
    assert engine.preregs["hyp_orphan"] in engine.active_preregistrations()


# ── the queue does not grow by repetition ────────────────────────────

def test_accepting_the_same_formal_hypothesis_twice_queues_it_once(engine, telemetry):
    _planted(telemetry)
    engine.ingest()
    statement = "aegis.reward.value ~ f(aegis.wm.surprise@lag1)"
    first = engine.accept_formal(statement, tick=1)
    before = len(engine.pending)
    second = engine.accept_formal(statement, tick=2)
    assert first.id == second.id
    assert len(engine.pending) == before


def test_an_unfittable_hypothesis_leaves_the_others_alone(engine, telemetry):
    """It is dropped from the queue so the loop can advance; everything else
    has to survive that. Dropping the *rest* instead would silently discard the
    hypotheses the scan had just paid for."""
    _planted(telemetry, count=400)
    engine.scan(tick=400)
    assert len(engine.pending) >= 2
    identifiers = [item.id for item in engine.pending]

    engine.pool.data["telemetry"] = engine.pool.data["telemetry"][:5]
    engine.fit_next(tick=400)

    remaining = [item.id for item in engine.pending]
    assert remaining == identifiers[1:], remaining


# ── the experiment finds the plan it was asked for ───────────────────

def test_an_experiment_runs_against_the_hypothesis_it_names(engine, telemetry):
    _planted(telemetry)
    engine.scan(tick=400)
    engine.fit_next(tick=400)
    prereg = engine.preregister_next(tick=400)
    _planted(telemetry, count=300, start=400)
    engine.ingest()

    assert engine.run_observational(prereg.hypothesis_id,
                                    tick=700)["status"] == "supported"
    assert engine.run_observational("hyp_that_does_not_exist",
                                    tick=700)["status"] == "invalid"


def test_an_experiment_with_a_plan_but_no_model_is_invalid(engine, telemetry):
    from aegis.layers.discovery import preregister

    _planted(telemetry)
    engine.scan(tick=400)
    item = engine.pending[0]
    engine.preregs[item.id] = preregister(item, None, tick=400)
    assert engine.run_observational(item.id, tick=700)["status"] == "invalid"


def test_the_recorded_window_begins_after_everything_already_counted(engine, telemetry):
    """Which rows a confirmation covers *is* the replication rule. A window
    computed over the wrong side of the boundary would overlap the previous one
    and every confirmation would be discarded as a re-read."""
    _planted(telemetry, count=400)
    engine.scan(tick=400)
    engine.fit_next(tick=400)
    prereg = engine.preregister_next(tick=400)

    _planted(telemetry, count=300, start=400)
    engine.ingest()
    engine.run_observational(prereg.hypothesis_id, tick=700)
    windows = engine.ledger.get(prereg.hypothesis_id).windows
    assert windows and windows[0][0] > 400


# ── the executors the registry calls ─────────────────────────────────

def test_the_fit_executor_reports_whether_it_fitted_anything(engine, telemetry):
    assert engine.fit(tick=1) == {"fitted": False, "pending": 0}

    _planted(telemetry)
    engine.scan(tick=400)
    report = engine.fit(tick=400)
    assert report["fitted"] is True
    assert report["expr"] and report["r2_valid"] > 0.9


def test_stepping_an_experiment_does_not_abort_by_default(engine):
    """The defaults are the healthy case. A kill-switch flag defaulting to true
    would abort every series the moment a caller omitted the argument."""
    knob = _Knob(0.15)
    engine.start_intervention("hyp_a", "explore_bonus", (0.10, 0.20), tick=0,
                              apply=knob.apply, read=knob.read)
    assert engine.step_experiment(tick=0, reward=1.0)["state"] == "running"
    assert knob.value == 0.10


def test_a_finished_intervention_hands_the_action_back(engine, telemetry):
    """One action, two designs. With no series running the action has to fall
    through to the observational plans, or a finished intervention would leave
    `run_experiment` doing nothing forever."""
    _planted(telemetry, count=400)
    engine.scan(tick=400)
    engine.fit_next(tick=400)
    engine.preregister_next(tick=400)
    _planted(telemetry, count=300, start=400)
    engine.ingest()

    assert engine.intervention is None
    assert engine.step_experiment(tick=700)["status"] in ("supported", "refuted")


def test_an_experiment_with_nothing_to_do_is_idle(engine):
    assert engine.step_experiment(tick=1) == {"state": "idle"}


def test_an_intervention_for_an_unknown_hypothesis_still_records_the_id(engine):
    """The id is what the ledger and the preregistration log are keyed on. An
    intervention started before its hypothesis was queued must still be
    attributable."""
    knob = _Knob()
    assert engine.start_intervention("hyp_not_queued", "explore_bonus",
                                     (0.1, 0.2), tick=0, apply=knob.apply,
                                     read=knob.read) is True
    assert engine.preregs["hyp_not_queued"].hypothesis_id == "hyp_not_queued"


# ── applying knowledge: each hook is checked on its own ──────────────

@pytest.mark.parametrize("system",
                         ["world_model", "policy", "evolution", "resources"])
def test_a_missing_system_is_simply_not_applied_to(engine, system):
    _confirm(engine)
    assert engine.applications(tick=1, **{system: None}) == []


@pytest.mark.parametrize("system",
                         ["world_model", "policy", "evolution", "resources"])
def test_a_system_without_the_hook_is_simply_not_applied_to(engine, system):
    _confirm(engine)
    assert engine.applications(tick=1, **{system: object()}) == []


def test_each_hook_is_reached_when_it_is_there(engine):
    _confirm(engine)
    recorder = _Recorder()
    applied = engine.applications(tick=1, world_model=recorder, policy=recorder,
                                  evolution=recorder, resources=recorder)
    kinds = {entry.split(chr(8594))[1].split(":")[0] for entry in applied}
    assert kinds == {"world_model", "policy", "evolution", "resources"}


def test_a_discovery_with_no_predictors_is_not_applied(engine):
    """A law with no left-hand side or no right-hand side is not a law, and
    handing one to the world model would be handing it an empty claim."""
    engine.ledger.propose({"id": "hyp_empty", "target": "y", "predictors": []},
                          {"expr": "0"})
    for window in [(0, 100), (200, 300)]:
        engine.ledger.record_result("hyp_empty",
                                    {"status": "supported", "effect_size": 1.0,
                                     "p_value": 0.001}, window=window)
    assert engine.applications(tick=1, world_model=_Recorder()) == []


def test_only_applied_discoveries_are_sent_back_after_a_regression(engine):
    """A discovery that was never applied cannot be the reason the metric fell,
    and re-testing it would spend an experiment answering a question nobody
    asked."""
    _confirm(engine, "hyp_applied")
    _confirm(engine, "hyp_untouched")
    engine.ledger.note_application("hyp_applied", "world_model", tick=1)

    sent = engine.review_applications(1.0, 0.4, tick=9)
    assert sent == ["hyp_applied"]
    assert engine.ledger.get("hyp_untouched").status == "replicated"


# ── the counts a panel reads ─────────────────────────────────────────

def test_replicated_counts_the_laws_as_well(engine):
    """A law is a replicated discovery that kept replicating. Reporting the two
    separately and then subtracting would make the number fall as the evidence
    grew."""
    _confirm(engine, "hyp_replicated")
    engine.ledger.propose({"id": "hyp_law", "target": "y", "predictors": ["x"]},
                          {"expr": "x"})
    for window in [(0, 100), (200, 300), (400, 500)]:
        engine.ledger.record_result("hyp_law",
                                    {"status": "supported", "effect_size": 1.0,
                                     "p_value": 0.001}, window=window)

    counts = engine.ledger.counts()
    assert counts["law"] == 1 and counts["replicated"] == 1
    assert engine.status()["replicated"] == 2


def test_an_experiment_needs_the_hypothesis_it_names_to_be_queued(engine, telemetry):
    """The lookup has to match, not merely find something.

    With one hypothesis queued a lookup that took anything *other* than the
    named one finds nothing and reports invalid; with several it would quietly
    score the wrong formula against the wrong plan. This is the shape that
    makes the first case visible.
    """
    _planted(telemetry, count=400)
    engine.scan(tick=400)
    engine.fit_next(tick=400)
    prereg = engine.preregister_next(tick=400)

    # Exactly one candidate left, so a mismatched lookup has nothing to fall
    # back on and the failure is unambiguous.
    engine.pending = [item for item in engine.pending
                      if item.id == prereg.hypothesis_id]
    assert len(engine.pending) == 1

    _planted(telemetry, count=300, start=400)
    engine.ingest()
    assert engine.run_observational(prereg.hypothesis_id,
                                    tick=700)["status"] == "supported"


def test_an_intervention_is_registered_against_the_hypothesis_it_names(engine, telemetry):
    """The plan carries the id the caller asked for, whether or not that
    hypothesis happens to be queued — it is what the ledger and the
    preregistration log are keyed on."""
    _planted(telemetry, count=400)
    engine.scan(tick=400)
    wanted = engine.pending[0].id
    knob = _Knob()

    assert engine.start_intervention(wanted, "explore_bonus", (0.1, 0.2),
                                     tick=0, apply=knob.apply,
                                     read=knob.read) is True
    assert engine.preregs[wanted].hypothesis_id == wanted
    assert engine.ledger.get(wanted) is not None


def test_the_published_replication_count_includes_the_laws(engine, telemetry):
    """The telemetry series, not only the status dict. A law is a replicated
    discovery that kept replicating, so subtracting the laws would make the
    published curve fall exactly as the evidence accumulated — the worst
    possible direction for a metric the discovery engine itself reads back."""
    from aegis.telemetry.metrics import DISC_REPLICATED

    _confirm(engine, "hyp_replicated")
    engine.ledger.propose({"id": "hyp_law", "target": "y", "predictors": ["x"]},
                          {"expr": "x"})
    for window in [(0, 100), (200, 300), (400, 500)]:
        engine.ledger.record_result("hyp_law",
                                    {"status": "supported", "effect_size": 1.0,
                                     "p_value": 0.001}, window=window)

    engine.publish_metrics(tick=10)
    counts = engine.ledger.counts()
    assert counts["law"] == 1 and counts["replicated"] == 1
    assert telemetry.series(DISC_REPLICATED).last() == 2.0
