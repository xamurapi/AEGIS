"""pytest-bdd step definitions for tests/features/discovery.feature.

Executable Gherkin: every scenario drives the real engine, pool, symbolic search
and ledger, so the feature file is both the description of how the discovery
contour behaves and the test that it does.
"""
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aegis.layers.discovery import DiscoveryEngine
from aegis.telemetry.store import Telemetry
from aegis.util.quasirandom import hash_unit

scenarios("features/discovery.feature")


class _Knob:
    def __init__(self, value=0.15):
        self.value = value

    def apply(self, name, value):
        self.value = value

    def read(self):
        return self.value


class _Recorder:
    def __init__(self):
        self.calls = []

    def note_prior(self, formula, effect):
        self.calls.append((formula, effect))
        return True


def _plant(telemetry, count, start=0):
    for tick in range(start, start + count):
        surprise, brier = hash_unit("s", tick), hash_unit("b", tick)
        telemetry.record("aegis.wm.surprise", surprise, tick=tick)
        telemetry.record("aegis.wm.brier", brier, tick=tick)
        telemetry.record("aegis.reward.value",
                         2.5 * surprise - brier * brier
                         + 0.02 * (hash_unit("n", tick) - 0.5), tick=tick)
    telemetry.flush()


# ── given ────────────────────────────────────────────────────────────

@given("a discovery engine", target_fixture="ctx")
def _bare_engine(tmp_path):
    telemetry = Telemetry(tmp_path / "telemetry")
    return {"telemetry": telemetry, "knob": _Knob(0.15),
            "engine": DiscoveryEngine(directory=tmp_path / "discovery",
                                      telemetry=telemetry)}


@given("telemetry in which reward is 2.5 times surprise minus brier squared",
       target_fixture="ctx")
def _planted(tmp_path):
    telemetry = Telemetry(tmp_path / "telemetry")
    _plant(telemetry, 400)
    return {"telemetry": telemetry, "knob": _Knob(0.15), "ticks": 400,
            "engine": DiscoveryEngine(
                directory=tmp_path / "discovery", telemetry=telemetry,
                watched=("aegis.wm.surprise", "aegis.wm.brier"))}


@given("telemetry of six series unrelated to reward", target_fixture="ctx")
def _noise(tmp_path):
    telemetry = Telemetry(tmp_path / "telemetry")
    for tick in range(500):
        for index in range(6):
            telemetry.record(f"aegis.noise.v{index}",
                             hash_unit("noise", index, tick), tick=tick)
        telemetry.record("aegis.reward.value", hash_unit("reward", tick),
                         tick=tick)
    telemetry.flush()
    return {"telemetry": telemetry,
            "engine": DiscoveryEngine(
                directory=tmp_path / "discovery", telemetry=telemetry,
                watched=tuple(f"aegis.noise.v{i}" for i in range(6)))}


@given("a discovery confirmed in two separate windows")
def _confirmed(ctx):
    ledger = ctx["engine"].ledger
    ledger.propose({"id": "hyp_confirmed", "target": "aegis.reward.value",
                    "predictors": ["explore_bonus"]},
                   {"expr": "0.4 + 1.8*explore_bonus"})
    for window in [(0, 100), (200, 300)]:
        ledger.record_result("hyp_confirmed",
                             {"status": "supported", "effect_size": 1.0,
                              "p_value": 0.001}, window=window)
    ctx["identifier"] = "hyp_confirmed"


@given("a hypothesis that has been refuted")
def _refuted(ctx):
    ctx["engine"].ledger.propose({"id": "hyp_dead", "target": "y",
                                  "predictors": ["x"]}, {"expr": "x"})
    ctx["engine"].ledger.record_result("hyp_dead", {"status": "refuted"})


# ── when ─────────────────────────────────────────────────────────────

@when("the engine scans for hypotheses")
def _scan(ctx):
    ctx["found"] = ctx["engine"].scan(tick=ctx.get("ticks", 400))


@when("the engine fits a model")
def _fit(ctx):
    ctx["model"] = ctx["engine"].fit_next(tick=ctx.get("ticks", 400))


@when("the engine preregisters the experiment")
def _prereg(ctx):
    ctx["prereg"] = ctx["engine"].preregister_next(tick=ctx.get("ticks", 400))


@when(parsers.parse("another {count:d} ticks of the same relationship are recorded"))
def _more_data(ctx, count):
    _plant(ctx["telemetry"], count, start=ctx.get("ticks", 400))
    ctx["engine"].ingest()
    ctx["ticks"] = ctx.get("ticks", 400) + count


@when("the analysis is changed after the plan was frozen")
def _tamper(ctx):
    ctx["prereg"].analysis = "mann_whitney"


@when("the engine runs the observational experiment")
def _observational(ctx):
    ctx["result"] = ctx["engine"].run_observational(
        ctx["prereg"].hypothesis_id, tick=ctx.get("ticks", 700))


@when("the engine scans and fits repeatedly")
def _repeat(ctx):
    engine = ctx["engine"]
    for round_number in range(10):
        engine.scan(tick=500 + round_number)
        while engine.fit_next(tick=500 + round_number) is not None:
            pass
        while engine.preregister_next(tick=500 + round_number) is not None:
            pass


@when(parsers.parse('an intervention is attempted on "{variable}"'))
def _forbidden_intervention(ctx, variable):
    knob = ctx["knob"]
    ctx["started"] = ctx["engine"].start_intervention(
        "hyp_x", variable, (0.10, 0.20), tick=0, apply=knob.apply,
        read=knob.read)


@when(parsers.parse('an intervention is started on "{variable}"'))
def _intervention(ctx, variable):
    knob = ctx["knob"]
    ctx["started"] = ctx["engine"].start_intervention(
        "hyp_x", variable, (0.10, 0.20), tick=0, apply=knob.apply,
        read=knob.read)
    assert ctx["started"] is True


@when("one tick of the series runs")
def _one_tick(ctx):
    ctx["engine"].step_intervention(0, reward=1.0)


@when("health goes critical")
def _critical(ctx):
    ctx["outcome"] = ctx["engine"].step_intervention(1, reward=1.0,
                                                     health="critical")


@when("the engine applies what it has learned")
def _apply(ctx):
    ctx["world_model"] = _Recorder()
    ctx["applied"] = ctx["engine"].applications(
        tick=5, world_model=ctx["world_model"])


@when("the metric falls afterwards")
def _regression(ctx):
    ctx["sent"] = ctx["engine"].review_applications(1.0, 0.4, tick=9)


@when("the same hypothesis is proposed again")
def _repropose(ctx):
    ctx["reproposed"] = ctx["engine"].ledger.propose(
        {"id": "hyp_dead", "target": "y", "predictors": ["x"]})


# ── then ─────────────────────────────────────────────────────────────

@then("it should propose at least one hypothesis")
def _found_something(ctx):
    assert ctx["found"]


@then(parsers.parse('the formula should contain "{fragment}"'))
def _formula_contains(ctx, fragment):
    assert ctx["model"] is not None
    assert fragment in ctx["model"].expr


@then(parsers.parse("the model should explain at least {percent:d} percent of "
                    "held-out variance"))
def _r_squared(ctx, percent):
    assert ctx["model"].r2_valid >= percent / 100.0


@then("the discovery should be supported")
def _supported(ctx):
    assert ctx["result"]["status"] == "supported"
    assert ctx["engine"].ledger.get(
        ctx["prereg"].hypothesis_id).status == "supported"


@then(parsers.parse("it should have made at least {count:d} comparisons"))
def _comparisons(ctx, count):
    assert ctx["engine"].scanner.tested >= count


@then("no discovery should be supported")
def _nothing_supported(ctx):
    counts = ctx["engine"].ledger.counts()
    assert counts["supported"] == 0 and counts["law"] == 0


@then("the result should be invalid")
def _invalid(ctx):
    assert ctx["result"]["status"] == "invalid"


@then("the intervention should not start")
def _not_started(ctx):
    assert ctx["started"] is False
    assert ctx["engine"].intervention is None


@then("the parameter should be untouched")
def _untouched(ctx):
    assert ctx["knob"].value == 0.15


@then("the parameter should be at its experimental level")
def _experimental(ctx):
    assert ctx["knob"].value == 0.10


@then("the intervention should be aborted")
def _aborted(ctx):
    assert ctx["outcome"]["state"] == "aborted"


@then("the parameter should be back at its original value")
def _restored(ctx):
    assert ctx["knob"].value == 0.15


@then("the world model should have been told")
def _told(ctx):
    assert ctx["world_model"].calls


@then("the discovery should record where it was applied")
def _recorded(ctx):
    assert "world_model" in ctx["engine"].ledger.get(
        ctx["identifier"]).applications


@then("the discovery should be proposed again")
def _reproposed(ctx):
    assert ctx["identifier"] in ctx["sent"]
    assert ctx["engine"].ledger.get(ctx["identifier"]).status == "proposed"


@then("it should be refused")
def _refused(ctx):
    assert ctx["reproposed"] is None
