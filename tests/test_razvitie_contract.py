"""One scenario per line of ``развитие.txt`` (spec §VII.2).

The development text is short and every line of it is a claim about what the
system should be able to do. This file is the contract: each test names one
line and drives the real contour that answers it, end to end. Nothing here is a
unit test — the unit tests live beside each module and are more thorough. What
these check is that the contours are *connected*: that a forecast reaches a
plan, that an experience reaches a rule that reaches a decision, that a lease
is what makes an action run.

A contour can be perfectly implemented and completely disconnected, and every
unit test in the tree would stay green. That is the failure this file exists to
catch.
"""
import pytest

from aegis.layers.discovery import DiscoveryEngine
from aegis.layers.evolution.genome import Genome
from aegis.layers.evolution.population import Population
from aegis.layers.motivation.priority import Candidate, PriorityScheduler
from aegis.layers.motivation.resources import ResourceCost, ResourceManager
from aegis.layers.policy import BehaviourPolicy
from aegis.layers.reasoning import ReasoningEngine
from aegis.layers.world.state import StateKey
from aegis.layers.world_model import PredictiveWorldModel
from aegis.telemetry.store import Telemetry
from aegis.util.quasirandom import hash_unit

STATE = StateKey(energy="mid", error="low", mood="curious", mode="focused",
                 focus_kind="knowledge", perf="flat", load="lo")
OTHER = StateKey(energy="hi", error="none", mood="calm", mode="focused",
                 focus_kind="competence", perf="up", load="lo")


# ── «модель мира → прогноз → решение» ────────────────────────────────

def test_a_forecast_is_made_before_the_action_and_scored_after_it(tmp_path):
    """The first line of the text. A forecast that were only produced after the
    outcome would be a description, not a prediction, and the error that drives
    learning would be unmeasurable."""
    wm = PredictiveWorldModel(store_path=tmp_path / "wm" / "model.json")
    for _ in range(20):
        wm.observe_outcome(STATE.key(), "run_benchmark", True, reward=1.0)
        wm.observe_transition(STATE.key(), "run_benchmark", OTHER.key())

    forecast = wm.make_prediction(STATE, "run_benchmark", tick=1)
    assert 0.0 <= forecast.p_success <= 1.0
    assert forecast.state == STATE.key()

    score = wm.score_prediction(forecast.id, True, 1.0, OTHER)
    assert score is not None
    assert wm.calibration()["scored"] == 1


def test_the_forecast_is_what_a_decision_is_made_from(tmp_path):
    """The arrow from prognosis to decision: an action the model expects to
    fail must not rank above one it expects to succeed."""
    wm = PredictiveWorldModel(store_path=tmp_path / "wm" / "model.json")
    for _ in range(30):
        wm.observe_outcome(STATE.key(), "good_action", True, reward=1.0)
        wm.observe_outcome(STATE.key(), "bad_action", False, reward=0.0)

    good = wm.predict_outcome(STATE.key(), "good_action")
    bad = wm.predict_outcome(STATE.key(), "bad_action")
    assert good.p_success > bad.p_success
    assert good.expected_reward > bad.expected_reward


# ── «действие → результат → оценка → знание → изменение поведения» ───

def test_experience_becomes_a_rule_that_changes_the_choice(tmp_path):
    """The fifth arrow, which is the one the text is really about. Everything
    before it existed already; what did not was behaviour that provably
    changed because of what was learned."""
    policy = BehaviourPolicy(store_dir=tmp_path / "policy")
    for tick in range(60):
        policy.observe(STATE.key(), "failing_action", reward=0.0, success=False,
                       tick=tick)
        policy.observe(STATE.key(), "working_action", reward=1.0, success=True,
                       tick=tick)

    assert policy.delta(STATE.key(), "working_action") > \
        policy.delta(STATE.key(), "failing_action"), \
        "experience did not move the preference at all"

    mined = policy.mine(100, safety_critical=[])
    assert mined is not None


def test_a_rule_never_suppresses_safety_critical_work(tmp_path):
    """The limit on the fifth arrow. Behaviour may change; keeping itself alive
    may not become negotiable (§M3.5)."""
    policy = BehaviourPolicy(store_dir=tmp_path / "policy")
    protected = ["checkpoint", "health_check"]
    for tick in range(60):
        policy.observe(STATE.key(), "checkpoint", reward=0.0, success=False,
                       tick=tick)
    policy.mine(100, safety_critical=protected)
    for rule in policy.lifecycle.ordered():
        assert rule.action not in protected or rule.effect != "suppress"


# ── «цель → ценность → приоритет → ресурс → действие» ────────────────

def test_the_motivation_chain_ends_in_a_lease(tmp_path):
    """The two links the text names that did not exist: priority as a number,
    and resource as something that runs out. Without the last one motivation is
    an opinion — everything is equally wanted and nothing is paid for."""
    resources = ResourceManager(store_path=tmp_path / "budgets.json",
                                limits={"llm_tokens": 10_000})
    priority = PriorityScheduler(resources=resources)

    candidates = [Candidate(objective="cheap_and_dull", value=0.1),
                  Candidate(objective="valuable", value=1.0)]
    ordered = priority.order(candidates)
    assert ordered[0].objective == "valuable", "value did not become priority"

    lease = resources.reserve(ResourceCost(llm_tokens=500), ordered[0].objective,
                              priority=ordered[0].priority)
    assert lease is not None, "priority did not become a resource"
    resources.commit(lease, ResourceCost(llm_tokens=120))
    assert resources.spent("llm_tokens") == 120


def test_without_a_resource_the_action_does_not_happen(tmp_path):
    resources = ResourceManager(store_path=tmp_path / "budgets.json",
                                limits={"llm_tokens": 0})
    assert resources.reserve(ResourceCost(llm_tokens=100), "anything") is None


# ── «создать 10 вариантов → проверить → оставить лучший» ─────────────

def test_a_generation_is_ten_variants_and_the_best_is_kept():
    """"Not self-modification, but automatic evolution" — the text's own
    correction. One mutation judged against one benchmark is the former."""
    population = Population(size=10)
    elites = [Genome({"plan_beam": 4}), Genome({"plan_beam": 7})]
    genomes = population.build(elites, generation=1)
    assert len(genomes) == 10

    class _Report:
        def __init__(self, identifier, fitness):
            self.genome_id = identifier
            self.fitness = fitness

    reports = [_Report(f"g{index}", index / 10.0) for index in range(10)]
    ranked = population.rank(reports)
    assert ranked[0][1].fitness == pytest.approx(0.9)


def test_evolution_cannot_reach_what_must_not_change():
    """The one thing automatic evolution must never be automatic about."""
    from aegis.layers.evolution.genome import GENES
    from aegis.safety.immutable import IMMUTABLE_PARAMS, normalize

    immutable = {normalize(name) for name in IMMUTABLE_PARAMS}
    assert not [gene for gene in GENES if normalize(gene.name) in immutable]


# ── «Kimi K3/DeepSeek/GPT как кору мозга» ────────────────────────────

def test_the_provider_is_changed_by_configuration_and_not_by_code(monkeypatch):
    """The text names a specific model; the point of the line is that naming a
    model must not be a code change. Kimi is reachable as configuration."""
    from aegis.cortex.router import Cortex

    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.setenv("KIMI_MODEL", "kimi-latest")
    cortex = Cortex()
    cortex.configure_routes({"deep": ["kimi"], "fast": ["kimi"],
                             "code": ["kimi"], "judge": ["kimi"]})
    assert "kimi" in cortex.providers


def test_the_core_still_thinks_with_no_model_at_all(tmp_path):
    """"A cortex on top of a core" — on top of, not in place of. Every contour
    has a deterministic path, so a system with no key configured still runs."""
    engine = ReasoningEngine(store_path=tmp_path / "strategies.json")
    engine.solve(8)
    assert engine.attempts == 8
    assert engine.solved_count > 0, "nothing was solved without a model"


# ── «найти слабость → новые алгоритмы → проверить → оставить» ────────

def test_a_weakness_is_found_and_a_strategy_is_written_for_it(tmp_path):
    """Thinking as data. The system has to be able to say where it is bad, in
    terms specific enough to write a strategy against."""
    engine = ReasoningEngine(store_path=tmp_path / "strategies.json")
    engine.solve(240)

    weaknesses = engine.detector.scan(engine.results)
    assert weaknesses, "no weakness was found in 240 attempts"

    candidates = engine.propose_strategy(tick=1)
    assert candidates, "no strategy was written for the weakness"


def test_a_strategy_is_only_kept_if_it_helps_on_held_out(tmp_path):
    """"Check, then keep the improvement" — checked on problems it was not
    tuned on, or the gain is the candidate fitting its own examples."""
    engine = ReasoningEngine(store_path=tmp_path / "strategies.json")
    engine.solve(240)
    engine.detector.scan(engine.results)
    engine.propose_strategy(tick=1)

    verdict = engine.evaluate_candidate(tick=2)
    assert verdict is not None
    assert "holdout_gain" in verdict
    if verdict["accepted"]:
        assert verdict["holdout_gain"] >= 0.0


# ── «данные → гипотеза → мат. модель → эксперимент → открытие» ───────

def test_a_law_planted_in_the_telemetry_is_discovered(tmp_path):
    """The last line of the text, end to end: the system produces knowledge
    nobody gave it, written as a formula and confirmed out of sample."""
    telemetry = Telemetry(tmp_path / "telemetry")
    for tick in range(700):
        surprise, brier = hash_unit("s", tick), hash_unit("b", tick)
        telemetry.record("aegis.wm.surprise", surprise, tick=tick)
        telemetry.record("aegis.wm.brier", brier, tick=tick)
        telemetry.record("aegis.reward.value",
                         2.5 * surprise - brier * brier
                         + 0.02 * (hash_unit("n", tick) - 0.5), tick=tick)
    telemetry.flush()

    engine = DiscoveryEngine(directory=tmp_path / "discovery",
                             telemetry=telemetry,
                             watched=("aegis.wm.surprise", "aegis.wm.brier"))
    assert engine.scan(tick=400), "no hypothesis came out of the data"

    model = engine.fit_next(tick=400)
    assert model is not None, "no mathematical model was produced"
    assert model.r2_valid >= 0.9
    assert model.expr, "the model is not written down as a formula"

    prereg = engine.preregister_next(tick=400)
    assert prereg is not None and prereg.intact()

    result = engine.run_observational(prereg.hypothesis_id, tick=700)
    assert result["status"] == "supported"
    assert engine.ledger.get(prereg.hypothesis_id).status == "supported"


def test_noise_produces_no_discovery(tmp_path):
    """The other half of the same line. Knowledge that is produced from
    anything is not knowledge."""
    telemetry = Telemetry(tmp_path / "telemetry")
    for tick in range(600):
        for index in range(6):
            telemetry.record(f"aegis.noise.v{index}",
                             hash_unit("noise", index, tick), tick=tick)
        telemetry.record("aegis.reward.value", hash_unit("reward", tick),
                         tick=tick)
    telemetry.flush()

    engine = DiscoveryEngine(
        directory=tmp_path / "discovery", telemetry=telemetry,
        watched=tuple(f"aegis.noise.v{index}" for index in range(6)))
    for round_number in range(6):
        engine.scan(tick=600 + round_number)
        while engine.fit_next(tick=600 + round_number) is not None:
            pass
        while engine.preregister_next(tick=600 + round_number) is not None:
            pass

    assert engine.ledger.counts()["supported"] == 0
    assert engine.ledger.counts()["law"] == 0
