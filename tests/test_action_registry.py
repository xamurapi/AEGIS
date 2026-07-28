"""The action registry (spec Appendix A).

The registry is the single source of truth about what the system can do, and
three properties have to hold or that claim is empty: every declared action can
actually be performed, the safety-critical set is complete, and the reserved
floor is large enough to fund it.
"""
import pytest

import aegis.config as cfg
from aegis.layers.actions import (
    DELIVERED_STAGE,
    ACTIONS, ACTIONS_BY_NAME, ActionSpace, ActionSpec, PREDICATES,
    evaluate_precondition,
)
from aegis.layers.motivation.resources import ResourceCost
from aegis.layers.substrate import Substrate

#: Appendix A, in full. Written out rather than derived from the registry, so
#: that deleting a row from the registry fails a test instead of passing one.
APPENDIX_A = {
    "perceive_world", "health_check", "checkpoint", "backup_state",
    "capacity_regulate", "env_step", "run_benchmark", "synthesize_skill",
    "synthesize_coding", "optimize_skill", "reason_task", "evolve_generation",
    "mine_rules", "review_rules", "scan_weakness", "synthesize_strategy",
    "evaluate_strategy", "scan_hypotheses", "fit_model", "run_experiment",
    "learn_external", "run_agents", "evolve_agents", "curiosity_explore",
    "evaluate_state_llm", "reflect_llm", "self_inspect", "consolidate_memory",
    "parametric_self_mod", "train_weights", "code_self_mod", "dream", "rest",
}

SAFETY_CRITICAL = {"perceive_world", "health_check", "checkpoint",
                   "backup_state", "capacity_regulate"}

#: Appendix A's cost table, transcribed: name -> (drive, tok, ms, proc, net,
#: train, min_interval). Written out here so the registry is checked against
#: the specification rather than against itself — a number quietly edited in
#: the source has to fail a test, not merely change one.
APPENDIX_A_COSTS = {
    "perceive_world":      ("coherence",     0,     5, 0, 0, 0,    1),
    "health_check":        ("stability",     0,     3, 0, 0, 0,    1),
    "checkpoint":          ("stability",     0,    40, 0, 0, 0,   10),
    "backup_state":        ("stability",     0,    80, 0, 0, 0,   50),
    "capacity_regulate":   ("stability",     0,     5, 0, 0, 0,   50),
    "env_step":            ("competence",    0,   300, 1, 0, 0,    2),
    "run_benchmark":       ("competence",    0,  4000, 1, 0, 0,   50),
    "synthesize_skill":    ("competence", 2500,  6000, 1, 0, 0,  200),
    "synthesize_coding":   ("competence", 3000,  8000, 1, 0, 0,  200),
    "optimize_skill":      ("competence", 2000,  5000, 1, 0, 0,  200),
    "evolve_generation":   ("competence",  800, 20000, 4, 0, 0,  250),
    "mine_rules":          ("coherence",     0,   400, 0, 0, 0,  200),
    "review_rules":        ("coherence",     0,   150, 0, 0, 0, 1000),
    "scan_weakness":       ("coherence",     0,   600, 0, 0, 0,  300),
    "synthesize_strategy": ("competence", 2000,  5000, 0, 0, 0,  300),
    "evaluate_strategy":   ("competence", 1500,  9000, 1, 0, 0,  300),
    "scan_hypotheses":     ("knowledge",     0,  1500, 0, 0, 0, 1000),
    "fit_model":           ("knowledge",     0,  3000, 1, 0, 0, 1000),
    "run_experiment":      ("knowledge",     0,   200, 0, 0, 0,    1),
    "learn_external":      ("knowledge",     0,  1200, 0, 1, 0,   40),
    "run_agents":          ("knowledge",     0,  2000, 0, 3, 0,    5),
    "evolve_agents":       ("knowledge",     0,   300, 0, 0, 0,  100),
    "curiosity_explore":   ("knowledge",   900,  3000, 0, 0, 0,   15),
    "evaluate_state_llm":  ("coherence",  1200,  3000, 0, 0, 0,    3),
    "reflect_llm":         ("coherence",  1200,  3000, 0, 0, 0,    3),
    "self_inspect":        ("coherence",     0,    60, 0, 0, 0,   20),
    "consolidate_memory":  ("coherence",     0,   250, 0, 0, 0,    8),
    "parametric_self_mod": ("competence", 1000,  2000, 0, 0, 0,   15),
    "train_weights":       ("competence",    0,     0, 0, 0, 1, 1000),
    "code_self_mod":       ("competence", 4000,  6000, 0, 0, 0,  500),
    "dream":               ("stability",     0,   100, 0, 0, 0,   50),
    "rest":                ("stability",     0,    10, 0, 0, 0,    1),
}


@pytest.mark.parametrize("name", sorted(APPENDIX_A_COSTS))
def test_each_action_matches_the_appendix_exactly(name):
    drive, tok, ms, proc, net, train, min_int = APPENDIX_A_COSTS[name]
    spec = ACTIONS_BY_NAME[name]
    assert spec.drive == drive
    assert spec.cost.llm_tokens == tok
    assert spec.cost.wall_ms == ms
    assert spec.cost.subprocess_slots == proc
    assert spec.cost.net_calls == net
    assert spec.cost.training_slots == train
    assert spec.min_interval == min_int


def test_the_one_action_with_a_cost_range_stays_inside_it():
    # Appendix A prices `reason_task` as a range (400–4000 tokens, 200–8000 ms,
    # 0–1 subprocess) because a reasoning strategy's cost depends on which one
    # is chosen. The registry has to reserve the upper bound.
    spec = ACTIONS_BY_NAME["reason_task"]
    assert 400 <= spec.cost.llm_tokens <= 4000
    assert 200 <= spec.cost.wall_ms <= 8000
    assert 0 <= spec.cost.subprocess_slots <= 1
    assert spec.min_interval == 1


@pytest.fixture
def substrate(isolated_state):
    return Substrate()


@pytest.fixture
def space():
    return ActionSpace()


# ── completeness ─────────────────────────────────────────────────────

def test_every_action_in_the_appendix_is_declared():
    assert APPENDIX_A <= set(ACTIONS_BY_NAME)


def test_no_action_is_declared_that_the_appendix_does_not_name():
    assert set(ACTIONS_BY_NAME) <= APPENDIX_A


def test_action_names_are_unique():
    names = [spec.name for spec in ACTIONS]
    assert len(names) == len(set(names))


def test_every_action_serves_a_known_drive():
    drives = {"competence", "knowledge", "coherence", "stability"}
    assert all(spec.drive in drives for spec in ACTIONS)


def test_every_action_declares_a_positive_rate_limit():
    assert all(spec.min_interval >= 1 for spec in ACTIONS)


def test_every_action_declares_a_cost():
    assert all(isinstance(spec.cost, ResourceCost) for spec in ACTIONS)


def test_an_action_cannot_be_rewritten_at_runtime():
    # The registry is the single source of truth about what the system may do.
    # If a spec were mutable, any contour could quietly raise its own budget or
    # clear its own safety flag — and the declaration would stop being a rule.
    spec = ACTIONS_BY_NAME["health_check"]
    with pytest.raises(Exception):
        spec.safety_critical = False
    with pytest.raises(Exception):
        spec.cost = ResourceCost(llm_tokens=10 ** 6)


def test_source_rewriting_is_the_only_irreversible_action():
    # Everything else can be undone; rewriting the package on disk cannot be,
    # which is why it stays opt-in and separately gated.
    irreversible = {spec.name for spec in ACTIONS if not spec.reversible}
    assert irreversible == {"code_self_mod"}


# ── safety-critical set ──────────────────────────────────────────────

def test_the_safety_critical_set_is_exactly_the_appendix_set(space):
    assert {spec.name for spec in space.safety_critical()} == SAFETY_CRITICAL


def test_perception_and_health_are_safety_critical():
    # These are what the system uses to notice it is in trouble; a policy that
    # could suppress them could suppress its own alarm.
    assert ACTIONS_BY_NAME["perceive_world"].safety_critical
    assert ACTIONS_BY_NAME["health_check"].safety_critical


def test_no_safety_critical_action_needs_a_model():
    # Safety work must survive a total cortex outage.
    for spec in ActionSpace().safety_critical():
        assert spec.cost.llm_tokens == 0, spec.name


def test_the_safety_floor_covers_every_safety_critical_action():
    # If the reserved slice could not fund the whole safety set at once, the
    # floor would be a number rather than a guarantee.
    total_wall = sum(spec.reservation_cost().wall_ms
                     for spec in ActionSpace().safety_critical())
    floor = cfg.RES_WALL_MS_PER_TICK * cfg.RESOURCE_SAFETY_FLOOR
    assert total_wall <= floor


# ── executors ────────────────────────────────────────────────────────

def test_every_delivered_action_has_a_live_executor(space, substrate):
    """Actions whose stage has landed must be performable, not just declared."""
    delivered = [spec for spec in ACTIONS if spec.stage <= DELIVERED_STAGE]
    unwired = [spec.name for spec in delivered if not space.is_wired(spec, substrate)]
    assert unwired == []


def test_actions_from_later_stages_are_simply_unavailable(space, substrate):
    # A contour still under construction must not be schedulable — but it must
    # also not crash anything by being named in the registry.
    pending = {spec.name for spec in ACTIONS if spec.stage > DELIVERED_STAGE}
    assert set(space.unwired(substrate)) == pending


def test_an_unwired_action_never_appears_as_available(space, substrate):
    available = {spec.name for spec in space.available(substrate)}
    assert not (available & set(space.unwired(substrate)))


def test_an_executor_path_that_does_not_resolve_yields_none(space, substrate):
    spec = ActionSpec("ghost", "stability", ResourceCost(), "no.such.attribute")
    assert space.executor_for(spec, substrate) is None


def test_a_non_callable_attribute_is_not_an_executor(space, substrate):
    spec = ActionSpec("ghost", "stability", ResourceCost(), "tick_count")
    assert space.executor_for(spec, substrate) is None


# ── preconditions ────────────────────────────────────────────────────

def test_every_declared_precondition_is_implemented():
    # A typo here would read as "always unmet" and silently disable an action
    # forever, which is the kind of failure nothing else would report.
    named = {p for spec in ACTIONS for p in spec.preconditions}
    assert named <= set(PREDICATES)


def test_an_unknown_precondition_is_unmet_not_permissive(substrate, caplog):
    # Silently granting an action its licence is the worse failure by far.
    assert evaluate_precondition("no_such_predicate", substrate) is False


def test_a_precondition_that_raises_is_unmet(substrate, monkeypatch):
    monkeypatch.setitem(PREDICATES, "explodes",
                        lambda s, ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    assert evaluate_precondition("explodes", substrate) is False


def test_a_comparison_precondition_is_understood(substrate):
    substrate.emotions.energy = 0.9
    assert evaluate_precondition("emotions.energy>0.2", substrate) is True
    assert evaluate_precondition("emotions.energy<0.2", substrate) is False


def test_every_comparison_operator_is_supported(substrate):
    substrate.emotions.energy = 0.5
    assert evaluate_precondition("emotions.energy>=0.5", substrate) is True
    assert evaluate_precondition("emotions.energy<=0.5", substrate) is True
    assert evaluate_precondition("emotions.energy==0.5", substrate) is True
    assert evaluate_precondition("emotions.energy!=0.5", substrate) is False


def test_a_comparison_against_a_missing_attribute_is_unmet(substrate):
    assert evaluate_precondition("nothing.here>1", substrate) is False


def test_a_comparison_with_an_unparseable_threshold_is_unmet(substrate):
    assert evaluate_precondition("emotions.energy>lots", substrate) is False


def test_checkpoint_is_due_on_its_cadence(substrate):
    substrate.tick_count = cfg.CHECKPOINT_EVERY_N_TICKS
    assert evaluate_precondition("checkpoint_due", substrate) is True
    substrate.tick_count = cfg.CHECKPOINT_EVERY_N_TICKS + 1
    assert evaluate_precondition("checkpoint_due", substrate) is False


def test_source_rewriting_is_gated_on_the_opt_in(substrate):
    assert evaluate_precondition("code_self_mod_enabled", substrate) \
        == cfg.CODE_SELF_MOD_ENABLED


# ── availability ─────────────────────────────────────────────────────

def test_available_is_sorted(space, substrate):
    names = [spec.name for spec in space.available(substrate)]
    assert names == sorted(names)


def test_a_rate_limited_action_is_withheld(space, substrate):
    substrate.tick_count = 100
    spec = ACTIONS_BY_NAME["evolve_agents"]        # min_interval 100
    space.mark_run(spec, 100)
    substrate.tick_count = 150
    assert spec not in space.available(substrate)


def test_the_rate_limit_expires(space, substrate):
    spec = ACTIONS_BY_NAME["evolve_agents"]
    space.mark_run(spec, 0)
    substrate.tick_count = spec.min_interval
    assert space.rate_limited(spec, substrate.tick_count) is False


def test_a_never_run_action_is_not_rate_limited(space):
    assert space.rate_limited(ACTIONS_BY_NAME["rest"], 0) is False


def test_an_action_with_unmet_preconditions_is_withheld(space, substrate):
    substrate.tick_count = 1        # checkpoint is not due
    assert ACTIONS_BY_NAME["checkpoint"] not in space.available(substrate)


def test_blocking_reasons_are_counted(space):
    space.note_blocked("resources")
    space.note_blocked("resources")
    assert space.status()["blocked"]["resources"] == 2


# ── external work and the tick budget ────────────────────────────────

def test_external_wall_time_is_not_charged_to_the_tick():
    # §3.4 already excludes external work from the phase budgets; charging it
    # to the per-tick resource would make the two measurements contradict.
    spec = ACTIONS_BY_NAME["evaluate_state_llm"]
    assert spec.external is True
    assert spec.cost.wall_ms > 0
    assert spec.reservation_cost().wall_ms == 0


def test_local_wall_time_is_charged_in_full():
    spec = ACTIONS_BY_NAME["health_check"]
    assert spec.external is False
    assert spec.reservation_cost().wall_ms == spec.cost.wall_ms


#: Actions whose wall time is spent waiting on something outside this process
#: — a hosted model, the network, a subprocess, or a detached task. Their wall
#: time is excluded from the per-tick budget for the same reason §3.4 excludes
#: it from the phase budgets: it measures the provider, not the cognitive cycle.
EXTERNAL = {
    "env_step", "run_benchmark", "synthesize_skill", "synthesize_coding",
    "optimize_skill", "reason_task", "evolve_generation", "evaluate_strategy",
    "fit_model", "learn_external", "run_agents", "curiosity_explore",
    "evaluate_state_llm", "reflect_llm", "parametric_self_mod",
    "train_weights", "code_self_mod", "synthesize_strategy",
}


def test_the_external_set_is_exactly_what_it_should_be():
    assert {spec.name for spec in ACTIONS if spec.external} == EXTERNAL


def test_purely_local_work_is_not_marked_external():
    # Mislabelling local work as external would exempt it from the tick budget
    # and let a slow local computation stall the cycle unmeasured.
    local = {spec.name for spec in ACTIONS if not spec.external}
    assert "health_check" in local
    assert "consolidate_memory" in local
    assert "self_inspect" in local
    assert local & EXTERNAL == set()


def test_any_action_that_spends_tokens_is_external():
    # Spending tokens means waiting on a model. An action that claimed to be
    # local while doing that would charge its whole round-trip to the tick's
    # 2500 ms and could never be afforded.
    mislabelled = [spec.name for spec in ACTIONS
                   if spec.cost.llm_tokens and not spec.external]
    assert mislabelled == []


def test_every_local_action_fits_inside_one_tick_budget():
    for spec in ACTIONS:
        if not spec.external:
            assert spec.reservation_cost().wall_ms <= cfg.RES_WALL_MS_PER_TICK, spec.name


def test_the_declared_cost_is_preserved_for_reporting():
    spec = ACTIONS_BY_NAME["run_benchmark"]
    assert spec.describe()["cost"]["wall_ms"] == 4000
    assert spec.describe()["reserved"]["wall_ms"] == 0


# ── reporting ────────────────────────────────────────────────────────

def test_status_groups_actions_by_drive(space, substrate):
    grouped = space.status(substrate)["by_drive"]
    assert set(grouped) == {"competence", "knowledge", "coherence", "stability"}
    assert sum(len(names) for names in grouped.values()) == len(ACTIONS)


def test_by_drive_is_sorted(space):
    names = [spec.name for spec in space.by_drive("stability")]
    assert names == sorted(names)


def test_describe_covers_the_declared_fields():
    described = ACTIONS_BY_NAME["rest"].describe()
    assert set(described) >= {"name", "drive", "cost", "reserved", "executor",
                              "preconditions", "min_interval", "reversible",
                              "safety_critical", "external", "stage"}
