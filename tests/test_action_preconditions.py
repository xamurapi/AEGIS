"""Every precondition, in both of its states (spec Appendix A).

A precondition stuck at False silently disables an action forever, and nothing
else in the system reports that: the action simply never appears in the
available set, which looks exactly like "the planner did not choose it". So
each predicate is exercised in the state where it permits and the state where
it withholds.
"""
import asyncio

import pytest

import aegis.config as cfg
from aegis.layers.actions import PREDICATES, evaluate_precondition
from aegis.layers.substrate import Substrate


@pytest.fixture
def s(isolated_state):
    substrate = Substrate()
    substrate._regulation_directives = {}
    return substrate


def check(name, substrate):
    return evaluate_precondition(name, substrate)


# ── cadence ──────────────────────────────────────────────────────────

def test_checkpoint_due_tracks_its_cadence(s):
    s.tick_count = cfg.CHECKPOINT_EVERY_N_TICKS * 3
    assert check("checkpoint_due", s) is True
    s.tick_count += 1
    assert check("checkpoint_due", s) is False


def test_backup_due_is_five_times_rarer_than_a_checkpoint(s):
    period = cfg.CHECKPOINT_EVERY_N_TICKS * 5
    s.tick_count = period
    assert check("backup_due", s) is True
    s.tick_count = cfg.CHECKPOINT_EVERY_N_TICKS      # a checkpoint, not a backup
    assert check("backup_due", s) is False


# ── measurement availability ─────────────────────────────────────────

def test_capacity_regulation_waits_for_latency_samples(s):
    s.health.tick_durations.clear()
    assert check("has_latency_samples", s) is False
    s.health.record_tick(5.0, success=True)
    assert check("has_latency_samples", s) is True


def test_the_environment_needs_at_least_one_skill(s):
    assert check("skills_available", s) is True
    s.skill_library.skills.clear()
    assert check("skills_available", s) is False


def test_a_second_benchmark_waits_for_the_first(s):
    assert check("no_benchmark_running", s) is True

    async def _hold():
        await asyncio.sleep(3600)

    async def _arrange():
        s._eval_task = asyncio.ensure_future(_hold())
        assert check("no_benchmark_running", s) is False
        s._eval_task.cancel()

    asyncio.run(_arrange())


def test_a_finished_benchmark_no_longer_blocks(s):
    async def _done():
        return None

    async def _arrange():
        s._eval_task = asyncio.ensure_future(_done())
        await s._eval_task
        assert check("no_benchmark_running", s) is True

    asyncio.run(_arrange())


# ── capability gaps ──────────────────────────────────────────────────

def test_synthesis_waits_for_something_to_fix(s):
    s.evaluator.failing_kinds_cached = lambda: ["is_prime"]
    assert check("has_failing_kind", s) is True
    assert check("no_failing_kind", s) is False


def test_optimisation_waits_until_nothing_is_failing(s):
    s.evaluator.failing_kinds_cached = lambda: []
    assert check("has_failing_kind", s) is False
    assert check("no_failing_kind", s) is True


def test_a_probe_that_raises_reads_as_no_failing_kind(s):
    def explode():
        raise RuntimeError("evaluator down")

    s.evaluator.failing_kinds_cached = explode
    assert check("has_failing_kind", s) is False


def test_coding_synthesis_waits_for_an_unsolved_task(s):
    s.evaluator.unsolved_coding_cached = lambda: ["task"]
    assert check("has_unsolved_coding", s) is True
    s.evaluator.unsolved_coding_cached = lambda: []
    assert check("has_unsolved_coding", s) is False


# ── contours delivered by later stages ───────────────────────────────

def test_reasoning_preconditions_are_unmet_before_the_contour_exists(s):
    for name in ("has_queued_task", "enough_results", "has_weakness",
                 "has_candidate_strategy"):
        assert check(name, s) is False, name


def test_discovery_preconditions_are_unmet_before_the_contour_exists(s):
    for name in ("enough_telemetry", "has_unmodelled_hypothesis", "has_prereg"):
        assert check(name, s) is False, name


def test_a_policy_precondition_is_unmet_before_the_contour_exists(s):
    assert check("has_active_rules", s) is False


def test_a_present_contour_that_says_no_is_believed(s):
    """Presence of a contour is not the answer — its answer is.

    Reading "the module exists" as "the condition holds" would schedule every
    reasoning and discovery action on the tick its contour was attached,
    regardless of whether there was anything to do.
    """
    class Silent:
        def has_queued_task(self):
            return False

        def result_count(self):
            return 0

        def top_weakness(self):
            return None

        def pending_candidates(self):
            return []

        def data_points(self):
            return 0

        def unmodelled_hypotheses(self):
            return []

        def active_preregistrations(self):
            return []

        def active_rules(self):
            return []

    s.reasoning = s.discovery = s.policy = Silent()
    for name in ("has_queued_task", "enough_results", "has_weakness",
                 "has_candidate_strategy", "enough_telemetry",
                 "has_unmodelled_hypothesis", "has_prereg", "has_active_rules"):
        assert check(name, s) is False, name


def test_a_present_policy_with_rules_permits_review(s):
    class Policy:
        def active_rules(self):
            return [{"id": "r1"}]

    s.policy = Policy()
    assert check("has_active_rules", s) is True


def test_a_present_discovery_with_work_permits_it(s):
    class Discovery:
        def data_points(self):
            return cfg.DISC_MIN_N

        def unmodelled_hypotheses(self):
            return [{"id": "h1"}]

        def active_preregistrations(self):
            return [{"id": "p1"}]

    s.discovery = Discovery()
    assert check("enough_telemetry", s) is True
    assert check("has_unmodelled_hypothesis", s) is True
    assert check("has_prereg", s) is True


def test_a_present_contour_is_asked_for_its_answer(s):
    class Reasoning:
        def has_queued_task(self):
            return True

        def result_count(self):
            return cfg.DISC_MIN_N

        def top_weakness(self):
            return {"rank": 1}

        def pending_candidates(self):
            return ["candidate"]

    s.reasoning = Reasoning()
    assert check("has_queued_task", s) is True
    assert check("enough_results", s) is True
    assert check("has_weakness", s) is True
    assert check("has_candidate_strategy", s) is True


def test_a_result_window_below_half_the_minimum_is_not_enough(s):
    class Reasoning:
        def result_count(self):
            return cfg.DISC_MIN_N // 2 - 1

    s.reasoning = Reasoning()
    assert check("enough_results", s) is False


def test_a_result_window_at_half_the_minimum_is_enough(s):
    class Reasoning:
        def result_count(self):
            return cfg.DISC_MIN_N // 2

    s.reasoning = Reasoning()
    assert check("enough_results", s) is True


def test_telemetry_below_the_minimum_is_not_enough(s):
    class Discovery:
        def data_points(self):
            return cfg.DISC_MIN_N - 1

    s.discovery = Discovery()
    assert check("enough_telemetry", s) is False


def test_telemetry_at_the_minimum_is_enough(s):
    class Discovery:
        def data_points(self):
            return cfg.DISC_MIN_N

    s.discovery = Discovery()
    assert check("enough_telemetry", s) is True


def test_a_generation_already_running_blocks_the_next(s):
    assert check("no_active_generation", s) is True
    s.evolution.generation_running = True
    assert check("no_active_generation", s) is False


# ── experience and evidence ──────────────────────────────────────────

def test_rule_mining_waits_for_enough_experience(s):
    s.feedback_loop.resolved = cfg.POLICY_MIN_SUPPORT * 5 - 1
    assert check("enough_experiences", s) is False
    s.feedback_loop.resolved = cfg.POLICY_MIN_SUPPORT * 5
    assert check("enough_experiences", s) is True


def test_self_inspection_waits_for_a_decision_trace(s):
    s.introspection.decision_trace = [{"decision": "x"}] * 4
    assert check("has_decision_trace", s) is False
    s.introspection.decision_trace = [{"decision": "x"}] * 5
    assert check("has_decision_trace", s) is True


def test_training_waits_for_a_large_enough_dataset(s):
    s.memory.semantic = {f"c{i}": {} for i in range(cfg.TRAIN_MIN_DATASET_SIZE - 1)}
    assert check("dataset_large_enough", s) is False
    s.memory.semantic[f"c{cfg.TRAIN_MIN_DATASET_SIZE}"] = {}
    assert check("dataset_large_enough", s) is True


# ── health and regulation ────────────────────────────────────────────

def test_experiments_require_full_health(s):
    s.health._prev_status = "healthy"
    assert check("health_ok", s) is True
    s.health._prev_status = "warning"
    assert check("health_ok", s) is False


def test_self_modification_only_stops_at_critical(s):
    s.health._prev_status = "warning"
    assert check("health_not_critical", s) is True
    s.health._prev_status = "critical"
    assert check("health_not_critical", s) is False


def test_learning_respects_the_regulator(s, monkeypatch):
    import aegis.config as cfg

    # Evolution is off across the suite (see conftest): a generation evaluates
    # ten variants in other processes, so starting one by accident costs
    # minutes. This test is about the *regulator*, so the switch is turned back
    # on for it and the regulator is what is varied.
    monkeypatch.setattr(cfg, "EVO_ENABLED", True)
    s._regulation_directives = {}
    assert check("not_skip_learning", s) is True
    assert check("evolution_allowed", s) is True
    s._regulation_directives = {"skip_learning": True}
    assert check("not_skip_learning", s) is False
    assert check("evolution_allowed", s) is False


def test_training_is_blocked_by_the_kill_switch(s):
    assert check("training_ethics_ok", s) is True
    s.ethics.kill_switch_active = True
    assert check("training_ethics_ok", s) is False


def test_external_learning_respects_the_network_permission(s):
    s.world.permissions["network"] = True
    assert check("network_allowed", s) is True
    s.world.permissions["network"] = False
    assert check("network_allowed", s) is False


def test_an_undeclared_network_permission_reads_as_allowed(s):
    # The permission set is the world interface's, not this registry's. Reading
    # an absent key as "forbidden" would disable external learning on any build
    # whose permission list happens to be spelled differently.
    s.world.permissions.pop("network", None)
    assert check("network_allowed", s) is True


# ── model availability ───────────────────────────────────────────────

def test_a_model_role_is_unavailable_with_no_provider_and_no_legacy_client(s):
    s.llm.enabled = False
    s.llm.cortex.configure_routes({})
    for name in ("role_fast_available", "role_deep_available", "role_code_available"):
        assert check(name, s) is False, name


def test_a_legacy_client_alone_makes_the_roles_available(s):
    s.llm.enabled = True
    s.llm.cortex.configure_routes({})
    for name in ("role_fast_available", "role_deep_available", "role_code_available"):
        assert check(name, s) is True, name


def test_a_cortex_route_alone_makes_its_role_available(s):
    from aegis.cortex.router import Cortex
    from tests.cortex_fakes import ScriptedProvider

    s.llm.enabled = False
    s.llm.cortex = Cortex(providers={"p": ScriptedProvider("p")},
                          routes={"code": ["p"]})
    assert check("role_code_available", s) is True
    assert check("role_deep_available", s) is False


# ── affect ───────────────────────────────────────────────────────────

def test_dreaming_needs_low_energy_or_a_reflective_mode(s):
    s.emotions.energy = 0.9
    s.consciousness.mode = "focused"
    assert check("low_energy_or_reflective", s) is False

    s.emotions.energy = 0.3
    assert check("low_energy_or_reflective", s) is True

    s.emotions.energy = 0.9
    s.consciousness.mode = "reflective"
    assert check("low_energy_or_reflective", s) is True


def test_the_dream_energy_threshold_is_where_the_spec_puts_it(s):
    s.consciousness.mode = "focused"
    s.emotions.energy = 0.4
    assert check("low_energy_or_reflective", s) is False
    s.emotions.energy = 0.39
    assert check("low_energy_or_reflective", s) is True


# ── opt-in gates ─────────────────────────────────────────────────────

def test_source_rewriting_follows_the_operator_switch(s, monkeypatch):
    monkeypatch.setattr(cfg, "CODE_SELF_MOD_ENABLED", False)
    assert check("code_self_mod_enabled", s) is False
    monkeypatch.setattr(cfg, "CODE_SELF_MOD_ENABLED", True)
    assert check("code_self_mod_enabled", s) is True


# ── the set itself ───────────────────────────────────────────────────

def test_every_predicate_answers_a_boolean(s):
    for name in sorted(PREDICATES):
        assert isinstance(check(name, s), bool), name


def test_no_predicate_raises_on_a_bare_substrate(s):
    for name in sorted(PREDICATES):
        check(name, s)
