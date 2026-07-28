"""A rule's life: candidate → trial → active → retired or refuted (spec M3.5).

A mined rule is a hypothesis, not a decision. Between the two sits a controlled
trial, and the transitions out of it are where the safety of this contour lives:

* nothing activates without having *helped* in a measured comparison;
* an active rule that stops being significant is retired, because the world may
  simply have changed;
* a rule that measurably *hurt* is refuted and archived forever, so the miner
  cannot rediscover it next generation and start the argument again;
* no rule, however well evidenced, may suppress a safety-critical action.
"""
import pytest

from aegis.layers.policy.rules import (
    ACTIVE, PREFER, REFUTED, RETIRED, SUPPRESS, TRIAL, Rule, RuleLifecycle,
    rule_id,
)
from aegis.layers.world.state import StateKey


_DEFAULT_STATE = {"energy": "lo"}


def make_rule(action="learn_external", effect=SUPPRESS, state=_DEFAULT_STATE, **kw):
    # `state` must be able to be an EMPTY dict — that is how "never do this at
    # all" is written — so it cannot be defaulted with `or`.
    condition = {"state": dict(state), "action": action}
    return Rule(id=rule_id(condition, effect), condition=condition,
                effect=effect, support=30, success_rate=0.0, wilson_low=0.0,
                wilson_high=0.1, base_rate=0.8, p_value=0.001, **kw)


@pytest.fixture
def lifecycle(tmp_path):
    return RuleLifecycle(store_path=tmp_path / "rules.json")


LOW = StateKey(energy="lo")
HIGH = StateKey(energy="hi")


# ── matching ─────────────────────────────────────────────────────────

def test_a_rule_matches_only_its_own_state_and_action():
    rule = make_rule()
    assert rule.matches(LOW, "learn_external")
    assert not rule.matches(HIGH, "learn_external")
    assert not rule.matches(LOW, "rest")


def test_an_empty_state_condition_matches_everywhere():
    """"Never do this at all" has to be expressible."""
    rule = make_rule(state={})
    assert rule.matches(LOW, "learn_external")
    assert rule.matches(HIGH, "learn_external")


def test_a_two_field_condition_needs_both(tmp_path):
    rule = make_rule(state={"energy": "lo", "mood": "tired"})
    assert rule.matches(StateKey(energy="lo", mood="tired"), "learn_external")
    assert not rule.matches(StateKey(energy="lo", mood="calm"), "learn_external")


def test_a_state_string_matches_the_same_as_a_state_key():
    rule = make_rule()
    assert rule.matches_state(LOW.key())


def test_nothing_matches_no_state():
    assert not make_rule().matches_state(None)


# ── admission ────────────────────────────────────────────────────────

def test_an_admitted_candidate_goes_on_trial(lifecycle):
    admitted = lifecycle.admit([make_rule()], tick=100)
    assert len(admitted) == 1
    assert admitted[0].status == TRIAL
    assert admitted[0].trial_started == 100
    assert lifecycle.on_trial() == admitted


def test_a_safety_critical_action_can_never_be_suppressed(lifecycle):
    """§M3.5. However convincing the evidence, the policy does not get to
    switch off the things that keep the system alive."""
    admitted = lifecycle.admit([make_rule(action="health_check")], tick=1,
                               safety_critical=["health_check", "checkpoint"])
    assert admitted == []
    assert lifecycle.rules == {}


def test_a_safety_critical_action_may_still_be_preferred(lifecycle):
    """The protection is against suppression, not against evidence."""
    admitted = lifecycle.admit([make_rule(action="health_check", effect=PREFER)],
                               tick=1, safety_critical=["health_check"])
    assert len(admitted) == 1


def test_re_mining_refreshes_the_evidence_without_restarting_the_trial(lifecycle):
    lifecycle.admit([make_rule()], tick=100)
    again = make_rule()
    again.support = 60
    again.p_value = 0.0001
    assert lifecycle.admit([again], tick=250) == []

    held = lifecycle.on_trial()[0]
    assert held.support == 60 and held.p_value == 0.0001
    assert held.trial_started == 100          # the clock did not restart


def test_a_refuted_rule_is_never_re_admitted(lifecycle):
    rule = make_rule()
    lifecycle.admit([rule], tick=100)
    lifecycle.conclude_trial(rule, applied=[0.1] * 20, withheld=[0.9] * 20,
                             tick=400)
    assert rule.status == REFUTED
    assert lifecycle.admit([make_rule()], tick=800) == []


# ── the trial verdict ────────────────────────────────────────────────

def test_a_rule_that_helped_becomes_active(lifecycle):
    rule = make_rule()
    lifecycle.admit([rule], tick=100)
    verdict = lifecycle.conclude_trial(
        rule, applied=[0.8, 0.85, 0.82, 0.79, 0.84, 0.81],
        withheld=[0.4, 0.42, 0.38, 0.41, 0.39, 0.40], tick=400)
    assert verdict == ACTIVE
    assert rule.status == ACTIVE and rule.activated_tick == 400
    assert rule.measured_effect == pytest.approx(0.415, abs=0.01)
    assert lifecycle.activations == 1


def test_a_rule_that_changed_nothing_is_retired(lifecycle):
    rule = make_rule()
    lifecycle.admit([rule], tick=100)
    verdict = lifecycle.conclude_trial(
        rule, applied=[0.5, 0.52, 0.48, 0.51], withheld=[0.5, 0.49, 0.51, 0.5],
        tick=400)
    assert verdict == RETIRED
    assert rule.status == RETIRED
    assert lifecycle.retirements == 1


def test_a_rule_that_hurt_is_refuted_and_archived(lifecycle):
    rule = make_rule()
    lifecycle.admit([rule], tick=100)
    verdict = lifecycle.conclude_trial(
        rule, applied=[0.2, 0.18, 0.22, 0.19, 0.21],
        withheld=[0.8, 0.82, 0.78, 0.81, 0.79], tick=400)
    assert verdict == REFUTED
    assert lifecycle.refutations == 1
    assert rule.id in lifecycle.refuted
    assert rule.id not in lifecycle.rules          # off the live rule base
    assert "opposite" in lifecycle.refuted[rule.id]["reason"]


def test_a_real_but_negligible_effect_does_not_activate(lifecycle):
    """Significance without a worthwhile size is not a reason to change
    behaviour — the effect floor is what stops a policy of tiny rules."""
    rule = make_rule()
    lifecycle.admit([rule], tick=100)
    applied = [0.500 + 0.001 * (i % 2) for i in range(60)]
    withheld = [0.499 + 0.001 * (i % 2) for i in range(60)]
    assert lifecycle.conclude_trial(rule, applied, withheld, tick=400,
                                    min_effect=0.03) == RETIRED


# ── review of an active rule ─────────────────────────────────────────

def test_an_active_rule_that_still_works_stays(lifecycle):
    rule = make_rule(status=ACTIVE, activated_tick=400)
    lifecycle.rules[rule.id] = rule
    verdict = lifecycle.review(rule, applied=[0.8] * 5 + [0.82, 0.79],
                               withheld=[0.4] * 5 + [0.42, 0.39], tick=1400)
    assert verdict == ACTIVE and rule.status == ACTIVE


def test_an_active_rule_that_stopped_mattering_is_retired_not_refuted(lifecycle):
    """The world may simply have changed. A rule that stopped applying is not
    the same thing as one that was wrong, and only the second should be
    remembered as disproved."""
    rule = make_rule(status=ACTIVE, activated_tick=400)
    lifecycle.rules[rule.id] = rule
    verdict = lifecycle.review(rule, applied=[0.5, 0.51, 0.49, 0.5],
                               withheld=[0.5, 0.49, 0.51, 0.5], tick=1400)
    assert verdict == RETIRED
    assert rule.id not in lifecycle.refuted
    assert rule.id in lifecycle.rules              # kept, with its history


def test_an_active_rule_that_reversed_is_refuted(lifecycle):
    rule = make_rule(status=ACTIVE, activated_tick=400)
    lifecycle.rules[rule.id] = rule
    verdict = lifecycle.review(rule, applied=[0.2, 0.18, 0.22, 0.19, 0.21],
                               withheld=[0.8, 0.82, 0.78, 0.81, 0.79], tick=1400)
    assert verdict == REFUTED
    assert "reversed" in lifecycle.refuted[rule.id]["reason"]


# ── capacity ─────────────────────────────────────────────────────────

def test_active_rules_are_never_evicted(tmp_path):
    lifecycle = RuleLifecycle(store_path=tmp_path / "rules.json", max_rules=3)
    keeper = make_rule(action="kept", status=ACTIVE, activated_tick=1)
    lifecycle.rules[keeper.id] = keeper
    lifecycle.admit([make_rule(action=f"a{i}") for i in range(10)], tick=1)
    assert len(lifecycle.rules) <= 3
    assert keeper.id in lifecycle.rules


def test_the_least_evidenced_rule_is_evicted_first(tmp_path):
    lifecycle = RuleLifecycle(store_path=tmp_path / "rules.json", max_rules=2)
    strong = make_rule(action="strong")
    strong.support = 500
    weak = make_rule(action="weak")
    weak.support = 1
    middle = make_rule(action="middle")
    middle.support = 50
    lifecycle.admit([strong, weak, middle], tick=1)
    assert weak.id not in lifecycle.rules
    assert strong.id in lifecycle.rules


# ── persistence ──────────────────────────────────────────────────────

def test_the_rule_base_survives_a_restart(tmp_path):
    path = tmp_path / "rules.json"
    lifecycle = RuleLifecycle(store_path=path)
    rule = make_rule()
    lifecycle.admit([rule], tick=100)
    lifecycle.conclude_trial(rule, applied=[0.8] * 6, withheld=[0.4] * 6 ,
                             tick=400)
    other = make_rule(action="dream")
    lifecycle.admit([other], tick=100)
    lifecycle.conclude_trial(other, applied=[0.2] * 6, withheld=[0.8] * 6,
                             tick=400)
    lifecycle.save()

    restored = RuleLifecycle(store_path=path)
    assert [r.id for r in restored.ordered()] == [r.id for r in lifecycle.ordered()]
    assert other.id in restored.refuted
    assert restored.activations == lifecycle.activations
    assert restored.refutations == lifecycle.refutations


def test_a_refuted_rule_stays_refused_after_a_restart(tmp_path):
    path = tmp_path / "rules.json"
    lifecycle = RuleLifecycle(store_path=path)
    rule = make_rule()
    lifecycle.admit([rule], tick=100)
    lifecycle.conclude_trial(rule, applied=[0.2] * 6, withheld=[0.8] * 6, tick=400)
    lifecycle.save()

    restored = RuleLifecycle(store_path=path)
    assert restored.admit([make_rule()], tick=900) == []


@pytest.mark.parametrize("row", [
    None, "not a row", {}, {"condition": {}},
    {"id": "x", "condition": {"action": "a"}, "effect": "delete_everything"},
    {"id": "x", "condition": {"action": ""}, "effect": SUPPRESS},
    {"id": "x", "condition": {"action": "a"}, "effect": SUPPRESS,
     "support": "many"},
])
def test_an_unreadable_rule_is_discarded_not_obeyed(row):
    """A rule that cannot be read must not be allowed to suppress anything."""
    assert Rule.from_dict(row) is None


def test_a_well_formed_row_round_trips():
    rule = make_rule(status=ACTIVE, activated_tick=400, created_tick=100)
    rule.measured_effect = 0.42
    rule.provenance = ["exp_1", "exp_2"]
    restored = Rule.from_dict(rule.to_dict())
    assert restored.id == rule.id
    assert restored.condition == rule.condition
    assert restored.status == ACTIVE
    assert restored.measured_effect == pytest.approx(0.42)
    assert restored.provenance == ["exp_1", "exp_2"]


def test_a_torn_store_leaves_the_rest_readable(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(
        '{"schema_version": 2, "rules": ["nonsense", '
        '{"id": "keep", "condition": {"state": {}, "action": "rest"},'
        ' "effect": "suppress"}], "refuted": ["also nonsense",'
        ' {"id": "gone"}], "activations": "lots"}',
        encoding="utf-8")
    lifecycle = RuleLifecycle(store_path=path)
    assert list(lifecycle.rules) == ["keep"]
    assert list(lifecycle.refuted) == ["gone"]
    assert lifecycle.activations == 0


def test_status_counts_every_stage(tmp_path):
    lifecycle = RuleLifecycle(store_path=tmp_path / "rules.json")
    active = make_rule(action="a", status=ACTIVE, activated_tick=1)
    retired = make_rule(action="b", status=RETIRED)
    lifecycle.rules[active.id] = active
    lifecycle.rules[retired.id] = retired
    lifecycle.admit([make_rule(action="c")], tick=1)

    status = lifecycle.status()
    assert status["total"] == 3
    assert status["active"] == 1 and status["retired"] == 1 and status["trial"] == 1
    assert len(status["rules"]) == 3
