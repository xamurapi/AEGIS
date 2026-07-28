"""pytest-bdd step definitions for tests/features/behaviour_change.feature.

Executable Gherkin: every scenario drives the real behaviour policy, so the
feature file is both the statement of what "experience changes behaviour" means
here and the test that it happens that way.
"""
import pytest
from pytest_bdd import given, scenarios, then, when

import aegis.config as cfg
from aegis.layers.planner import Plan
from aegis.layers.policy import ACTIVE, REFUTED, RETIRED, SUPPRESS, TRIAL, BehaviourPolicy
from aegis.layers.world.state import StateKey
from aegis.util.quasirandom import hash_unit

scenarios("features/behaviour_change.feature")

LOW = StateKey(energy="lo", mood="tired", mode="focused")
HIGH = StateKey(energy="hi", mood="curious", mode="focused")
BAD = "learn_external"
GOOD = "consolidate_memory"


def _plans(*actions, safety_critical=()):
    built = []
    for index, action in enumerate(actions):
        plan = Plan(objective=f"objective_{action}", steps=[action],
                    safety_critical=action in safety_critical)
        plan.score = 1.0 - index * 0.1
        built.append(plan)
    return built


@pytest.fixture
def ctx(tmp_path):
    return {"policy": BehaviourPolicy(store_dir=tmp_path, min_support=10,
                                      trial_ticks=40),
            "root": tmp_path, "tick": 0, "admitted": []}


# ── Background ───────────────────────────────────────────────────────

@given("a behaviour policy with no rules yet")
def _no_rules(ctx):
    assert ctx["policy"].active_rules() == []


# ── preferences ──────────────────────────────────────────────────────

@when("the action pays better than its state usually does")
def _pays_better(ctx):
    policy = ctx["policy"]
    policy.observe(LOW, GOOD, reward=0.2, success=False, tick=1)
    policy.observe(LOW, GOOD, reward=0.9, success=True, tick=2)


@then("the preference for that action should rise")
def _preference_rose(ctx):
    assert ctx["policy"].store.weight_for(LOW, GOOD) > 0


@given("an action that pays the same amount in a rich state and a poor one")
def _same_pay_two_states(ctx):
    policy = ctx["policy"]
    tick = 0
    for _ in range(5):
        policy.observe(HIGH, "other", reward=0.9, success=True, tick=tick)
        policy.observe(LOW, "other", reward=0.1, success=False, tick=tick + 1)
        tick += 2
    for _ in range(5):
        policy.observe(HIGH, GOOD, reward=0.5, success=True, tick=tick)
        policy.observe(LOW, GOOD, reward=0.5, success=True, tick=tick + 1)
        tick += 2


@then("it should be preferred in the poor state and not in the rich one")
def _advantage_not_reward(ctx):
    policy = ctx["policy"]
    assert policy.store.weight_for(LOW, GOOD) > 0
    assert policy.store.weight_for(HIGH, GOOD) < 0


@when("the action fails repeatedly in one state")
def _fails_repeatedly(ctx):
    policy, tick = ctx["policy"], ctx["tick"]
    for _ in range(30):
        policy.observe(LOW, BAD, reward=0.05, success=False, tick=tick)
        policy.observe(HIGH, BAD, reward=0.95, success=True, tick=tick + 1)
        policy.observe(LOW, GOOD, reward=0.6, success=True, tick=tick + 2)
        tick += 3
    ctx["tick"] = tick


@then("both courses of action should still be offered")
def _both_offered(ctx):
    surviving = ctx["policy"].apply_rules(LOW, _plans(BAD, GOOD), ctx["tick"])
    assert {plan.action for plan in surviving} == {BAD, GOOD}


@then("the preference for the failing action should be negative")
def _preference_negative(ctx):
    assert ctx["policy"].delta(LOW, BAD) < 0


# ── mining ───────────────────────────────────────────────────────────

@when("the policy mines its experience")
def _mine(ctx):
    ctx["admitted"] = ctx["policy"].mine(ctx["tick"], safety_critical=[])


@when("the policy mines its experience protecting that action")
def _mine_protected(ctx):
    ctx["admitted"] = ctx["policy"].mine(ctx["tick"], safety_critical=[BAD])


@then("a suppression rule for that state and action should be on trial")
def _rule_on_trial(ctx):
    rule = _suppression(ctx)
    assert rule is not None
    assert rule.status == TRIAL
    assert rule.matches(LOW, BAD)
    ctx["rule"] = rule


@then("no suppression rule for that action should exist")
def _no_suppression(ctx):
    assert all(not (rule.effect == SUPPRESS and rule.action == BAD)
               for rule in ctx["admitted"])
    assert ctx["policy"].suppressed_actions(LOW) == []


@then("no rules should have been proposed")
def _nothing_proposed(ctx):
    assert ctx["admitted"] == []
    assert ctx["policy"].miner.tested > 50      # it really did look


@when("outcomes are independent of the state")
def _pure_noise(ctx):
    policy = ctx["policy"]
    energies, moods = ("lo", "mid", "hi"), ("curious", "tired", "calm")
    actions = (BAD, GOOD, "rest", "dream")
    for index in range(1200):
        state = StateKey(energy=energies[index % 3],
                         mood=moods[(index // 3) % 3], mode="focused")
        policy.observe(state, actions[(index // 9) % 4],
                       reward=hash_unit("bdd_noise_r", index),
                       success=hash_unit("bdd_noise", index) < 0.5, tick=index)
    ctx["tick"] = 1200


# ── the trial ────────────────────────────────────────────────────────

@given("a suppression rule on trial")
def _on_trial(ctx):
    _fails_repeatedly(ctx)
    _mine(ctx)
    rule = _suppression(ctx)
    assert rule is not None
    ctx["rule"] = rule


@when("suppressing the action pays better than allowing it")
def _suppression_pays(ctx):
    _run_trial(ctx, applied=0.8, withheld=0.3)


@when("the policy reviews its trials")
def _review(ctx):
    ctx["outcomes"] = ctx["policy"].review(ctx["tick"])


@then("the rule should be active")
def _rule_active(ctx):
    assert ctx["rule"].status == ACTIVE


@then("the action should no longer be offered in that state")
def _not_offered(ctx):
    surviving = ctx["policy"].apply_rules(LOW, _plans(BAD, GOOD), 90_000)
    assert [plan.action for plan in surviving] == [GOOD]


@given("an active suppression rule")
def _active_rule(ctx):
    _on_trial(ctx)
    _suppression_pays(ctx)
    _review(ctx)
    assert ctx["rule"].status == ACTIVE


@then("the action should still be offered in a different state")
def _offered_elsewhere(ctx):
    surviving = ctx["policy"].apply_rules(HIGH, _plans(BAD, GOOD), 90_000)
    assert BAD in [plan.action for plan in surviving]


@then("a safety-critical plan for that action should still be offered")
def _safety_critical_survives(ctx):
    surviving = ctx["policy"].apply_rules(
        LOW, _plans(BAD, GOOD, safety_critical=[BAD]), 90_000)
    assert BAD in [plan.action for plan in surviving]


# ── measurement ──────────────────────────────────────────────────────

@when("the system decides repeatedly in that state")
def _decides_repeatedly(ctx):
    for offset in range(50):
        ctx["policy"].apply_rules(LOW, _plans(BAD, GOOD), 50_000 + offset)


@then("the behaviour-change rate should be above zero")
def _delta_rate_positive(ctx):
    assert ctx["policy"].behaviour_delta_rate() > 0


# ── retirement and refutation ────────────────────────────────────────

@when("suppressing the action stops making any difference")
def _no_longer_matters(ctx):
    policy, rule = ctx["policy"], ctx["rule"]
    start = rule.activated_tick + 1
    # Every active rule about this action, not only the tracked one: the miner
    # legitimately finds several equivalent descriptions of one fact, and the
    # world changing back changed it for all of them.
    watched = [r for r in policy.lifecycle.active() if r.action == BAD]
    for offset in range(80):
        tick = start + offset
        policy.apply_rules(LOW, _plans(BAD, GOOD), tick)
        for active in watched:
            policy.shadow.record(
                active.id, policy.shadow.applies_this_tick(active, tick),
                0.7, tick)
    ctx["tick"] = rule.activated_tick + cfg.POLICY_REVIEW_TICKS


@when("suppressing the action starts making things worse")
def _makes_things_worse(ctx):
    policy, rule = ctx["policy"], ctx["rule"]
    start = rule.activated_tick + 1
    for offset in range(60):
        applied = offset % 2 == 0
        policy.shadow.record(rule.id, applied, 0.2 if applied else 0.9,
                             start + offset)
    ctx["tick"] = rule.activated_tick + cfg.POLICY_REVIEW_TICKS


@then("the rule should be retired")
def _retired(ctx):
    assert ctx["rule"].status == RETIRED


@then("the rule should be refuted")
def _refuted(ctx):
    assert ctx["rule"].status == REFUTED
    assert ctx["rule"].id in ctx["policy"].lifecycle.refuted


@then("the action should be offered again")
def _offered_again(ctx):
    surviving = ctx["policy"].apply_rules(LOW, _plans(BAD, GOOD), 99_999)
    assert BAD in [plan.action for plan in surviving]


@then("re-mining the same evidence should not bring it back")
def _never_returns(ctx):
    ctx["tick"] = 200_000
    _fails_repeatedly(ctx)
    admitted = ctx["policy"].mine(300_000, safety_critical=[])
    assert ctx["rule"].id not in [rule.id for rule in admitted]


# ── shared helpers ───────────────────────────────────────────────────

def _suppression(ctx):
    return next((rule for rule in ctx["admitted"]
                 if rule.effect == SUPPRESS and rule.action == BAD), None)


def _run_trial(ctx, applied, withheld):
    policy, rule = ctx["policy"], ctx["rule"]
    tick = ctx["tick"]
    for offset in range(policy.shadow.trial_ticks + 1):
        current = tick + offset
        surviving = policy.apply_rules(LOW, _plans(BAD, GOOD), current)
        reward = applied if policy.shadow.applies_this_tick(rule, current) \
            else withheld
        chosen = surviving[0].action if surviving else GOOD
        policy.observe(LOW, chosen, reward=reward, success=reward > 0.5,
                       tick=current)
    ctx["tick"] = tick + policy.shadow.trial_ticks + 1
