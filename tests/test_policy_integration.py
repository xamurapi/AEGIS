"""The acceptance scenarios of §M3.8: experience actually changing behaviour.

These drive the whole contour — observe, mine, trial, review, apply — rather
than any one piece of it, because the claim being tested is end-to-end:

    действие → результат → оценка → новое знание → изменение поведения

Two scenarios, and the second matters as much as the first. A system that can
only learn to stop doing things is not learning, it is decaying: the rule has to
come back off again when the world changes.
"""
import pytest

import aegis.config as cfg
from aegis.layers.policy import ACTIVE, SUPPRESS, TRIAL, BehaviourPolicy
from aegis.layers.planner import Plan
from aegis.layers.world.state import StateKey

LOW = StateKey(energy="lo", mood="tired", mode="focused")
HIGH = StateKey(energy="hi", mood="curious", mode="focused")

#: The action that will be learned about, and a harmless alternative.
BAD = "learn_external"
GOOD = "consolidate_memory"


def plans(*actions, safety_critical=()):
    out = []
    for index, action in enumerate(actions):
        plan = Plan(objective=f"objective_{action}", steps=[action],
                    safety_critical=action in safety_critical)
        plan.score = 1.0 - index * 0.1
        out.append(plan)
    return out


@pytest.fixture
def policy(tmp_path):
    return BehaviourPolicy(store_dir=tmp_path, min_support=10, trial_ticks=40)


def teach_failure(policy, repetitions=30, start=0):
    """`BAD` fails in LOW and works in HIGH — a state-conditional fact."""
    tick = start
    for _ in range(repetitions):
        policy.observe(LOW, BAD, reward=0.05, success=False, tick=tick)
        tick += 1
        policy.observe(HIGH, BAD, reward=0.95, success=True, tick=tick)
        tick += 1
        policy.observe(LOW, GOOD, reward=0.6, success=True, tick=tick)
        tick += 1
    return tick


def run_trial(policy, rule, first_tick, ticks, reward_when_applied,
              reward_when_withheld):
    """Drive a rule's ABAB trial and hand each arm its reward."""
    for offset in range(ticks):
        tick = first_tick + offset
        surviving = policy.apply_rules(LOW, plans(BAD, GOOD), tick)
        applied = policy.shadow.applies_this_tick(rule, tick)
        reward = reward_when_applied if applied else reward_when_withheld
        chosen = surviving[0].action if surviving else GOOD
        policy.observe(LOW, chosen, reward=reward, success=reward > 0.5,
                       tick=tick)
    return first_tick + ticks


# ── scenario 1: suppression ──────────────────────────────────────────

def test_a_repeatedly_failing_action_is_suppressed_where_it_fails(policy):
    """§M3.8, first scenario, in full."""
    tick = teach_failure(policy)

    admitted = policy.mine(tick, safety_critical=["health_check"])
    rule = next((r for r in admitted
                 if r.effect == SUPPRESS and r.action == BAD), None)
    assert rule is not None, "30 failures in one state have to be findable"
    assert rule.state_condition.get("energy") == "lo"
    assert rule.status == TRIAL

    # On trial it acts only on its "applied" blocks; suppressing BAD pays.
    tick = run_trial(policy, rule, tick, ticks=policy.shadow.trial_ticks + 1,
                     reward_when_applied=0.8, reward_when_withheld=0.3)

    outcomes = policy.review(tick)
    assert outcomes[ACTIVE] >= 1
    assert rule.status == ACTIVE
    assert rule.measured_effect > 0

    # One finding, not a dozen restatements of it: the miner must not emit a
    # two-field rule whose one-field parent already says the same thing.
    conditions = [frozenset(r.state_condition.items()) for r in admitted
                  if r.effect == SUPPRESS and r.action == BAD]
    assert not any(a < b for a in conditions for b in conditions)


def test_an_active_rule_removes_the_action_where_it_fails(policy):
    rule = _activate(policy)
    surviving = policy.apply_rules(LOW, plans(BAD, GOOD), tick=10_000)
    assert [plan.action for plan in surviving] == [GOOD]
    assert policy.suppressed_actions(LOW) == [BAD]


def test_the_same_action_stays_available_in_other_states(policy):
    """The point of a state condition: the rule is about *there*, not about the
    action. Suppressing it everywhere would be a much larger claim than the
    evidence supports."""
    _activate(policy)
    surviving = policy.apply_rules(HIGH, plans(BAD, GOOD), tick=10_000)
    assert BAD in [plan.action for plan in surviving]
    assert policy.suppressed_actions(HIGH) == []


def test_the_whole_scenario_fits_inside_the_specified_window(policy):
    """§M3.8 asks for this within `mine_every + trial_ticks` ticks."""
    tick = teach_failure(policy)
    started = tick
    admitted = policy.mine(tick, safety_critical=[])
    rule = next(r for r in admitted if r.effect == SUPPRESS and r.action == BAD)
    tick = run_trial(policy, rule, tick, policy.shadow.trial_ticks + 1,
                     reward_when_applied=0.8, reward_when_withheld=0.3)
    policy.review(tick)
    assert rule.status == ACTIVE
    assert tick - started <= cfg.POLICY_MINE_EVERY_N_TICKS + policy.shadow.trial_ticks


# ── scenario 2: the world changes back ───────────────────────────────

def test_a_rule_retires_once_it_stops_paying(policy):
    """§M3.8, second scenario. The action starts working again, so suppressing
    it no longer helps, and the rule has to come off."""
    rule = _activate(policy)
    activated_at = rule.activated_tick

    # Now both arms pay the same: suppression buys nothing any more.
    tick = activated_at + 1
    for offset in range(80):
        policy.apply_rules(LOW, plans(BAD, GOOD), tick + offset)
        policy.shadow.record(rule.id,
                             policy.shadow.applies_this_tick(rule, tick + offset),
                             0.7, tick + offset)

    outcomes = policy.review(activated_at + cfg.POLICY_REVIEW_TICKS)
    assert outcomes["retired"] >= 1
    assert rule.status == "retired"
    # And the action is offerable again.
    surviving = policy.apply_rules(LOW, plans(BAD, GOOD), tick=99_999)
    assert BAD in [plan.action for plan in surviving]


def test_a_rule_that_reverses_is_refuted_and_never_returns(policy):
    rule = _activate(policy)
    activated_at = rule.activated_tick
    policy.shadow.forget(rule.id)
    for offset in range(60):
        applied = offset % 2 == 0
        policy.shadow.record(rule.id, applied,
                             0.2 if applied else 0.9, activated_at + offset)

    policy.review(activated_at + cfg.POLICY_REVIEW_TICKS)
    assert rule.status == "refuted"
    assert rule.id in policy.lifecycle.refuted

    # Re-mining the same evidence must not bring it back.
    teach_failure(policy, repetitions=30, start=200_000)
    admitted = policy.mine(300_000)
    assert rule.id not in [r.id for r in admitted]


# ── safety ───────────────────────────────────────────────────────────

def test_a_safety_critical_plan_survives_a_suppression(policy):
    """Even if a rule for the action somehow exists, the plan is protected."""
    rule = _activate(policy)
    assert rule.effect == SUPPRESS
    surviving = policy.apply_rules(LOW, plans(BAD, GOOD, safety_critical=[BAD]),
                                   tick=10_000)
    assert BAD in [plan.action for plan in surviving]


def test_the_miner_is_never_offered_a_safety_critical_suppression(policy):
    tick = teach_failure(policy)
    admitted = policy.mine(tick, safety_critical=[BAD])
    assert all(not (rule.effect == SUPPRESS and rule.action == BAD)
               for rule in admitted)


# ── measurement ──────────────────────────────────────────────────────

def test_behaviour_delta_rate_rises_once_a_rule_acts(policy):
    assert policy.behaviour_delta_rate() == 0.0
    _activate(policy)
    for tick in range(50):
        policy.apply_rules(LOW, plans(BAD, GOOD), 20_000 + tick)
    assert policy.behaviour_delta_rate() > 0


def test_behaviour_delta_rate_stays_zero_without_rules(policy):
    for tick in range(50):
        policy.apply_rules(LOW, plans(BAD, GOOD), tick)
    assert policy.behaviour_delta_rate() == 0.0


def test_preferences_alone_never_remove_an_option(policy):
    """A weight shifts a ranking; only a rule can take a choice away. Keeping
    those powers separate is what stops a run of bad luck from silently
    disabling something."""
    teach_failure(policy, repetitions=20)
    surviving = policy.apply_rules(LOW, plans(BAD, GOOD), tick=1)
    assert len(surviving) == 2
    assert policy.delta(LOW, BAD) < 0        # it did learn, quietly


def test_a_preferred_rule_promotes_rather_than_removes(policy, tmp_path):
    from aegis.layers.policy.rules import PREFER, PREFER_BONUS, Rule, rule_id

    condition = {"state": {"energy": "lo"}, "action": GOOD}
    rule = Rule(id=rule_id(condition, PREFER), condition=condition,
                effect=PREFER, status=ACTIVE, activated_tick=1)
    policy.lifecycle.rules[rule.id] = rule

    candidates = plans(BAD, GOOD)
    before = candidates[1].score
    surviving = policy.apply_rules(LOW, candidates, tick=10)
    assert len(surviving) == 2
    assert surviving[0].action == GOOD                 # promoted to the top
    assert surviving[0].score == pytest.approx(before + PREFER_BONUS)


# ── persistence ──────────────────────────────────────────────────────

def test_the_whole_contour_survives_a_restart(policy, tmp_path):
    rule = _activate(policy)
    policy.save()

    restored = BehaviourPolicy(store_dir=tmp_path, min_support=10, trial_ticks=40)
    assert rule.id in [r.id for r in restored.active_rules()]
    assert restored.experiences
    assert restored.delta(LOW, BAD) == pytest.approx(policy.delta(LOW, BAD))
    surviving = restored.apply_rules(LOW, plans(BAD, GOOD), tick=10_000)
    assert [plan.action for plan in surviving] == [GOOD]


def test_a_torn_experience_log_costs_one_row(tmp_path):
    path = tmp_path / "experiences.jsonl"
    path.write_text('{"state": "s", "action": "a", "success": true}\n'
                    '\n'                        # a blank line from a flush
                    'not json\n'                # a line torn by a crash
                    '{"no action here": 1}\n'   # a row from an older schema
                    '{"state": "s", "action": "b", "success": false}\n',
                    encoding="utf-8")
    policy = BehaviourPolicy(store_dir=tmp_path)
    assert [row["action"] for row in policy.experiences] == ["a", "b"]


def test_the_experience_log_is_bounded(policy, monkeypatch):
    import aegis.layers.policy as module

    monkeypatch.setattr(module, "MAX_EXPERIENCE_ROWS", 25)
    for tick in range(100):
        policy.observe(LOW, GOOD, reward=0.5, success=True, tick=tick)
    assert len(policy.experiences) == 25
    assert policy.experiences[-1]["tick"] == 99


# ── genome and reporting ─────────────────────────────────────────────

def test_the_genome_retunes_the_policy(policy):
    policy.set_genome({"policy_weight": 0.75, "policy_min_support": 42})
    assert policy.store.weight == 0.75
    assert policy.miner.min_support == 42


def test_an_unusable_gene_is_ignored(policy):
    policy.set_genome({"policy_weight": "strong", "policy_min_support": None})
    assert policy.store.weight == cfg.POLICY_WEIGHT
    assert policy.miner.min_support == 10


def test_the_policy_weight_gene_is_bounded(policy):
    policy.set_genome({"policy_weight": 9.0})
    assert policy.store.weight == 1.0


def test_status_covers_every_part(policy):
    _activate(policy)
    status = policy.status()
    assert status["rules"]["active"] >= 1
    assert status["preferences"]["preferences"] > 0
    assert status["shadow"]["ticks_seen"] > 0
    assert status["experiences"] > 0
    assert status["suppressions"] >= 0


def test_metrics_are_published_for_every_required_name(policy):
    from aegis.telemetry import metrics as M

    recorded = []
    policy.telemetry = type("T", (), {
        "record": staticmethod(
            lambda name, value, tick, tags=None: recorded.append(name))})()
    policy.publish_metrics(1)
    assert set(recorded) == {
        M.POLICY_BEHAVIOUR_DELTA_RATE, M.POLICY_ACTIVE_RULES, M.POLICY_REFUTED,
        M.POLICY_CANDIDATES, M.POLICY_REGRET}


def test_a_failing_telemetry_sink_does_not_break_the_tick(policy):
    def _explode(*a, **k):
        raise RuntimeError("sink down")

    policy.telemetry = type("T", (), {"record": staticmethod(_explode)})()
    policy.publish_metrics(1)          # swallowed, logged, the tick continues


def test_no_telemetry_is_not_an_error(policy):
    policy.telemetry = None
    policy.publish_metrics(1)


def test_applying_rules_to_nothing_is_nothing(policy):
    assert policy.apply_rules(LOW, [], tick=1) == []


def test_a_plan_without_an_action_is_left_alone(policy):
    _activate(policy)
    empty = Plan(objective="nothing", steps=[])
    surviving = policy.apply_rules(LOW, [empty], tick=10_000)
    assert surviving == [empty]


def test_observing_without_a_state_or_action_records_nothing(policy):
    policy.observe(None, GOOD, 0.5, True, 1)
    policy.observe(LOW, "", 0.5, True, 1)
    assert policy.experiences == []


# ── shared setup ─────────────────────────────────────────────────────

def _activate(policy):
    """Take one rule all the way to `active` — the state most tests start from."""
    tick = teach_failure(policy)
    admitted = policy.mine(tick, safety_critical=[])
    rule = next(r for r in admitted if r.effect == SUPPRESS and r.action == BAD)
    tick = run_trial(policy, rule, tick, policy.shadow.trial_ticks + 1,
                     reward_when_applied=0.8, reward_when_withheld=0.3)
    policy.review(tick)
    assert rule.status == ACTIVE
    return rule


# ── the remaining degraded paths ─────────────────────────────────────

def test_an_experience_id_is_carried_into_the_evidence(policy):
    policy.observe(LOW, GOOD, reward=0.5, success=True, tick=1,
                   experience_id="exp_00000042")
    assert policy.experiences[-1]["id"] == "exp_00000042"


def test_a_trial_that_has_not_finished_is_left_alone(policy):
    _on_trial = teach_failure(policy)
    admitted = policy.mine(_on_trial, safety_critical=[])
    rule = next(r for r in admitted if r.effect == SUPPRESS and r.action == BAD)
    outcomes = policy.review(rule.trial_started + 1)
    assert outcomes == {ACTIVE: 0, "retired": 0, "refuted": 0}
    assert rule.status == TRIAL


def test_an_active_rule_is_not_re_judged_before_its_review_window(policy):
    rule = _activate(policy)
    outcomes = policy.review(rule.activated_tick + 1)
    assert outcomes[ACTIVE] == 0
    assert rule.status == ACTIVE


def test_a_surviving_review_restarts_the_clock(policy):
    rule = _activate(policy)
    for offset in range(60):
        applied = offset % 2 == 0
        policy.shadow.record(rule.id, applied, 0.9 if applied else 0.2,
                             rule.activated_tick + offset)
    review_tick = rule.activated_tick + cfg.POLICY_REVIEW_TICKS
    assert policy.review(review_tick)[ACTIVE] == 1
    assert rule.activated_tick == review_tick


def test_an_unreadable_experience_log_leaves_the_policy_empty(tmp_path, monkeypatch):
    (tmp_path / "experiences.jsonl").write_text("{}\n", encoding="utf-8")

    def _explode(*a, **k):
        raise OSError("unreadable")

    monkeypatch.setattr("pathlib.Path.open", _explode)
    assert BehaviourPolicy(store_dir=tmp_path).experiences == []


def test_an_unwritable_experience_log_costs_the_log_not_the_tick(policy, monkeypatch):
    policy.observe(LOW, GOOD, reward=0.5, success=True, tick=1)

    def _explode(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.write_text", _explode)
    policy.save()          # swallowed, logged, the caller continues


def test_regret_is_published_once_it_exists(policy):
    from aegis.telemetry import metrics as M

    recorded = {}
    policy.telemetry = type("T", (), {"record": staticmethod(
        lambda name, value, tick, tags=None: recorded.__setitem__(name, value))})()
    policy.shadow.note_regret(0.4, 0.9)
    policy.publish_metrics(1)
    assert recorded[M.POLICY_REGRET] == pytest.approx(0.5)
