"""Mutation-killing tests for the behaviour policy (spec §3.7).

Fifteen mutants survived the first harness run against this contour, and they
fell into three groups worth naming, because each is a different way for a test
suite to be reassuring and empty:

* **Default store paths.** Every store took ``store_path or DIR / "file.json"``,
  and nothing ever constructed one without a path — so the fallback was never
  executed and could have pointed anywhere.
* **Arithmetic nobody pinned.** Retention scores, success rates, elapsed ticks,
  truncation thresholds. The tests asserted that a number came out and that it
  had the right sign; a factor of ten either way passed.
* **Boundary conditions on the review window and the trial arms.** These decide
  *whether a rule is judged at all* and *which arm a sample lands in* — the two
  places where a wrong answer is silent by construction.
"""
import json

import pytest

import aegis.config as cfg
from aegis.layers.planner import Plan
from aegis.layers.policy import ACTIVE, SUPPRESS, BehaviourPolicy
from aegis.layers.policy.counterfactual import ShadowEvaluator
from aegis.layers.policy.rules import (
    RETIRED, Rule, RuleLifecycle, RuleMiner, rule_id,
)
from aegis.layers.policy.store import PolicyStore
from aegis.layers.world.state import StateKey

LOW = StateKey(energy="lo", mood="tired", mode="focused")
HIGH = StateKey(energy="hi", mood="curious", mode="focused")
BAD = "learn_external"
GOOD = "consolidate_memory"


def plans(*actions):
    built = []
    for index, action in enumerate(actions):
        plan = Plan(objective=f"objective_{action}", steps=[action])
        plan.score = 1.0 - index * 0.1
        built.append(plan)
    return built


# ── the default paths, which nothing had ever taken ──────────────────

def test_every_store_defaults_under_the_configured_policy_directory(monkeypatch,
                                                                    tmp_path):
    """`store_path or POLICY_DIR / name` — the fallback branch.

    Every test passed an explicit path, so this expression had never run. A
    store whose default landed somewhere else would persist to the wrong place
    in production and nowhere in the tests.
    """
    monkeypatch.setattr(cfg, "POLICY_DIR", tmp_path)
    assert PolicyStore()._store_path == tmp_path / "preferences.json"
    assert RuleLifecycle()._store_path == tmp_path / "rules.json"
    assert ShadowEvaluator()._store_path == tmp_path / "shadow.jsonl"

    policy = BehaviourPolicy()
    assert policy.store._store_path == tmp_path / "preferences.json"
    assert policy.lifecycle._store_path == tmp_path / "rules.json"
    assert policy.shadow._store_path == tmp_path / "shadow.jsonl"
    assert policy._experience_path == tmp_path / "experiences.jsonl"


# ── arithmetic ───────────────────────────────────────────────────────

def test_the_retention_score_is_strength_times_confidence():
    """`|w| · min(1, n/10)`.

    Both halves matter: a strong weight seen twice is not knowledge, and ten
    observations of nothing are not either. The cap at ten stops a much-visited
    but uninformative pair from outranking a decisive one.
    """
    score = PolicyStore._retention_score
    assert score({"w": 0.5, "n": 5}) == pytest.approx(0.25)
    assert score({"w": 0.5, "n": 10}) == pytest.approx(0.5)
    assert score({"w": 0.5, "n": 1000}) == pytest.approx(0.5)   # capped
    assert score({"w": -0.8, "n": 20}) == pytest.approx(0.8)    # sign-blind
    assert score({"w": 0.0, "n": 100}) == 0.0
    assert score({}) == 0.0


def test_a_mined_rule_reports_the_proportion_it_actually_saw():
    """`successes / len(matched)` — a rate, not a count."""
    miner = RuleMiner(min_support=10, max_condition_size=1, alpha=0.05)
    rows = []
    for index in range(30):                      # 6 of 30 succeed -> 0.2
        rows.append({"tick": index, "state": LOW.key(), "action": BAD,
                     "success": index < 6, "reward": 0.5})
    for index in range(30):                      # 27 of 30 succeed -> 0.9
        rows.append({"tick": 100 + index, "state": HIGH.key(), "action": BAD,
                     "success": index < 27, "reward": 0.5})

    found = miner.mine(rows, tick=1)
    suppress = next(rule for rule in found
                    if rule.effect == SUPPRESS and rule.state_condition.get("energy") == "lo")
    assert suppress.success_rate == pytest.approx(0.2)
    assert suppress.support == 30
    assert suppress.base_rate == pytest.approx(0.55)


def test_elapsed_ticks_are_measured_from_the_trial_start():
    """`tick − started`, not `tick + started`.

    Adding instead of subtracting still yields blocks that alternate, so every
    test that only checked "the arms alternate" passed. What it destroys is the
    *phase*: the first block of the trial stops being the first block.
    """
    shadow = ShadowEvaluator(store_path=None, trial_ticks=40)

    class _Rule:
        id = "r"
        trial_started = 5

    rule = _Rule()
    assert shadow.applies_this_tick(rule, 5) is True      # tick 0 of the trial
    assert shadow.applies_this_tick(rule, 14) is True     # still block 0
    assert shadow.applies_this_tick(rule, 15) is False    # block 1
    # A tick before the trial began cannot be negative into a previous block.
    assert shadow.applies_this_tick(rule, 0) is True

    # The same arithmetic governs the monitoring holdout of an active rule, and
    # there the phase is what decides whether the action is offered at all.
    class _Active:
        id = "a"
        activated_tick = 5

    active = _Active()
    assert shadow.acts_while_active(active, 5) is True     # block 0
    assert shadow.acts_while_active(active, 35) is False   # block 3 — holdout
    assert shadow.acts_while_active(active, 45) is True    # block 4


def test_the_shadow_log_truncates_at_twice_its_budget(tmp_path, monkeypatch):
    """The hysteresis is the point: truncate at 2×, keep 1×.

    Rewriting the file on every append past the limit would put a full read and
    replace on the critical path of a tick; a threshold that is not twice the
    budget either truncates constantly or never.
    """
    import aegis.layers.policy.counterfactual as module

    monkeypatch.setattr(module, "MAX_SHADOW_ROWS", 10)
    path = tmp_path / "shadow.jsonl"
    shadow = ShadowEvaluator(store_path=path)

    for index in range(20):
        shadow.record("r", True, 0.5, tick=index)
        shadow.flush()
    # Exactly at 2× and not a row over: nothing has been rewritten yet.
    assert len(path.read_text(encoding="utf-8").splitlines()) == 20
    assert shadow.truncations == 0

    shadow.record("r", True, 0.5, tick=20)
    shadow.flush()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10                                # cut back to 1×
    assert json.loads(lines[-1])["tick"] == 20             # the recent end
    assert json.loads(lines[0])["tick"] == 11
    assert shadow.truncations == 1


def test_a_drifted_row_count_is_corrected_without_rewriting(tmp_path, monkeypatch):
    """`_truncate` is also how the tracked count recovers from drift.

    A restart, or anything that edited the file underneath, leaves the counter
    disagreeing with reality. Correcting it must not cost the log its contents.
    """
    import aegis.layers.policy.counterfactual as module

    monkeypatch.setattr(module, "MAX_SHADOW_ROWS", 10)
    path = tmp_path / "shadow.jsonl"
    path.write_text("".join(json.dumps({"tick": i}) + "\n" for i in range(6)),
                    encoding="utf-8")
    shadow = ShadowEvaluator(store_path=path)
    shadow._rows_written = 9999          # wildly wrong

    shadow._truncate()
    assert shadow._rows_written == 6
    assert shadow.truncations == 0
    assert len(path.read_text(encoding="utf-8").splitlines()) == 6


def test_the_shadow_log_creates_its_own_directory(tmp_path):
    """`mkdir(parents=True, exist_ok=True)`. The store path may be several
    levels below anything that exists yet, and a second flush must not trip over
    the directory it made itself."""
    path = tmp_path / "deep" / "deeper" / "shadow.jsonl"
    shadow = ShadowEvaluator(store_path=path)
    shadow.record("r", True, 0.5, tick=1)
    assert shadow.flush() == 1
    shadow.record("r", False, 0.4, tick=2)
    assert shadow.flush() == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_the_experience_log_creates_its_own_directory(tmp_path):
    """Isolated from the other stores on purpose.

    `save()` writes preferences first, and `write_store` already creates the
    tree — so with the whole facade running, this line's own `mkdir` is never
    the one that matters and its arguments are unobservable. The experience log
    has to stand on its own, because it is the only store here that is not
    written through `write_store`.
    """
    root = tmp_path / "deep" / "deeper"
    policy = BehaviourPolicy(store_dir=root)
    policy.observe(LOW, GOOD, reward=0.5, success=True, tick=1)
    policy.store.save = lambda: None
    policy.lifecycle.save = lambda: None

    policy.save()
    policy.save()          # again: the directory now exists
    rows = (root / "experiences.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1


# ── the review window ────────────────────────────────────────────────

def _active_rule(policy, activated_tick):
    condition = {"state": {"energy": "lo"}, "action": BAD}
    rule = Rule(id=rule_id(condition, SUPPRESS), condition=condition,
                effect=SUPPRESS, status=ACTIVE, activated_tick=activated_tick,
                support=30, p_value=0.001)
    policy.lifecycle.rules[rule.id] = rule
    for offset in range(40):
        policy.shadow.record(rule.id, offset % 2 == 0, 0.7,
                             activated_tick + offset)
    return rule


def test_a_rule_is_judged_on_time_since_it_activated(tmp_path):
    """`tick − (activated_tick or 0) < REVIEW_TICKS`.

    Two mutations hide here and both are invisible early in a run: adding
    instead of subtracting, and reading the activation tick as zero. Either one
    starts judging rules the moment the *absolute* tick counter passes the
    review interval, regardless of how long the rule has actually been active —
    so a rule activated at tick 1500 would be reviewed on its very next tick.
    """
    policy = BehaviourPolicy(store_dir=tmp_path, min_support=10, trial_ticks=40)
    rule = _active_rule(policy, activated_tick=1500)

    # 100 ticks in: far past the absolute interval, nowhere near its own.
    assert policy.review(1600)["retired"] == 0
    assert rule.status == ACTIVE

    # And once its own window has passed, it is judged.
    assert policy.review(1500 + cfg.POLICY_REVIEW_TICKS)["retired"] == 1
    assert rule.status == RETIRED


def test_a_significant_but_negligible_effect_still_retires_an_active_rule(tmp_path):
    """`not significant OR effect < min_effect` — either is enough to retire.

    Requiring both would keep a rule alive on a difference that is real and
    far too small to justify removing an option from consideration.
    """
    lifecycle = RuleLifecycle(store_path=tmp_path / "rules.json")
    condition = {"state": {"energy": "lo"}, "action": BAD}
    rule = Rule(id=rule_id(condition, SUPPRESS), condition=condition,
                effect=SUPPRESS, status=ACTIVE, activated_tick=1)
    lifecycle.rules[rule.id] = rule

    # Highly significant (tight, separated samples) but only 0.005 apart.
    applied = [0.500 + 0.0002 * (index % 3) for index in range(60)]
    withheld = [0.495 + 0.0002 * (index % 3) for index in range(60)]
    verdict = lifecycle.review(rule, applied, withheld, tick=2000,
                               min_effect=0.03)
    assert verdict == RETIRED


# ── which arm a sample lands in ──────────────────────────────────────

def test_a_withheld_rule_records_into_the_withheld_arm(tmp_path):
    """`setdefault(rule.id, False)` — the default is *did not act*.

    A rule held back for its ABAB block that matches nothing on offer this tick
    is still part of the trial, and its reward belongs to the arm it was in.
    Defaulting to True would file every such tick as evidence that the rule
    acted, which is the one mistake that makes a trial argue for itself.
    """
    policy = BehaviourPolicy(store_dir=tmp_path, min_support=10, trial_ticks=40)
    condition = {"state": {"energy": "lo"}, "action": BAD}
    rule = Rule(id=rule_id(condition, SUPPRESS), condition=condition,
                effect=SUPPRESS, status="trial", trial_started=0)
    policy.lifecycle.rules[rule.id] = rule

    # Tick 10 falls in a withheld block, and nothing on offer is the rule's
    # action — so the rule is eligible-but-inert and matched no plan.
    assert policy.shadow.applies_this_tick(rule, 10) is False
    policy.apply_rules(LOW, plans(GOOD), tick=10)
    policy.observe(LOW, GOOD, reward=0.9, success=True, tick=10)

    applied, withheld = policy.shadow.arms(rule.id)
    assert applied == []
    assert withheld == [0.9]
