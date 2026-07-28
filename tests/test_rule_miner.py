"""The rule miner (spec M3.4).

The single most important test in this file is
:func:`test_pure_noise_yields_no_rules`. A miner that enumerates every feature
subset against every action runs hundreds of comparisons per generation, and at
α = 0.05 roughly one in twenty pure-noise combinations clears an uncorrected
threshold. Without false-discovery control such a system would *always* find
rules — and a policy full of descriptions of nothing is worse than no policy,
because it looks like knowledge.

Everything else here checks that a real signal still gets through, and that a
rule carries enough evidence to be argued with.
"""
import pytest

from aegis.layers.policy.rules import PREFER, SUPPRESS, RuleMiner, rule_id
from aegis.layers.world.state import StateKey
from aegis.util.quasirandom import hash_unit


def _state(**fields) -> str:
    return StateKey(**fields).key()


def rows(state_fields, action, successes, failures, start=0):
    """A block of experiences for one (state, action) cell."""
    key = _state(**state_fields)
    out = []
    for index in range(successes):
        out.append({"tick": start + index, "state": key, "action": action,
                    "success": True, "reward": 0.9, "id": f"exp_{start + index}"})
    for index in range(failures):
        position = start + successes + index
        out.append({"tick": position, "state": key, "action": action,
                    "success": False, "reward": 0.1, "id": f"exp_{position}"})
    return out


@pytest.fixture
def miner():
    return RuleMiner(min_support=10, max_condition_size=2, alpha=0.05)


# ── the property the whole contour depends on ────────────────────────

def test_pure_noise_yields_no_rules():
    """Success independent of state, over hundreds of comparisons.

    The outcome is a hash of the row index alone — nothing about the state or
    the action carries any information about it. A miner that returns anything
    here is manufacturing knowledge.
    """
    miner = RuleMiner(min_support=10, max_condition_size=2, alpha=0.05)
    energies = ("lo", "mid", "hi")
    moods = ("curious", "tired", "calm")
    actions = ("rest", "dream", "learn_external", "self_inspect")

    experiences = []
    for index in range(1200):
        experiences.append({
            "tick": index,
            "state": _state(energy=energies[index % 3],
                            mood=moods[(index // 3) % 3],
                            mode="focused"),
            "action": actions[(index // 9) % 4],
            "success": hash_unit("noise", index) < 0.5,
            "reward": hash_unit("noise_r", index),
        })

    assert miner.mine(experiences, tick=100) == []
    assert miner.tested > 100          # it really did look at many combinations


def test_noise_survives_repeated_generations():
    """Mining repeatedly must not eventually find something by persistence."""
    miner = RuleMiner(min_support=10, max_condition_size=2, alpha=0.05)
    found = []
    for generation in range(5):
        experiences = [{
            "tick": index,
            "state": _state(energy=("lo", "mid", "hi")[index % 3],
                            mode="focused"),
            "action": ("rest", "dream")[(index // 3) % 2],
            "success": hash_unit("noise", generation, index) < 0.5,
            "reward": 0.5,
        } for index in range(600)]
        found.extend(miner.mine(experiences, tick=generation))
    assert found == []


# ── a real signal gets through ───────────────────────────────────────

def test_an_action_that_fails_in_one_state_is_suppressed(miner):
    """The acceptance scenario of §M3.8, at the miner's level."""
    experiences = (rows({"energy": "lo"}, "learn_external", 0, 30)
                   + rows({"energy": "hi"}, "learn_external", 28, 2, start=100))
    found = miner.mine(experiences, tick=200)

    suppressions = [rule for rule in found if rule.effect == SUPPRESS]
    assert suppressions, "a 0/30 failure rate in one state has to be findable"
    rule = suppressions[0]
    assert rule.action == "learn_external"
    assert rule.state_condition == {"energy": "lo"}
    assert rule.support == 30
    assert rule.success_rate == 0.0
    assert rule.wilson_high < rule.base_rate


def test_an_action_that_shines_in_one_state_is_preferred(miner):
    experiences = (rows({"energy": "hi"}, "env_step", 30, 0)
                   + rows({"energy": "lo"}, "env_step", 2, 28, start=100))
    found = miner.mine(experiences, tick=200)
    preferences = [rule for rule in found if rule.effect == PREFER]
    assert preferences
    assert preferences[0].state_condition == {"energy": "hi"}
    assert preferences[0].wilson_low > preferences[0].base_rate


def test_a_rule_carries_the_evidence_that_produced_it(miner):
    experiences = (rows({"energy": "lo"}, "learn_external", 0, 30)
                   + rows({"energy": "hi"}, "learn_external", 28, 2, start=100))
    rule = miner.mine(experiences, tick=7)[0]
    assert rule.created_tick == 7
    assert rule.provenance and all(p.startswith("exp_") for p in rule.provenance)
    assert len(rule.provenance) <= 10
    assert 0.0 <= rule.wilson_low <= rule.wilson_high <= 1.0
    assert 0.0 <= rule.p_value <= 1.0


# ── the thresholds ───────────────────────────────────────────────────

def test_too_little_evidence_is_no_rule():
    """Support below the threshold is not a finding, however extreme."""
    miner = RuleMiner(min_support=50, max_condition_size=2)
    experiences = (rows({"energy": "lo"}, "learn_external", 0, 30)
                   + rows({"energy": "hi"}, "learn_external", 28, 2, start=100))
    assert miner.mine(experiences, tick=1) == []


def test_an_empty_log_is_mined_without_incident(miner):
    assert miner.mine([], tick=1) == []
    assert miner.mine(None, tick=1) == []
    assert miner.generations == 2


def test_unusable_rows_are_skipped(miner):
    experiences = [
        "not a row", {"no": "action"}, {"action": "rest"},
        {"action": "rest", "state": None, "success": True},
    ] + rows({"energy": "lo"}, "learn_external", 0, 30) \
      + rows({"energy": "hi"}, "learn_external", 28, 2, start=100)
    assert miner.mine(experiences, tick=1)          # the good rows still work


def test_a_cell_with_no_comparison_group_is_skipped(miner):
    """A subset compared against a pool that contains it is compared with
    itself; if nothing is left over there is no comparison to make."""
    experiences = rows({"energy": "lo"}, "learn_external", 0, 30)
    assert miner.mine(experiences, tick=1) == []


def test_a_difference_too_small_to_act_on_is_not_a_rule():
    """Statistically detectable is not the same as worth changing behaviour
    over: the interval has to sit clear of the base rate."""
    miner = RuleMiner(min_support=10, max_condition_size=1, alpha=0.5)
    experiences = (rows({"energy": "lo"}, "rest", 15, 15)
                   + rows({"energy": "hi"}, "rest", 16, 14, start=100))
    for rule in miner.mine(experiences, tick=1):
        if rule.effect == SUPPRESS:
            assert rule.wilson_high < rule.base_rate
        else:
            assert rule.wilson_low > rule.base_rate


# ── the shape of the search ──────────────────────────────────────────

def test_conditions_are_tried_smallest_first(miner):
    shapes = list(miner._condition_shapes())
    widths = [len(shape) for shape in shapes]
    assert widths == sorted(widths)
    assert widths[0] == 1 and max(widths) == 2


def test_the_condition_width_is_bounded_by_configuration():
    assert max(len(shape) for shape in
               RuleMiner(max_condition_size=1)._condition_shapes()) == 1
    assert max(len(shape) for shape in
               RuleMiner(max_condition_size=3)._condition_shapes()) == 3


def test_an_absurd_condition_width_is_clamped_to_the_fields_available():
    from aegis.layers.world.state import FIELDS

    widest = max(len(shape) for shape in
                 RuleMiner(max_condition_size=99)._condition_shapes())
    assert widest == len(FIELDS)


def test_mining_the_same_log_twice_gives_the_same_rules(miner):
    experiences = (rows({"energy": "lo"}, "learn_external", 0, 30)
                   + rows({"energy": "hi"}, "learn_external", 28, 2, start=100))
    first = [rule.id for rule in miner.mine(experiences, tick=1)]
    second = [rule.id for rule in
              RuleMiner(min_support=10, max_condition_size=2).mine(experiences, 1)]
    assert first == second


def test_rules_come_back_strongest_first(miner):
    experiences = (rows({"energy": "lo"}, "learn_external", 0, 40)
                   + rows({"energy": "hi"}, "learn_external", 38, 2, start=100)
                   + rows({"mood": "tired"}, "dream", 5, 25, start=300)
                   + rows({"mood": "calm"}, "dream", 20, 10, start=400))
    found = miner.mine(experiences, tick=1)
    assert [rule.p_value for rule in found] == sorted(rule.p_value for rule in found)


# ── identity ─────────────────────────────────────────────────────────

def test_a_rule_id_is_derived_from_what_it_says():
    """Stable identity, so a rule re-mined next generation is recognised as the
    one already on trial rather than proposed all over again."""
    first = rule_id({"state": {"energy": "lo", "mood": "tired"},
                     "action": "rest"}, SUPPRESS)
    second = rule_id({"state": {"mood": "tired", "energy": "lo"},
                      "action": "rest"}, SUPPRESS)
    assert first == second
    assert rule_id({"state": {"energy": "hi"}, "action": "rest"},
                   SUPPRESS) != first
    assert rule_id({"state": {"energy": "lo", "mood": "tired"},
                    "action": "rest"}, PREFER) != first


def test_status_reports_what_the_miner_has_done(miner):
    miner.mine(rows({"energy": "lo"}, "learn_external", 0, 30)
               + rows({"energy": "hi"}, "learn_external", 28, 2, start=100), 1)
    status = miner.status()
    assert status["generations"] == 1
    assert status["tested"] > 0
    assert status["proposed"] >= 1
    assert status["min_support"] == 10 and status["alpha"] == 0.05
