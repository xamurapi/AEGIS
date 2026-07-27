"""Looking ahead (spec M1.5, M1.9).

The rollout is what turns a model of the world into a decision, so the
properties that matter are: it gives the same answer twice, it prefers what the
evidence supports, it does not run away, and it fits inside the latency budget
the planner has to live in.
"""
import time

import pytest

from aegis.layers.world.outcome import OutcomeModel
from aegis.layers.world.simulate import RolloutResult, Simulator
from aegis.layers.world.state import StateKey
from aegis.layers.world.transition import TransitionModel


def state(name: str) -> StateKey:
    return StateKey(energy=name)


def build(*, half_life=0, explore_bonus=0.0):
    transitions = TransitionModel(half_life=half_life, min_n=3)
    outcomes = OutcomeModel(half_life=half_life, min_n=3)
    simulator = Simulator(transitions, outcomes, explore_bonus=explore_bonus,
                          discount=0.9, branch=3)
    return transitions, outcomes, simulator


def teach(transitions, outcomes, source, action, target, *, reward, success=True,
          times=10, cost=0.0):
    for _ in range(times):
        transitions.observe(source, action, target)
        outcomes.observe(source, action, success=success, reward=reward, cost=cost)


# ── choosing ─────────────────────────────────────────────────────────

def test_the_better_paying_action_wins():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "rich", state("s"), reward=0.9)
    teach(transitions, outcomes, state("s"), "poor", state("s"), reward=0.1)
    assert simulator.rollout(state("s"), ["rich", "poor"], depth=1).sequence[0] == "rich"


def test_a_reliable_action_beats_an_erratic_one():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "steady", state("s"),
          reward=0.6, success=True, times=20)
    for i in range(20):
        transitions.observe(state("s"), "erratic", state("s"))
        outcomes.observe(state("s"), "erratic", success=(i % 2 == 0), reward=0.6)
    assert simulator.rollout(state("s"), ["steady", "erratic"], depth=1).sequence[0] \
        == "steady"


def test_an_expensive_action_is_penalised():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "cheap", state("s"), reward=0.6, cost=0.0)
    teach(transitions, outcomes, state("s"), "dear", state("s"), reward=0.6, cost=5.0)
    assert simulator.rollout(state("s"), ["cheap", "dear"], depth=1).sequence[0] \
        == "cheap"


def test_a_future_payoff_is_found_through_a_barren_step():
    # Depth is the point: a step worth nothing now that unlocks a rich state
    # should still win, and a one-step search cannot see that.
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "invest", state("rich"), reward=0.0)
    teach(transitions, outcomes, state("s"), "nibble", state("s"), reward=0.2)
    teach(transitions, outcomes, state("rich"), "invest", state("rich"), reward=1.0)
    teach(transitions, outcomes, state("rich"), "nibble", state("rich"), reward=1.0)

    assert simulator.rollout(state("s"), ["invest", "nibble"], depth=1).sequence[0] \
        == "nibble"
    assert simulator.rollout(state("s"), ["invest", "nibble"], depth=3).sequence[0] \
        == "invest"


def test_the_exploration_bonus_favours_the_unknown():
    transitions, outcomes, simulator = build(explore_bonus=0.5)
    teach(transitions, outcomes, state("s"), "known", state("s"), reward=0.3)
    assert simulator.rollout(state("s"), ["known", "untried"], depth=1).sequence[0] \
        == "untried"


def test_without_a_bonus_the_known_option_wins():
    transitions, outcomes, simulator = build(explore_bonus=0.0)
    teach(transitions, outcomes, state("s"), "known", state("s"), reward=0.9)
    assert simulator.rollout(state("s"), ["known", "untried"], depth=1).sequence[0] \
        == "known"


# ── determinism (§3.1) ───────────────────────────────────────────────

def test_the_same_question_gives_the_same_answer():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "a", state("t"), reward=0.5)
    teach(transitions, outcomes, state("s"), "b", state("t"), reward=0.5)
    first = simulator.rollout(state("s"), ["a", "b"], depth=3)
    second = simulator.rollout(state("s"), ["a", "b"], depth=3)
    assert first.sequence == second.sequence
    assert first.value == second.value


def test_the_order_the_actions_are_offered_in_does_not_matter():
    transitions, outcomes, simulator = build()
    for name in ("a", "b", "c"):
        teach(transitions, outcomes, state("s"), name, state("t"), reward=0.5)
    assert simulator.rollout(state("s"), ["a", "b", "c"], depth=2).sequence == \
        simulator.rollout(state("s"), ["c", "b", "a"], depth=2).sequence


def test_a_duplicate_action_changes_nothing():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "a", state("t"), reward=0.5)
    assert simulator.rollout(state("s"), ["a", "a", "a"], depth=2).sequence == \
        simulator.rollout(state("s"), ["a"], depth=2).sequence


# ── limits ───────────────────────────────────────────────────────────

def test_no_actions_means_no_plan():
    _, _, simulator = build()
    result = simulator.rollout(state("s"), [], depth=3)
    assert result.sequence == [] and result.value == 0.0


def test_zero_depth_means_no_plan():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "a", state("t"), reward=0.5)
    assert simulator.rollout(state("s"), ["a"], depth=0).sequence == []


def test_the_plan_is_no_longer_than_the_depth():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "a", state("s"), reward=0.5)
    assert len(simulator.rollout(state("s"), ["a"], depth=3).sequence) <= 3


def test_expansion_is_capped():
    # A pathological branching factor must not turn a 30 ms budget into a stall.
    transitions = TransitionModel(half_life=0)
    outcomes = OutcomeModel(half_life=0)
    for i in range(40):
        for j in range(40):
            transitions.observe(state(f"s{i}"), f"a{j}", state(f"s{(i + j) % 40}"))
            outcomes.observe(state(f"s{i}"), f"a{j}", success=True, reward=0.5)
    simulator = Simulator(transitions, outcomes, max_nodes=50)
    result = simulator.rollout(state("s0"), [f"a{j}" for j in range(40)], depth=5)
    assert result.nodes_expanded <= 50
    assert result.truncated is True


def test_a_normal_search_is_not_truncated():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "a", state("s"), reward=0.5)
    assert simulator.rollout(state("s"), ["a"], depth=3).truncated is False


def test_repeated_states_are_memoised():
    transitions, outcomes, simulator = build()
    for name in ("a", "b", "c"):
        teach(transitions, outcomes, state("s"), name, state("s"), reward=0.5)
    assert simulator.rollout(state("s"), ["a", "b", "c"], depth=4).memo_hits > 0


# ── latency (§M1.9) ──────────────────────────────────────────────────

def test_a_depth_three_rollout_fits_the_budget():
    transitions = TransitionModel(half_life=0)
    outcomes = OutcomeModel(half_life=0)
    actions = [f"a{i}" for i in range(12)]
    for i in range(20):
        for action in actions:
            transitions.observe(state(f"s{i}"), action, state(f"s{(i + 3) % 20}"))
            outcomes.observe(state(f"s{i}"), action, success=True, reward=0.5)
    simulator = Simulator(transitions, outcomes)

    simulator.rollout(state("s0"), actions, depth=3, beam=5)     # warm the caches
    started = time.perf_counter()
    for _ in range(10):
        simulator.rollout(state("s0"), actions, depth=3, beam=5)
    average_ms = (time.perf_counter() - started) / 10 * 1000
    assert average_ms <= 15.0, f"rollout took {average_ms:.1f} ms"


# ── explaining ───────────────────────────────────────────────────────

def test_the_plan_comes_with_its_reasoning():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "a", state("t"), reward=0.7)
    teach(transitions, outcomes, state("t"), "a", state("t"), reward=0.7)
    result = simulator.rollout(state("s"), ["a"], depth=2)
    assert len(result.steps) == len(result.sequence)
    assert set(result.steps[0]) >= {"state", "action", "p_success",
                                    "expected_reward", "known", "next"}


def test_the_result_serialises():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "a", state("t"), reward=0.5)
    assert set(simulator.rollout(state("s"), ["a"], depth=1).as_dict()) >= \
        {"sequence", "value", "steps", "nodes_expanded", "elapsed_ms"}


def test_best_sequence_is_just_the_plan():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "a", state("t"), reward=0.5)
    assert simulator.best_sequence(state("s"), ["a"], depth=2) == \
        simulator.rollout(state("s"), ["a"], depth=2).sequence


# ── pricing someone else's plan ──────────────────────────────────────

def test_a_proposed_sequence_can_be_priced():
    # A plan suggested by the cortex has to be measured by the same yardstick
    # as the planner's own, not trusted.
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "good", state("s"), reward=0.9)
    teach(transitions, outcomes, state("s"), "bad", state("s"), reward=0.1)
    assert simulator.evaluate(state("s"), ["good", "good"]) > \
        simulator.evaluate(state("s"), ["bad", "bad"])


def test_pricing_an_empty_sequence_is_zero():
    _, _, simulator = build()
    assert simulator.evaluate(state("s"), []) == 0.0


def test_pricing_stops_where_the_model_runs_out():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "a", state("t"), reward=0.5)
    assert simulator.evaluate(state("s"), ["a", "a", "a"]) > 0


def test_a_bare_state_key_is_accepted():
    transitions, outcomes, simulator = build()
    teach(transitions, outcomes, state("s"), "a", state("t"), reward=0.5)
    assert simulator.rollout(state("s").key(), ["a"], depth=1).sequence == ["a"]


def test_status_reports_the_search_settings():
    _, _, simulator = build()
    simulator.rollout(state("s"), ["a"], depth=1)
    status = simulator.status()
    assert status["rollouts"] == 1
    assert status["branch"] == 3 and status["discount"] == 0.9
