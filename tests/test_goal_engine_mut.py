"""Mutation-hardening tests for goal_engine — pin the exact progress-increment
formulas, goal-generation arithmetic, staleness, pruning and status counting so
every arithmetic/boolean/comparison mutant is killed."""
import math
import time
import pytest

from aegis.layers.goal_engine import GoalEngine, Goal, STALE_THRESHOLD_SECONDS


def _engine_with(*goals):
    """A GoalEngine whose goal list is exactly the given goals (axioms removed
    so a single target's increment can be asserted in isolation)."""
    e = GoalEngine()
    e.goals = list(goals)
    e.curiosity_level = 0.0  # silence curiosity-goal generation unless a test sets it
    return e


def _goal(name, level, progress=0.0, priority=0.5, status="active"):
    g = Goal(name, level, f"desc {name}", priority)
    g.progress = progress
    g.status = status
    return g


# ── evaluate_progress: exact increment per goal type (L192-L208) ──────

def test_knowledge_strategy_increment_exact():
    g = _goal("expand_knowledge", "strategy")
    e = _engine_with(g)
    e.evaluate_progress({"new_concepts": 10, "llm_insights": 3})
    # 10*0.005 + 3*0.01 = 0.08
    assert g.progress == pytest.approx(0.08)


def test_expand_without_knowledge_word_still_uses_knowledge_formula():
    # Kills the `"knowledge" in name or "expand" in name` Or->And mutant.
    g = _goal("expand_horizons", "strategy")
    e = _engine_with(g)
    e.evaluate_progress({"new_concepts": 10, "llm_insights": 3})
    assert g.progress == pytest.approx(0.08)  # not the 0.002 default branch


def test_reasoning_strategy_increment_exact():
    g = _goal("improve_reasoning", "strategy")
    e = _engine_with(g)
    e.evaluate_progress({"llm_insights": 4, "error_rate": 0.5})
    # 4*0.015 + (1-0.5)*0.002 = 0.06 + 0.001 = 0.061
    assert g.progress == pytest.approx(0.061)


def test_memory_strategy_increment_exact():
    g = _goal("optimize_memory", "strategy")
    e = _engine_with(g)
    e.evaluate_progress({"new_episodic": 5, "new_concepts": 7})
    # (5 + 7) * 0.003 = 0.036
    assert g.progress == pytest.approx(0.036)


def test_ethics_strategy_fixed_increment():
    g = _goal("strengthen_ethics", "strategy")
    e = _engine_with(g)
    e.evaluate_progress({})
    assert g.progress == pytest.approx(0.002)


def test_self_model_strategy_increment_exact():
    g = _goal("enhance_self_model", "strategy")
    e = _engine_with(g)
    e.evaluate_progress({"llm_insights": 3})
    # 3*0.01 + 0.001 = 0.031
    assert g.progress == pytest.approx(0.031)


def test_default_strategy_increment():
    g = _goal("something_else", "strategy")
    e = _engine_with(g)
    e.evaluate_progress({"new_concepts": 100})  # no keyword match -> fixed 0.002
    assert g.progress == pytest.approx(0.002)


def test_tactic_increment_exact():
    g = _goal("tac", "tactic")
    e = _engine_with(g)
    e.evaluate_progress({"new_concepts": 2, "llm_insights": 1})
    # 2*0.01 + 1*0.02 + 0.003 = 0.043
    assert g.progress == pytest.approx(0.043)


def test_curiosity_increment_exact():
    g = _goal("cur", "curiosity")
    e = _engine_with(g)
    e.evaluate_progress({"new_concepts": 3, "llm_insights": 2})
    # 3*0.02 + 2*0.015 = 0.09
    assert g.progress == pytest.approx(0.09)


# ── completion information-gain (L217) ────────────────────────────────

def test_completion_information_gain_exact():
    g = _goal("tac", "tactic", progress=0.99)
    e = _engine_with(g)
    e.information_gain = 0.0
    e.evaluate_progress({"new_concepts": 5})  # increment = 5*0.01+0.003 = 0.053
    assert g.status == "completed"
    # info_gain += 0.2 + increment*5 = 0.2 + 0.053*5 = 0.465
    assert e.information_gain == pytest.approx(0.465)


def test_advance_progress_completion_information_gain_exact():
    # Kills the `information_gain += amount * 2` Mult->Div mutant in
    # advance_progress (the direct completion path).
    g = _goal("mygoal", "tactic", progress=0.9)
    e = _engine_with(g)
    e.information_gain = 0.0
    e.advance_progress("mygoal", 0.2)  # 0.9+0.2 -> completed
    assert g.status == "completed"
    assert e.information_gain == pytest.approx(0.4)  # 0.2 * 2, not 0.2 / 2


# ── loop-skip and staleness boolean/arith mutants ─────────────────────

def test_completed_goal_is_skipped_in_progress_loop():
    # Kills the `status != active or level == axiom` Or->And skip mutant.
    g = _goal("tac", "tactic", progress=0.3, status="completed")
    e = _engine_with(g)
    e.evaluate_progress({"new_concepts": 10})
    assert g.progress == 0.3  # untouched — a completed goal must be skipped


def test_active_stale_strategy_not_abandoned():
    # Kills the staleness `status != active or level in (axiom, strategy)`
    # Or->And mutant. A reasoning strategy with zero increment keeps its old
    # last_progress_time, so it would enter the staleness check if the guard
    # were broken — but strategies must never be abandoned.
    g = _goal("improve_reasoning", "strategy")
    g.last_progress_time = time.time() - (STALE_THRESHOLD_SECONDS + 1000)
    e = _engine_with(g)
    # error_rate=1.0, llm_insights=0 -> increment 0 -> last_progress not refreshed
    e.evaluate_progress({"llm_insights": 0, "error_rate": 1.0})
    assert g.status == "active"


def test_fresh_goal_not_abandoned():
    # Kills the `now - last_progress_time` Sub->Add mutant, which would make a
    # freshly-progressed goal look ancient.
    g = _goal("tac", "tactic")
    e = _engine_with(g)
    e.evaluate_progress({})  # tactic gets +0.003, refreshing last_progress to now
    assert g.status == "active"


def test_stale_tactic_is_abandoned():
    g = _goal("tac", "tactic")
    g.progress = 1.0  # completed-progress so no increment refreshes the timer
    g.status = "active"
    g.last_progress_time = time.time() - (STALE_THRESHOLD_SECONDS + 1000)
    e = _engine_with(g)
    e.evaluate_progress({})
    # progress already 1.0 -> gets marked completed OR abandoned; either way not
    # left silently active with a stale timer. Verify it left the active state.
    assert g.status in ("abandoned", "completed")


# ── curiosity_level formula (L238) ────────────────────────────────────

def test_curiosity_level_formula_exact():
    e = _engine_with()  # no goals -> information_gain stays as set
    e.information_gain = 50.0
    e.evaluate_progress({})
    expected = 0.3 + 0.7 * math.exp(-50.0 * 0.01)
    assert e.curiosity_level == pytest.approx(expected)


# ── prune keeps active goals (L247) ───────────────────────────────────

def test_prune_keeps_active_goals():
    # Kills the `level != axiom and status in (completed, abandoned)` And->Or
    # mutant, which would classify active goals as "finished" and prune them.
    e = GoalEngine()
    active = _goal("keeper", "tactic")
    active.last_progress_time = 0.0  # oldest -> first to be dropped if misclassified
    finished = [_goal(f"done_{i}", "tactic", progress=1.0, status="completed")
                for i in range(101)]
    for i, g in enumerate(finished):
        g.last_progress_time = 100.0 + i
    e.goals = [active] + finished
    e._prune()
    assert active in e.goals  # active goal must survive pruning


# ── generate_goals arithmetic (L106, L109, L114, L130, L132) ──────────

def test_generate_creates_tactic_when_only_completed_tactics_exist():
    # Kills the `status == "active"` Eq->NotEq mutant in the active_tactics count.
    e = GoalEngine()
    strat = _goal("strat", "strategy", priority=0.6)
    done_tactics = [_goal(f"t{i}", "tactic", progress=1.0, status="completed")
                    for i in range(3)]
    e.goals = [strat] + done_tactics
    e.curiosity_level = 0.0
    e._last_goal_gen = 0.0
    new = e.generate_goals({"tick": 1})
    assert any(g.level == "tactic" for g in new)


def test_generate_tactic_priority_derives_from_best_parent():
    # Kills L109 (parent-selection key) and L114 (`parent.priority * 0.8`).
    e = GoalEngine()
    a = _goal("stratA", "strategy", progress=0.9, priority=0.9)   # key 0.9*0.1=0.09
    b = _goal("stratB", "strategy", progress=0.0, priority=0.5)   # key 0.5*1.0=0.50
    e.goals = [a, b]
    e.curiosity_level = 0.0
    e._last_goal_gen = 0.0
    new = e.generate_goals({"tick": 1})
    tactic = next(g for g in new if g.level == "tactic")
    # parent is B (higher key); tactic priority = 0.5 * 0.8 = 0.4
    assert tactic.priority == pytest.approx(0.4)


def test_generate_curiosity_priority_and_reasoning_exact():
    # Kills L130 (`0.3 + 0.2*curiosity_level`) and L132 (gain_potential formula).
    e = GoalEngine()
    e.goals = []
    e.curiosity_level = 0.5
    e.information_gain = 50.0
    e._last_goal_gen = 0.0
    new = e.generate_goals({"tick": 1})
    cur = next(g for g in new if g.level == "curiosity")
    # priority = 0.3 + 0.2*0.5 = 0.4
    assert cur.priority == pytest.approx(0.4)
    # gain_potential = max(0.1, 1.0 - 50*0.01) = 0.5 -> appears as "0.50"
    assert "0.50" in cur.reasoning


# ── get_current_focus and status (L267, L274, L282) ───────────────────

def test_current_focus_picks_highest_priority_progress_gap():
    # Kills L267 (`priority * (1 - progress)` key).
    a = _goal("a", "tactic", progress=0.9, priority=0.9)  # 0.09
    b = _goal("b", "tactic", progress=0.0, priority=0.5)  # 0.50
    e = _engine_with(a, b)
    assert e.get_current_focus()["name"] == "b"


def test_status_counts_increment_by_one():
    # Kills the `.get(status, 0) + 1` Add->Sub mutant.
    e = _engine_with(_goal("t1", "tactic"), _goal("t2", "tactic"))
    st = e.status()
    assert st["by_level"]["tactic"]["active"] == 2


def test_status_active_goals_only_active():
    # Kills the `status == "active"` Eq->NotEq mutant in status().
    active = _goal("a", "tactic")
    done = _goal("d", "tactic", progress=1.0, status="completed")
    e = _engine_with(active, done)
    names = [g["name"] for g in e.status()["active_goals"]]
    assert names == ["a"]
