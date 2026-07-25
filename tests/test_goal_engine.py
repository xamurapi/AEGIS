"""Tests for the Goal Engine — generation, progress, staleness, pruning, conflicts."""
import time
from aegis.layers.goal_engine import (
    Goal, GoalEngine, AXIOM_GOAL_SPECS, STRATEGY_TEMPLATES,
    CURIOSITY_TOPICS, STALE_THRESHOLD_SECONDS,
)


# ── Goal object ──────────────────────────────────────────────────

def test_goal_ids_are_unique_and_to_dict():
    g1 = Goal("a", "strategy", "desc", 0.6)
    g2 = Goal("b", "strategy", "desc2", 0.7)
    assert g1.id != g2.id
    d = g1.to_dict()
    assert d["name"] == "a"
    assert d["level"] == "strategy"
    assert d["status"] == "active"
    assert d["progress"] == 0.0


# ── Engine init ──────────────────────────────────────────────────

def test_engine_starts_with_axiom_goals():
    e = GoalEngine()
    assert len(e.goals) == len(AXIOM_GOAL_SPECS)
    assert all(g.level == "axiom" for g in e.goals)


# ── generate_goals ───────────────────────────────────────────────

def test_generate_rate_limited_returns_empty_when_called_twice():
    e = GoalEngine()
    first = e.generate_goals({"tick": 1})
    # Immediate second call is within the 10s window → nothing generated.
    second = e.generate_goals({"tick": 2})
    assert first  # first produced something
    assert second == []


def test_generate_creates_strategy_and_curiosity():
    e = GoalEngine()
    new = e.generate_goals({"tick": 5, "memory_size": 10})
    levels = {g.level for g in new}
    assert "strategy" in levels
    assert "curiosity" in levels


def test_generate_creates_tactic_when_strategy_exists():
    e = GoalEngine()
    # Seed an active strategy so the tactic branch fires.
    strat = Goal("expand_knowledge", "strategy", "Explore", 0.7)
    e.goals.append(strat)
    e._last_goal_gen = 0
    new = e.generate_goals({"tick": 9})
    assert any(g.level == "tactic" for g in new)
    tactic = next(g for g in new if g.level == "tactic")
    assert tactic.parent == strat.id


def test_generate_skips_duplicate_strategy():
    e = GoalEngine()
    # Pre-create the first strategy template as active so it is skipped.
    first_tmpl = STRATEGY_TEMPLATES[0][0]
    e.goals.append(Goal(first_tmpl, "strategy", "dup", 0.7))
    e._last_goal_gen = 0
    new = e.generate_goals({"tick": 3})
    # No second strategy with the same name should be generated.
    strat_names = [g.name for g in new if g.level == "strategy"]
    assert first_tmpl not in strat_names


def test_generate_skips_duplicate_curiosity():
    e = GoalEngine()
    topic = CURIOSITY_TOPICS[0]
    e.goals.append(Goal("dup", "curiosity", f"Investigate: {topic}", 0.4))
    e._last_goal_gen = 0
    new = e.generate_goals({"tick": 4})
    curiosity_descs = [g.description for g in new if g.level == "curiosity"]
    assert f"Investigate: {topic}" not in curiosity_descs


def test_generate_no_curiosity_when_level_low():
    e = GoalEngine()
    e.curiosity_level = 0.2  # below 0.4 threshold
    new = e.generate_goals({"tick": 1})
    assert all(g.level != "curiosity" for g in new)


def test_generate_skips_strategy_when_two_active():
    e = GoalEngine()
    e.goals.append(Goal("s1", "strategy", "d", 0.7))
    e.goals.append(Goal("s2", "strategy", "d", 0.7))
    e._last_goal_gen = 0
    new = e.generate_goals({"tick": 1})
    assert all(g.level != "strategy" for g in new)


# ── advance_progress ─────────────────────────────────────────────

def test_advance_progress_partial():
    e = GoalEngine()
    g = Goal("mytask", "strategy", "d", 0.6)
    e.goals.append(g)
    e.advance_progress("mytask", 0.3)
    assert abs(g.progress - 0.3) < 1e-9
    assert g.status == "active"


def test_advance_progress_completes_goal():
    e = GoalEngine()
    g = Goal("finisher", "strategy", "d", 0.6)
    e.goals.append(g)
    e.advance_progress("finisher", 1.5)  # over-complete → clamped
    assert g.progress == 1.0
    assert g.status == "completed"
    assert e.information_gain > 0


def test_advance_progress_unknown_goal_is_noop():
    e = GoalEngine()
    e.advance_progress("does_not_exist", 0.5)  # should not raise


# ── evaluate_progress ────────────────────────────────────────────

def test_evaluate_progress_increments_by_level():
    e = GoalEngine()
    know = Goal("expand_knowledge", "strategy", "d", 0.7)
    reason = Goal("improve_reasoning", "strategy", "d", 0.6)
    mem = Goal("optimize_memory", "strategy", "d", 0.65)
    eth = Goal("strengthen_ethics", "strategy", "d", 0.75)
    selfm = Goal("enhance_self_model", "strategy", "d", 0.55)
    other = Goal("misc_strategy", "strategy", "d", 0.5)
    tac = Goal("tactic_x", "tactic", "d", 0.5)
    cur = Goal("explore_x", "curiosity", "d", 0.4)
    for g in (know, reason, mem, eth, selfm, other, tac, cur):
        e.goals.append(g)
    e.evaluate_progress({
        "new_concepts": 5, "new_episodic": 4,
        "error_rate": 0.1, "llm_insights": 3,
    })
    for g in (know, reason, mem, eth, selfm, other, tac, cur):
        assert g.progress > 0


def test_evaluate_progress_completes_goal():
    e = GoalEngine()
    g = Goal("expand_knowledge", "strategy", "d", 0.7)
    g.progress = 0.999
    e.goals.append(g)
    e.evaluate_progress({"new_concepts": 100, "llm_insights": 50})
    assert g.status == "completed"
    assert g.progress == 1.0


def test_evaluate_progress_abandons_stale_curiosity():
    e = GoalEngine()
    stale = Goal("explore_old", "curiosity", "d", 0.4)
    stale.last_progress_time = time.time() - (STALE_THRESHOLD_SECONDS + 100)
    e.goals.append(stale)
    # Empty metrics → curiosity increment is 0 → last_progress_time unchanged.
    e.evaluate_progress({})
    assert stale.status == "abandoned"


def test_evaluate_progress_skips_axiom_goals():
    e = GoalEngine()
    axiom_goal = e.goals[0]
    before = axiom_goal.progress
    e.evaluate_progress({"new_concepts": 10, "llm_insights": 10})
    assert axiom_goal.progress == before


def test_evaluate_progress_updates_curiosity_level():
    e = GoalEngine()
    e.information_gain = 50.0
    e.evaluate_progress({})
    # curiosity_level = 0.3 + 0.7*exp(-0.5) ≈ 0.724
    assert 0.3 <= e.curiosity_level <= 1.0


def test_evaluate_progress_none_metrics():
    e = GoalEngine()
    e.evaluate_progress(None)  # must not raise


# ── _prune ───────────────────────────────────────────────────────

def test_prune_drops_old_finished_goals():
    e = GoalEngine()
    for i in range(150):
        g = Goal(f"done_{i}", "curiosity", "d", 0.3)
        g.status = "completed"
        g.last_progress_time = time.time() + i  # increasing recency
        e.goals.append(g)
    e._prune()
    finished = [g for g in e.goals if g.status == "completed"]
    assert len(finished) == 100  # MAX_FINISHED cap


def test_prune_caps_goal_log():
    e = GoalEngine()
    e._max_goal_log = 10
    e.goal_log = [{"i": i} for i in range(50)]
    e._prune()
    assert len(e.goal_log) == 10


# ── resolve_conflict ─────────────────────────────────────────────

def test_resolve_conflict_prefers_lower_level():
    e = GoalEngine()
    ax = Goal("a", "axiom", "d", 0.1)
    cur = Goal("c", "curiosity", "d", 0.9)
    assert e.resolve_conflict(ax, cur) is ax
    assert e.resolve_conflict(cur, ax) is ax


def test_resolve_conflict_same_level_prefers_priority():
    e = GoalEngine()
    a = Goal("a", "strategy", "d", 0.9)
    b = Goal("b", "strategy", "d", 0.4)
    assert e.resolve_conflict(a, b) is a
    assert e.resolve_conflict(b, a) is a


# ── get_current_focus ────────────────────────────────────────────

def test_get_current_focus_none_when_only_axioms():
    e = GoalEngine()
    assert e.get_current_focus() is None


def test_get_current_focus_picks_best():
    e = GoalEngine()
    low = Goal("low", "strategy", "d", 0.3)
    high = Goal("high", "strategy", "d", 0.9)
    e.goals.extend([low, high])
    focus = e.get_current_focus()
    assert focus["name"] == "high"


# ── status ───────────────────────────────────────────────────────

def test_status_reports_structure():
    e = GoalEngine()
    e.goals.append(Goal("s", "strategy", "d", 0.7))
    st = e.status()
    assert st["total_goals"] == len(e.goals)
    assert "by_level" in st
    assert "axiom" in st["by_level"]
    assert "curiosity_level" in st
    assert "active_goals" in st
