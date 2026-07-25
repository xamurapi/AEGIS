"""Tests for MetaGoalGenerator."""
from aegis.layers.meta_goal_generator import MetaGoalGenerator


def test_initial_state():
    g = MetaGoalGenerator()
    assert g.generation_cycles == 0
    assert g.active_meta_goals == []


def test_no_triggers_no_goals():
    g = MetaGoalGenerator()
    ctx = {
        "memory_total": 0, "mood_valence": 0.5, "learning_sessions": 10,
        "avg_tick_ms": 0, "tick": 1, "active_agents": 5, "error_rate": 0.0,
    }
    goals = g.generate_goals(ctx)
    assert goals == []
    assert g.generation_cycles == 1


def test_memory_optimization_trigger():
    g = MetaGoalGenerator()
    ctx = {"memory_total": 2000, "mood_valence": 0.5, "learning_sessions": 10,
           "avg_tick_ms": 0, "tick": 1, "active_agents": 5, "error_rate": 0.0}
    goals = g.generate_goals(ctx)
    assert any(x["domain"] == "memory_optimization" for x in goals)


def test_multiple_triggers():
    g = MetaGoalGenerator()
    ctx = {"memory_total": 2000, "mood_valence": 0.2, "learning_sessions": 1,
           "avg_tick_ms": 5000, "tick": 1000, "active_agents": 0, "error_rate": 0.5}
    goals = g.generate_goals(ctx)
    domains = {x["domain"] for x in goals}
    assert "emotional_balance" in domains
    assert "knowledge_expansion" in domains
    assert "performance_tuning" in domains
    assert "architecture_evolution" in domains
    assert "agent_expansion" in domains
    assert "error_recovery" in domains


def test_no_duplicate_active_goals():
    g = MetaGoalGenerator()
    ctx = {"memory_total": 2000, "mood_valence": 0.5, "learning_sessions": 10,
           "avg_tick_ms": 0, "tick": 1, "active_agents": 5, "error_rate": 0.0}
    g.generate_goals(ctx)
    goals2 = g.generate_goals(ctx)
    # already active, so not re-added
    assert not any(x["domain"] == "memory_optimization" for x in goals2)


def test_trigger_exception_is_swallowed():
    g = MetaGoalGenerator()
    # mood_valence as a string makes the emotional_balance lambda raise TypeError,
    # which must be swallowed (continue) rather than propagate.
    ctx = {"memory_total": 0, "mood_valence": "bad", "learning_sessions": 10,
           "avg_tick_ms": 0, "tick": 1, "active_agents": 5, "error_rate": 0.0}
    goals = g.generate_goals(ctx)
    assert goals == []


def test_active_goals_trimmed_to_ten():
    g = MetaGoalGenerator()
    # Seed 12 fake active goals, then trigger one more domain.
    for i in range(12):
        g.active_meta_goals.append(
            {"domain": f"fake{i}", "description": "x", "priority": 0.1,
             "created_at": 0.0, "status": "pending"})
    ctx = {"memory_total": 2000, "mood_valence": 0.5, "learning_sessions": 10,
           "avg_tick_ms": 0, "tick": 1, "active_agents": 5, "error_rate": 0.0}
    g.generate_goals(ctx)
    assert len(g.active_meta_goals) == 10


def test_build_prompt_valid_template():
    g = MetaGoalGenerator()
    p = g.build_prompt("code_optimization", code="print(1)")
    assert "print(1)" in p
    assert g.prompts_generated == 1


def test_build_prompt_unknown_template():
    g = MetaGoalGenerator()
    p = g.build_prompt("does_not_exist", foo="bar")
    assert "recommendations" in p
    # unknown template returns early, counter not incremented
    assert g.prompts_generated == 0


def test_build_prompt_missing_key_returns_raw():
    g = MetaGoalGenerator()
    # strategy_generation needs several keys; omit them to hit KeyError branch
    p = g.build_prompt("strategy_generation")
    assert "{mood}" in p  # raw, unformatted template
    assert g.prompts_generated == 1


def test_complete_goal():
    g = MetaGoalGenerator()
    ctx = {"memory_total": 2000, "mood_valence": 0.5, "learning_sessions": 10,
           "avg_tick_ms": 0, "tick": 1, "active_agents": 5, "error_rate": 0.0}
    g.generate_goals(ctx)
    g.complete_goal("memory_optimization")
    assert g.completed_goals == 1
    assert not any(x["domain"] == "memory_optimization" for x in g.active_meta_goals)


def test_complete_goal_unknown_domain_noop():
    g = MetaGoalGenerator()
    g.complete_goal("nonexistent")
    assert g.completed_goals == 0


def test_get_top_priority_empty():
    g = MetaGoalGenerator()
    assert g.get_top_priority() is None


def test_get_top_priority_returns_highest():
    g = MetaGoalGenerator()
    g.active_meta_goals = [
        {"domain": "a", "description": "x", "priority": 0.3, "status": "pending"},
        {"domain": "b", "description": "y", "priority": 0.9, "status": "pending"},
    ]
    top = g.get_top_priority()
    assert top["domain"] == "b"


def test_status():
    g = MetaGoalGenerator()
    ctx = {"memory_total": 2000, "mood_valence": 0.5, "learning_sessions": 10,
           "avg_tick_ms": 0, "tick": 1, "active_agents": 5, "error_rate": 0.0}
    g.generate_goals(ctx)
    st = g.status()
    assert st["generation_cycles"] == 1
    assert st["active_goals"] >= 1
    assert st["top_priority"] is not None
    assert isinstance(st["active_meta_goals"], list)
