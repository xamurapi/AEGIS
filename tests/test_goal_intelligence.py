"""Unit tests for System 4: GoalIntelligence (value-driven motivation)."""
from aegis.layers.goal_intelligence import GoalIntelligence


def _gi(tmp_path):
    return GoalIntelligence(store_path=tmp_path / "gi.json")


def test_classify_drive():
    gi = GoalIntelligence.__new__(GoalIntelligence)
    assert gi._classify_drive("solve coding task") == "competence"
    assert gi._classify_drive("explore new knowledge") == "knowledge"
    assert gi._classify_drive("reduce error rate") == "coherence"
    assert gi._classify_drive("recharge energy") == "stability"


def test_choose_returns_highest_value(tmp_path):
    gi = _gi(tmp_path)
    choice = gi.choose(["solve_task", "rest"], {"tick": 1})
    assert choice["objective"] in ("solve_task", "rest")
    assert "expected_value" in choice
    assert isinstance(choice["alternatives"], list)


def test_choose_empty_returns_none(tmp_path):
    gi = _gi(tmp_path)
    assert gi.choose([], {}) is None


def test_reward_updates_utility(tmp_path):
    gi = _gi(tmp_path)
    gi.choose(["explore_topic"], {"tick": 1})
    before = gi.values["explore_topic"]["utility"]
    gi.reward(1.0, "explore_topic")
    after = gi.values["explore_topic"]["utility"]
    assert after > before  # positive reward raises utility


def test_reward_credits_last_choice_when_unspecified(tmp_path):
    gi = _gi(tmp_path)
    gi.choose(["objective_a"], {"tick": 1})
    gi.reward(1.0)  # no explicit objective
    assert gi.values["objective_a"]["utility"] > 0.5


def test_learning_converges_toward_reward(tmp_path):
    gi = _gi(tmp_path)
    gi.choose(["obj"], {})
    for _ in range(50):
        gi.reward(0.9, "obj")
    assert abs(gi.values["obj"]["utility"] - 0.9) < 0.05


def test_low_energy_boosts_stability_drive(tmp_path):
    gi = _gi(tmp_path)
    # Prime a stability objective and a knowledge objective at equal utility.
    normal = gi.expected_value("recharge_energy", {"energy": 1.0})
    boosted = gi.expected_value("recharge_energy", {"energy": 0.1})
    assert boosted > normal


def test_high_error_boosts_coherence_drive(tmp_path):
    gi = _gi(tmp_path)
    normal = gi.expected_value("fix_error_consistency", {"error_rate": 0.0})
    boosted = gi.expected_value("fix_error_consistency", {"error_rate": 0.5})
    assert boosted > normal


def test_choice_increments_attempts(tmp_path):
    gi = _gi(tmp_path)
    gi.choose(["obj"], {})
    gi.choose(["obj"], {})
    assert gi.values["obj"]["attempts"] == 2


def test_reward_clamped(tmp_path):
    gi = _gi(tmp_path)
    gi.choose(["obj"], {})
    gi.reward(5.0, "obj")  # out of range
    assert gi.values["obj"]["utility"] <= 1.0


def test_pruning_bounds_values(tmp_path, monkeypatch):
    import aegis.layers.goal_intelligence as gimod
    monkeypatch.setattr(gimod, "MAX_VALUE_ENTRIES", 10)
    gi = _gi(tmp_path)
    for i in range(50):
        gi.choose([f"obj_{i}"], {})
    assert len(gi.values) <= 11  # allow the just-added one before prune settles


def test_persistence_round_trip(tmp_path):
    p = tmp_path / "gi.json"
    gi = GoalIntelligence(store_path=p)
    gi.choose(["obj"], {})
    gi.reward(0.8, "obj")
    gi.save()
    gi2 = GoalIntelligence(store_path=p)
    assert "obj" in gi2.values
    assert gi2.total_reward > 0


# ── mutation-hardening tests ──────────────────────────────────────────

def test_default_store_path_is_used(tmp_path, monkeypatch):
    import aegis.layers.goal_intelligence as gimod
    monkeypatch.setattr(gimod, "GOAL_INTEL_DIR", tmp_path)
    gi = GoalIntelligence()
    assert gi._store_path == tmp_path / "values.json"


def test_high_curiosity_boosts_knowledge_drive(tmp_path):
    # Kills the L118 Gt/And/Eq mutants: the knowledge boost applies only when
    # curiosity > 0.6 AND the drive is knowledge.
    gi = _gi(tmp_path)
    base = gi.expected_value("explore_new_topic", {"curiosity": 0.0})
    boosted = gi.expected_value("explore_new_topic", {"curiosity": 0.9})
    assert boosted > base
    # A non-knowledge objective must NOT be boosted by curiosity.
    stab_lo = gi.expected_value("recharge_energy", {"curiosity": 0.0})
    stab_hi = gi.expected_value("recharge_energy", {"curiosity": 0.9})
    assert stab_hi == stab_lo


def test_choose_picks_strictly_highest_value(tmp_path):
    # Kills the `scored.sort(reverse=True)` direction mutant.
    gi = _gi(tmp_path)
    # Pre-bias utilities so ordering is unambiguous.
    gi.choose(["explore_knowledge"], {})
    for _ in range(20):
        gi.reward(0.95, "explore_knowledge")
    gi.choose(["rest_stability"], {})
    for _ in range(20):
        gi.reward(0.05, "rest_stability")
    choice = gi.choose(["explore_knowledge", "rest_stability"], {})
    assert choice["objective"] == "explore_knowledge"


def test_choose_with_none_context_does_not_crash(tmp_path):
    # Kills the `(context or {}).get("tick")` Or->And mutant, which would turn
    # None into an AttributeError.
    gi = _gi(tmp_path)
    choice = gi.choose(["obj"], None)
    assert choice is not None
    assert choice["tick"] is None


def test_values_prune_keeps_exactly_max(tmp_path, monkeypatch):
    # Kills the prune boundary (<=) and slice-count (Sub->Add) mutants.
    import aegis.layers.goal_intelligence as gimod
    monkeypatch.setattr(gimod, "MAX_VALUE_ENTRIES", 10)
    gi = _gi(tmp_path)
    for i in range(60):
        gi._value_of(f"obj_{i}")   # direct valuation avoids choose's attempt bump
    assert len(gi.values) == 10


def test_decisions_log_is_capped_at_200(tmp_path):
    # Kills the `len(decisions) > 200` boundary mutant.
    gi = _gi(tmp_path)
    for i in range(250):
        gi.choose([f"obj_{i}"], {})
    assert len(gi.decisions) == 200


def test_status_top_valued_sorted_desc(tmp_path):
    # Kills the status sort-direction mutant.
    gi = _gi(tmp_path)
    gi.choose(["high"], {})
    for _ in range(20):
        gi.reward(0.9, "high")
    gi.choose(["low"], {})
    for _ in range(20):
        gi.reward(0.1, "low")
    top = gi.status()["top_valued"]
    assert top[0]["objective"] == "high"
    assert top[0]["utility"] >= top[1]["utility"]
