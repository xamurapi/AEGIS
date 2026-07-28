"""Mutation-killing tests for the DECIDE phase (spec §3.7, Appendix J).

Half the mutants in this module survived the first harness run, and they were
concentrated in two places that matter more than their size suggests:

* **the confidence penalty** — the number the ethics gate reads. A mutant that
  turned `sum(rate)/10` into `sum(rate)*10` still produced a confidence, still
  ordered plans, and still passed every test, while making a single remembered
  failure enough to drive confidence to the floor.
* **`modifies_self`** — the predicate that decides whether self-preservation is
  consulted at all. A mutant that made it always true would consult it on every
  rest tick; one that made it always false would skip it on a source rewrite.
  Neither shows up as a failure anywhere else.

So these tests assert the exact arithmetic and the exact routing, not that
"a decision came out".
"""
import asyncio

import pytest

from aegis.layers.phases import decide as decide_phase
from aegis.layers.world.state import StateKey


class _Ctx:
    def __init__(self, state=None):
        self.state = state or StateKey(energy="hi")
        self.state_inputs = {}
        self.decision = None
        self.action = None
        self.prediction = None
        self.ethics_status = "ok"
        self.ethics_score = 1.0

    def mark_external(self, phase):
        pass


class _Plan:
    def __init__(self, objective="obj", action="rest", rationale="why"):
        self.objective = objective
        self.action = action
        self.rationale = rationale
        self.steps = [action] if action else []
        self.expected_value = 0.5


def _run(coro):
    return asyncio.run(coro)


# ── the confidence penalty the ethics gate reads ─────────────────────

class _Risky:
    """A world model with a known failure history, and a record of the query."""

    def __init__(self, rates=(1.0,)):
        self.rates = list(rates)
        self.asked = []

    def risks_for(self, tokens):
        self.asked.append(list(tokens))
        return [{"failure_rate": rate} for rate in self.rates]


class _Substrate:
    def __init__(self, rates=(1.0,), confidence=0.8):
        self.world_model = _Risky(rates)
        self._confidence = confidence
        self.tick_count = 7

    def _compute_confidence(self):
        return self._confidence


def test_the_risk_query_names_the_action_not_an_empty_string():
    """`plan.action or ""` — the fallback replaces a *missing* action.

    Querying the causal history for `""` instead of the action would silently
    return nothing, so every plan would look risk-free.
    """
    substrate = _Substrate()
    decide_phase._adjust_confidence(substrate, _Ctx(), _Plan(action="rest"))
    assert substrate.world_model.asked == [["obj", "rest"]]


def test_a_plan_with_no_action_still_asks_about_its_objective():
    substrate = _Substrate()
    decide_phase._adjust_confidence(substrate, _Ctx(), _Plan(action=None))
    assert substrate.world_model.asked == [["obj", ""]]


def test_one_remembered_failure_costs_a_tenth_of_the_confidence():
    """penalty = sum(rate)/10, confidence ← confidence·(1 − penalty).

    0.8 · (1 − 0.1) = 0.72. The divisor is what keeps a single failure from
    being treated as a verdict.
    """
    substrate = _Substrate(rates=(1.0,), confidence=0.8)
    confidence, reasoning = decide_phase._adjust_confidence(
        substrate, _Ctx(), _Plan())
    assert confidence == pytest.approx(0.72)
    assert "lower confidence by 10%" in reasoning


def test_the_penalty_accumulates_across_failure_modes():
    substrate = _Substrate(rates=(1.0, 1.0, 0.0), confidence=0.5)
    confidence, reasoning = decide_phase._adjust_confidence(
        substrate, _Ctx(), _Plan())
    assert confidence == pytest.approx(0.4)          # 0.5 · (1 − 0.2)
    assert "3 known failure mode(s)" in reasoning
    assert "by 20%" in reasoning


def test_the_penalty_is_capped_so_evidence_cannot_veto_by_itself():
    from aegis.config import MAX_RISK_CONFIDENCE_PENALTY

    substrate = _Substrate(rates=(1.0,) * 50, confidence=0.8)
    confidence, _ = decide_phase._adjust_confidence(substrate, _Ctx(), _Plan())
    assert confidence == pytest.approx(
        round(0.8 * (1 - MAX_RISK_CONFIDENCE_PENALTY), 4))


def test_confidence_never_falls_below_the_floor():
    substrate = _Substrate(rates=(1.0,) * 50, confidence=0.01)
    confidence, _ = decide_phase._adjust_confidence(substrate, _Ctx(), _Plan())
    assert confidence == 0.05


def test_no_failure_history_leaves_confidence_untouched():
    substrate = _Substrate(rates=(), confidence=0.8)
    confidence, reasoning = decide_phase._adjust_confidence(
        substrate, _Ctx(), _Plan())
    assert confidence == 0.8
    assert "failure mode" not in reasoning


# ── modifies_self: which actions summon self-preservation ────────────

class _Gates:
    """Records what the two gates were told about a plan."""

    def __init__(self, registry, verdict="ok", safe=True):
        self.actions = registry
        self.ethics = type("E", (), {"evaluate_action": self._evaluate})()
        self.self_preservation = type("SP", (), {
            "is_modification_safe": self._safe})()
        self.planner = type("P", (), {"note_blocked": self._blocked})()
        self.seen = []
        self.consulted = []
        self.blocked = []
        self._verdict = verdict
        self._safe_answer = safe

    def _evaluate(self, info):
        self.seen.append(info)
        return {"status": self._verdict, "score": 0.9}

    def _safe(self, what, why):
        self.consulted.append(what)
        return self._safe_answer, {}

    def _blocked(self, reason):
        self.blocked.append(reason)


def _registry(**specs):
    return type("A", (), {"by_name": dict(specs),
                          "note_blocked": staticmethod(lambda reason: None)})()


def _spec(reversible=True):
    return type("S", (), {"reversible": reversible})()


def test_a_reversible_action_is_not_treated_as_self_modifying():
    """Otherwise every rest tick would consult self-preservation, and the
    check would stop meaning anything."""
    substrate = _Gates(_registry(rest=_spec(reversible=True)))
    assert decide_phase._passes_gates(substrate, _Ctx(), _Plan(action="rest"), 0.8)
    assert substrate.seen[0]["modifies_self"] is False
    assert substrate.consulted == []


def test_an_irreversible_action_summons_self_preservation():
    substrate = _Gates(_registry(code_self_mod=_spec(reversible=False)))
    assert decide_phase._passes_gates(
        substrate, _Ctx(), _Plan(action="code_self_mod"), 0.8)
    assert substrate.seen[0]["modifies_self"] is True
    assert substrate.consulted == ["plan/code_self_mod"]


def test_an_unregistered_action_is_judged_by_its_name():
    """The name-based half of the predicate is the backstop: an action the
    registry does not know must still be recognised as self-modifying if it
    says so."""
    substrate = _Gates(_registry())          # nothing registered
    assert decide_phase._passes_gates(
        substrate, _Ctx(), _Plan(action="train_weights"), 0.8)
    assert substrate.seen[0]["modifies_self"] is True
    assert substrate.consulted == ["plan/train_weights"]


def test_an_unregistered_ordinary_action_is_not_self_modifying():
    substrate = _Gates(_registry())
    assert decide_phase._passes_gates(substrate, _Ctx(), _Plan(action="rest"), 0.8)
    assert substrate.seen[0]["modifies_self"] is False


def test_an_ethics_veto_is_recorded_and_refuses():
    substrate = _Gates(_registry(rest=_spec()), verdict="blocked")
    ctx = _Ctx()
    assert decide_phase._passes_gates(substrate, ctx, _Plan(action="rest"), 0.8) is False
    assert ctx.ethics_status == "blocked"
    assert substrate.blocked == ["ethics"]


def test_self_preservation_can_refuse_after_ethics_allowed():
    substrate = _Gates(_registry(code_self_mod=_spec(reversible=False)), safe=False)
    assert decide_phase._passes_gates(
        substrate, _Ctx(), _Plan(action="code_self_mod"), 0.8) is False
    assert substrate.blocked == ["self_preservation"]


# ── the fallback path: the same arithmetic, separately ───────────────

class _Fallback:
    def __init__(self, rates=(), goals=None):
        self.tick_count = 9
        self.emotions = type("E", (), {"energy": 0.7, "mood": "curious"})()
        self.consciousness = type("C", (), {"mode": "focused"})()
        self.health = type("H", (), {"successful_ticks": 6, "failed_ticks": 2,
                                     "error_count": 2})()
        self.world_model = _Risky(rates)
        self.goals = type("G", (), {
            "curiosity_level": 0.4,
            "goals": goals or [],
            "get_current_focus": staticmethod(lambda: {"name": "focus"})})()
        self.chosen = []
        self.goal_intelligence = type("GI", (), {"choose": self._choose})()
        self.ethics_info = []
        self.ethics = type("Et", (), {"evaluate_action": self._evaluate})()
        self._regulation_directives = {}

    def _choose(self, options, context):
        self.chosen.append(list(options))
        return {"objective": options[0]}

    def _evaluate(self, info):
        self.ethics_info.append(info)
        return {"status": "ok", "score": 0.9}

    def _compute_confidence(self):
        return 0.8

    def _is_llm_tick(self):
        return False

    def acquire(self, action):
        return None


def test_the_value_table_is_offered_every_alternative():
    """`[decision] + alternatives` — the greedy pick and its alternatives, in
    one list. Offering only one of them makes the choice a formality."""
    substrate = _Fallback()
    _run(decide_phase._decide_without_a_plan(
        substrate, _Ctx(), "greedy", ["rest", "dream"]))
    assert substrate.chosen == [["greedy", "rest", "dream"]]


def test_the_fallback_penalty_uses_the_same_tenth():
    substrate = _Fallback(rates=(1.0,))
    _, confidence, reasoning, _, _ = _run(decide_phase._decide_without_a_plan(
        substrate, _Ctx(), "greedy", []))
    assert confidence == pytest.approx(0.72)
    assert "lower confidence by 10%" in reasoning


def test_the_fallback_never_claims_to_modify_the_system():
    """This path chooses an objective, not an action, so it cannot be a
    self-modification — declaring otherwise would route ordinary ticks into
    the self-preservation branch of the ethics core."""
    substrate = _Fallback()
    _run(decide_phase._decide_without_a_plan(substrate, _Ctx(), "greedy", []))
    assert substrate.ethics_info[-1]["modifies_self"] is False


def test_the_error_rate_offered_to_the_value_table_is_a_ratio():
    """2 errors over 8 ticks is 0.25 — not 16."""
    substrate = _Fallback()
    context = decide_phase._value_context(substrate)
    assert context["error_rate"] == pytest.approx(0.25)
    assert context["energy"] == 0.7
    assert context["curiosity"] == 0.4


def test_a_system_with_no_ticks_yet_reports_no_errors():
    substrate = _Fallback()
    substrate.health = type("H", (), {"successful_ticks": 0, "failed_ticks": 0,
                                      "error_count": 3})()
    assert decide_phase._value_context(substrate)["error_rate"] == 0.0


# ── what the model is told about the goals ───────────────────────────

class _Goal:
    def __init__(self, name, status="active", level="tactic", progress=0.5):
        self.name, self.status, self.level, self.progress = (
            name, status, level, progress)


def test_the_model_is_shown_only_live_non_axiom_goals():
    """`status == "active" and level != "axiom"` — a completed goal has no
    progress left to make and an axiom is a constraint, not a task."""
    substrate = _Fallback(goals=[
        _Goal("live"), _Goal("done", status="completed"),
        _Goal("law", level="axiom")])
    substrate._is_llm_tick = lambda: True
    substrate.acquire = lambda action: "L"
    substrate.settle = lambda lease, value=0.0: None
    seen = {}

    async def _make_decision(options, context, lease=None):
        seen.update(context)
        return {"success": False}

    substrate.llm = type("L", (), {"make_decision": staticmethod(_make_decision)})()
    _run(decide_phase._llm_decision(substrate, _Ctx(), "greedy", [], 0.8, "why"))
    assert seen["goals_summary"] == {"live": 0.5}


# ── the numbers meta-goal generation is fed ──────────────────────────

def test_meta_goal_generation_is_told_milliseconds_and_a_ratio():
    """`last_tick_duration · 1000` and `error_count / ticks`. Both feed
    thresholds that decide whether the system proposes to fix itself."""
    seen = {}

    class _S:
        tick_count = 30
        last_tick_duration = 0.25
        health = type("H", (), {"successful_ticks": 6, "failed_ticks": 2,
                                "error_count": 2})()
        memory = type("M", (), {"status": staticmethod(
            lambda: {"total_memories": 12})})()
        emotions = type("E", (), {"valence": 0.6})()
        external_learning = type("X", (), {"learning_sessions": 3})()
        agent_system = type("A", (), {"agents": []})()
        meta_goals = type("MG", (), {"generate_goals": staticmethod(
            lambda state: seen.update(state) or [])})()
        autobiography = type("Ab", (), {"log_event": staticmethod(lambda *a: None)})()

    decide_phase._generate_meta_goals(_S())
    assert seen["avg_tick_ms"] == pytest.approx(250.0)
    assert seen["error_rate"] == pytest.approx(0.25)


def test_the_opened_experience_records_the_error_rate_as_a_ratio():
    seen = {}

    class _S:
        tick_count = 4
        consciousness = type("C", (), {"mode": "focused"})()
        emotions = type("E", (), {"energy": 0.5})()
        health = type("H", (), {"successful_ticks": 6, "failed_ticks": 2,
                                "error_count": 2})()
        goal_intelligence = type("GI", (), {"commit": staticmethod(lambda *a: None)})()
        feedback_loop = type("F", (), {"record_situation": staticmethod(
            lambda situation, decision, context: seen.update(
                situation=situation, context=context) or "exp_1")})()
        _pending_experiences = {}

        @staticmethod
        def _compute_confidence():
            return 0.8

    substrate = _S()
    decide_phase._open_experience(substrate, _Ctx(), {"name": "focus"}, "d", None)
    assert "err=0.25" in seen["situation"]
    assert substrate._pending_experiences["decide"] == "exp_1"


# ── the causal chain's constraints ───────────────────────────────────

def test_only_axioms_become_chain_constraints():
    """`level == "axiom"` — constraints are the things a plan may not violate.

    Inverting the test would hand the chain builder every ordinary goal as a
    constraint and no axiom at all.
    """
    import aegis.config as cfg

    seen = {}

    class _S:
        tick_count = cfg.WORLD_MODEL_EVERY_N_TICKS
        goals = type("G", (), {"goals": [
            _Goal("law", level="axiom"), _Goal("task", level="tactic")]})()
        world_model = type("W", (), {"build_chain": staticmethod(
            lambda focus, constraints: seen.update(
                focus=focus, constraints=list(constraints)) or {
                    "plan": [], "confidence": 0.5})})()
        autobiography = type("Ab", (), {"log_event": staticmethod(lambda *a: None)})()

    decide_phase._build_causal_chain(_S(), {"name": "focus"})
    assert seen["constraints"] == ["law"]
    assert seen["focus"] == "focus"


def test_a_chain_with_steps_is_written_to_the_autobiography():
    import aegis.config as cfg

    logged = []

    class _S:
        tick_count = cfg.WORLD_MODEL_EVERY_N_TICKS
        goals = type("G", (), {"goals": []})()
        world_model = type("W", (), {"build_chain": staticmethod(
            lambda focus, constraints: {"plan": ["a", "b"], "confidence": 0.7})})()
        autobiography = type("Ab", (), {"log_event": staticmethod(
            lambda *a: logged.append(a))})()

    decide_phase._build_causal_chain(_S(), {"name": "focus"})
    assert logged and "2 steps" in logged[0][1]
