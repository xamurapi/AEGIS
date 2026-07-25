"""pytest-bdd step definitions for tests/features/higher_systems.feature.

Executable Gherkin: each scenario drives the real system classes (isolated to
tmp_path), so the .feature file doubles as living documentation and a test.
"""
import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from aegis.layers.world_model import WorldModel
from aegis.layers.cognitive_graph import CognitiveGraph
from aegis.layers.evolution_engine import EvolutionEngine
from aegis.layers.goal_intelligence import GoalIntelligence
from aegis.layers.feedback_loop import FeedbackLoop

scenarios("features/higher_systems.feature")


@pytest.fixture
def ctx(tmp_path):
    return {
        "world_model": WorldModel(store_path=tmp_path / "wm.json"),
        "cognitive_graph": CognitiveGraph(store_path=tmp_path / "cg.json"),
        "evolution": EvolutionEngine(store_path=tmp_path / "ev.json"),
        "goal_intelligence": GoalIntelligence(store_path=tmp_path / "gi.json"),
        "feedback_loop": FeedbackLoop(store_path=tmp_path / "exp.jsonl"),
        "chain": None, "experience": None, "verdict": None, "exp_id": None,
    }


# ── Background ────────────────────────────────────────────────────────

@given("a fresh AEGIS instance with the five systems")
def _fresh(ctx):
    assert all(ctx[k] is not None for k in
               ("world_model", "cognitive_graph", "evolution",
                "goal_intelligence", "feedback_loop"))


# ── System 1: World Model ─────────────────────────────────────────────

@when(parsers.parse('I observe "{cause}" causing "{effect}" {n:d} times successfully'))
def _observe_success(ctx, cause, effect, n):
    for _ in range(n):
        ctx["world_model"].observe(cause, effect, success=True)


@when(parsers.parse('I observe "{cause}" causing "{effect}" {n:d} times unsuccessfully'))
def _observe_fail(ctx, cause, effect, n):
    for _ in range(n):
        ctx["world_model"].observe(cause, effect, success=False)


@when(parsers.parse('I build a causal chain for the objective "{objective}"'))
def _build_chain(ctx, objective):
    ctx["chain"] = ctx["world_model"].build_chain(objective)


@then(parsers.parse('the chain should predict "{effect}" as a likely effect'))
def _chain_predicts(ctx, effect):
    effects = [s["expected"] for s in ctx["chain"]["plan"]]
    assert effect in effects or ctx["chain"]["expected_result"] == effect


@then("the chain should contain at least one plan step")
def _chain_has_steps(ctx):
    assert len(ctx["chain"]["plan"]) >= 1


@then(parsers.parse('querying risks for "{token}" should surface a high failure rate'))
def _risks(ctx, token):
    risks = ctx["world_model"].risks_for([token])
    assert risks and risks[0]["failure_rate"] > 0.5


# ── System 2: Cognitive Graph ─────────────────────────────────────────

@when(parsers.parse('I add concepts "{a}", "{b}" and "{c}" linked {x}-{y} and {y2}-{z}'))
def _add_graph(ctx, a, b, c, x, y, y2, z):
    cg = ctx["cognitive_graph"]
    for n in (a, b, c):
        cg.add_node(n, "concept")
    cg.add_edge(x, y)
    cg.add_edge(y2, z)


@then(parsers.parse('there should be a path from "{start}" to "{goal}"'))
def _path(ctx, start, goal):
    assert ctx["cognitive_graph"].find_path(start, goal) is not None


# ── System 3: Evolution Engine ────────────────────────────────────────

@given(parsers.parse("a champion genome with fitness {fitness:f}"))
def _champion(ctx, fitness):
    ctx["evolution"].register_champion({"learning_rate": 0.01, "temperature": 0.7}, fitness)


@when(parsers.parse("I propose a mutation and the benchmark scores {score:f}"))
def _mutate_and_judge(ctx, score):
    m = ctx["evolution"].propose_mutation(tick=1)
    ctx["mutation"] = m
    ctx["verdict"] = ctx["evolution"].judge_candidate(score)


@then("the mutation should be accepted as the new champion")
def _accepted(ctx):
    assert ctx["verdict"]["decision"] == "accepted"


@then("the mutation should be rejected and the parameter reverted")
def _rejected(ctx):
    assert ctx["verdict"]["decision"] == "rejected"
    assert ctx["verdict"]["revert_to"] == ctx["mutation"]["old_value"]


# ── System 4: Goal Intelligence ───────────────────────────────────────

@when(parsers.parse('I choose the objective "{obj}" and receive reward {r:f} ten times'))
def _choose_reward(ctx, obj, r):
    gi = ctx["goal_intelligence"]
    gi.choose([obj], {})
    for _ in range(10):
        gi.reward(r, obj)
    ctx["obj"] = obj


@then(parsers.parse('the utility of "{obj}" should rise above its default'))
def _utility_rose(ctx, obj):
    assert ctx["goal_intelligence"].values[obj]["utility"] > 0.5


# ── System 5: Feedback Loop ───────────────────────────────────────────

@when(parsers.parse('I record a situation "{sit}" with decision "{dec}"'))
def _record_situation(ctx, sit, dec):
    ctx["exp_id"] = ctx["feedback_loop"].record_situation(sit, dec, {"tick": 1})


@when(parsers.parse("the real result comes back as failure with metric {metric:f}"))
def _record_result(ctx, metric):
    ctx["experience"] = ctx["feedback_loop"].record_result(
        ctx["exp_id"], success=False, metric=metric)


@then("the stored experience should explain why it failed")
def _explains(ctx):
    assert ctx["experience"] is not None
    assert ctx["experience"]["cause"]
    assert ctx["experience"]["success"] is False


@then("it should be exportable as a training example")
def _exportable(ctx):
    rows = ctx["feedback_loop"].export_examples()
    assert rows and "prompt" in rows[0] and "completion" in rows[0]
