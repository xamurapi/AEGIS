"""Behaviour closure: learned structures must CHANGE BEHAVIOUR, not just accumulate.

Each of the five higher-order systems recorded knowledge and persisted it, but
four of them were read by nothing except the dashboard. The chain

    action -> result -> evaluation -> new knowledge -> BEHAVIOUR CHANGE

stopped at the fourth step. These tests pin the fifth step: for each system,
seed the learned structure and assert the system's next decision/action differs
because of it. A test here fails if the wiring is removed, even though every
system-level unit test would stay green.
"""
import asyncio
import json
from pathlib import Path

from aegis.eval.skill_library import SkillLibrary
from aegis.layers import dataset_builder
from aegis.layers.cognitive_graph import CognitiveGraph
from aegis.layers.dataset_builder import DatasetBuilder
from aegis.layers.evolution_engine import EvolutionEngine
from aegis.layers.feedback_loop import FeedbackLoop
from aegis.layers.goal_intelligence import GoalIntelligence
from aegis.layers.memory import MemorySystem
from aegis.layers.substrate import Substrate
from aegis.layers.world_model import WorldModel


def _make_substrate(tmp_path):
    """Offline substrate with every asserted-on structure redirected to tmp_path.

    Substrate() reads the developer's real data/ directory, so any test that
    asserts on a learned structure must substitute it (docs/QA.md §6).
    """
    s = Substrate()

    async def _noop_agents():
        return []

    s.agent_system.run_due_agents = _noop_agents
    s.llm.enabled = False
    s.environment.step = lambda: {"reward": 1.0, "solved": True, "task": "t1",
                                  "kind": "arith", "winning_skill": "adder"}
    # HealthMonitor.check() reads real CPU: under full-suite load it reports
    # "critical" and MetaRegulation then suppresses learning (docs/QA.md §6).
    s.health.check = lambda: {"status": "healthy", "warnings": [],
                              "critical": [], "metrics": {}}
    s.goal_intelligence = GoalIntelligence(store_path=tmp_path / "values.json")
    s.world_model = WorldModel(store_path=tmp_path / "world.json")
    s.cognitive_graph = CognitiveGraph(store_path=tmp_path / "graph.json")
    # The graph and the recency slice are both read when picking a learning
    # topic, so the developer's real memory must not leak into the assertion.
    s.memory.semantic, s.memory.episodic = {}, []
    s.feedback_loop = FeedbackLoop(store_path=tmp_path / "experience.jsonl")
    s.evolution = EvolutionEngine(store_path=tmp_path / "lineage.json")
    s.skill_library = SkillLibrary(store_path=tmp_path / "skills.json")
    return s


# ── System 4: Goal Intelligence must select, not merely score ────────────

def test_decision_follows_learned_utility(tmp_path):
    """A learned high-utility objective becomes the decision.

    Fails while goal_intelligence.choose() runs after the decision is already
    final: the choice is recorded as a number and steers nothing.
    """
    async def run():
        s = _make_substrate(tmp_path)
        s.active_archetype = None
        # Freeze the goal set: _decide() generates goals, which can move the
        # focus between seeding and asserting.
        s.goals.generate_goals = lambda ctx: []

        # Observe the real option set, then learn from realized reward that only
        # "rest" pays. Every other option must be driven down explicitly: an
        # unseen objective starts at utility 0.5, and "rest" serves the
        # lowest-weighted drive (stability), so one high utility is not enough.
        await s._decide()
        focus = s.goals.get_current_focus()
        # The trace's decision is already the value-driven pick, so the
        # heuristic option it replaced must be added back to the loser set.
        losers = set(s.introspection.decision_trace[-1]["alternatives"])
        losers.add(s.introspection.decision_trace[-1]["decision"])
        losers.add(focus["name"] if focus else "idle_exploration")
        assert "rest" in losers
        losers.discard("rest")
        for _ in range(30):
            s.goal_intelligence.reward(1.0, "rest")
            for loser in losers:
                s.goal_intelligence.reward(0.0, loser)

        await s._decide()

        trace = s.introspection.decision_trace[-1]
        # Compare against the choice the system actually made rather than
        # re-scoring here: expected_value is drive-modulated by the tick
        # context, so a reimplementation in the test would drift from it.
        assert trace["decision"] == s.goal_intelligence._last_choice["objective"], (
            "decision ignored the value-driven choice"
        )
        assert trace["decision"] == "rest", (
            "learned utility did not determine the decision"
        )
    asyncio.run(run())


# ── System 1: World Model must make known risk cost something ────────────

def test_known_failure_risk_lowers_decision_confidence(tmp_path):
    """Causes observed to fail reduce the confidence carried into ethics.

    Fails while build_chain's output only reaches the autobiography and
    predict/explain/risks_for have no callers at all.
    """
    async def run():
        clean = _make_substrate(tmp_path / "clean")
        clean._compute_confidence = lambda: 0.8
        clean.goals.generate_goals = lambda ctx: []
        await clean._decide()
        baseline = clean.introspection.decision_trace[-1]["confidence"]
        assert baseline == 0.8, "baseline confidence should be untouched"

        risky = _make_substrate(tmp_path / "risky")
        risky._compute_confidence = lambda: 0.8
        risky.goals.generate_goals = lambda ctx: []  # keep the focus stable
        focus = risky.goals.get_current_focus()
        focus_name = focus["name"] if focus else "idle_exploration"
        # Observed repeatedly: this course of action fails.
        for _ in range(8):
            risky.world_model.observe(focus_name, "tick_failure", success=False)

        await risky._decide()

        assert risky.introspection.decision_trace[-1]["confidence"] < baseline, (
            "known failure history did not reduce confidence"
        )
    asyncio.run(run())


def test_confidence_penalty_is_bounded(tmp_path):
    """Risk lowers confidence but never drives it to zero or negative."""
    async def run():
        s = _make_substrate(tmp_path)
        s._compute_confidence = lambda: 0.8
        s.goals.generate_goals = lambda ctx: []
        focus = s.goals.get_current_focus()
        focus_name = focus["name"] if focus else "idle_exploration"
        for effect in range(20):
            for _ in range(8):
                s.world_model.observe(focus_name, f"failure_{effect}", success=False)

        await s._decide()

        conf = s.introspection.decision_trace[-1]["confidence"]
        assert 0.0 < conf < 0.8
    asyncio.run(run())


# ── System 2: Cognitive Graph must steer what gets learned ───────────────

def test_learning_topic_comes_from_the_graph(tmp_path):
    """The next learning topic is graph-connected to the current focus.

    Fails while the graph is written every 8 ticks and read by nothing: the
    topic is picked from a flat recency slice of semantic memory.
    """
    async def run():
        s = _make_substrate(tmp_path)
        focus = s.goals.get_current_focus()
        focus_name = focus["name"] if focus else "idle_exploration"

        # Recency list holds only unrelated concepts — the old path would pick one.
        for unrelated in ("quantum chromodynamics", "baroque music", "tide tables"):
            s.memory.add_semantic(unrelated, {"type": "external_learning"})
        connected = f"{focus_name} failure modes"
        s.cognitive_graph.add_node(connected, "concept", {"source": "test"})

        captured = {}

        async def _record(source, topic):
            captured["topic"] = topic
            return {"success": False}

        s.external_learning.learn_from_source = _record
        s.tick_count = 40  # external-learning cadence
        s._regulation_directives = {}

        await s._act()

        assert captured.get("topic") == connected, (
            f"learning topic {captured.get('topic')!r} ignored the knowledge graph"
        )
    asyncio.run(run())


def test_learning_falls_back_when_graph_has_nothing(tmp_path):
    """An empty graph must not break learning — the recency path still runs."""
    async def run():
        s = _make_substrate(tmp_path)
        s.memory.add_semantic("cellular automata", {"type": "external_learning"})
        captured = {}

        async def _record(source, topic):
            captured["topic"] = topic
            return {"success": False}

        s.external_learning.learn_from_source = _record
        s.tick_count = 40
        s._regulation_directives = {}

        await s._act()

        assert captured.get("topic"), "learning did not run without a graph hit"
    asyncio.run(run())


# ── System 5: Feedback Loop must reach the training data ─────────────────

def _isolated_memory():
    """MemorySystem() loads the developer's real data/ — start from empty."""
    memory = MemorySystem()
    memory.semantic, memory.episodic, memory.procedural = {}, [], []
    return memory


def _written_rows(dataset_dir):
    rows = []
    for name in ("train.jsonl", "val.jsonl"):
        path = dataset_dir / name
        if path.exists():
            rows += [json.loads(line) for line in
                     path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def test_dataset_carries_experience_causes(tmp_path, monkeypatch):
    """Resolved experiences (with their inferred cause) become training rows.

    Fails while export_examples() has no caller and the dataset is built from
    memory alone — the causes are recorded and then dropped.
    """
    monkeypatch.setattr(dataset_builder, "WEIGHT_DATASETS_DIR", tmp_path / "datasets")
    fl = FeedbackLoop(store_path=tmp_path / "experience.jsonl")
    # Distinct metrics -> distinct completions: identical rows would dedup to a
    # single sample, which the builder now (correctly) refuses to split.
    for i in range(4):
        exp_id = fl.record_situation("energy=0.2 err=0.40", "explore_topic")
        fl.record_result(exp_id, success=False, metric=0.05 * i)

    result = DatasetBuilder().build_from_memory(_isolated_memory(), None, feedback_loop=fl)

    assert result["success"], result.get("error")
    assert result["source_distribution"].get("experience"), (
        "no experience rows reached the dataset"
    )
    rows = _written_rows(Path(result["dataset_dir"]))
    experience_rows = [r for r in rows if r.get("source") == "experience"]
    assert experience_rows, "experience rows were counted but not written"
    assert any("Cause:" in r["output"] for r in experience_rows), (
        "experience rows carry no cause — the point of System 5"
    )


def test_training_cycle_forwards_the_experience_log(tmp_path):
    """The tick's training cycle hands the feedback loop to the dataset build.

    Without this the previous test proves only that build_from_memory *can*
    take experiences — the running system would still never pass them.
    """
    async def run():
        s = _make_substrate(tmp_path)
        captured = {}

        async def _fake_propose(memory, agent_system=None, ethics_core=None,
                                feedback_loop=None):
            captured["feedback_loop"] = feedback_loop
            return {"status": "stub"}

        s.self_mod.propose_weight_modification = _fake_propose

        await s._weight_training_cycle()

        assert captured.get("feedback_loop") is s.feedback_loop, (
            "training cycle built the dataset without the experience log"
        )
    asyncio.run(run())


def test_weight_modification_forwards_the_experience_log(tmp_path, monkeypatch):
    """propose_weight_modification passes the log down to the dataset builder."""
    monkeypatch.setattr(dataset_builder, "WEIGHT_DATASETS_DIR", tmp_path / "datasets")
    s = _make_substrate(tmp_path)
    captured = {}

    def _fake_build(memory, agent_system=None, feedback_loop=None):
        captured["feedback_loop"] = feedback_loop
        return {"success": False, "error": "stub"}

    s.self_mod.weight_modifier.can_train = lambda: (True, "ok")
    s.self_mod.dataset_builder.build_from_memory = _fake_build

    asyncio.run(s.self_mod.propose_weight_modification(
        s.memory, s.agent_system, s.ethics, feedback_loop=s.feedback_loop))

    assert captured.get("feedback_loop") is s.feedback_loop


def test_dataset_without_feedback_loop_still_builds(tmp_path, monkeypatch):
    """The feedback loop is optional: existing callers keep working."""
    monkeypatch.setattr(dataset_builder, "WEIGHT_DATASETS_DIR", tmp_path / "datasets")
    memory = _isolated_memory()
    memory.add_semantic("cellular automata", {
        "type": "external_learning",
        "summary": "Discrete models evolving on a grid by local rules.",
    })
    # A second concept keeps the build above the minimum-samples guard (one
    # sample alone would leave the training split empty and be refused).
    memory.add_semantic("strange attractors", {
        "type": "external_learning",
        "summary": "Fractal sets that chaotic trajectories settle onto.",
    })

    result = DatasetBuilder().build_from_memory(memory)

    assert result["success"], result.get("error")
    assert "experience" not in result["source_distribution"]
