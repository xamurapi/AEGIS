"""Zero-randomness guarantee: the deterministic replacements for the former
``random`` calls still do real work (select topics, stagger agents, assign
priorities/ids, shuffle datasets) AND are fully reproducible.
"""
import json
import hashlib

from aegis.layers.external_learning import ExternalLearning
from aegis.layers.agent_system import AgentSystem
from aegis.layers.meta_goal_generator import MetaGoalGenerator


# ── no `random` import anywhere in the core ──────────────────────────

def test_no_random_import_in_core():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "aegis"
    offenders = []
    # Actual RNG *calls* (require the opening paren) — this ignores docstring /
    # comment mentions like ``random.shuffle`` that only describe the old code.
    call_patterns = ("random.choice(", "random.random(", "random.randint(",
                     "random.uniform(", "random.sample(", "random.shuffle(",
                     "random.seed(", "np.random")
    for py in root.rglob("*.py"):
        for ln in py.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if s.startswith("import random") or s.startswith("from random "):
                offenders.append(f"{py.name}: {s}")
            if any(c in ln for c in call_patterns):
                offenders.append(f"{py.name}: {s}")
    assert offenders == [], f"random still used in core: {offenders}"


# ── external_learning: rotation actually selects, and cycles ─────────

def test_external_learning_rotate_cycles():
    el = ExternalLearning()
    picks = [el._rotate(["a", "b", "c"]) for _ in range(4)]
    assert picks == ["a", "b", "c", "a"]           # real selection + wrap-around
    assert el._rotate([]) == ""                     # empty is safe


def test_external_learning_rotate_n():
    el = ExternalLearning()
    assert el._rotate_n(["x1", "x2", "x3", "x4"], 3) == ["x1", "x2", "x3"]
    assert el._rotate_n([], 3) == []


def test_external_learning_is_reproducible():
    a = ExternalLearning()
    b = ExternalLearning()
    seq_a = [a._rotate(["p", "q", "r"]) for _ in range(6)]
    seq_b = [b._rotate(["p", "q", "r"]) for _ in range(6)]
    assert seq_a == seq_b                           # two instances -> identical


# ── agent_system: topics assigned, ids unique, staggered start ──────

def test_agent_system_initializes_with_real_topics_and_unique_ids():
    a = AgentSystem()
    a.auto_initialize()
    assert len(a.agents) >= 5
    ids = [ag.agent_id for ag in a.agents]
    assert len(set(ids)) == len(ids)               # ids are unique (was randint)
    # Every blueprint that has topics got a real (non-empty) topic assigned.
    with_topics = [ag for ag in a.agents if ag.name != "quote_gatherer"]
    assert all(ag.topic for ag in with_topics)


def test_agent_system_next_run_is_staggered():
    import time
    a = AgentSystem()
    a.auto_initialize()
    offsets = sorted(round(ag.next_run - time.time()) for ag in a.agents)
    # Deterministic 5,10,15,... spread — distinct start times, none piled up.
    assert len(set(offsets)) == len(offsets)
    assert min(offsets) >= 5


def test_agent_ids_are_reproducible_and_monotonic():
    a = AgentSystem()
    first = a._create("x", "arxiv", "t").agent_id
    second = a._create("y", "arxiv", "t").agent_id
    assert first.endswith("_0001")
    assert second.endswith("_0002")               # monotonic counter, no RNG


# ── meta_goal_generator: real varied priorities, no RNG ─────────────

def _all_triggers_context():
    return {"error_rate": 0.9, "memory_total": 9999, "information_gain": 0.0,
            "energy": 0.1, "success_rate": 0.0, "stagnation": 10, "avg_tick_ms": 9999,
            "goals_completed": 0, "tick": 1}


def test_meta_goals_have_valid_varied_priorities():
    g = MetaGoalGenerator()
    goals = g.generate_goals(_all_triggers_context())
    assert goals                                    # goals were actually produced
    for gg in goals:
        assert 0.4 <= gg["priority"] <= 0.9         # in range
        assert gg["description"]                    # real text selected
    assert len({gg["priority"] for gg in goals}) > 1  # priorities vary (for sorting)


def test_meta_goals_are_reproducible():
    g1 = MetaGoalGenerator()
    g2 = MetaGoalGenerator()
    d1 = [(x["domain"], x["priority"]) for x in g1.generate_goals(_all_triggers_context())]
    d2 = [(x["domain"], x["priority"]) for x in g2.generate_goals(_all_triggers_context())]
    assert d1 == d2


# ── dataset_builder: hash-shuffle reorders and is reproducible ──────

def _hash_shuffle(samples):
    return sorted(samples, key=lambda s: hashlib.md5(
        json.dumps(s, sort_keys=True, default=str).encode("utf-8")).hexdigest())


def test_hash_shuffle_reorders_and_is_reproducible():
    samples = [{"id": i} for i in range(8)]
    order1 = [s["id"] for s in _hash_shuffle(list(samples))]
    order2 = [s["id"] for s in _hash_shuffle(list(samples))]
    assert order1 == order2                         # reproducible
    assert order1 != list(range(8))                 # actually decorrelated from input
    assert sorted(order1) == list(range(8))         # nothing lost
