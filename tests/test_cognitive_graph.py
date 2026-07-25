"""Unit tests for System 2: CognitiveGraph (typed knowledge/experience graph)."""
import time
from aegis.layers.cognitive_graph import CognitiveGraph


def _cg(tmp_path):
    return CognitiveGraph(store_path=tmp_path / "cg.json")


class _FakeMemory:
    """Minimal stand-in for MemorySystem with the two fields the graph reads."""
    def __init__(self):
        self.semantic = {}
        self.episodic = []

    def add_semantic(self, concept, relations):
        self.semantic[concept] = {"relations": relations}

    def add_episodic(self, event, importance=0.5):
        self.episodic.append({"event": event, "importance": importance,
                              "timestamp": time.time()})


def test_add_node_and_edge(tmp_path):
    cg = _cg(tmp_path)
    cg.add_node("reinforcement learning", "concept")
    cg.add_node("reward signal", "concept")
    cg.add_edge("reinforcement learning", "reward signal", "relates_to", 0.6)
    nb = cg.neighbors("reinforcement learning")
    assert any(n["node"] == "reward signal" for n in nb)


def test_edge_reinforcement_increases_weight(tmp_path):
    cg = _cg(tmp_path)
    cg.add_node("a", "concept")
    cg.add_node("b", "concept")
    cg.add_edge("a", "b", "relates_to", 0.5)
    cg.add_edge("a", "b", "relates_to", 0.5)  # same edge again
    w = cg.edges["a"]["b"]["weight"]
    assert w > 0.5


def test_no_self_loops(tmp_path):
    cg = _cg(tmp_path)
    cg.add_node("x", "concept")
    cg.add_edge("x", "x", "relates_to")
    assert "x" not in cg.edges or "x" not in cg.edges.get("x", {})


def test_invalid_type_falls_back_to_concept(tmp_path):
    cg = _cg(tmp_path)
    cg.add_node("n", "not_a_type")
    assert cg.nodes["n"]["type"] == "concept"


def test_find_path(tmp_path):
    cg = _cg(tmp_path)
    for n in ("a", "b", "c"):
        cg.add_node(n, "concept")
    cg.add_edge("a", "b")
    cg.add_edge("b", "c")
    path = cg.find_path("a", "c")
    assert path == ["a", "b", "c"]


def test_find_path_none_when_disconnected(tmp_path):
    cg = _cg(tmp_path)
    cg.add_node("a", "concept")
    cg.add_node("z", "concept")
    assert cg.find_path("a", "z") is None


def test_related_ranks_by_overlap(tmp_path):
    cg = _cg(tmp_path)
    cg.add_node("neural network training", "concept")
    cg.add_node("photosynthesis biology", "concept")
    res = cg.related("how to train a neural network", k=3)
    assert res
    assert res[0]["node"] == "neural network training"


def test_ingest_memory_links_events_to_concepts(tmp_path):
    cg = _cg(tmp_path)
    mem = _FakeMemory()
    mem.add_semantic("transformers", {"type": "learned"})
    mem.add_episodic("studied transformers architecture", importance=0.8)
    cg.ingest_memory(mem)
    # concept node + event node both present, and linked
    assert "transformers" in cg.nodes
    ev = [n for n in cg.nodes if n.startswith("ev:")]
    assert ev
    nb = cg.neighbors(ev[0])
    assert any(n["node"] == "transformers" for n in nb)


def test_ingest_is_incremental(tmp_path):
    cg = _cg(tmp_path)
    mem = _FakeMemory()
    mem.add_episodic("first event")
    cg.ingest_memory(mem)
    n1 = len([n for n in cg.nodes if n.startswith("ev:")])
    cg.ingest_memory(mem)  # no new episodic — no new event nodes
    n2 = len([n for n in cg.nodes if n.startswith("ev:")])
    assert n1 == n2


def test_central_nodes(tmp_path):
    cg = _cg(tmp_path)
    for n in ("hub", "a", "b", "c"):
        cg.add_node(n, "concept")
    for leaf in ("a", "b", "c"):
        cg.add_edge("hub", leaf)
    central = cg.central_nodes(1)
    assert central[0]["node"] == "hub"


def test_pruning_bounds_nodes(tmp_path, monkeypatch):
    import aegis.layers.cognitive_graph as cgmod
    monkeypatch.setattr(cgmod, "MAX_NODES", 10)
    cg = _cg(tmp_path)
    for i in range(50):
        cg.add_node(f"node_{i}", "concept")
    assert len(cg.nodes) <= 10


def test_persistence_round_trip(tmp_path):
    p = tmp_path / "cg.json"
    cg = CognitiveGraph(store_path=p)
    cg.add_node("a", "concept")
    cg.add_node("b", "skill")
    cg.add_edge("a", "b", "requires", 0.7)
    cg.save()
    cg2 = CognitiveGraph(store_path=p)
    assert "a" in cg2.nodes and "b" in cg2.nodes
    assert cg2.edges["a"]["b"]["relation"] == "requires"


def test_status_by_type(tmp_path):
    cg = _cg(tmp_path)
    cg.add_node("c1", "concept")
    cg.add_node("s1", "skill")
    st = cg.status()
    assert st["by_type"]["concept"] == 1
    assert st["by_type"]["skill"] == 1


# ── mutation-hardening tests ──────────────────────────────────────────

def test_default_store_path_is_used(tmp_path, monkeypatch):
    # Kills the Path-division mutant in the default store_path branch.
    import aegis.layers.cognitive_graph as cgmod
    monkeypatch.setattr(cgmod, "COGNITIVE_GRAPH_DIR", tmp_path)
    cg = CognitiveGraph()
    assert cg._store_path == tmp_path / "graph.json"


def test_node_created_without_meta_gets_empty_dict(tmp_path):
    # Kills the `meta or {}` -> `meta and {}` mutant: a node created with no
    # meta must store {} (updatable), not None.
    cg = _cg(tmp_path)
    cg.add_node("n", "concept")             # no meta
    assert cg.nodes["n"]["meta"] == {}
    cg.add_node("n", "concept", {"k": 1})   # would raise if meta were None
    assert cg.nodes["n"]["meta"]["k"] == 1


def test_degree_is_in_plus_out(tmp_path):
    # Kills the `out_deg + in_deg` -> `out_deg - in_deg` mutant.
    cg = _cg(tmp_path)
    for n in ("mid", "a", "b"):
        cg.add_node(n, "concept")
    cg.add_edge("a", "mid")   # mid: 1 incoming
    cg.add_edge("mid", "b")   # mid: 1 outgoing
    # degree(mid) = 1 in + 1 out = 2, not 0
    central = {c["node"]: c["degree"] for c in cg.central_nodes(5)}
    assert central["mid"] == 2


def test_ingest_edge_weight_is_exact(tmp_path):
    # Kills the `0.3 + 0.4 * importance` Add/Mult mutants.
    cg = _cg(tmp_path)
    mem = _FakeMemory()
    mem.add_semantic("robotics", {"type": "learned"})
    mem.add_episodic("robotics research progress", importance=0.5)
    cg.ingest_memory(mem)
    ev = [n for n in cg.nodes if n.startswith("ev:")][0]
    weight = cg.edges[ev]["robotics"]["weight"]
    assert weight == 0.3 + 0.4 * 0.5   # 0.5, not 0.3-0.4*0.5 or 0.3+0.4/0.5


def test_ingest_does_not_link_unrelated_event(tmp_path):
    # Kills the `ctoks and ctoks & ev_tokens` And->Or mutant: an event with no
    # shared tokens must NOT be linked to the concept.
    cg = _cg(tmp_path)
    mem = _FakeMemory()
    mem.add_semantic("astronomy", {"type": "learned"})
    mem.add_episodic("cooking dinner tonight", importance=0.5)
    cg.ingest_memory(mem)
    ev = [n for n in cg.nodes if n.startswith("ev:")][0]
    # no token overlap -> no edge to astronomy
    assert "astronomy" not in cg.edges.get(ev, {})


def test_edge_not_added_when_target_missing(tmp_path):
    # Kills the `src not in or dst not in or self` guard mutants: an edge to a
    # non-existent node must be silently dropped.
    cg = _cg(tmp_path)
    cg.add_node("a", "concept")
    cg.add_edge("a", "ghost")              # ghost not a node
    assert "ghost" not in cg.edges.get("a", {})


def test_prune_keeps_exactly_max_nodes(tmp_path, monkeypatch):
    # Kills the prune boundary (<=) and slice-count (Sub->Add) mutants.
    import aegis.layers.cognitive_graph as cgmod
    monkeypatch.setattr(cgmod, "MAX_NODES", 10)
    cg = _cg(tmp_path)
    for i in range(50):
        cg.add_node(f"node_{i}", "concept")
    assert len(cg.nodes) == 10   # exactly the cap, not 0 and not 50


def test_related_uses_jaccard_not_product(tmp_path):
    # Kills the `overlap / len(union)` -> `overlap * len(union)` mutant. Crafted
    # so div and mult disagree on ORDER: the high-Jaccard small node must beat a
    # low-Jaccard node whose large token set would win under multiplication.
    cg = _cg(tmp_path)
    cg.add_node("alpha beta", "concept")                       # overlap 2, union 2
    cg.add_node("alpha one two three four five six", "concept")  # overlap 1, union 8
    res = cg.related("alpha beta", k=2)
    # Jaccard: 1.0 vs 0.125 -> "alpha beta" first. Product: 4 vs 8 -> would flip.
    assert res[0]["node"] == "alpha beta"


def test_related_degree_boost_raises_rank(tmp_path):
    # Kills the `1 + 0.05*degree` -> `1 - 0.05*degree` boost mutant: between two
    # equal-overlap nodes, the more-connected one must rank higher.
    cg = _cg(tmp_path)
    cg.add_node("signal alpha", "concept")
    cg.add_node("signal beta", "concept")
    # Give "signal alpha" a higher degree via edges to helper nodes.
    for i in range(3):
        cg.add_node(f"h{i}", "concept")
        cg.add_edge("signal alpha", f"h{i}")
    res = cg.related("signal", k=2)
    assert res[0]["node"] == "signal alpha"   # degree boost lifts it above beta
