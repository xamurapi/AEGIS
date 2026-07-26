"""Bounded growth: indexed lookup, cost-driven capacity, and what to forget.

Three properties, all of which the fixed module constants left unaddressed:

1. Looking up risks must not enumerate the whole link table. `risks_for` runs
   every tick now, so its cost must not scale with everything ever learned.
2. The caps must follow MEASURED cost, not a hand-picked number. The system
   already measures tick latency and already has a threshold for it.
3. Pruning must keep what is informative. Dropping by frequency alone throws
   away the rare decisive failure before the frequent unremarkable link — the
   exact inversion of what a risk memory is for.
"""
import asyncio

import pytest

from aegis.layers.cognitive_graph import CognitiveGraph
from aegis.layers.evolution_engine import EvolutionEngine
from aegis.layers.feedback_loop import FeedbackLoop
from aegis.layers.goal_intelligence import GoalIntelligence
from aegis.layers.substrate import Substrate
from aegis.layers.world_model import WorldModel


class CountingLinks(dict):
    """Records full-table enumerations so a scan cannot slip back in."""

    def __init__(self, *args):
        super().__init__(*args)
        self.enumerations = 0

    def items(self):
        self.enumerations += 1
        return super().items()


def _wm(tmp_path):
    return WorldModel(store_path=tmp_path / "world.json")


# ── 1. Indexed lookup ────────────────────────────────────────────────────

def test_risk_lookup_does_not_enumerate_every_link(tmp_path):
    """risks_for resolves candidates through an index, not a full scan."""
    wm = _wm(tmp_path)
    for i in range(50):
        for _ in range(3):
            wm.observe(f"unrelated_{i}", "effect", success=False)
    for _ in range(3):
        wm.observe("target_action", "collapse", success=False)

    wm.links = CountingLinks(wm.links)
    risks = wm.risks_for(["target"])

    assert risks and risks[0]["cause"] == "target_action"
    assert wm.links.enumerations == 0, (
        "risks_for walked the whole link table instead of using the index"
    )


def test_chain_plan_does_not_enumerate_every_link(tmp_path):
    """build_chain's plan search uses the same index."""
    wm = _wm(tmp_path)
    for i in range(50):
        for _ in range(4):
            wm.observe(f"unrelated_{i}", "effect", success=True)
    for _ in range(4):
        wm.observe("deploy service", "service_up", success=True)

    wm.links = CountingLinks(wm.links)
    chain = wm.build_chain("deploy service")

    assert chain["plan"], "plan lost its steps"
    assert wm.links.enumerations == 0, "build_chain walked the whole link table"


def test_index_survives_reload(tmp_path):
    """The index is derived, not persisted — it must be rebuilt on load."""
    wm = _wm(tmp_path)
    for _ in range(3):
        wm.observe("alpha_task", "fail", success=False)
    wm.save()

    reloaded = WorldModel(store_path=tmp_path / "world.json")

    assert reloaded.risks_for(["alpha"]), "index was not rebuilt after load"


def test_index_forgets_pruned_causes(tmp_path):
    """A cause dropped by pruning must not be reachable through the index."""
    wm = _wm(tmp_path)
    wm.max_links = 2
    for _ in range(2):
        wm.observe("doomed_cause", "x", success=False)
    for i in range(4):
        for _ in range(9):
            wm.observe(f"kept_{i}", "y", success=False)

    assert "doomed_cause" not in wm.links
    assert all(r["cause"] != "doomed_cause" for r in wm.risks_for(["doomed"])), (
        "index returned a cause that was pruned"
    )


# ── 3. Retention: keep what is informative ───────────────────────────────

def test_rare_severe_failure_outlives_frequent_neutral_link(tmp_path):
    """A decisive failure seen 3 times beats a coin-flip link seen 20 times.

    Fails while pruning sorts on observation count alone: the rare disaster is
    the first thing dropped, which is backwards for a memory of what fails.
    """
    wm = _wm(tmp_path)
    wm.max_links = 3
    for i in range(3):
        for n in range(20):
            wm.observe(f"routine_{i}", "mixed", success=(n % 2 == 0))
    for _ in range(3):
        wm.observe("rare_disaster", "data_loss", success=False)

    assert "rare_disaster" in wm.links, (
        "the rare decisive failure was pruned before frequent noise"
    )


def test_prune_still_respects_the_cap(tmp_path):
    """Retention scoring must not defeat the bound it is scoring inside."""
    wm = _wm(tmp_path)
    wm.max_links = 5
    for i in range(30):
        for _ in range(3):
            wm.observe(f"cause_{i}", "effect", success=(i % 2 == 0))

    assert sum(len(v) for v in wm.links.values()) <= 5


def test_retention_score_formula_is_exact(tmp_path):
    """Pin the arithmetic itself, not just the ordering it produces.

    Ordering tests cannot see a uniform rescaling of the score, so the scale is
    asserted directly: decisiveness (0..1) × evidence (0..1) × failure bias.
    """
    wm = _wm(tmp_path)
    #   strength     = (0 + 1) / (10 + 2) = 1/12
    #   decisiveness = |1/12 - 0.5| * 2   = 5/6
    #   evidence     = min(1, 10/10)      = 1.0
    #   bias         = 1.5 (a failure)
    score = wm._retention_score({"observations": 10, "successes": 0, "updated": 0.0})
    assert score == pytest.approx(1.25)

    #   a coin-flip link carries no information at all
    neutral = wm._retention_score({"observations": 10, "successes": 5, "updated": 0.0})
    assert neutral == pytest.approx(0.0, abs=0.05)


def test_failure_outranks_an_equally_decisive_success(tmp_path):
    """Same decisiveness, same evidence, opposite sign — the failure survives.

    Pins the direction of the failure bias. A memory of what goes wrong is
    worth more than an equally certain memory of what goes right, because the
    first prevents an action and the second only reassures.
    """
    wm = _wm(tmp_path)
    wm.max_links = 2
    for _ in range(8):
        wm.observe("reliable", "works", success=True)    # strength 0.9
    for _ in range(8):
        wm.observe("broken", "fails", success=False)     # strength 0.1
    wm.observe("trigger", "x", success=False)            # forces one eviction

    assert "broken" in wm.links, "the known failure was evicted"
    assert "reliable" not in wm.links


def test_well_evidenced_link_outranks_a_thin_one(tmp_path):
    """Equal decisiveness — the one backed by more observations survives.

    Pins evidence weighting: 2 observations must not count the same as 10.
    """
    wm = _wm(tmp_path)
    wm.max_links = 2
    for n in range(10):
        wm.observe("thick", "effect", success=(n < 2))   # 2/10 -> strength 0.25
    for _ in range(2):
        wm.observe("thin", "effect", success=False)      # 0/2  -> strength 0.25
    wm.observe("trigger", "x", success=False)

    assert "thick" in wm.links, "the better-evidenced link was evicted"
    assert "thin" not in wm.links


def test_decisive_success_is_also_kept(tmp_path):
    """Retention keeps informative links of BOTH signs, not just failures."""
    wm = _wm(tmp_path)
    wm.max_links = 2
    for n in range(20):
        wm.observe("noise", "mixed", success=(n % 2 == 0))
    for _ in range(8):
        wm.observe("reliable", "works", success=True)
    for _ in range(8):
        wm.observe("broken", "fails", success=False)

    assert "reliable" in wm.links and "broken" in wm.links
    assert "noise" not in wm.links


# ── 2. Capacity follows measured cost ────────────────────────────────────

def _make_substrate(tmp_path):
    s = Substrate()

    async def _noop_agents():
        return []

    async def _noop_learn(*a, **k):
        return {"success": False}

    s.agent_system.run_due_agents = _noop_agents
    s.external_learning.learn_from_source = _noop_learn
    s.llm.enabled = False
    s.health.check = lambda: {"status": "healthy", "warnings": [],
                              "critical": [], "metrics": {}}
    s.world_model = WorldModel(store_path=tmp_path / "world.json")
    s.cognitive_graph = CognitiveGraph(store_path=tmp_path / "graph.json")
    s.goal_intelligence = GoalIntelligence(store_path=tmp_path / "values.json")
    s.feedback_loop = FeedbackLoop(store_path=tmp_path / "experience.jsonl")
    s.evolution = EvolutionEngine(store_path=tmp_path / "lineage.json")
    return s


def test_capacity_grows_while_ticks_are_cheap(tmp_path):
    """Fast ticks buy a bigger memory horizon."""
    s = _make_substrate(tmp_path)
    before_links = s.world_model.max_links
    before_nodes = s.cognitive_graph.max_nodes
    s.health.tick_durations.extend([10.0] * 20)  # far below the 5000ms threshold

    s.regulate_capacity()

    assert s.world_model.max_links > before_links
    assert s.cognitive_graph.max_nodes > before_nodes


def test_capacity_shrinks_when_ticks_get_expensive(tmp_path):
    """Slow ticks give the memory horizon back."""
    s = _make_substrate(tmp_path)
    s.health.tick_durations.extend([10.0] * 20)
    s.regulate_capacity()
    grown = s.world_model.max_links

    s.health.tick_durations.clear()
    s.health.tick_durations.extend([9000.0] * 20)  # above the threshold
    s.regulate_capacity()

    assert s.world_model.max_links < grown


def test_capacity_never_falls_below_the_baseline(tmp_path):
    """Shrinking cannot erase the configured floor."""
    s = _make_substrate(tmp_path)
    baseline = s.world_model.max_links
    s.health.tick_durations.extend([9000.0] * 20)
    for _ in range(50):
        s.regulate_capacity()

    assert s.world_model.max_links == baseline


def test_capacity_never_exceeds_the_ceiling(tmp_path):
    """Growth is bounded — "measured cost" is not a licence to grow forever."""
    s = _make_substrate(tmp_path)
    baseline = s.world_model.max_links
    s.health.tick_durations.extend([1.0] * 20)
    for _ in range(200):
        s.regulate_capacity()

    assert s.world_model.max_links <= baseline * 20


def test_capacity_holds_without_measurements(tmp_path):
    """No tick history means no evidence — the caps must not drift."""
    s = _make_substrate(tmp_path)
    s.health.tick_durations.clear()
    before = s.world_model.max_links

    s.regulate_capacity()

    assert s.world_model.max_links == before


def test_capacity_is_reported_in_status(tmp_path):
    """An operator must be able to see the caps the system chose."""
    s = _make_substrate(tmp_path)
    capacity = s.full_status()["capacity"]
    assert capacity["world_model_max_links"] == s.world_model.max_links
    assert capacity["cognitive_graph_max_nodes"] == s.cognitive_graph.max_nodes


def test_tick_regulates_capacity_on_cadence(tmp_path):
    """The controller is actually wired into the loop, not just callable."""
    async def run():
        s = _make_substrate(tmp_path)
        called = []
        s.regulate_capacity = lambda: called.append(1)
        for _ in range(60):
            await s.tick()
        assert called, "capacity regulation never ran during ticks"
    asyncio.run(run())
