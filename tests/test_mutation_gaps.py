"""Tests written to kill specific surviving mutants outside the sandbox.

Each one pins a safety or correctness property that `python scripts/mutation_test.py`
showed was unpinned: the suite passed even when the property was inverted.
"""
import asyncio
import json

import pytest


# ══ event_bus: the veto must FAIL CLOSED ══════════════════════════════
#
# Surviving mutant: `allowed = False` in the veto's except handler flipped to
# True — i.e. a safety check that CRASHES would let the event through, and the
# whole suite still passed.

def test_event_is_blocked_when_the_veto_raises():
    from aegis.event_bus import EventBus, Event, Layer

    bus = EventBus()

    def exploding_veto(_event):
        raise RuntimeError("safety check itself failed")

    bus.set_veto(exploding_veto)
    event = Event(source=Layer.INTROSPECTION, target=None,
                  event_type="test", payload={"x": 1})

    delivered = asyncio.run(bus.publish(event))

    assert delivered is False, "a veto that raised let the event through"
    blocked = bus.get_blocked(5)
    assert blocked and blocked[0]["blocked"] is True


def test_event_passes_when_the_veto_allows_it():
    """Guard: fail-closed must not mean fail-always."""
    from aegis.event_bus import EventBus, Event, Layer

    bus = EventBus()
    bus.set_veto(lambda _e: True)
    event = Event(source=Layer.INTROSPECTION, target=None, event_type="test", payload={})
    assert asyncio.run(bus.publish(event)) is True


# ══ self_preservation: lethal calls hidden behind a from-import ════════
#
# Surviving mutants: `(node.module or "")` and `alias.asname or alias.name`
# flipped to `and`, which silently stops resolving from-import aliases — the
# exact bypass the round-2 audit closed.

@pytest.mark.parametrize("code", [
    "from os import kill\ndef f():\n    kill(1, 9)\n",
    "from shutil import rmtree\ndef f():\n    rmtree('/data')\n",
    "from os import _exit\ndef f():\n    _exit(0)\n",
])
def test_lethal_call_via_plain_from_import_is_detected(code):
    from aegis.layers.self_preservation import _ast_lethal_findings
    assert _ast_lethal_findings(code), f"undetected lethal call: {code!r}"


@pytest.mark.parametrize("code", [
    "from os import kill as k\ndef f():\n    k(1, 9)\n",
    "from shutil import rmtree as nuke\ndef f():\n    nuke('/data')\n",
])
def test_lethal_call_via_aliased_from_import_is_detected(code):
    """The binding to watch is the LOCAL name (`k`), not the original one."""
    from aegis.layers.self_preservation import _ast_lethal_findings
    assert _ast_lethal_findings(code), f"undetected aliased lethal call: {code!r}"


def test_harmless_from_import_is_not_flagged():
    """Guard against over-detection: `from math import ceil` is not lethal."""
    from aegis.layers.self_preservation import _ast_lethal_findings
    assert _ast_lethal_findings("from math import ceil\ndef f():\n    return ceil(1.2)\n") == []


# ══ cognitive_graph: in-degree bookkeeping during pruning ═════════════
#
# Surviving mutant: `if self._in_degree[dst] <= 0: del ...` flipped to `> 0`,
# which drops the counter for nodes that still HAVE incoming edges.

def _build_hub_graph(tmp_path, monkeypatch, leaves=3):
    """A hub with `leaves` incoming edges, built with pruning switched off."""
    from aegis.layers import cognitive_graph as cg

    monkeypatch.setattr(cg, "MAX_NODES", 1000)
    g = cg.CognitiveGraph(store_path=tmp_path / "graph.json")
    g.add_node("hub", "concept")
    for i in range(leaves):
        g.add_node(f"leaf{i}", "concept")
        g.add_edge(f"leaf{i}", "hub")
    assert g._in_degree["hub"] == leaves
    return cg, g


def test_pruning_decrements_shared_target_without_dropping_it(tmp_path, monkeypatch):
    cg, g = _build_hub_graph(tmp_path, monkeypatch)
    # Tighten the cap so the next insert evicts the degree-0 newcomer AND the
    # oldest leaf — the hub keeps two of its three incoming edges. The cap is
    # per-instance now (Substrate.regulate_capacity moves it at runtime), so it
    # is set on the graph rather than on the module constant.
    g.max_nodes = 3
    g.add_node("newcomer", "concept")

    assert len(g.nodes) == 3
    assert "hub" in g.nodes, "the hub should outrank the leaves on degree"
    assert g._in_degree["hub"] == 2, (
        "pruning one leaf must DECREMENT the hub's in-degree, not delete it")
    brute = sum(1 for dsts in g.edges.values() if "hub" in dsts)
    assert g._degree("hub") == len(g.edges.get("hub", {})) + brute


def test_in_degree_entry_is_dropped_when_it_reaches_zero(tmp_path, monkeypatch):
    cg, g = _build_hub_graph(tmp_path, monkeypatch, leaves=2)
    # Evict everything but the hub: its in-degree falls to 0 and the entry must
    # go away entirely rather than linger at zero.
    g.max_nodes = 1
    g.add_node("newcomer", "concept")

    assert set(g.nodes) == {"hub"}
    assert "hub" not in g._in_degree, "in-degree entry lingered after reaching zero"
    assert g._degree("hub") == 0
    assert set(g._in_degree) <= set(g.nodes), "stale in-degree entry survived pruning"


# ══ feedback_loop: truncation threshold and status defaults ═══════════
#
# Surviving mutants: `MAX_JSONL_ROWS * 2` flipped to `/ 2` (truncating far too
# eagerly, silently discarding training data), and the `success` default in
# status() flipped to True (reporting a missing outcome as a success).

def test_log_is_not_truncated_below_the_threshold(tmp_path, monkeypatch):
    from aegis.layers import feedback_loop as fl

    monkeypatch.setattr(fl, "MAX_JSONL_ROWS", 5)
    fb = fl.FeedbackLoop(store_path=tmp_path / "experiences.jsonl")
    for i in range(9):                        # 9 <= 2 * 5, so nothing may be dropped
        eid = fb.record_situation(f"s{i}", "d")
        fb.record_result(eid, success=True, metric=0.5)

    lines = [ln for ln in fb._store_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 9, "experiences were discarded before the cap was reached"
    assert fb._rows_on_disk == 9


def test_log_is_truncated_once_past_the_threshold(tmp_path, monkeypatch):
    from aegis.layers import feedback_loop as fl

    monkeypatch.setattr(fl, "MAX_JSONL_ROWS", 5)
    fb = fl.FeedbackLoop(store_path=tmp_path / "experiences.jsonl")
    for i in range(11):                       # crosses 2 * 5
        eid = fb.record_situation(f"s{i}", "d")
        fb.record_result(eid, success=True, metric=0.5)

    lines = [ln for ln in fb._store_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 5, "log was not bounded to MAX_JSONL_ROWS after crossing 2x"


def test_append_does_not_touch_the_file_below_the_threshold(tmp_path, monkeypatch):
    """The append-side threshold must be observable on its own.

    The two `MAX_JSONL_ROWS * 2` guards (in _append and in _truncate_if_needed)
    are redundant for the DATA outcome, so mutating either alone changed no
    assertion. What the append-side guard uniquely controls is whether the log
    is re-read at all — the O(n)-per-append cost that R3-6 removed.
    """
    from pathlib import Path
    from aegis.layers import feedback_loop as fl

    monkeypatch.setattr(fl, "MAX_JSONL_ROWS", 5)
    fb = fl.FeedbackLoop(store_path=tmp_path / "experiences.jsonl")

    reads = {"count": 0}
    real_open = Path.open

    def counting_open(self, mode="r", *a, **kw):
        if self == fb._store_path and "r" in mode:
            reads["count"] += 1
        return real_open(self, mode, *a, **kw)

    monkeypatch.setattr(Path, "open", counting_open)
    for i in range(9):                       # 9 <= 2 * 5 — must never re-read
        eid = fb.record_situation(f"s{i}", "d")
        fb.record_result(eid, success=True, metric=0.5)

    assert reads["count"] == 0, (
        f"log was re-read {reads['count']}x below the truncation threshold")


def test_truncation_resyncs_a_drifted_counter_without_losing_data(tmp_path, monkeypatch):
    """If the tracked row count drifts above reality (file replaced underneath
    us), _truncate_if_needed must correct the counter — not delete live rows."""
    from aegis.layers import feedback_loop as fl

    monkeypatch.setattr(fl, "MAX_JSONL_ROWS", 5)
    log = tmp_path / "experiences.jsonl"
    log.write_text("".join(
        json.dumps({"id": f"exp_{i:08d}", "success": True, "metric": 0.5}) + "\n"
        for i in range(6)), encoding="utf-8")

    fb = fl.FeedbackLoop(store_path=log)
    fb._rows_on_disk = 999                   # drifted counter
    fb._truncate_if_needed()

    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 6, "rows were discarded although the log was under the cap"
    assert fb._rows_on_disk == 6, "drifted counter was not resynced to the file"


def test_missing_outcome_is_reported_as_failure_not_success(tmp_path):
    """A row with no recorded outcome must never be counted as a success."""
    from aegis.layers.feedback_loop import FeedbackLoop

    log = tmp_path / "experiences.jsonl"
    log.write_text(json.dumps({"id": "exp_00000001", "situation": "s"}) + "\n",
                   encoding="utf-8")
    st = FeedbackLoop(store_path=log).status()
    assert st["recent"][0]["success"] is False
