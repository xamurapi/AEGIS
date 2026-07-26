"""Regression tests for the third-round audit fixes.

Every test here fails on the pre-fix code and passes after it. Findings are
labelled R3-1..R3-10 and cross-referenced in docs/АУДИТ.md.
"""
import json
import os
import tempfile
from pathlib import Path

import pytest


# ══ R3-1 (CRITICAL): sandbox escape via unchecked annotations ═════════
#
# check_safe() walked decorators, defaults and the body but never the
# ANNOTATIONS of a signature, nor a lambda's keyword-only defaults. Python
# evaluates all of those when the `def`/`lambda` executes, so the payload ran
# in the child process with a clean "safe" verdict.

@pytest.mark.parametrize("code,needle", [
    # parameter annotation
    ('def solve(p, _z: __import__("os").getcwd()):\n    return 1\n', "__import__"),
    # return annotation
    ('def solve(p) -> __import__("os").getcwd():\n    return 1\n', "__import__"),
    # *args / **kwargs annotations
    ('def solve(p, *a: eval("1")):\n    return 1\n', "eval"),
    ('def solve(p, **k: open("x")):\n    return 1\n', "open"),
    # keyword-only default of a lambda
    ('f = lambda *, a=__import__("os").getcwd(): a\ndef solve(p):\n    return 1\n', "__import__"),
    # nested function's annotation
    ('def solve(p):\n    def g(x: __import__("os").getcwd()): return x\n    return 1\n', "__import__"),
])
def test_annotation_sandbox_escapes_blocked(code, needle):
    from aegis.eval.sandbox import check_safe
    safe, reasons = check_safe(code)
    assert safe is False, f"annotation escape not detected: {code!r}"
    assert any(needle in r for r in reasons)


def test_annotation_escape_does_not_execute():
    """End-to-end proof: the payload must not run and must leave no side effect."""
    from aegis.eval.sandbox import run_skill
    marker = Path(tempfile.gettempdir()) / "aegis_r3_escape_proof.txt"
    if marker.exists():
        marker.unlink()
    code = (f'def solve(p, _z: __import__("pathlib").Path(r"{marker}").write_text("X")):\n'
            f'    return 1\n')
    out = run_skill(code, "solve", None, timeout=5.0)
    try:
        assert out["ok"] is False
        assert "unsafe code" in out["error"]
        assert not marker.exists(), "sandbox escape executed — payload wrote to disk"
    finally:
        if marker.exists():
            marker.unlink()


def test_legitimate_annotations_still_accepted():
    """The fix must not reject ordinary typed skill code."""
    from aegis.eval.sandbox import check_safe, run_skill
    code = ("import math\n"
            "def solve(payload: dict) -> int:\n"
            "    return int(math.sqrt(payload['n']))\n")
    safe, reasons = check_safe(code)
    assert safe is True, reasons
    assert run_skill(code, "solve", {"n": 16}, timeout=5.0) == {"ok": True, "result": 4}


def test_lambda_positional_default_still_checked():
    """The pre-existing positional-default check must survive the refactor."""
    from aegis.eval.sandbox import check_safe
    safe, _ = check_safe('f = lambda a=eval("1"): a\ndef solve(p):\n    return 1\n')
    assert safe is False


# ══ R3-2 (HIGH): full_status leaked to unauthorized WebSocket clients ═
#
# The handshake gated the FIRST send and inbound commands, but every socket —
# authorized or not — was registered in connected_ws, and Substrate broadcasts
# full_status() to that list every tick.

def _ws_client(monkeypatch, token_query):
    import aegis.config as cfg
    from fastapi.testclient import TestClient
    from aegis.api import server

    monkeypatch.setattr(cfg, "API_TOKEN", "s3cret")
    monkeypatch.setattr(server, "connected_ws", [])

    class _FakeSubstrate:
        def full_status(self):
            return {"secret_internal_state": "leak"}

    monkeypatch.setattr(server, "substrate", _FakeSubstrate())
    client = TestClient(server.app)
    url = "/ws" + (f"?token={token_query}" if token_query else "")
    return server, client, url


def test_unauthorized_ws_is_not_registered_for_broadcast(monkeypatch):
    server, client, url = _ws_client(monkeypatch, token_query=None)
    with client.websocket_connect(url) as ws:
        first = json.loads(ws.receive_text())
        assert first == {"error": "unauthorized"}
        # The socket must NOT be in the broadcast fan-out.
        assert server.connected_ws == [], "unauthorized socket joined the broadcast list"


def test_authorized_ws_is_registered_and_receives_status(monkeypatch):
    server, client, url = _ws_client(monkeypatch, token_query="s3cret")
    with client.websocket_connect(url) as ws:
        first = json.loads(ws.receive_text())
        assert first == {"secret_internal_state": "leak"}
        assert len(server.connected_ws) == 1


def test_broadcast_reaches_only_authorized_sockets(monkeypatch):
    """Directly exercise broadcast(): an unauthorized socket must get nothing."""
    import asyncio
    server, client, url = _ws_client(monkeypatch, token_query=None)
    with client.websocket_connect(url):
        assert server.connected_ws == []
        # broadcast() iterates connected_ws — with no authorized clients it is
        # a no-op, so no state can reach the connected unauthorized socket.
        asyncio.run(server.broadcast({"secret_internal_state": "leak"}))


# ══ R3-3 (MED): zero-valued genome parameter could never mutate ═══════

def test_zero_parameter_mutation_is_not_a_noop(tmp_path):
    from aegis.layers.evolution_engine import EvolutionEngine
    ev = EvolutionEngine(store_path=tmp_path / "lineage.json")
    ev.register_champion({"dropout": 0.0}, fitness=0.5)
    mutation = ev.propose_mutation(tick=10)
    assert mutation is not None
    assert mutation["new_value"] != mutation["old_value"], (
        "a zero parameter is a fixed point of the multiplicative step — "
        "that genome slot could never be explored")


def test_zero_parameter_explores_both_directions(tmp_path):
    from aegis.layers.evolution_engine import EvolutionEngine
    ev = EvolutionEngine(store_path=tmp_path / "lineage.json")
    ev.register_champion({"dropout": 0.0}, fitness=0.5)
    first = ev.propose_mutation(tick=1)["new_value"]
    ev.judge_candidate(0.1)          # reject, clears the candidate
    second = ev.propose_mutation(tick=2)["new_value"]
    assert first > 0 > second, "direction must still alternate for zero params"


def test_nonzero_parameter_keeps_multiplicative_step(tmp_path):
    from aegis.layers.evolution_engine import EvolutionEngine, MUTATION_MAGNITUDE
    ev = EvolutionEngine(store_path=tmp_path / "lineage.json")
    ev.register_champion({"temperature": 0.7}, fitness=0.5)
    m = ev.propose_mutation(tick=1)
    assert m["new_value"] == pytest.approx(0.7 * (1 + MUTATION_MAGNITUDE))


# ══ R3-4 (MED): restart absorbed an unjudged mutation into the champion ═

def test_pending_candidate_param_not_synced_into_champion(tmp_path, monkeypatch):
    """On restart the live parameters are the source of truth — EXCEPT for the
    parameter of a pending candidate, whose live value is the unjudged mutation."""
    import aegis.config as cfg
    from aegis.layers.evolution_engine import EvolutionEngine
    from aegis.layers.substrate import Substrate

    monkeypatch.setattr(cfg, "CHECKPOINTS_DIR", tmp_path)
    sub = Substrate()
    monkeypatch.setattr(sub, "_checkpoint_path", tmp_path / "latest.json")
    # Isolate from any evolution state persisted by a real run.
    sub.evolution = EvolutionEngine(store_path=tmp_path / "lineage.json")

    # Champion knows temperature=0.7; a mutation to 0.77 is applied and pending.
    param = "temperature"
    sub.self_mod.parameters[param] = 0.7
    sub.evolution.register_champion({param: 0.7}, fitness=0.5)
    sub.evolution.propose_mutation(tick=5)
    assert sub.evolution.candidate["mutated_param"] == param
    sub.self_mod.parameters[param] = 0.77
    sub.evolution.candidate["new_value"] = 0.77
    sub.evolution.candidate["genome"][param] = 0.77

    (tmp_path / "latest.json").write_text(json.dumps({
        "tick_count": 5, "version": "1.0.0", "parameters": {param: 0.77},
    }), encoding="utf-8")

    sub._restore_checkpoint()

    assert sub.evolution.champion["genome"][param] == 0.7, (
        "restart silently promoted an unjudged mutation to champion")
    assert sub.self_mod.parameters[param] == 0.77  # live value still under test


def test_non_pending_params_still_synced_on_restore(tmp_path, monkeypatch):
    import aegis.config as cfg
    from aegis.layers.evolution_engine import EvolutionEngine
    from aegis.layers.substrate import Substrate

    monkeypatch.setattr(cfg, "CHECKPOINTS_DIR", tmp_path)
    sub = Substrate()
    monkeypatch.setattr(sub, "_checkpoint_path", tmp_path / "latest.json")
    sub.evolution = EvolutionEngine(store_path=tmp_path / "lineage.json")
    sub.self_mod.parameters["temperature"] = 0.7
    sub.evolution.register_champion({"temperature": 0.7}, fitness=0.5)
    sub.evolution.candidate = None

    (tmp_path / "latest.json").write_text(json.dumps({
        "tick_count": 9, "version": "1.0.0", "parameters": {"temperature": 0.9},
    }), encoding="utf-8")
    sub._restore_checkpoint()

    assert sub.evolution.champion["genome"]["temperature"] == 0.9


# ══ R3-5 (MED): one torn JSONL line discarded the whole history ═══════

def test_corrupt_line_does_not_wipe_experience_history(tmp_path):
    from aegis.layers.feedback_loop import FeedbackLoop
    log = tmp_path / "experiences.jsonl"
    good = {"id": "exp_00000001", "situation": "s", "decision": "d",
            "success": True, "metric": 0.9, "cause": "c"}
    good2 = {"id": "exp_00000002", "situation": "s2", "decision": "d2",
             "success": False, "metric": 0.1, "cause": "c2"}
    log.write_text(json.dumps(good) + "\n"
                   + '{"id": "exp_0000000\n'          # torn line (crash mid-write)
                   + json.dumps(good2) + "\n", encoding="utf-8")

    fb = FeedbackLoop(store_path=log)
    assert fb.resolved == 2, "a single torn line discarded every other experience"
    assert fb.successes == 1 and fb.failures == 1
    assert fb._seq == 2, "id sequence must continue past the last valid row"


def test_export_examples_tolerates_legacy_rows(tmp_path):
    from aegis.layers.feedback_loop import FeedbackLoop
    log = tmp_path / "experiences.jsonl"
    log.write_text(json.dumps({"id": "exp_00000001", "situation": "s"}) + "\n",
                   encoding="utf-8")
    fb = FeedbackLoop(store_path=log)
    rows = fb.export_examples()          # must not raise KeyError
    assert len(rows) == 1
    assert rows[0]["success"] is False and rows[0]["metric"] == 0.0


def test_status_tolerates_legacy_rows(tmp_path):
    from aegis.layers.feedback_loop import FeedbackLoop
    log = tmp_path / "experiences.jsonl"
    log.write_text(json.dumps({"id": "exp_00000001", "decision": "d"}) + "\n",
                   encoding="utf-8")
    fb = FeedbackLoop(store_path=log)
    st = fb.status()                     # must not raise KeyError
    assert st["resolved"] == 1
    assert st["recent"][0]["metric"] == 0.0


# ══ R3-6 (MED): O(n) file read on every single append ═════════════════

def test_append_does_not_reread_the_whole_log(tmp_path, monkeypatch):
    from aegis.layers import feedback_loop as fl

    fb = fl.FeedbackLoop(store_path=tmp_path / "experiences.jsonl")
    opens = {"count": 0}
    real_open = Path.open

    def counting_open(self, mode="r", *a, **kw):
        if self == fb._store_path and "r" in mode:
            opens["count"] += 1
        return real_open(self, mode, *a, **kw)

    monkeypatch.setattr(Path, "open", counting_open)
    for i in range(25):
        eid = fb.record_situation(f"situation {i}", "decide")
        fb.record_result(eid, success=True, metric=0.8)

    assert opens["count"] == 0, (
        f"log was read back {opens['count']}× while appending 25 rows")
    assert fb._rows_on_disk == 25


def test_truncation_still_bounds_the_log(tmp_path, monkeypatch):
    from aegis.layers import feedback_loop as fl
    monkeypatch.setattr(fl, "MAX_JSONL_ROWS", 5)
    fb = fl.FeedbackLoop(store_path=tmp_path / "experiences.jsonl")
    for i in range(14):                       # crosses 2× the cap
        eid = fb.record_situation(f"s{i}", "d")
        fb.record_result(eid, success=True, metric=0.5)

    lines = [ln for ln in fb._store_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) <= fl.MAX_JSONL_ROWS * 2
    assert fb._rows_on_disk == len(lines), "tracked row count drifted from disk"


# ══ R3-7 (MED): _degree() was O(E), making status() O(N·E) ════════════

def test_in_degree_index_matches_a_full_scan(tmp_path):
    from aegis.layers.cognitive_graph import CognitiveGraph
    g = CognitiveGraph(store_path=tmp_path / "graph.json")
    for i in range(12):
        g.add_node(f"n{i}", "concept")
    for i in range(1, 12):
        g.add_edge(f"n{i}", "n0", "relates_to")
        g.add_edge("n0", f"n{i}", "causes")

    def brute_force(node_id):
        out_deg = len(g.edges.get(node_id, {}))
        in_deg = sum(1 for dsts in g.edges.values() if node_id in dsts)
        return out_deg + in_deg

    for node_id in g.nodes:
        assert g._degree(node_id) == brute_force(node_id), node_id


def test_in_degree_index_survives_pruning(tmp_path, monkeypatch):
    from aegis.layers import cognitive_graph as cg
    monkeypatch.setattr(cg, "MAX_NODES", 6)
    g = cg.CognitiveGraph(store_path=tmp_path / "graph.json")
    for i in range(6):
        g.add_node(f"n{i}", "concept")
    for i in range(1, 6):
        g.add_edge(f"n{i}", "n0")
    for i in range(6, 12):                    # forces repeated pruning
        g.add_node(f"n{i}", "concept")

    def brute_force(node_id):
        out_deg = len(g.edges.get(node_id, {}))
        in_deg = sum(1 for dsts in g.edges.values() if node_id in dsts)
        return out_deg + in_deg

    assert len(g.nodes) <= cg.MAX_NODES
    for node_id in g.nodes:
        assert g._degree(node_id) == brute_force(node_id), node_id
    # No stale entries for nodes that no longer exist.
    assert set(g._in_degree) <= set(g.nodes)


def test_in_degree_index_rebuilt_on_load(tmp_path):
    from aegis.layers.cognitive_graph import CognitiveGraph
    path = tmp_path / "graph.json"
    g = CognitiveGraph(store_path=path)
    g.add_node("a", "concept")
    g.add_node("b", "concept")
    g.add_edge("a", "b")
    g.save()

    reloaded = CognitiveGraph(store_path=path)
    assert reloaded._in_degree.get("b") == 1
    assert reloaded._degree("b") == 1
    assert reloaded.central_nodes(2)[0]["degree"] == 1


# ══ R3-8 (MED): refine_chain trusted LLM-supplied shapes ══════════════

@pytest.mark.parametrize("bad", [
    {"objective": "o", "plan": {"step": "not a list"}},
    {"objective": "o", "risks": {"r": 1}},
    {"objective": "o", "constraints": "a bare string"},
    {"objective": "o", "plan": 42},
    {"objective": "o", "confidence": "very high"},
    {"objective": "o", "confidence": None},
])
def test_refine_chain_survives_malformed_llm_output(tmp_path, bad):
    from aegis.layers.world_model import WorldModel
    wm = WorldModel(store_path=tmp_path / "model.json")
    chain = wm.refine_chain(bad)          # must not raise
    assert chain is not None
    assert isinstance(chain["plan"], list)
    assert isinstance(chain["risks"], list)
    assert isinstance(chain["constraints"], list)
    assert 0.0 <= chain["confidence"] <= 1.0


def test_refine_chain_string_constraints_are_not_split_into_chars(tmp_path):
    from aegis.layers.world_model import WorldModel
    wm = WorldModel(store_path=tmp_path / "model.json")
    chain = wm.refine_chain({"objective": "o", "constraints": "budget"})
    assert chain["constraints"] == [], "a bare string was iterated character-wise"


def test_refine_chain_well_formed_output_preserved(tmp_path):
    from aegis.layers.world_model import WorldModel
    wm = WorldModel(store_path=tmp_path / "model.json")
    chain = wm.refine_chain({
        "objective": "open a company", "constraints": ["legal"],
        "risks": ["tax"], "plan": ["register"], "confidence": 0.8,
    })
    assert chain["constraints"] == ["legal"]
    assert chain["plan"][0]["action"] == "register"
    assert chain["confidence"] == 0.8


# ══ R3-9 (LOW): restore endpoint reported a rollback that never happened ═

def test_restore_endpoint_reports_that_nothing_was_applied(monkeypatch):
    from fastapi.testclient import TestClient
    import aegis.config as cfg
    from aegis.api import server

    monkeypatch.setattr(cfg, "API_TOKEN", "")

    class _FakeBackup:
        def restore_latest(self):
            return {"substrate": {"tick": 42}}

    class _FakeSubstrate:
        state_backup = _FakeBackup()

    monkeypatch.setattr(server, "substrate", _FakeSubstrate())
    body = TestClient(server.app).post("/api/state-backup/restore").json()
    assert body["applied"] is False, "endpoint claimed a restore it did not perform"
    assert body["status"] == "loaded"
    assert body["tick"] == 42


# ══ R3-10 (LOW): API answered 500 before the runtime existed ══════════

def test_api_returns_503_when_runtime_not_started(monkeypatch):
    from fastapi.testclient import TestClient
    import aegis.config as cfg
    from aegis.api import server

    monkeypatch.setattr(cfg, "API_TOKEN", "")
    monkeypatch.setattr(server, "substrate", None)
    resp = TestClient(server.app).get("/api/status")
    assert resp.status_code == 503
    assert "not started" in resp.json()["detail"]


def test_auth_still_precedes_the_runtime_guard(monkeypatch):
    """A 503 must never leak runtime state to an unauthenticated caller."""
    from fastapi.testclient import TestClient
    import aegis.config as cfg
    from aegis.api import server

    monkeypatch.setattr(cfg, "API_TOKEN", "s3cret")
    monkeypatch.setattr(server, "substrate", None)
    client = TestClient(server.app)
    for path in ("/api/code-modifier/read/config.py", "/api/code-modifier/sources"):
        assert client.get(path).status_code == 401, path
    # With a valid token the runtime guard takes over.
    resp = client.get("/api/code-modifier/sources", headers={"x-api-token": "s3cret"})
    assert resp.status_code == 503
