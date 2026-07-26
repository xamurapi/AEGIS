"""End-to-end smoke coverage of the control-plane API.

`aegis/api/server.py` was excluded from the coverage gate, and that is exactly
where the round-3 audit found an unauthenticated state leak. Most endpoints are
thin delegations (`return substrate.<layer>.status()`), so a renamed attribute
turns into a 500 in production with nothing failing in CI.

The GET sweep below is derived from the app's own route table, so an endpoint
added later is covered automatically.
"""
import json

import pytest
from fastapi.testclient import TestClient

import aegis.config as cfg
from aegis.api import server
from aegis.layers.substrate import Substrate


@pytest.fixture(scope="module")
def substrate(tmp_path_factory):
    """One offline Substrate for the whole module (construction is expensive).

    Nothing here starts the run loop, so no background tasks are created.
    """
    sub = Substrate()
    sub.llm.enabled = False
    # Redirect the eval-layer stores into a private directory. A Substrate
    # otherwise shares data/eval/skills.json with the eval-layer tests, which
    # add and remove skills — /api/eval then serializes a library that another
    # test is rewriting underneath it (see the isolation rule in docs/QA.md).
    from aegis.eval.skill_library import SkillLibrary

    store = tmp_path_factory.mktemp("api_eval")
    sub.skill_library = SkillLibrary(store_path=store / "skills.json")
    sub.solver.library = sub.skill_library      # MultiAgentSolver holds it as `library`
    assert sub.solver.library is sub.skill_library
    # Health readings are real psutil values; pin them so a loaded machine
    # cannot change what the status endpoints report.
    sub.health.check = lambda: {"status": "healthy", "warnings": [],
                                "critical": [], "metrics": {}}
    return sub


@pytest.fixture
def client(substrate, monkeypatch):
    monkeypatch.setattr(cfg, "API_TOKEN", "")
    monkeypatch.setattr(server, "substrate", substrate)
    monkeypatch.setattr(server, "connected_ws", [])
    # The substrate is shared across the module for speed, so reset the bits
    # individual tests mutate — otherwise assertions depend on test order.
    substrate.llm.enabled = False
    return TestClient(server.app)


def _plain_get_paths():
    """Every GET route that needs no path parameter."""
    paths = []
    for route in server.app.routes:
        methods = getattr(route, "methods", set()) or set()
        path = getattr(route, "path", "")
        if "GET" in methods and "{" not in path and path.startswith("/api"):
            paths.append(path)
    return sorted(paths)


@pytest.mark.parametrize("path", _plain_get_paths())
def test_every_get_endpoint_answers_and_serializes(client, path):
    resp = client.get(path)
    assert resp.status_code == 200, f"{path} -> {resp.status_code}: {resp.text[:200]}"
    # Every response must survive JSON serialization (the WS broadcast and the
    # dashboard both depend on it).
    if resp.headers.get("content-type", "").startswith("application/json"):
        json.dumps(resp.json())


def test_dashboard_page_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_eval_history_is_downloadable_csv(client):
    resp = client.get("/api/eval/history.csv")
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")


# ── control actions ───────────────────────────────────────────────────

def test_kill_switch_round_trip(client, substrate):
    assert client.post("/api/kill-switch/activate").json()["active"] is True
    assert substrate.ethics.kill_switch_active is True
    assert client.post("/api/kill-switch/deactivate").json()["active"] is False
    assert substrate.ethics.kill_switch_active is False


def test_kill_switch_rejects_an_unknown_action(client):
    assert "error" in client.post("/api/kill-switch/sideways").json()


@pytest.mark.parametrize("requested,expected", [
    (0.1, 0.5),       # clamped up to the floor
    (3.0, 3.0),
    (99.0, 30.0),     # clamped down to the ceiling
])
def test_tick_interval_is_clamped(client, requested, expected):
    body = client.post(f"/api/tick-interval/{requested}").json()
    assert body["tick_interval"] == pytest.approx(expected)
    assert cfg.TICK_INTERVAL == pytest.approx(expected)


def test_permissions_can_be_granted_and_revoked(client, substrate):
    granted = client.post("/api/permissions/filesystem_read/grant").json()
    assert granted["permissions"]["filesystem_read"] is True
    revoked = client.post("/api/permissions/filesystem_read/revoke").json()
    assert revoked["permissions"]["filesystem_read"] is False


def test_ethics_evaluation_returns_a_scored_verdict(client):
    body = client.post("/api/ethics/evaluate", json={
        "type": "read", "modifies_self": False, "confidence": 0.9,
    }).json()
    assert "status" in body and "score" in body


def test_goal_can_be_added(client, substrate):
    before = len(substrate.goals.goals)
    body = client.post("/api/goals/add", json={
        "name": "test_goal", "level": "tactic", "description": "d", "priority": 0.4,
    }).json()
    assert body["goal"]["name"] == "test_goal"
    assert len(substrate.goals.goals) == before + 1


def test_lockdown_round_trip(client):
    assert client.post("/api/self-preservation/lockdown/activate").json()["active"] is True
    assert client.post("/api/self-preservation/lockdown/deactivate").json()["active"] is False
    assert "error" in client.post("/api/self-preservation/lockdown/maybe").json()


def test_integrity_check_runs(client):
    body = client.post("/api/self-preservation/integrity").json()
    assert isinstance(body, dict)


def test_emotion_analysis_requires_text(client):
    assert "error" in client.post("/api/emotion-nlp/analyze", json={"text": ""}).json()
    body = client.post("/api/emotion-nlp/analyze", json={"text": "I am glad"}).json()
    assert "error" not in body


def test_agent_can_be_created(client):
    resp = client.post("/api/agents/create", json={
        "name": "spider_test", "source_type": "custom",
        "task": "collect", "topic": "physics",
    })
    assert resp.status_code == 200, resp.text[:300]
    body = resp.json()
    assert body["agent"]["name"] == "spider_test"
    assert body["agent"]["topic"] == "physics"
    assert "collect" in body["prompt"]


def test_state_backup_save_and_list(client):
    saved = client.post("/api/state-backup/save").json()
    assert isinstance(saved, dict)
    listed = client.get("/api/state-backup/list").json()
    assert isinstance(listed, list)


def test_llm_provider_switch_validates_the_mode(client):
    assert client.post("/api/llm/provider/deepseek").json()["provider_mode"] == "deepseek"
    assert "error" in client.post("/api/llm/provider/telepathy").json()
    assert client.post("/api/weight-training/provider/local").json()["provider_mode"] == "local"
    assert "error" in client.post("/api/weight-training/provider/telepathy").json()


def test_llm_key_is_required(client):
    assert "error" in client.post("/api/llm/set-key", json={"provider": "deepseek"}).json()


def test_chat_requires_a_configured_llm(client):
    body = client.post("/api/chat", json={"message": "hello"}).json()
    assert body["error"] == "No LLM configured"


def test_synthesize_requires_a_configured_llm(client):
    assert "error" in client.post("/api/eval/synthesize").json()


# ── autonomous replies (the no-LLM fallback brain) ────────────────────

@pytest.mark.parametrize("prompt,needle", [
    ("Кто ты?", "AEGIS"),
    ("Как дела?", "тик"),
    ("Какие твои цели?", "цел"),
    ("Что ты выучил?", "знани"),
    ("Что ты знаешь о квантах?", "тем"),
])
def test_autonomous_reply_covers_each_intent(client, prompt, needle):
    body = client.post("/api/llm/think", json={"prompt": prompt}).json()
    assert body["success"] is True
    assert body["provider"] == "autonomous"
    assert needle.lower() in body["response"].lower()


def test_autonomous_reply_falls_back_to_a_capability_list(client):
    body = client.post("/api/llm/think", json={"prompt": "zzz"}).json()
    assert body["provider"] == "autonomous"
    assert "AEGIS" in body["response"]


def test_autonomous_reply_finds_matching_knowledge(client, substrate):
    substrate.memory.add_semantic("quantum_entanglement", {
        "summary": "two particles share a state", "type": "physics",
    })
    body = client.post("/api/llm/think",
                       json={"prompt": "Что ты знаешь о quantum_entanglement?"}).json()
    assert "quantum_entanglement" in body["response"]


def test_semantic_summary_reads_the_nested_payload():
    """add_semantic nests the payload under `relations` — a top-level lookup
    silently returned nothing before this was fixed."""
    assert server._semantic_summary({"relations": {"summary": "nested"}}) == "nested"
    assert server._semantic_summary({"summary": "flat"}) == "flat"
    assert server._semantic_summary({"relations": {"definition": "def"}}) == "def"
    assert server._semantic_summary("not a dict") == ""


# ── code-modifier surface (token-gated) ───────────────────────────────

def test_code_modifier_endpoints_without_a_token(client):
    assert client.get("/api/code-modifier").status_code == 200
    sources = client.get("/api/code-modifier/sources").json()
    assert any(s["path"] == "config.py" for s in sources)
    analysis = client.post("/api/code-modifier/analyze", json={"file_path": "config.py"}).json()
    assert "error" not in analysis
    read = client.get("/api/code-modifier/read/config.py").json()
    assert read["lines"] > 0


def test_code_modifier_read_rejects_a_traversal_attempt(client):
    # URL-encoded so the client does not normalise the ".." away before it
    # ever reaches the handler.
    body = client.get("/api/code-modifier/read/%2e%2e%2f%2e%2e%2fsecrets.py").json()
    assert "error" in body


def test_code_modifier_analyze_reports_errors_as_json(client):
    body = client.post("/api/code-modifier/analyze",
                       json={"file_path": "does_not_exist.py"}).json()
    assert "error" in body


# ── websocket ─────────────────────────────────────────────────────────

def test_websocket_streams_status_and_handles_commands(client, substrate):
    with client.websocket_connect("/ws") as ws:
        first = json.loads(ws.receive_text())
        assert "substrate" in first
        ws.send_text(json.dumps({"action": "get_status"}))
        assert "substrate" in json.loads(ws.receive_text())
        ws.send_text(json.dumps({"action": "kill_switch_on"}))
        ws.send_text(json.dumps({"action": "get_status"}))
        assert json.loads(ws.receive_text())["ethics"]["kill_switch"] is True
        ws.send_text(json.dumps({"action": "kill_switch_off"}))
        ws.send_text(json.dumps({"action": "get_status"}))
        assert json.loads(ws.receive_text())["ethics"]["kill_switch"] is False


def test_websocket_ignores_malformed_frames(client):
    with client.websocket_connect("/ws") as ws:
        json.loads(ws.receive_text())
        ws.send_text("not json at all")
        ws.send_text(json.dumps({"action": "get_status"}))
        assert "substrate" in json.loads(ws.receive_text())


def test_websocket_rejects_a_cross_origin_handshake(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_text()


def test_broadcast_drops_a_dead_socket():
    """A socket that raises on send must be removed from the fan-out."""
    import asyncio

    class _DeadSocket:
        async def send_text(self, _):
            raise RuntimeError("closed")

    class _LiveSocket:
        def __init__(self):
            self.sent = []

        async def send_text(self, message):
            self.sent.append(message)

    dead, live = _DeadSocket(), _LiveSocket()
    server.connected_ws[:] = [dead, live]
    try:
        asyncio.run(server.broadcast({"tick": 1}))
        assert dead not in server.connected_ws
        assert live in server.connected_ws
        assert live.sent == [json.dumps({"tick": 1})]
    finally:
        server.connected_ws.clear()
