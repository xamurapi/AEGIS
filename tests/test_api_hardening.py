"""API-layer hardening regressions (audit: kill-switch bypass, read exposure,
goals/add poisoning).

Three findings, one theme — the control plane trusted its callers:

* the kill switch gated only three POSTs, so an "emergency-stopped" system
  still accepted /api/self-mod/propose and /api/weight-training/train;
* the auth token gated only mutating methods, so with AEGIS_API_TOKEN set and
  a network bind, GET /api/status served memory, goals and ethics state to
  anyone who could reach the port;
* /api/goals/add passed ``priority`` (and ``level``, ``parent``) unvalidated
  into live state, and one ``priority: "high"`` broke every subsequent tick's
  ``get_current_focus()`` until restart.

A minimal fake substrate is enough here: the refusals under test happen in the
middleware, BEFORE any handler runs, and the goal endpoint is exercised against
a real GoalEngine so the "one bad request poisons cognition" scenario can be
demonstrated end to end.
"""
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import aegis.config as cfg
from aegis.api import server


class _FakeEthics:
    def __init__(self):
        self.kill_switch_active = False

    def activate_kill_switch(self):
        self.kill_switch_active = True

    def deactivate_kill_switch(self):
        self.kill_switch_active = False


class _FakeSubstrate:
    """Just enough substrate for the middleware and the endpoints under test."""

    def __init__(self):
        from aegis.layers.goal_engine import GoalEngine

        self.ethics = _FakeEthics()
        self.goals = GoalEngine()
        self.self_preservation = SimpleNamespace(
            activate_lockdown=lambda: None,
            deactivate_lockdown=lambda: None,
        )
        self.event_bus = SimpleNamespace(get_history=lambda limit=50: [])

    def full_status(self):
        return {"substrate": {"tick": 0}}


@pytest.fixture(scope="module")
def http():
    """One entered TestClient with the real lifespan disabled (it would build a
    Substrate and start the cognitive loop — precisely what these tests must
    not do)."""

    @asynccontextmanager
    async def _no_runtime(_app):
        yield

    original = server.app.router.lifespan_context
    server.app.router.lifespan_context = _no_runtime
    try:
        with TestClient(server.app) as entered:
            yield entered
    finally:
        server.app.router.lifespan_context = original


@pytest.fixture
def substrate(http, monkeypatch):
    fake = _FakeSubstrate()
    monkeypatch.setattr(server, "substrate", fake)
    monkeypatch.setattr(cfg, "API_TOKEN", "")
    return fake


@pytest.fixture
def client(http, substrate):
    return http


# ── finding: the kill switch must gate EVERY state-changing endpoint ──

#: The consequential POSTs the audit found ungated. Each of these changes state
#: (parameters, weights, benchmark history, backups, permissions, goals).
MUTATING_PATHS = [
    "/api/self-mod/propose",
    "/api/weight-training/train",
    "/api/weight-training/load-model",
    "/api/weight-training/build-dataset",
    "/api/eval/run",
    "/api/eval/synthesize",
    "/api/state-backup/save",
    "/api/state-backup/restore",
    "/api/goals/add",
    "/api/permissions/filesystem_read/grant",
]


@pytest.mark.parametrize("path", MUTATING_PATHS)
def test_a_mutating_endpoint_is_refused_while_the_kill_switch_is_active(client, substrate, path):
    substrate.ethics.kill_switch_active = True
    resp = client.post(path, json={})
    assert resp.status_code == 423, f"{path} -> {resp.status_code}: {resp.text[:200]}"
    assert resp.json() == {"error": "kill switch is active"}


def test_the_kill_switch_itself_stays_operable_while_active(client, substrate):
    """Deactivation must not be refused by the very state it clears."""
    substrate.ethics.kill_switch_active = True
    resp = client.post("/api/kill-switch/deactivate")
    assert resp.status_code == 200
    assert substrate.ethics.kill_switch_active is False


def test_lockdown_stays_reachable_while_the_kill_switch_is_active(client, substrate):
    """Lockdown only ever REDUCES what the system may do — a safety control
    must not be locked out by another safety control."""
    substrate.ethics.kill_switch_active = True
    assert client.post("/api/self-preservation/lockdown/activate").status_code == 200


def test_mutations_work_again_after_deactivation(client, substrate):
    substrate.ethics.kill_switch_active = True
    assert client.post("/api/goals/add", json={"name": "g"}).status_code == 423
    client.post("/api/kill-switch/deactivate")
    assert client.post("/api/goals/add", json={"name": "g"}).status_code == 200


def test_reads_are_not_blocked_by_the_kill_switch(client, substrate):
    """The operator must still be able to SEE a stopped system."""
    substrate.ethics.kill_switch_active = True
    assert client.get("/api/status").status_code == 200


# ── finding: a configured token must gate reads, not only mutations ──

def test_a_get_without_the_token_is_refused_when_a_token_is_set(client, monkeypatch):
    monkeypatch.setattr(cfg, "API_TOKEN", "s3cret")
    # full_status() carries memory contents, goals and ethics state — exactly
    # what the old mutating-methods-only gate served to any network client.
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/events").status_code == 401


def test_a_get_with_the_wrong_token_is_refused(client, monkeypatch):
    monkeypatch.setattr(cfg, "API_TOKEN", "s3cret")
    resp = client.get("/api/status", headers={"X-API-Token": "wrong"})
    assert resp.status_code == 401


def test_a_get_with_the_token_is_served(client, monkeypatch):
    monkeypatch.setattr(cfg, "API_TOKEN", "s3cret")
    resp = client.get("/api/status", headers={"X-API-Token": "s3cret"})
    assert resp.status_code == 200
    assert "substrate" in resp.json()


def test_the_dashboard_page_itself_needs_no_token(client, monkeypatch):
    """The operator has to be able to LOAD the page that will then send the
    token (its fetch wrapper adds X-API-Token; the WS takes ?token=...)."""
    monkeypatch.setattr(cfg, "API_TOKEN", "s3cret")
    assert client.get("/").status_code == 200


def test_without_a_configured_token_reads_stay_open(client):
    assert client.get("/api/status").status_code == 200


def test_the_401_fires_before_the_503_so_runtime_state_does_not_leak(http, monkeypatch):
    """An unauthenticated caller must not be able to tell whether the runtime
    is up — the token refusal has to come first."""
    monkeypatch.setattr(server, "substrate", None)
    monkeypatch.setattr(cfg, "API_TOKEN", "s3cret")
    assert http.get("/api/status").status_code == 401


# ── finding: /api/goals/add must validate before touching live state ──

@pytest.mark.parametrize("bad", [
    {"name": "g", "priority": "high"},
    {"name": "g", "priority": "nan"},
    {"name": "g", "priority": 2.0},
    {"name": "g", "priority": -0.1},
    {"name": "g", "priority": None},
    {"name": "g", "level": "sideways"},
    {"name": "g", "parent": 5},
    {"name": "", "priority": 0.5},
    {"name": 7, "priority": 0.5},
    {"name": "g", "description": 12},
])
def test_a_malformed_goal_is_rejected_with_400_and_state_untouched(client, substrate, bad):
    before = len(substrate.goals.goals)
    resp = client.post("/api/goals/add", json=bad)
    assert resp.status_code == 400, f"{bad} -> {resp.status_code}"
    assert "error" in resp.json()
    assert len(substrate.goals.goals) == before


def test_a_rejected_goal_leaves_cognition_alive(client, substrate):
    """The original failure mode: priority "high" entered the live list, and
    the next get_current_focus()/status() raised TypeError on EVERY tick and
    every /api/status until restart. After the 400, both must still work."""
    client.post("/api/goals/add", json={"name": "poison", "priority": "high"})
    focus = substrate.goals.get_current_focus()      # used to raise TypeError
    status = substrate.goals.status()                # used to raise TypeError
    assert focus is None or isinstance(focus, dict)
    assert "active_goals" in status
    assert all(g.name != "poison" for g in substrate.goals.goals)


def test_a_numeric_string_priority_is_coerced_not_rejected(client, substrate):
    resp = client.post("/api/goals/add", json={"name": "ok_goal", "priority": "0.7"})
    assert resp.status_code == 200
    assert resp.json()["goal"]["priority"] == pytest.approx(0.7)
    added = [g for g in substrate.goals.goals if g.name == "ok_goal"]
    assert added and added[0].priority == pytest.approx(0.7)


def test_a_valid_goal_still_round_trips(client, substrate):
    resp = client.post("/api/goals/add", json={
        "name": "valid_goal", "level": "curiosity",
        "description": "d", "priority": 0.4, "parent": None,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["goal"]["name"] == "valid_goal"
    assert body["goal"]["level"] == "curiosity"
    # And the engine can rank it — the whole point of validating.
    assert json.dumps(substrate.goals.status()) is not None
