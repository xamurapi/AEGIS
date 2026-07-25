"""Security regression tests for the API layer.

Covers audit findings:
  M8 — config.network_exposure_warning flags 0.0.0.0 + empty token.
  H4 — WebSocket rejects cross-origin (CSWSH) handshakes.
"""
import aegis.config as cfg
from aegis.api.server import _ws_origin_allowed


# ── M8: network-exposure warning ──────────────────────────────────────

def test_exposure_warning_on_public_host_without_token():
    assert cfg.network_exposure_warning("0.0.0.0", "") is not None


def test_no_warning_on_loopback_without_token():
    assert cfg.network_exposure_warning("127.0.0.1", "") is None
    assert cfg.network_exposure_warning("localhost", "") is None


def test_no_warning_on_public_host_with_token():
    assert cfg.network_exposure_warning("0.0.0.0", "secret") is None


# ── H4: WebSocket origin gate ─────────────────────────────────────────

def test_ws_allows_missing_origin():
    # Non-browser clients (curl, native) send no Origin — allowed.
    assert _ws_origin_allowed(None) is True


def test_ws_allows_loopback_origins():
    assert _ws_origin_allowed("http://127.0.0.1:8888") is True
    assert _ws_origin_allowed("http://localhost:3000") is True


def test_ws_rejects_foreign_origin():
    assert _ws_origin_allowed("https://evil.example.com") is False
    assert _ws_origin_allowed("http://attacker.test") is False


def test_ws_allows_configured_cors_origin(monkeypatch):
    monkeypatch.setattr(cfg, "API_CORS_ORIGINS", ["https://dash.mycompany.com"])
    assert _ws_origin_allowed("https://dash.mycompany.com") is True


def test_ws_handshake_rejected_cross_origin():
    # End to end via the ASGI app: a cross-origin WS handshake is closed.
    import pytest
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    from aegis.api.server import app

    # Don't run the real substrate lifespan (it would start the tick loop); the
    # origin check happens before any substrate access, so drive the endpoint
    # directly through the test client without lifespan startup.
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "https://evil.example.com"}) as ws:
            ws.receive_text()
