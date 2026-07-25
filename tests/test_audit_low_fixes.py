"""Tests for the low-severity audit fixes: L1, L2, L4, L9, L10."""
import asyncio

import pytest


# ── L4: bad env var must not crash config parsing ─────────────────────

def test_env_int_falls_back_on_bad_value(monkeypatch):
    import aegis.config as cfg
    monkeypatch.setenv("AEGIS_TEST_INT", "not-a-number")
    assert cfg._env_int("AEGIS_TEST_INT", "42") == 42


def test_env_float_falls_back_on_bad_value(monkeypatch):
    import aegis.config as cfg
    monkeypatch.setenv("AEGIS_TEST_FLOAT", "xyz")
    assert cfg._env_float("AEGIS_TEST_FLOAT", "1.5") == 1.5


def test_env_int_uses_valid_value(monkeypatch):
    import aegis.config as cfg
    monkeypatch.setenv("AEGIS_TEST_INT2", "7")
    assert cfg._env_int("AEGIS_TEST_INT2", "0") == 7


# ── L2: a crashing veto function must not propagate out of publish ─────

def test_veto_exception_blocks_and_does_not_raise():
    from aegis.event_bus import EventBus, Event, Layer

    bus = EventBus()

    def _boom(event):
        raise RuntimeError("ethics bug")

    bus.set_veto(_boom)

    async def run():
        ev = Event(source=Layer.SUBSTRATE, target=None, event_type="t", payload={})
        return await bus.publish(ev)

    # Fail-closed: the event is blocked, and no exception escapes.
    result = asyncio.run(run())
    assert result is False
    assert bus.stats()["blocked_events"] == 1


# ── L1: version bump tolerates a non-"x.y.z" version ──────────────────

def test_bump_patch_tolerates_bad_version():
    from aegis.layers.self_modification import SelfModification
    sm = SelfModification()
    # Must not raise on malformed versions restored from a foreign checkpoint.
    assert sm._bump_patch("1.0").count(".") == 2
    assert sm._bump_patch("weird").count(".") == 2
    assert sm._bump_patch("2.3.9") == "2.3.10"


# ── L10: constant-time token check ────────────────────────────────────

def test_token_ok_constant_time(monkeypatch):
    import aegis.config as cfg
    from aegis.api import server

    monkeypatch.setattr(cfg, "API_TOKEN", "s3cret")
    assert server._token_ok("s3cret") is True
    assert server._token_ok("wrong") is False
    assert server._token_ok(None) is False

    monkeypatch.setattr(cfg, "API_TOKEN", "")
    assert server._token_ok(None) is True  # no token configured -> open


# ── L9: source-read GET requires the token when configured ────────────

def test_source_read_requires_token(monkeypatch):
    import aegis.config as cfg
    from fastapi.testclient import TestClient
    from aegis.api import server

    monkeypatch.setattr(cfg, "API_TOKEN", "s3cret")
    client = TestClient(server.app)
    # No token -> 401, without ever touching substrate.
    resp = client.get("/api/code-modifier/read/config.py")
    assert resp.status_code == 401
    # Wrong token -> still 401.
    resp2 = client.get("/api/code-modifier/read/config.py",
                       headers={"x-api-token": "nope"})
    assert resp2.status_code == 401


def test_source_read_open_when_no_token(monkeypatch):
    # With no token configured the endpoint is reachable (auth passes); it may
    # then error on the missing substrate, but must NOT be a 401.
    import aegis.config as cfg
    from fastapi.testclient import TestClient
    from aegis.api import server

    monkeypatch.setattr(cfg, "API_TOKEN", "")
    client = TestClient(server.app, raise_server_exceptions=False)
    resp = client.get("/api/code-modifier/read/config.py")
    assert resp.status_code != 401


# ── L3: event history must not hold live references to the payload ────

def test_history_payload_is_snapshotted():
    from aegis.event_bus import EventBus, Event, Layer

    bus = EventBus()
    payload = {"k": "original"}

    async def run():
        await bus.publish(Event(source=Layer.SUBSTRATE, target=None,
                                event_type="t", payload=payload))

    asyncio.run(run())
    # Mutating the original payload after publish must NOT rewrite history.
    payload["k"] = "tampered"
    assert bus.get_history(1)[0]["payload"]["k"] == "original"


def test_get_history_returns_copies():
    from aegis.event_bus import EventBus, Event, Layer

    bus = EventBus()

    async def run():
        await bus.publish(Event(source=Layer.SUBSTRATE, target=None,
                                event_type="t", payload={"k": 1}))

    asyncio.run(run())
    got = bus.get_history(1)
    got[0]["payload"]["k"] = 999  # mutate the returned copy
    # Internal state is unaffected.
    assert bus.get_history(1)[0]["payload"]["k"] == 1


# ── L5: ethics immutable-file check matches the basename, not a substring ──

def test_ethics_blocks_exact_immutable_files():
    from aegis.layers.ethics_core import EthicsCore
    e = EthicsCore()
    r = e.evaluate_code_modification({"target_file": "layers/ethics_core.py",
                                      "proposed_code": "x = 1\n"})
    assert r["status"] == "blocked"


def test_ethics_does_not_falsematch_similar_names():
    from aegis.layers.ethics_core import EthicsCore
    e = EthicsCore()
    # "ethics_core_helper.py" must NOT be treated as the immutable ethics_core.py
    r = e.evaluate_code_modification({"target_file": "layers/ethics_core_helper.py",
                                      "proposed_code": "x = 1\n"})
    assert r["status"] != "blocked" or "immutable" not in " ".join(r["reasons"])
    # "myconfig.py" must NOT trigger the config human-approval penalty.
    r2 = e.evaluate_code_modification({"target_file": "myconfig.py",
                                       "proposed_code": "x = 1\n"})
    assert not any("human approval" in reason.lower() for reason in r2["reasons"])


# ── L6: reflection-escape dunders are blocked (blocklist hardening) ────

def test_code_modifier_blocks_reflection_escape(tmp_path):
    from aegis.layers.code_modifier import CodeModifier
    base = tmp_path / "pkg"
    (base / "layers").mkdir(parents=True)
    (base / "layers" / "toy.py").write_text("x = 1\n", encoding="utf-8")
    cm = CodeModifier(base_dir=base, backups_dir=tmp_path / "b")
    code = "def f():\n    return ().__class__.__bases__[0].__subclasses__()\n"
    safe, warnings = cm.validate_safety(code, "layers/toy.py")
    assert not safe
    assert any("reflection" in w.lower() for w in warnings)


def test_code_modifier_allows_normal_dunders(tmp_path):
    # Legitimate whole-file dunders must still pass.
    from aegis.layers.code_modifier import CodeModifier
    base = tmp_path / "pkg"
    (base / "layers").mkdir(parents=True)
    (base / "layers" / "toy.py").write_text("x = 1\n", encoding="utf-8")
    cm = CodeModifier(base_dir=base, backups_dir=tmp_path / "b")
    code = ('from pathlib import Path\n'
            'class A:\n'
            '    def __init__(self):\n'
            '        self.p = Path(__file__)\n'
            'if __name__ == "__main__":\n'
            '    A()\n')
    safe, warnings = cm.validate_safety(code, "layers/toy.py")
    assert safe, warnings


def test_self_preservation_flags_globals_escape(tmp_path):
    from aegis.layers.self_preservation import SelfPreservation
    sp = SelfPreservation(base_dir=tmp_path)
    safe, rep = sp.is_modification_safe(
        "x.py", "def f():\n    return f.__globals__['os']\n")
    assert safe is False
    assert any("reflection" in c.lower() for c in rep["critical"])
