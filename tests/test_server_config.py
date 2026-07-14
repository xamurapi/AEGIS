"""Tests for server wiring and security-relevant config defaults."""
import aegis.config as cfg


def test_default_bind_is_loopback():
    # The control plane must not be network-exposed by default.
    assert cfg.API_HOST == "127.0.0.1"


def test_substrate_not_created_at_import():
    import aegis.api.server as server
    # Created in lifespan, not at import — keeps imports cheap and testable.
    assert server.substrate is None


def test_app_has_auth_middleware():
    import aegis.api.server as server
    # The mutating-method auth guard must be installed. The @app.middleware("http")
    # decorator registers a BaseHTTPMiddleware whose dispatch is auth_middleware.
    dispatches = [
        getattr(m, "kwargs", {}).get("dispatch") for m in server.app.user_middleware
    ]
    names = [getattr(d, "__name__", "") for d in dispatches if d is not None]
    assert "auth_middleware" in names, f"auth_middleware not installed; found {names}"


def test_semantic_summary_handles_plain_and_nested():
    from aegis.api.server import _semantic_summary
    assert _semantic_summary({"relations": {"definition": "d"}}) == "d"
    assert _semantic_summary({"summary": "s"}) == "s"
    assert _semantic_summary({}) == ""
