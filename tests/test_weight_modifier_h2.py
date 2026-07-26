"""Regression test for audit H2: model loading must not block the event loop.

train() must offload the heavy load_model() to an executor thread, not call it
inline in the coroutine.
"""
import asyncio
import threading

import aegis.layers.weight_modifier as wm_mod
from aegis.layers.weight_modifier import WeightModifier


def test_load_model_runs_off_the_event_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(wm_mod, "WEIGHT_CHECKPOINTS_DIR", tmp_path)
    wm = WeightModifier()
    wm._stats_path = tmp_path / "training_stats.json"

    main_thread = threading.current_thread()
    seen = {}

    def fake_load():
        seen["thread"] = threading.current_thread()
        # Return failure so train() exits before the (mocked-out) _train_sync.
        return {"success": False, "error": "stub"}

    wm.model_loaded = False
    wm.load_model = fake_load
    wm.can_train = lambda: (True, "ok")

    result = asyncio.run(wm.train(tmp_path, ethics_approved=True))

    assert result["success"] is False
    assert seen["thread"] is not main_thread  # ran in an executor thread
    # training flag reset in finally.
    assert wm.training_in_progress is False


def test_train_requires_ethics_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(wm_mod, "WEIGHT_CHECKPOINTS_DIR", tmp_path)
    wm = WeightModifier()
    wm._stats_path = tmp_path / "training_stats.json"
    result = asyncio.run(wm.train(tmp_path, ethics_approved=False))
    assert result["success"] is False
    assert "approval" in result["error"].lower()
