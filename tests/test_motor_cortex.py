"""Tests for the MotorCortex.

pyttsx3 (TTS) is always faked so no real speech engine is touched.
"""
import sys
import types
import time
import pytest
from aegis.layers.motor_cortex import MotorCortex


class _FakeEngine:
    def __init__(self):
        self.props = {}
        self.said = []
        self.ran = 0

    def setProperty(self, key, value):
        self.props[key] = value

    def say(self, text):
        self.said.append(text)

    def runAndWait(self):
        self.ran += 1


def _install_fake_pyttsx3(monkeypatch, engine=None, raise_on_init=False):
    mod = types.ModuleType("pyttsx3")
    eng = engine or _FakeEngine()

    def init(*args, **kwargs):
        if raise_on_init:
            raise RuntimeError("no audio device")
        return eng

    mod.init = init
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)
    return eng


def test_init_voice_success(monkeypatch):
    eng = _install_fake_pyttsx3(monkeypatch)
    m = MotorCortex()
    assert m.voice_enabled is True
    assert m.voice_engine is eng
    assert eng.props.get("rate") == 150


def test_init_voice_failure(monkeypatch):
    _install_fake_pyttsx3(monkeypatch, raise_on_init=True)
    m = MotorCortex()
    assert m.voice_enabled is False
    assert m.voice_engine is None


def test_execute_speak_runs_async(monkeypatch):
    eng = _install_fake_pyttsx3(monkeypatch)
    m = MotorCortex()
    result = m.execute("speak", {"text": "hello world"})
    assert result["success"] is True
    assert "Speaking (async)" in result["output"]
    # wait for the daemon speech thread to run
    deadline = time.time() + 2.0
    while eng.ran == 0 and time.time() < deadline:
        time.sleep(0.01)
    assert eng.said == ["hello world"]
    assert eng.ran == 1


def test_execute_speak_swallows_engine_error(monkeypatch):
    class _BoomEngine(_FakeEngine):
        def runAndWait(self):
            raise RuntimeError("engine exploded")

    eng = _install_fake_pyttsx3(monkeypatch, engine=_BoomEngine())
    m = MotorCortex()
    result = m.execute("speak", {"text": "boom"})
    assert "Speaking (async)" in result["output"]
    # thread swallows the exception; wait for it to have attempted say()
    deadline = time.time() + 2.0
    while not eng.said and time.time() < deadline:
        time.sleep(0.01)
    assert eng.said == ["boom"]
    # no exception propagates to the caller
    assert result["success"] is True


def test_execute_speak_ignored_when_voice_disabled(monkeypatch):
    _install_fake_pyttsx3(monkeypatch, raise_on_init=True)
    m = MotorCortex()  # voice disabled
    result = m.execute("speak", {"text": "hi"})
    # falls through to default dispatch branch
    assert "Action dispatched: speak" in result["output"]


def test_execute_log():
    m = MotorCortex()
    m.voice_enabled = False
    result = m.execute("log", {"message": "test message"})
    assert result["output"] == "Logged: test message"


def test_execute_log_truncates():
    m = MotorCortex()
    m.voice_enabled = False
    long_msg = "x" * 200
    result = m.execute("log", {"message": long_msg})
    assert result["output"] == "Logged: " + "x" * 100


def test_execute_internal_computation():
    m = MotorCortex()
    m.voice_enabled = False
    result = m.execute("internal_computation")
    assert result["output"] == "Internal computation cycle completed"


def test_execute_alert():
    m = MotorCortex()
    m.voice_enabled = False
    result = m.execute("alert", {"level": "warning", "message": "danger"})
    assert result["output"] == "[WARNING] danger"


def test_execute_alert_defaults():
    m = MotorCortex()
    m.voice_enabled = False
    result = m.execute("alert")
    assert result["output"] == "[INFO] "


def test_execute_recharge():
    m = MotorCortex()
    m.voice_enabled = False
    result = m.execute("recharge", {"amount": 0.3})
    assert "Recharge requested: +0.3" in result["output"]


def test_execute_unknown_action():
    m = MotorCortex()
    m.voice_enabled = False
    result = m.execute("wiggle")
    assert result["output"] == "Action dispatched: wiggle"


def test_execute_with_archetype_and_goal():
    m = MotorCortex()
    m.voice_enabled = False
    result = m.execute("log", {"message": "m"}, archetype="Explorer", goal="explore")
    assert result["archetype"] == "Explorer"
    assert result["goal"] == "explore"


def test_execute_increments_counter_and_log():
    m = MotorCortex()
    m.voice_enabled = False
    m.execute("log", {"message": "a"})
    m.execute("log", {"message": "b"})
    assert m.actions_executed == 2
    assert len(m.action_log) == 2


def test_action_log_capped():
    m = MotorCortex()
    m.voice_enabled = False
    for i in range(120):
        m.execute("log", {"message": str(i)})
    assert len(m.action_log) <= 100


def test_status_shape():
    m = MotorCortex()
    m.voice_enabled = False
    m.execute("log", {"message": "hello"})
    s = m.status()
    assert s["actions_executed"] == 1
    assert "voice_enabled" in s
    assert len(s["recent_actions"]) == 1
    assert s["recent_actions"][0]["type"] == "log"


def test_execute_payload_none_default():
    m = MotorCortex()
    m.voice_enabled = False
    result = m.execute("log")  # no payload
    assert result["output"] == "Logged: "
