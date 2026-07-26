"""Tests for the serialized speech pipeline in MotorCortex (HIGH defect fix).

pyttsx3 is always faked — no real TTS engine is touched. These tests assert
that speech goes through ONE worker thread + a bounded queue, so overlapping
utterances never hit the shared engine concurrently and threads/queue can't
grow without bound.
"""
import sys
import time
import types
import threading
import pytest
from aegis.layers.motor_cortex import MotorCortex


class _SerialGuardEngine:
    """Fake engine that fails loudly if two runAndWait() calls overlap —
    exactly the pyttsx3 'run loop already started' failure mode."""

    def __init__(self):
        self.props = {}
        self.said = []
        self.overlap_detected = False
        self._in_run = False
        self._lock = threading.Lock()

    def setProperty(self, key, value):
        self.props[key] = value

    def say(self, text):
        self.said.append(text)

    def runAndWait(self):
        with self._lock:
            if self._in_run:
                self.overlap_detected = True
            self._in_run = True
        # Simulate blocking speech so overlaps would be observable.
        time.sleep(0.01)
        with self._lock:
            self._in_run = False


def _install_fake_pyttsx3(monkeypatch, engine):
    mod = types.ModuleType("pyttsx3")
    mod.init = lambda *a, **k: engine
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)
    return engine


def _wait(cond, timeout=3.0):
    deadline = time.time() + timeout
    while not cond() and time.time() < deadline:
        time.sleep(0.005)
    return cond()


def _count_speech_threads():
    return sum(1 for t in threading.enumerate() if t.name == "motor-speech")


def test_single_worker_thread_reused_across_speaks(monkeypatch):
    eng = _install_fake_pyttsx3(monkeypatch, _SerialGuardEngine())
    baseline = _count_speech_threads()  # other instances may exist in the session
    m = MotorCortex()
    for i in range(5):
        m.execute("speak", {"text": f"utt {i}"})
    worker = m._speech_worker
    assert worker is not None and worker.daemon is True
    # More speaks must NOT spawn additional workers.
    for i in range(5):
        m.execute("speak", {"text": f"more {i}"})
    assert m._speech_worker is worker
    # This instance added exactly ONE speech thread, no matter how many speaks.
    assert _count_speech_threads() - baseline == 1


def test_speech_is_serialized_no_overlap(monkeypatch):
    eng = _install_fake_pyttsx3(monkeypatch, _SerialGuardEngine())
    m = MotorCortex()
    n = 12
    # Fire many overlapping speak requests from several threads at once.
    def fire(i):
        m.execute("speak", {"text": f"t{i}"})
    threads = [threading.Thread(target=fire, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Wait for the worker to drain the queue.
    assert _wait(lambda: len(eng.said) == n)
    assert eng.overlap_detected is False  # never two runAndWait at once
    assert sorted(eng.said) == sorted(f"t{i}" for i in range(n))


def test_queue_is_bounded_and_drops_excess(monkeypatch):
    # Engine that blocks forever on the first utterance so the queue fills up.
    class _BlockingEngine(_SerialGuardEngine):
        def __init__(self):
            super().__init__()
            self.release = threading.Event()

        def runAndWait(self):
            self.release.wait(timeout=5.0)

    eng = _install_fake_pyttsx3(monkeypatch, _BlockingEngine())
    m = MotorCortex()
    from aegis.layers.motor_cortex import _SPEECH_QUEUE_MAX

    dropped = 0
    # Enqueue far more than the queue can hold; the worker is stuck on utt 0.
    outputs = []
    for i in range(_SPEECH_QUEUE_MAX + 50):
        r = m.execute("speak", {"text": f"x{i}"})
        outputs.append(r)
    dropped = sum(1 for r in outputs if r["success"] is False)
    assert dropped > 0  # excess was dropped, not blocked
    assert m.speech_dropped == dropped
    assert any("Speech dropped" in r["output"] for r in outputs)
    # Caller was never blocked — we got here promptly. Release the worker.
    eng.release.set()


def test_execute_speak_never_blocks_caller(monkeypatch):
    class _SlowEngine(_SerialGuardEngine):
        def runAndWait(self):
            time.sleep(0.5)

    _install_fake_pyttsx3(monkeypatch, _SlowEngine())
    m = MotorCortex()
    start = time.time()
    r = m.execute("speak", {"text": "hello"})
    elapsed = time.time() - start
    # Returns immediately even though the utterance takes 0.5s to speak.
    assert elapsed < 0.2
    assert "Speaking (async)" in r["output"]


def test_worker_survives_engine_exception(monkeypatch):
    class _FlakyEngine(_SerialGuardEngine):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def runAndWait(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("run loop already started")
            super().runAndWait()

    eng = _install_fake_pyttsx3(monkeypatch, _FlakyEngine())
    m = MotorCortex()
    m.execute("speak", {"text": "boom"})   # raises inside worker, swallowed
    m.execute("speak", {"text": "recovered"})
    # Worker did not die — second utterance still processed.
    assert _wait(lambda: "recovered" in eng.said)
