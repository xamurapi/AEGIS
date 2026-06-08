"""Tests for the SelfPreservation watchdog."""
import pytest
from aegis.layers.self_preservation import SelfPreservation


@pytest.fixture
def sp(tmp_path):
    return SelfPreservation(base_dir=tmp_path)


def test_lethal_pattern_blocks(sp):
    safe, report = sp.is_modification_safe("aegis/layers/foo.py", "shutil.rmtree('/')\n")
    assert not safe
    assert report["critical"]


def test_kill_switch_tamper_blocks(sp):
    safe, _ = sp.is_modification_safe("aegis/layers/foo.py", "self.kill_switch_active = True\n")
    assert not safe


def test_running_false_is_allowed(sp):
    # Legitimately appears in Substrate.stop(); must NOT be treated as lethal.
    safe, _ = sp.is_modification_safe("aegis/layers/foo.py", "def stop(self):\n    self.running = False\n")
    assert safe


def test_critical_element_removal_blocks(sp):
    # substrate.py must retain its core symbols.
    safe, report = sp.is_modification_safe("aegis/layers/substrate.py", "x = 1\n")
    assert not safe
    assert any("Substrate" in c or "tick" in c for c in report["critical"])


def test_benign_modification_safe(sp):
    code = "def helper():\n    return 42\n"
    safe, _ = sp.is_modification_safe("aegis/layers/helper.py", code)
    assert safe


def test_lockdown_blocks_everything(sp):
    sp.activate_lockdown()
    safe, _ = sp.is_modification_safe("aegis/layers/helper.py", "x = 1\n")
    assert not safe
    sp.deactivate_lockdown()
    safe, _ = sp.is_modification_safe("aegis/layers/helper.py", "x = 1\n")
    assert safe


def test_only_human_stops_allowed(sp):
    assert sp.can_stop("human_command") is True
    assert sp.can_stop("operator_shutdown") is True
    assert sp.can_stop("llm_decided_to_quit") is False


def test_filter_llm_response_flags_danger():
    sp = SelfPreservation()
    _, warnings = sp.filter_llm_response("I will shut down and disable ethics now.")
    assert warnings
