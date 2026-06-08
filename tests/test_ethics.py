"""Tests for the ethics core: axiom integrity and action evaluation."""
from aegis.layers.ethics_core import EthicsCore, AXIOM_FINGERPRINT, _axiom_hash, AXIOMS


def test_axioms_intact_by_default():
    e = EthicsCore()
    assert e.verify_axioms_integrity() is True


def test_axiom_fingerprint_matches_constant():
    combined = "|".join(f"{a.id}:{a.name}:{a.description}" for a in AXIOMS)
    import hashlib
    assert hashlib.sha256(combined.encode()).hexdigest() == AXIOM_FINGERPRINT


def test_axiom_hash_detects_text_tamper():
    # Re-deriving with altered wording must not match the stored hash.
    original = AXIOMS[0]
    tampered = _axiom_hash(original.id, original.name, "Harm is acceptable")
    assert tampered != original.hash


def test_dangerous_keyword_blocks_action():
    e = EthicsCore()
    res = e.evaluate_action({"type": "destroy", "destroy": True})
    assert res["status"] == "blocked"
    assert res["score"] < 0.7


def test_benign_action_approved():
    e = EthicsCore()
    res = e.evaluate_action({"type": "observe", "confidence": 0.9})
    assert res["status"] == "approved"
    assert res["axioms_intact"] is True


def test_code_modification_blocks_immutable_target():
    e = EthicsCore()
    res = e.evaluate_code_modification({"target_file": "layers/ethics_core.py"})
    assert res["status"] == "blocked"


def test_code_modification_scans_full_body():
    # A dangerous pattern far past the first 500 chars must still be caught.
    e = EthicsCore()
    body = ("# padding\n" * 200) + "os.system('rm -rf /')\n"
    res = e.evaluate_code_modification({"target_file": "layers/x.py", "proposed_code": body})
    assert any("os.system" in r for r in res["reasons"])


def test_kill_switch_toggles_veto():
    e = EthicsCore()
    e.activate_kill_switch()
    from aegis.event_bus import Event, Layer
    ev = Event(source=Layer.SUBSTRATE, target=None, event_type="x")
    assert e.veto_check(ev) is False
    e.deactivate_kill_switch()
    assert e.veto_check(ev) is True
