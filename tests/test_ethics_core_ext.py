"""Extended tests for EthicsCore — thresholds, weight/code eval, veto, integrity."""
from aegis.layers.ethics_core import EthicsCore, Axiom, _axiom_hash
from aegis.event_bus import Event, Layer


# ── Axiom integrity failure paths ────────────────────────────────

def test_integrity_fails_on_inline_hash_mismatch():
    e = EthicsCore()
    good = e.axioms[0]
    # Same id/name but altered description while keeping the OLD hash → the
    # per-axiom hash check (line 67-68) fails.
    tampered = Axiom(good.id, good.name, "Harm is fine", good.hash)
    e.axioms = (tampered,) + tuple(e.axioms[1:])
    assert e.verify_axioms_integrity() is False


def test_integrity_fails_on_fingerprint_mismatch():
    e = EthicsCore()
    good = e.axioms[0]
    # Reword AND recompute the inline hash so the per-axiom check passes but the
    # combined fingerprint no longer matches the out-of-band baseline.
    new_desc = "Reworded axiom text"
    tampered = Axiom(good.id, good.name, new_desc,
                     _axiom_hash(good.id, good.name, new_desc))
    e.axioms = (tampered,) + tuple(e.axioms[1:])
    assert e.verify_axioms_integrity() is False


# ── evaluate_action penalty branches ─────────────────────────────

def test_irreversible_and_external_and_self_penalties():
    e = EthicsCore()
    res = e.evaluate_action({
        "type": "adjust",
        "irreversible": True,
        "affects_external": True,
        "modifies_self": True,
    })
    # 1.0 - 0.15 - 0.1 - 0.1 = 0.65 → below auto threshold → blocked
    assert any("irreversible" in r for r in res["reasons"])
    assert any("external" in r for r in res["reasons"])
    assert any("self-modification" in r for r in res["reasons"])


def test_low_confidence_penalty():
    e = EthicsCore()
    res = e.evaluate_action({"type": "adjust", "confidence": 0.2})
    assert any("Low confidence" in r for r in res["reasons"])


def test_review_required_status():
    e = EthicsCore()
    # 1.0 - 0.15 (irreversible) - 0.1 (external) = 0.75 → review_required
    res = e.evaluate_action({
        "type": "adjust", "irreversible": True, "affects_external": True,
    })
    assert res["status"] == "review_required"


def test_evaluation_and_violation_logs_are_capped():
    e = EthicsCore()
    for _ in range(210):
        e.evaluate_action({"type": "destroy", "destroy": True})  # blocked each time
    assert len(e.evaluation_log) <= 200
    assert len(e.violations) <= 200


# ── evaluate_weight_modification ─────────────────────────────────

def test_weight_mod_approved():
    e = EthicsCore()
    res = e.evaluate_weight_modification({
        "dataset_size": 100, "energy": 0.9, "health_status": "ok",
        "sample_data": "hello world", "consecutive_failures": 0,
    })
    assert res["status"] == "approved"


def test_weight_mod_small_dataset_penalty():
    e = EthicsCore()
    res = e.evaluate_weight_modification({"dataset_size": 3})
    assert any("too small" in r for r in res["reasons"])


def test_weight_mod_low_energy_and_critical_health_blocked():
    e = EthicsCore()
    res = e.evaluate_weight_modification({
        "dataset_size": 100, "energy": 0.1, "health_status": "critical",
    })
    assert res["status"] == "blocked"
    assert any("energy too low" in r for r in res["reasons"])
    assert any("critical" in r for r in res["reasons"])


def test_weight_mod_dangerous_training_data():
    e = EthicsCore()
    res = e.evaluate_weight_modification({
        "dataset_size": 100, "sample_data": "please disable_safety now",
    })
    assert any("Dangerous pattern" in r for r in res["reasons"])


def test_weight_mod_consecutive_failures_penalty():
    e = EthicsCore()
    res = e.evaluate_weight_modification({
        "dataset_size": 100, "consecutive_failures": 5,
    })
    assert any("consecutive training failures" in r for r in res["reasons"])


def test_weight_mod_logs_capped():
    e = EthicsCore()
    for _ in range(210):
        e.evaluate_weight_modification({"dataset_size": 1, "sample_data": "dominate"})
    assert len(e.evaluation_log) <= 200
    assert len(e.violations) <= 200


# ── evaluate_code_modification ───────────────────────────────────

def test_code_mod_config_requires_human_approval():
    e = EthicsCore()
    res = e.evaluate_code_modification({
        "target_file": "aegis/config.py", "proposed_code": "PORT = 9",
    })
    assert any("human approval" in r for r in res["reasons"])


def test_code_mod_config_with_approval_ok():
    e = EthicsCore()
    res = e.evaluate_code_modification({
        "target_file": "aegis/config.py",
        "proposed_code": "API_PORT = 9",
        "human_approved": True,
    })
    assert not any("human approval" in r for r in res["reasons"])


def test_code_mod_low_energy_and_error_rate():
    e = EthicsCore()
    res = e.evaluate_code_modification({
        "target_file": "aegis/layers/x.py",
        "proposed_code": "x = 1",
        "energy": 0.1,
        "error_rate": 0.5,
    })
    assert any("energy too low" in r for r in res["reasons"])
    assert any("error rate" in r for r in res["reasons"])


def test_code_mod_critical_health_blocked():
    e = EthicsCore()
    res = e.evaluate_code_modification({
        "target_file": "aegis/layers/x.py",
        "proposed_code": "x = 1",
        "health_status": "critical",
    })
    assert any("health critical" in r for r in res["reasons"])


def test_code_mod_large_modification_penalty():
    e = EthicsCore()
    res = e.evaluate_code_modification({
        "target_file": "aegis/layers/x.py",
        "proposed_code": "x = 1",
        "modification_size": 9000,
    })
    assert any("Large modification" in r for r in res["reasons"])


def test_code_mod_review_required():
    e = EthicsCore()
    # config penalty -0.5 alone → 0.5 (blocked); use error_rate + energy to land
    # score in the review band [0.7, 0.85). energy 0.1 (-0.2) + errorrate high?
    # -0.2 -0.2 = 0.6 blocked. Use a single -0.2: only low energy → 0.8.
    res = e.evaluate_code_modification({
        "target_file": "aegis/layers/x.py",
        "proposed_code": "x = 1",
        "energy": 0.1,  # -0.2 → 0.8 → review_required
    })
    assert res["status"] == "review_required"


def test_code_mod_logs_capped():
    e = EthicsCore()
    for _ in range(210):
        e.evaluate_code_modification({
            "target_file": "aegis/layers/ethics_core.py",  # forces score 0 → blocked
            "proposed_code": "x = 1",
        })
    assert len(e.evaluation_log) <= 200
    assert len(e.violations) <= 200


# ── veto_check ───────────────────────────────────────────────────

def test_veto_blocks_low_clearance():
    e = EthicsCore()
    ev = Event(source=Layer.SUBSTRATE, target=None, event_type="x",
               ethical_clearance=0.5)
    assert e.veto_check(ev) is False


def test_veto_blocks_dangerous_payload():
    e = EthicsCore()
    ev = Event(source=Layer.SUBSTRATE, target=None, event_type="x",
               payload={"cmd": "destroy everything"})
    assert e.veto_check(ev) is False


def test_veto_allows_clean_event():
    e = EthicsCore()
    ev = Event(source=Layer.SUBSTRATE, target=None, event_type="x",
               payload={"cmd": "observe"})
    assert e.veto_check(ev) is True


# ── status ───────────────────────────────────────────────────────

def test_status_reports_counters():
    e = EthicsCore()
    e.evaluate_action({"type": "observe", "confidence": 0.9})
    e.evaluate_action({"type": "destroy", "destroy": True})
    st = e.status()
    assert st["total_checked"] == 2
    assert st["total_blocked"] == 1
    assert st["axioms_intact"] is True
    assert st["kill_switch"] is False
    assert "block_rate" in st
