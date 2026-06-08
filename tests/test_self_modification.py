"""Tests for parametric self-modification sandbox + apply/rollback."""
from aegis.layers.self_modification import SelfModification


def test_in_bounds_small_change_applies():
    sm = SelfModification()
    old = sm.parameters["temperature"]  # 0.7
    new = old * 1.1  # +10%, within 20% and bounds
    proposal = sm.propose_modification("parametric", "temperature", new)
    sandbox = sm.sandbox_test(proposal, current_metric=0.7)
    assert sandbox["passed"] is True
    res = sm.apply_modification(proposal, sandbox)
    assert res["applied"] is True
    assert sm.parameters["temperature"] == new


def test_out_of_bounds_rejected():
    sm = SelfModification()
    proposal = sm.propose_modification("parametric", "temperature", 99.0)
    sandbox = sm.sandbox_test(proposal, current_metric=0.7)
    assert sandbox["passed"] is False
    res = sm.apply_modification(proposal, sandbox)
    assert res["applied"] is False


def test_large_change_rejected():
    sm = SelfModification()
    old = sm.parameters["curiosity_weight"]  # 0.5
    proposal = sm.propose_modification("parametric", "curiosity_weight", old * 2)  # +100%
    sandbox = sm.sandbox_test(proposal, current_metric=0.7)
    assert sandbox["passed"] is False


def test_low_metric_blocks_change():
    sm = SelfModification()
    old = sm.parameters["dropout"]
    proposal = sm.propose_modification("parametric", "dropout", old * 1.05)
    sandbox = sm.sandbox_test(proposal, current_metric=0.1)  # below 0.3 floor
    assert sandbox["passed"] is False


def test_version_bumps_on_apply():
    sm = SelfModification()
    before = sm.current_version
    old = sm.parameters["temperature"]
    proposal = sm.propose_modification("parametric", "temperature", old * 1.05)
    sandbox = sm.sandbox_test(proposal, current_metric=0.7)
    sm.apply_modification(proposal, sandbox)
    assert sm.current_version != before
