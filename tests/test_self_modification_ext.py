"""Extended tests for SelfModification — sandbox branches, weight mod pipeline."""
import asyncio
import pytest
from aegis.layers.self_modification import SelfModification


# ── Fakes for the weight-modification pipeline ───────────────────

class FakeDatasetBuilder:
    def __init__(self, result=None):
        self._result = result if result is not None else {
            "success": True, "total_size": 100, "dataset_dir": "/tmp/ds",
        }

    def build_from_memory(self, memory, agent_system=None):
        return self._result

    def status(self):
        return {"builder": "ok"}


class FakeWeightModifier:
    def __init__(self, can_train=(True, ""), train_result=None):
        self._can_train = can_train
        self._train_result = train_result if train_result is not None else {
            "success": True, "checkpoint": "/tmp/ckpt",
            "train_loss": 0.1, "val_loss": 0.2,
        }
        self.rollback_ckpt_called = None
        self.rollback_base_called = False

    def can_train(self):
        return self._can_train

    async def train(self, dataset_dir, ethics_approved=False):
        return self._train_result

    def rollback_to_checkpoint(self, path):
        self.rollback_ckpt_called = path
        return {"success": True, "restored": path}

    def rollback_to_base(self):
        self.rollback_base_called = True
        return self._rollback_base_result

    _rollback_base_result = {"success": True, "restored": "base"}

    def status(self):
        return {"weights": "ok"}


class FakeEthics:
    def __init__(self, status="approved", score=0.9):
        self._status = status
        self._score = score

    def evaluate_action(self, action):
        return {"status": self._status, "score": self._score}


# ── sandbox_test uncovered branches ──────────────────────────────

def test_sandbox_unknown_param_and_none_old_value():
    sm = SelfModification()
    proposal = sm.propose_modification("parametric", "unknown_param", 1.23)
    assert proposal["old_value"] is None
    sandbox = sm.sandbox_test(proposal, current_metric=0.7)
    # unknown param: no bounds → passes bounds test; None old_value → passes mag.
    assert "Out of bounds" not in " ".join(sandbox["reasons"])


def test_sandbox_declining_trend_fails():
    sm = SelfModification()
    sm._metric_history = [0.9, 0.8, 0.5]  # trend = -0.4 < -0.05
    old = sm.parameters["temperature"]
    proposal = sm.propose_modification("parametric", "temperature", old * 1.05)
    sandbox = sm.sandbox_test(proposal, current_metric=0.7)
    assert any("Declining metric trend" in r for r in sandbox["reasons"])


def test_sandbox_learning_rate_sanity():
    sm = SelfModification()
    proposal = sm.propose_modification("parametric", "learning_rate", 0.008)
    sandbox = sm.sandbox_test(proposal, current_metric=0.7)
    assert any("Learning rate dangerously high" in r for r in sandbox["reasons"])


def test_sandbox_dropout_sanity():
    sm = SelfModification()
    proposal = sm.propose_modification("parametric", "dropout", 0.45)
    sandbox = sm.sandbox_test(proposal, current_metric=0.7)
    assert any("Dropout too high" in r for r in sandbox["reasons"])


def test_sandbox_memory_decay_sanity():
    sm = SelfModification()
    proposal = sm.propose_modification("parametric", "memory_decay", 0.09)
    sandbox = sm.sandbox_test(proposal, current_metric=0.7)
    assert any("Memory decay too fast" in r for r in sandbox["reasons"])


def test_sandbox_metric_history_is_capped():
    sm = SelfModification()
    proposal = sm.propose_modification("parametric", "temperature", 0.7)
    for _ in range(60):
        sm.sandbox_test(proposal, current_metric=0.6)
    assert len(sm._metric_history) <= 50


# ── apply_modification uncovered branches ────────────────────────

def test_apply_degradation_rejected():
    sm = SelfModification()
    proposal = sm.propose_modification("parametric", "temperature", 0.72)
    fake_sandbox = {"passed": True, "degradation": 0.2, "proposal_id": proposal["id"]}
    res = sm.apply_modification(proposal, fake_sandbox)
    assert res["applied"] is False
    assert res["reason"] == "degradation_too_high"
    assert proposal["status"] == "rejected_degradation"
    assert sm.rollbacks  # audit entry recorded


def test_apply_target_not_found():
    sm = SelfModification()
    proposal = sm.propose_modification("parametric", "nonexistent", 1.0)
    fake_sandbox = {"passed": True, "degradation": 0.0, "proposal_id": proposal["id"]}
    res = sm.apply_modification(proposal, fake_sandbox)
    assert res["applied"] is False
    assert proposal["status"] == "target_not_found"


def test_apply_rejected_when_sandbox_failed():
    sm = SelfModification()
    proposal = sm.propose_modification("parametric", "temperature", 0.72)
    res = sm.apply_modification(proposal, {"passed": False})
    assert res["applied"] is False
    assert proposal["status"] == "rejected"


# ── version helpers ──────────────────────────────────────────────

def test_version_parts_short_and_nonnumeric():
    assert SelfModification._version_parts("1.0") == [1, 0, 0]
    assert SelfModification._version_parts("1.x.0") == [1, 0, 0]
    assert SelfModification._version_parts("2.3.4-beta") == [2, 3, 4]


def test_bump_minor_and_patch():
    assert SelfModification._bump_minor("1.2.3") == "1.3.0"
    assert SelfModification._bump_patch("1.2.3") == "1.2.4"


# ── _rollback ────────────────────────────────────────────────────

def test_rollback_restores_checkpoint():
    sm = SelfModification()
    sm._stable_checkpoint = dict(sm.parameters)
    sm.parameters["temperature"] = 9.9  # drift
    proposal = sm.propose_modification("parametric", "temperature", 9.9)
    sm._rollback(proposal)
    assert sm.parameters["temperature"] != 9.9
    assert proposal["status"] == "rolled_back"
    assert sm.rollbacks


# ── propose_weight_modification pipeline ─────────────────────────

def test_weight_mod_no_modifier_configured():
    sm = SelfModification()
    res = asyncio.run(sm.propose_weight_modification(memory={}))
    assert res["success"] is False


def test_weight_mod_can_train_false():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier(can_train=(False, "cooldown"))
    sm.dataset_builder = FakeDatasetBuilder()
    res = asyncio.run(sm.propose_weight_modification(memory={}))
    assert res["success"] is False
    assert res["error"] == "cooldown"


def test_weight_mod_dataset_build_fails():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier()
    sm.dataset_builder = FakeDatasetBuilder({"success": False, "error": "no data"})
    res = asyncio.run(sm.propose_weight_modification(memory={},
                                                     ethics_core=FakeEthics()))
    assert res["status"] == "dataset_failed"
    assert res["error"] == "no data"


def test_weight_mod_ethics_unavailable_fails_closed():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier()
    sm.dataset_builder = FakeDatasetBuilder()
    res = asyncio.run(sm.propose_weight_modification(memory={}, ethics_core=None))
    assert res["status"] == "ethics_unavailable"


def test_weight_mod_ethics_blocked():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier()
    sm.dataset_builder = FakeDatasetBuilder()
    res = asyncio.run(sm.propose_weight_modification(
        memory={}, ethics_core=FakeEthics(status="blocked", score=0.2)))
    assert res["status"] == "ethics_blocked"
    assert res["ethics_score"] == 0.2


def test_weight_mod_ethics_review_required():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier()
    sm.dataset_builder = FakeDatasetBuilder()
    res = asyncio.run(sm.propose_weight_modification(
        memory={}, ethics_core=FakeEthics(status="review_required", score=0.8)))
    assert res["status"] == "ethics_review_required"


def test_weight_mod_train_success_applies_and_bumps_version():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier()
    sm.dataset_builder = FakeDatasetBuilder()
    before = sm.current_version
    res = asyncio.run(sm.propose_weight_modification(
        memory={}, ethics_core=FakeEthics()))
    assert res["status"] == "applied"
    assert sm.weight_mod_success == 1
    assert res["version"] != before


def test_weight_mod_train_failure():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier(
        train_result={"success": False, "error": "OOM"})
    sm.dataset_builder = FakeDatasetBuilder()
    res = asyncio.run(sm.propose_weight_modification(
        memory={}, ethics_core=FakeEthics()))
    assert res["status"] == "failed"
    assert res["error"] == "OOM"


def test_weight_mod_train_rolled_back():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier(
        train_result={"success": False, "error": "degraded — rolled_back to base"})
    sm.dataset_builder = FakeDatasetBuilder()
    res = asyncio.run(sm.propose_weight_modification(
        memory={}, ethics_core=FakeEthics()))
    assert res["status"] == "rolled_back"
    assert sm.weight_mod_rollbacks == 1


# ── rollback_weights ─────────────────────────────────────────────

def test_rollback_weights_no_modifier():
    sm = SelfModification()
    res = sm.rollback_weights()
    assert res["success"] is False


def test_rollback_weights_to_checkpoint():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier()
    res = sm.rollback_weights(checkpoint_path="/tmp/ckpt1")
    assert res["success"] is True
    assert sm.weight_modifier.rollback_ckpt_called == "/tmp/ckpt1"
    assert sm.weight_mod_rollbacks == 1


def test_rollback_weights_to_base():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier()
    res = sm.rollback_weights()
    assert res["success"] is True
    assert sm.weight_modifier.rollback_base_called is True


def test_rollback_weights_unsuccessful_does_not_count():
    sm = SelfModification()
    wm = FakeWeightModifier()
    wm._rollback_base_result = {"success": False, "error": "no checkpoint"}
    sm.weight_modifier = wm
    res = sm.rollback_weights()
    assert res["success"] is False
    assert sm.weight_mod_rollbacks == 0  # failed rollback not counted


# ── status ───────────────────────────────────────────────────────

def test_status_without_weight_modules():
    sm = SelfModification()
    st = sm.status()
    assert st["version"] == "1.0.0"
    assert st["weight_modifier"] == {}
    assert st["dataset_builder"] == {}


def test_status_with_weight_modules():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier()
    sm.dataset_builder = FakeDatasetBuilder()
    st = sm.status()
    assert st["weight_modifier"] == {"weights": "ok"}
    assert st["dataset_builder"] == {"builder": "ok"}
