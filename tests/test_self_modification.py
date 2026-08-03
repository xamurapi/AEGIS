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


def test_proposal_ids_stay_unique_once_the_record_cap_is_reached():
    """The id was derived from ``len(self.modifications)``, which _cap pins at
    _MAX_RECORDS — so every proposal after the 200th was "mod_0200" and
    rollbacks/sandbox results could no longer be matched to their proposal.
    The id now comes from a monotonic counter, as CodeModifier's does."""
    from aegis.layers.self_modification import _MAX_RECORDS

    sm = SelfModification()
    sm.modifications = [{"status": "applied"}] * _MAX_RECORDS  # list at the cap

    first = sm.propose_modification("parametric", "temperature", 99.0)
    sm.apply_modification(first, {"passed": False})   # recorded via _cap
    second = sm.propose_modification("parametric", "temperature", 99.0)
    assert first["id"] != second["id"]


def test_weight_training_degradation_rollback_is_counted():
    """WeightModifier reports degradation as "Training caused degradation —
    rolled back" (with a SPACE) plus a structured ``rolled_back`` flag; the old
    check looked for the substring "rolled_back", never matched, and
    weight_mod_rollbacks stayed 0 forever."""
    import asyncio
    import types

    sm = SelfModification()

    async def _train(dataset_dir, ethics_approved=False):
        # The exact shape WeightModifier._train_sync returns on degradation.
        return {"success": False, "rolled_back": True,
                "error": "Training caused degradation — rolled back",
                "train_loss": 0.5, "val_loss": 5.0, "baseline_val_loss": 0.2}

    sm.weight_modifier = types.SimpleNamespace(
        can_train=lambda: (True, "Ready"), train=_train)
    sm.dataset_builder = types.SimpleNamespace(
        build_from_memory=lambda memory, agent_system, feedback_loop=None: {
            "success": True, "total_size": 60, "dataset_dir": "datasets/d1"})
    ethics = types.SimpleNamespace(
        evaluate_action=lambda info: {"status": "approved", "score": 1.0})

    record = asyncio.run(sm.propose_weight_modification(None, ethics_core=ethics))
    assert record["status"] == "rolled_back"
    assert sm.weight_mod_rollbacks == 1


def test_weight_training_rollback_recognised_from_text_alone():
    """Older-shaped results carry only the error text; the fallback must match
    the words the trainer actually says, space and all."""
    import asyncio
    import types

    sm = SelfModification()

    async def _train(dataset_dir, ethics_approved=False):
        return {"success": False,
                "error": "Training caused degradation — rolled back"}

    sm.weight_modifier = types.SimpleNamespace(
        can_train=lambda: (True, "Ready"), train=_train)
    sm.dataset_builder = types.SimpleNamespace(
        build_from_memory=lambda memory, agent_system, feedback_loop=None: {
            "success": True, "total_size": 60, "dataset_dir": "datasets/d1"})
    ethics = types.SimpleNamespace(
        evaluate_action=lambda info: {"status": "approved", "score": 1.0})

    record = asyncio.run(sm.propose_weight_modification(None, ethics_core=ethics))
    assert record["status"] == "rolled_back"
    assert sm.weight_mod_rollbacks == 1
