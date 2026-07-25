"""Tests for SelfModification hardening (MEDIUM defect fixes):

- audit lists (modifications/sandbox_results/rollbacks/weight_modifications)
  are capped at _MAX_RECORDS;
- propose_weight_modification survives an exception from train() with a
  recorded result and consistent counters;
- target_not_found counts as a rejection in success_rate.
"""
import asyncio
import pytest
from aegis.layers.self_modification import SelfModification, _MAX_RECORDS
from tests.test_self_modification_ext import (
    FakeDatasetBuilder, FakeWeightModifier, FakeEthics,
)


# ── record-list capping ─────────────────────────────────────────

def test_modifications_and_sandbox_lists_capped():
    sm = SelfModification()
    old = sm.parameters["temperature"]
    for _ in range(_MAX_RECORDS + 120):
        proposal = sm.propose_modification("parametric", "temperature", old * 1.02)
        sandbox = sm.sandbox_test(proposal, current_metric=0.7)
        sm.apply_modification(proposal, sandbox)
    assert len(sm.modifications) <= _MAX_RECORDS
    assert len(sm.sandbox_results) <= _MAX_RECORDS


def test_rollbacks_list_capped():
    sm = SelfModification()
    for i in range(_MAX_RECORDS + 50):
        proposal = sm.propose_modification("parametric", "temperature", 0.72)
        sm.apply_modification(proposal, {"passed": True, "degradation": 0.2,
                                         "proposal_id": proposal["id"]})
    assert len(sm.rollbacks) <= _MAX_RECORDS
    assert len(sm.modifications) <= _MAX_RECORDS


def test_weight_modifications_list_capped():
    sm = SelfModification()
    sm.weight_modifier = FakeWeightModifier()
    sm.dataset_builder = FakeDatasetBuilder({"success": False, "error": "no data"})
    for _ in range(_MAX_RECORDS + 40):
        asyncio.run(sm.propose_weight_modification(memory={}, ethics_core=FakeEthics()))
    assert len(sm.weight_modifications) <= _MAX_RECORDS


def test_cap_preserves_most_recent():
    sm = SelfModification()
    for i in range(_MAX_RECORDS + 10):
        sm._cap(sm.modifications, {"id": i, "status": "applied"})
    ids = [m["id"] for m in sm.modifications]
    assert ids[-1] == _MAX_RECORDS + 9
    assert len(ids) == _MAX_RECORDS


# ── train() exception handling ──────────────────────────────────

class _ExplodingWeightModifier(FakeWeightModifier):
    async def train(self, dataset_dir, ethics_approved=False):
        raise RuntimeError("CUDA out of memory")


def test_train_exception_records_failed_and_keeps_counters_consistent():
    sm = SelfModification()
    sm.weight_modifier = _ExplodingWeightModifier()
    sm.dataset_builder = FakeDatasetBuilder()
    res = asyncio.run(sm.propose_weight_modification(memory={}, ethics_core=FakeEthics()))
    # Record is returned and appended, not stuck at "training".
    assert res["status"] == "failed"
    assert "training_exception" in res["error"]
    assert sm.weight_modifications and sm.weight_modifications[-1]["status"] == "failed"
    # total incremented, success NOT — counters stay consistent.
    assert sm.weight_mod_total == 1
    assert sm.weight_mod_success == 0


def test_train_returns_non_dict_is_handled():
    class _WeirdModifier(FakeWeightModifier):
        async def train(self, dataset_dir, ethics_approved=False):
            return None

    sm = SelfModification()
    sm.weight_modifier = _WeirdModifier()
    sm.dataset_builder = FakeDatasetBuilder()
    res = asyncio.run(sm.propose_weight_modification(memory={}, ethics_core=FakeEthics()))
    assert res["status"] == "failed"
    assert sm.weight_mod_success == 0


# ── success_rate accounting ─────────────────────────────────────

def test_target_not_found_counts_as_rejected_in_success_rate():
    sm = SelfModification()
    # One genuine applied modification.
    old = sm.parameters["temperature"]
    p1 = sm.propose_modification("parametric", "temperature", old * 1.02)
    s1 = sm.sandbox_test(p1, current_metric=0.7)
    sm.apply_modification(p1, s1)
    # One target_not_found.
    p2 = sm.propose_modification("parametric", "does_not_exist", 1.0)
    sm.apply_modification(p2, {"passed": True, "degradation": 0.0, "proposal_id": p2["id"]})

    st = sm.status()
    assert st["applied"] == 1
    assert st["rejected"] == 1  # target_not_found now counted
    assert st["success_rate"] == 50.0
