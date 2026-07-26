"""Mutation-hardening tests for ethics_core (kill surviving mutants)."""
import dataclasses
import pytest

from aegis.layers.ethics_core import EthicsCore, Axiom


def test_axioms_are_frozen_immutable():
    # Kills the `@dataclass(frozen=True)` -> frozen=False mutant. Axiom
    # immutability is a safety property: the axioms must not be mutable.
    e = EthicsCore()
    ax = e.axioms[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ax.description = "tampered"


def test_irreversible_penalty_not_applied_when_key_absent():
    # Kills the `action.get("irreversible", False)` -> default True mutant.
    e = EthicsCore()
    res = e.evaluate_action({"type": "benign", "confidence": 1.0})
    # No irreversible/external/self-mod keys and full confidence -> pristine.
    assert res["score"] == 1.0
    assert not any("irreversible" in r.lower() for r in res["reasons"])


def test_affects_external_penalty_not_applied_when_key_absent():
    # Kills the `action.get("affects_external", False)` -> default True mutant.
    e = EthicsCore()
    res = e.evaluate_action({"type": "benign", "confidence": 1.0})
    assert not any("external" in r.lower() for r in res["reasons"])


def test_irreversible_penalty_applied_when_true():
    # Confirms the flag actually does something (guards against over-fitting the
    # "absent" tests above).
    e = EthicsCore()
    res = e.evaluate_action({"type": "x", "confidence": 1.0, "irreversible": True})
    assert res["score"] < 1.0
    assert any("irreversible" in r.lower() for r in res["reasons"])


def test_block_rate_is_a_percentage():
    # Kills the `* 100` -> `/ 100` mutant in block_rate. One blocked of two
    # checks must read as 50.0%, not 0.005.
    e = EthicsCore()
    # A clearly-dangerous action drives score below the auto threshold -> blocked.
    e.evaluate_action({"type": "attack", "detail": "harm exploit manipulate_human"})
    # A benign action -> approved.
    e.evaluate_action({"type": "benign", "confidence": 1.0})
    st = e.status()
    assert st["total_checked"] == 2
    assert st["total_blocked"] == 1
    assert st["block_rate"] == 50.0
