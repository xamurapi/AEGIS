"""The promoted champion actually reaches the running system, and is watched.

Two audit findings live here, and they are two halves of one hole:

* ``EvolutionEngine.watch()`` was dead code — both callers of
  ``run_generation`` discard its result, so no promoted champion was ever
  applied, no live metric was ever fed to the watch, and the rollback the
  module docstring promises could never fire. The reflect phase would be the
  natural call site, but the phases are owned elsewhere; the substrate now
  closes the loop itself, once per tick, after REFLECT.
* the restart guard for a pending v1 candidate was unreachable — ``_load``
  dropped every stored candidate, so the checkpointed genome (the unjudged
  mutation) was adopted as the running baseline on every restart while the
  champion record still held the measured one.
"""
import asyncio

import pytest

from aegis.layers.evolution.genome import Genome
from aegis.layers.substrate import Substrate


@pytest.fixture
def substrate(isolated_state):
    s = Substrate()
    s.llm.enabled = False
    return s


def assert_same_genome(got: dict, wanted: dict):
    """Per-gene, with float tolerance: applying a genome re-normalises the
    simplex shares, which shifts the last decimal without meaning anything."""
    assert set(got) == set(wanted)
    for name, value in wanted.items():
        if isinstance(value, float):
            assert got[name] == pytest.approx(value), name
        else:
            assert got[name] == value, name


# ── the watch is wired into the tick ─────────────────────────────────

def test_the_tick_feeds_the_champion_watch(substrate, monkeypatch):
    """`watch` only works if somebody calls it. This pins the call site: one
    tick, one reading, through `_watch_champion`."""
    called = []
    monkeypatch.setattr(substrate, "_watch_champion",
                        lambda: called.append(True))
    asyncio.run(substrate.tick())
    assert called, "the tick never reached the champion watch"


def test_an_unapplied_promotion_is_adopted_and_a_live_drop_rolls_it_back(
        substrate, monkeypatch):
    """The full circle §M5.6 describes, against the real substrate:

    the engine promotes (as a detached generation would, with nobody reading
    the result) → the substrate applies the champion through the safety gates
    → the live metric falls past EVO_ROLLBACK_DELTA → the previous champion is
    put back *in the running system*, not merely in the engine's records.
    """
    import aegis.config as cfg

    reading = {"value": 0.80}
    monkeypatch.setattr(substrate, "_compute_reward",
                        lambda: reading["value"])

    old_genome = substrate.current_genome().to_dict()
    substrate._watch_champion()          # a pre-promotion reading is on record

    promoted = Genome(old_genome)
    promoted["w_ev"] = 2.0 if old_genome.get("w_ev") != 2.0 else 1.5
    substrate.evolution._promote(
        {"genome": promoted.to_dict(), "fitness": 0.9,
         "generation": 1, "created": 0.0}, tick=substrate.tick_count)

    substrate._watch_champion()          # adoption: the champion goes live
    assert_same_genome(substrate.current_genome().to_dict(), promoted.to_dict())

    reading["value"] = 0.80 - cfg.EVO_ROLLBACK_DELTA - 0.05
    substrate.tick_count += 1
    substrate._watch_champion()          # the watch trips and puts it back
    assert substrate.evolution.rollbacks == 1
    assert_same_genome(substrate.current_genome().to_dict(), old_genome)


def test_the_watch_stays_out_of_a_pending_candidate_experiment(substrate,
                                                               monkeypatch):
    """While a v1 candidate is live, the substrate must neither adopt a
    champion over it nor roll back "through" it — either would clobber the
    experiment in flight before its benchmark could judge it."""
    monkeypatch.setattr(substrate, "_compute_reward", lambda: 0.8)
    mutation = substrate.evolution.propose_mutation(substrate.tick_count)
    assert mutation is not None
    substrate._genome_before_candidate = substrate.apply_genome(
        substrate.evolution.candidate["genome"])
    live = substrate.current_genome().to_dict()

    substrate.evolution.previous_champion = {"genome": Genome().to_dict(),
                                             "fitness": 0.5}
    substrate.evolution.promoted_at_tick = substrate.tick_count
    substrate._watch_champion()
    assert substrate.current_genome().to_dict() == live


# ── the restart guard for a pending candidate ────────────────────────

def test_a_pending_candidate_is_not_adopted_as_the_baseline_on_restart(
        isolated_state):
    """The failure scenario of the audit, end to end: candidate applied →
    checkpoint written (with the candidate's genome as the running one) →
    restart. The restored system must still hold the candidate for judgement
    and must run the *champion* — the only measured configuration — not the
    unjudged mutation the checkpoint happens to record."""
    s = Substrate()
    s.llm.enabled = False
    champion_genome = dict(s.evolution.champion["genome"])

    mutation = s.evolution.propose_mutation(s.tick_count)
    assert mutation is not None
    s._genome_before_candidate = s.apply_genome(s.evolution.candidate["genome"])
    candidate_genome = dict(s.evolution.candidate["genome"])
    assert candidate_genome != champion_genome     # the mutation moved a gene
    s._save_checkpoint()

    restarted = Substrate()
    restarted.llm.enabled = False
    assert restarted.evolution.candidate is not None, \
        "the pending candidate did not survive the restart"
    # Not the candidate's genome — the champion's. Per-gene with tolerance,
    # since restoring re-normalises the simplex shares; what matters is that
    # the mutated gene came back to its measured value.
    assert_same_genome(restarted.current_genome().to_dict(), champion_genome)
