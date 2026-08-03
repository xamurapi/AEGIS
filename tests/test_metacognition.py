"""The metacognition facade (spec M11.7, M11.10, M11.11).

Pinned here: the novelty quota is a quota and not a bonus; the acceptance
gates of M6.8 are untouched; an ``unsupported`` explanation steers nothing;
a contradicting cortex contests and changes no number; the ablation is routed
through the evaluation pool and never through the tick path; the genes read
back from their consumers and each moves behaviour; and with
``META_ENABLED=False`` the reasoning contour is exactly what it was before
M11 (acceptance criterion 11).
"""
import asyncio
import math

import pytest

import aegis.config as cfg
from aegis.layers.metacognition import MetaCognition
from aegis.layers.metacognition.attribution import attribute_edits, Edit
from aegis.layers.metacognition.mechanism import (
    FIXED_TRANSFORM_ORDER, mechanism_of_transform,
)
from aegis.layers.reasoning import ReasoningEngine
from aegis.layers.reasoning.synthesis import TRANSFORMS


def _engine(tmp_path):
    return ReasoningEngine(store_path=tmp_path / "strategies.json")


def _meta(tmp_path, engine=None, *, enabled=True, pool=None, cortex=None):
    engine = engine or _engine(tmp_path)
    return engine, MetaCognition(reasoning=engine, pool=pool, cortex=cortex,
                                 store_dir=tmp_path / "meta", enabled=enabled)


def _warm(engine, tasks=96):
    engine.solve(tasks)
    engine.scan_weakness()


# ── criterion 11: the system works without the module ────────────────

def test_disabled_metacognition_leaves_the_synthesiser_alone(tmp_path):
    engine, meta = _meta(tmp_path / "a", enabled=False)
    assert engine.synthesiser.order_hook is None
    assert meta.invent() == []
    meta.on_reflect(0)                       # a no-op, not an error
    assert meta.pending_attributions() == []


def test_disabled_metacognition_reproduces_the_pre_m11_run(tmp_path):
    """Two engines, one shadowed by a disabled contour: identical candidates
    in identical order, identical verdicts."""
    plain = _engine(tmp_path / "plain")
    shadowed, _meta_off = _meta(tmp_path / "shadowed", enabled=False)

    for engine in (plain, shadowed):
        _warm(engine)
        engine.propose_strategy(tick=1)

    assert [c.name for c in plain.candidates] \
        == [c.name for c in shadowed.candidates]


def test_the_interpreter_is_not_extended():
    """Criterion 10: twelve operations, before and after M11."""
    from aegis.layers.reasoning.dsl import OPS

    assert len(OPS) == 12


# ── the order hook (M11.6.3) ─────────────────────────────────────────

def test_enabled_metacognition_installs_the_order_hook(tmp_path):
    engine, meta = _meta(tmp_path, enabled=True)
    assert engine.synthesiser.order_hook == meta.order_for


def test_credit_reorders_the_actual_proposals(tmp_path):
    """Behaviour, not a report: the same weakness yields candidates in a
    different order once credit has accumulated."""
    engine, meta = _meta(tmp_path, enabled=True)
    _warm(engine)
    weakness = engine.found[0]
    baseline = [c.transform for c in
                engine.synthesiser.propose(weakness, engine.library, 1)]
    assert baseline == [name for name, _ in TRANSFORMS
                        if name in set(baseline)]

    features = tuple(weakness.combo)
    last = baseline[-1]
    for name in FIXED_TRANSFORM_ORDER:
        for _ in range(4):
            meta.credit.note_attempt(mechanism_of_transform(name), features)
    meta.credit.note_accepted(mechanism_of_transform(last), features, 0.2)

    fresh_engine = _engine(tmp_path / "fresh")
    fresh_engine.synthesiser.order_hook = meta.order_for
    _warm(fresh_engine)
    reordered = [c.transform for c in
                 fresh_engine.synthesiser.propose(weakness,
                                                  fresh_engine.library, 1)]
    assert reordered[0] == last
    assert meta.credit.order_differs(features, FIXED_TRANSFORM_ORDER)


def test_an_unsupported_explanation_steers_nothing(tmp_path):
    """Criterion 6: unsupported explanations put nothing into the credit
    table, so the synthesis order is exactly the credit-free order."""
    engine, meta = _meta(tmp_path, enabled=True)
    from aegis.layers.metacognition.attribution import conclude

    before = meta.order_for("family=arithmetic_chain")
    explanation = conclude("s", "direct", "family=arithmetic_chain", 0.1,
                           attribute_edits(
                               (Edit(kind="insert", position=0, op="VERIFY"),),
                               {}))
    assert explanation.status == "unsupported"
    meta.explanations.append(explanation)
    assert meta.order_for("family=arithmetic_chain") == before
    assert not meta.credit.rows


# ── the far quota (M11.6.4) ──────────────────────────────────────────

def test_the_quota_is_a_ceiling_share_of_the_round(tmp_path):
    engine, meta = _meta(tmp_path, enabled=True)
    k = engine.synthesiser.max_candidates
    meta.set_genome({"meta_far_share": 0.25})
    assert meta.far_quota() == math.ceil(k * 0.25)
    meta.set_genome({"meta_far_share": 0.5})
    assert meta.far_quota() == math.ceil(k * 0.5)


def test_invent_fills_the_quota_with_far_candidates(tmp_path):
    engine, meta = _meta(tmp_path, enabled=True)
    _warm(engine)
    issued = meta.invent(tick=1)
    assert len(issued) == meta.far_quota()
    for row in issued:
        assert row["transform"].startswith("skeleton:")
    # Every issued candidate joined the SAME queue the arena drains.
    names = {c.name for c in engine.candidates}
    assert all(row["name"] in names for row in issued)


def test_far_candidates_are_far_from_the_whole_archive(tmp_path):
    from aegis.layers.metacognition.distance import distance

    engine, meta = _meta(tmp_path, enabled=True)
    _warm(engine)
    # Measure against the archive as it stood BEFORE invention added its own.
    archived = [list(entry["steps"]) for entry in meta.archive.entries]
    issued = meta.invent(tick=1)
    assert issued
    for row in issued:
        candidate = next(c for c in engine.candidates
                         if c.name == row["name"])
        nearest = min(distance(candidate.steps, steps) for steps in archived)
        assert nearest >= cfg.META_FAR, (row["name"], nearest)


def test_acceptance_gates_are_not_softened_for_far_candidates(tmp_path):
    """M11.6.4: a far candidate faces the same arena and the same thresholds.
    The arena the far candidate meets IS the engine's arena object with its
    configured min_gain — nothing swaps in a milder judge."""
    engine, meta = _meta(tmp_path, enabled=True)
    _warm(engine)
    meta.invent(tick=1)
    min_gain_before = engine.arena.min_gain
    verdict = engine.evaluate_candidate(tick=1)
    assert verdict is not None
    assert engine.arena.min_gain == min_gain_before == cfg.REASON_MIN_GAIN
    if not verdict["accepted"]:
        assert verdict["reasons"], "a rejection carries its reasons"


def test_retired_pairs_are_not_reinvented(tmp_path):
    engine, meta = _meta(tmp_path, enabled=True)
    _warm(engine)
    weakness = engine.found[0]
    features = tuple(weakness.combo)
    issued = meta.invent(tick=1)
    assert issued
    first = issued[0]["transform"].split(":", 1)[1]
    for _ in range(int(cfg.META_RETIRE_AFTER)):
        meta.skeletons.note_failure(first, features)
    engine.candidates.clear()
    again = meta.invent(tick=2)
    assert all(row["transform"] != f"skeleton:{first}" for row in again)


# ── credit from verdicts, through the tick-side hook ─────────────────

def test_on_reflect_folds_verdicts_into_credit(tmp_path):
    engine, meta = _meta(tmp_path, enabled=True)
    _warm(engine)
    engine.propose_strategy(tick=1)
    while engine.pending_candidates():
        engine.evaluate_candidate(tick=1)
    meta.on_reflect(1)
    assert meta.credit.rows, "arena verdicts must earn attempts"
    attempts = sum(row.attempts for row in meta.credit.rows.values())
    assert attempts > 0
    # And accepted strategies are queued for attribution.
    accepted = [v for v in engine.verdicts if v["accepted"]]
    if accepted:
        assert meta.pending_attributions()


def test_on_reflect_never_runs_the_interpreter(tmp_path):
    """M11.7.3: the tick-side part is bookkeeping. If it ever scores a task,
    this raises and the budget claim is false."""
    engine, meta = _meta(tmp_path, enabled=True)
    _warm(engine)
    engine.propose_strategy(tick=1)
    while engine.pending_candidates():
        engine.evaluate_candidate(tick=1)

    def _forbidden(*args, **kwargs):
        raise AssertionError("on_reflect reached the interpreter")

    engine.interpreter.run = _forbidden
    meta.on_reflect(2)


def test_the_verdict_cursor_survives_a_trim(tmp_path):
    engine, meta = _meta(tmp_path, enabled=True)
    meta._verdict_cursor = 500
    engine.verdicts.clear()
    meta.on_reflect(1)                       # must not raise or double-count
    assert meta._verdict_cursor == 0


# ── attribution routing (M11.7.3): the pool, never the tick ──────────

class _SpyPool:
    def __init__(self):
        self.batches = 0

    def map(self, function, items, purpose=""):
        from aegis.eval.pool import PoolResult

        self.batches += 1
        return [PoolResult(index=index, value=function(item))
                for index, item in enumerate(items)]


def _accepted_strategy(engine, meta, tick=1):
    """Admit a known-good candidate as if the arena had accepted it."""
    steps = [
        {"op": "SOLVE"},
        {"op": "VERIFY", "checker": "confidence"},
        {"op": "BRANCH", "cond": "insufficient",
         "then": [{"op": "ABSTAIN", "reason": "guess"}]},
    ]
    strategy = engine.library.admit(
        "planted", steps, origin="synth", parent="direct", tick=tick,
        status="trial", weakness="family=missing_data",
        family="missing_data", incumbent="direct")
    meta.on_reflect(tick)
    return strategy


def test_attribution_goes_through_the_evaluation_pool(tmp_path):
    pool = _SpyPool()
    engine, meta = _meta(tmp_path, enabled=True, pool=pool)
    _accepted_strategy(engine, meta)
    assert "planted" in meta.pending_attributions()
    report = asyncio.run(meta.attribute(tick=1))
    assert report is not None
    assert pool.batches == 1, "the ablation must be submitted to the pool"
    assert report["strategy"] == "planted"
    assert report["status"] in ("supported", "unsupported", "contested")


def test_a_planted_win_is_attributed_and_supported(tmp_path):
    engine, meta = _meta(tmp_path, enabled=True)
    _accepted_strategy(engine, meta)
    report = asyncio.run(meta.attribute(tick=1))
    assert report["status"] == "supported"
    assert report["mechanism"] != ""
    confirmed = [edit for edit in report["edits"] if edit["confirmed"]]
    assert confirmed, "the planted cause must be confirmed"


def test_a_contradicting_cortex_contests_but_rewrites_nothing(tmp_path):
    """Criterion 7, end to end through the facade."""

    class _ContraryCortex:
        def role_available(self, role):
            return True

        async def structured(self, role, messages, schema):
            return {"narrative": "a beautiful story",
                    "mechanism": "voting_reduced_variance"}

    engine, meta = _meta(tmp_path, enabled=True, cortex=_ContraryCortex())
    _accepted_strategy(engine, meta)
    report = asyncio.run(meta.attribute(tick=1))
    assert report["status"] == "contested"
    # The computed mechanism stands; the proposal is nowhere in the record.
    assert report["mechanism"] == "abstention_avoided_confident_error"
    assert report["narrative"] == "a beautiful story"
    # Ordering keeps using the computed mechanism: the contested explanation
    # credited the computed one, not the proposed one.
    assert all(key[0] != "voting_reduced_variance"
               or row.total_effect == 0.0
               for key, row in meta.credit.rows.items())


def test_an_agreeing_cortex_leaves_the_status_supported(tmp_path):
    class _AgreeingCortex:
        def role_available(self, role):
            return True

        async def structured(self, role, messages, schema):
            return {"narrative": "the abstain branch stopped the guessing",
                    "mechanism": "abstention_avoided_confident_error"}

    engine, meta = _meta(tmp_path, enabled=True, cortex=_AgreeingCortex())
    _accepted_strategy(engine, meta)
    report = asyncio.run(meta.attribute(tick=1))
    assert report["status"] == "supported"


def test_too_many_edits_yield_unsupported_without_an_ablation(tmp_path):
    pool = _SpyPool()
    engine, meta = _meta(tmp_path, enabled=True, pool=pool)
    steps = [{"op": "PREDICT", "horizon": 1},
             {"op": "RETRIEVE", "source": "memory", "k": 5},
             {"op": "DECOMPOSE", "max_parts": 4},
             {"op": "SOLVE"},
             {"op": "VERIFY", "checker": "type"},
             {"op": "BRANCH", "cond": "insufficient",
              "then": [{"op": "ABSTAIN"}]}]
    engine.library.admit("sprawling", steps, origin="synth", parent="direct",
                         status="trial", weakness="family=missing_data",
                         family="missing_data", incumbent="direct")
    meta.on_reflect(1)
    report = asyncio.run(meta.attribute(tick=1))
    assert report["status"] == "unsupported"
    assert pool.batches == 0, "refusal must not cost an ablation"


# ── genes (M11.7.4): consumer read-back and behaviour ────────────────

def test_genes_read_back_from_their_consumers(tmp_path):
    _engine_, meta = _meta(tmp_path, enabled=True)
    meta.set_genome({"meta_far_share": 0.4, "meta_min_effect": 0.05,
                     "meta_ablation_n": 80, "meta_mechanism_c": 1.3})
    assert meta.far_share == pytest.approx(0.4)
    assert meta.min_effect == pytest.approx(0.05)
    assert meta.ablation_n == 80
    assert meta.mechanism_c == pytest.approx(1.3)
    # mechanism_c lives on the object that consumes it — the credit table.
    assert meta.credit.mechanism_c == pytest.approx(1.3)


def test_far_share_changes_the_issue_count(tmp_path):
    """Behavioural, not read-back: the quota consumed the value."""
    engine, meta = _meta(tmp_path, enabled=True)
    _warm(engine)
    meta.set_genome({"meta_far_share": 0.5})
    high = len(meta.invent(tick=1))
    engine.candidates.clear()
    meta.archive = type(meta.archive)()      # forget, so the same shapes pass
    for strategy in engine.library.strategies.values():
        meta.archive.add(strategy.name, strategy.steps)
    meta.set_genome({"meta_far_share": 0.17})
    low = len(meta.invent(tick=2))
    assert high > low > 0


def test_mechanism_c_changes_the_order(tmp_path):
    _engine_, meta = _meta(tmp_path, enabled=True)
    features = ("family=arithmetic_chain",)
    for name in FIXED_TRANSFORM_ORDER:
        mechanism = mechanism_of_transform(name)
        attempts = 2 if name == "add_abstain" else 20
        for _ in range(attempts):
            meta.credit.note_attempt(mechanism, features)
    meta.credit.note_accepted(mechanism_of_transform("add_verify"),
                              features, 0.1)
    meta.set_genome({"meta_mechanism_c": 0.0})
    exploit = meta.order_for(type("W", (), {"combo": features})())
    meta.set_genome({"meta_mechanism_c": 2.0})
    explore = meta.order_for(type("W", (), {"combo": features})())
    assert exploit != explore


def test_min_effect_changes_the_confirmation_count():
    edit = Edit(kind="insert", position=0, op="VERIFY")
    measured = {edit.signature(): {"candidate_solved": 45,
                                   "reverted_solved": 30, "n": 60}}
    lenient = attribute_edits((edit,), measured, fdr_q=0.10, min_effect=0.03)
    strict = attribute_edits((edit,), measured, fdr_q=0.10, min_effect=0.30)
    assert lenient[0].confirmed and not strict[0].confirmed


def test_ablation_n_changes_the_sample_size(tmp_path):
    pool = _SpyPool()
    engine, meta = _meta(tmp_path, enabled=True, pool=pool)
    _accepted_strategy(engine, meta)
    meta.set_genome({"meta_ablation_n": 25})
    report = asyncio.run(meta.attribute(tick=1))
    assert all(edit["n"] == 25 for edit in report["edits"])


# ── persistence and eviction (M11.7.8) ───────────────────────────────

def test_state_round_trips_through_the_store(tmp_path):
    engine, meta = _meta(tmp_path, enabled=True)
    _accepted_strategy(engine, meta)
    asyncio.run(meta.attribute(tick=3))
    meta.credit.note_attempt("verification_caught_error", ("f",))
    meta.skeletons.note_failure("vote_of_alternatives", ("f",))
    meta.save()

    engine2 = _engine(tmp_path / "second")
    reloaded = MetaCognition(reasoning=engine2, store_dir=tmp_path / "meta",
                             enabled=True)
    assert [e.strategy for e in reloaded.explanations] \
        == [e.strategy for e in meta.explanations]
    assert reloaded.credit.rows.keys() == meta.credit.rows.keys()
    assert reloaded.skeletons.to_dict() == meta.skeletons.to_dict()


def test_a_future_schema_is_refused(tmp_path):
    import json

    engine, meta = _meta(tmp_path, enabled=True)
    meta.save()
    path = meta.store.explanations_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    reloaded = MetaCognition(reasoning=_engine(tmp_path / "b"),
                             store_dir=tmp_path / "meta", enabled=True)
    assert reloaded.explanations == []


def test_eviction_drops_unsupported_first():
    from aegis.layers.metacognition.attribution import Explanation
    from aegis.layers.metacognition.store import evict

    rows = [Explanation(strategy=f"s{i}", incumbent="direct", weakness="w",
                        gain=0.1, status="supported", created_tick=i)
            for i in range(3)]
    rows.insert(1, Explanation(strategy="cheap", incumbent="direct",
                               weakness="w", gain=0.0, status="unsupported",
                               created_tick=99))
    kept = evict(rows, cap=3)
    assert all(e.strategy != "cheap" for e in kept)
    assert len(kept) == 3


# ── the snapshot (M11.7.1, §M9.4) ────────────────────────────────────

def test_the_snapshot_carries_the_registries(tmp_path):
    engine, meta = _meta(tmp_path, enabled=True)
    _accepted_strategy(engine, meta)
    asyncio.run(meta.attribute(tick=1))
    snapshot = meta.snapshot()
    assert snapshot["explanations"], "explanations are state"
    assert "credit" in snapshot and "retired" in snapshot
    from aegis.util.canonical import digest_of

    assert digest_of(snapshot) == digest_of(meta.snapshot())


# ── determinism (M11.8, criterion 9) ─────────────────────────────────

def test_two_runs_from_one_state_are_byte_identical(tmp_path):
    """The module-level determinism claim: same inputs, same registries."""
    def run(root):
        engine, meta = _meta(root, enabled=True)
        _warm(engine)
        engine.propose_strategy(tick=1)
        meta.invent(tick=1)
        while engine.pending_candidates():
            engine.evaluate_candidate(tick=1)
        meta.on_reflect(1)
        while meta.pending_attributions():
            asyncio.run(meta.attribute(tick=1))
        return meta.snapshot()

    from aegis.util.canonical import digest_of

    assert digest_of(run(tmp_path / "one")) == digest_of(run(tmp_path / "two"))
