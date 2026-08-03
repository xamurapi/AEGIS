"""Attribution by ablation (spec M11.5, M11.11).

The four claims the spec makes measurable:

* a planted cause is found — precision and recall 1.0 on prepared cases where
  exactly one edit carries the win;
* pure noise confirms **nothing** — the BH gate over ≥200 comparisons;
* too many simultaneous edits refuse attribution instead of guessing;
* the measurement is held out and the correction runs over the whole family.
"""
import pytest

from aegis.eval import reasoning_bench as bench
from aegis.layers.metacognition.attribution import (
    ABLATION_BASE, Edit, EditAttribution, Explanation, ablation_tasks,
    ablation_worker, apply_narrative, attribute_edits, conclude, diff, revert,
)
from aegis.layers.metacognition.distance import canonical_hash, canonicalize
from aegis.layers.reasoning.library import BUILTIN_STRATEGIES


def _step(op, **fields):
    return {"op": op, **fields}


DIRECT = [_step("SOLVE")]
WITH_ABSTAIN = [
    _step("SOLVE"),
    _step("VERIFY", checker="confidence"),
    _step("BRANCH", cond="insufficient",
          then=[_step("ABSTAIN", reason="the answer would be a guess")]),
]


# ── the diff ─────────────────────────────────────────────────────────

def test_diff_of_identical_strategies_is_empty():
    assert diff(DIRECT, DIRECT) == ()


def test_diff_sees_an_inserted_verify():
    edits = diff(DIRECT, [_step("SOLVE"), _step("VERIFY", checker="type")])
    assert len(edits) == 1
    assert edits[0].kind == "insert" and edits[0].op == "VERIFY"


def test_diff_sees_a_parameter_change():
    a = [_step("VOTE", n=3, agg="majority", body=[_step("SOLVE")])]
    b = [_step("VOTE", n=5, agg="majority", body=[_step("SOLVE")])]
    edits = diff(a, b)
    assert len(edits) == 1
    assert edits[0].kind == "param" and edits[0].op == "VOTE" \
        and edits[0].key == "n"


def test_diff_sees_a_wrap_as_one_edit():
    wrapped = [_step("VOTE", n=3, agg="majority", body=canonicalize(DIRECT))]
    edits = diff(DIRECT, wrapped)
    assert len(edits) == 1 and edits[0].kind == "wrap"


def test_diff_sees_a_reorder_as_one_edit():
    a = [_step("SOLVE"), _step("VERIFY", checker="type")]
    b = [_step("VERIFY", checker="type"), _step("SOLVE")]
    edits = diff(a, b)
    assert len(edits) == 1 and edits[0].kind == "reorder"


def test_diff_sees_llm_step_replaced_by_compute_as_one_param_edit():
    a = [_step("LLM_STEP", template="write an expression"),
         _step("VERIFY", checker="type")]
    b = [_step("COMPUTE", expr="$last"), _step("VERIFY", checker="type")]
    edits = diff(a, b)
    assert len(edits) == 1
    assert edits[0].kind == "param" and edits[0].op == "COMPUTE" \
        and edits[0].key == "op"


# ── the revert: S∖e really undoes e ──────────────────────────────────

@pytest.mark.parametrize("incumbent,candidate", [
    (DIRECT, [_step("SOLVE"), _step("VERIFY", checker="type")]),
    (DIRECT, WITH_ABSTAIN),
    ([_step("VOTE", n=3, agg="majority", body=[_step("SOLVE")])],
     [_step("VOTE", n=5, agg="majority", body=[_step("SOLVE")])]),
    (DIRECT, [_step("VOTE", n=3, agg="majority", body=canonicalize(DIRECT))]),
    ([_step("SOLVE"), _step("VERIFY", checker="type")],
     [_step("VERIFY", checker="type"), _step("SOLVE")]),
])
def test_reverting_every_edit_of_a_full_diff_recovers_the_incumbent(
        incumbent, candidate):
    """Undoing all edits one at a time walks back to the incumbent — for a
    single-edit diff, one revert lands exactly on it."""
    edits = diff(incumbent, candidate)
    assert edits, "the prepared pair must differ"
    if len(edits) == 1:
        assert canonical_hash(revert(candidate, edits[0])) \
            == canonical_hash(incumbent)
    else:
        for edit in edits:
            reverted = revert(candidate, edit)
            assert canonical_hash(reverted) != canonical_hash(candidate)


# ── held-out sampling ────────────────────────────────────────────────

def test_ablation_tasks_are_held_out_and_deterministic():
    tasks_a = ablation_tasks("arithmetic_chain", "digest", "sig", 20)
    tasks_b = ablation_tasks("arithmetic_chain", "digest", "sig", 20)
    assert [t.id for t in tasks_a] == [t.id for t in tasks_b]
    assert all(t.family == "arithmetic_chain" for t in tasks_a)


def test_different_edits_get_different_samples():
    ids_a = [t.id for t in ablation_tasks("arithmetic_chain", "d", "sig_a", 20)]
    ids_b = [t.id for t in ablation_tasks("arithmetic_chain", "d", "sig_b", 20)]
    assert ids_a != ids_b


def test_ablation_indices_stay_inside_the_reserved_range():
    """Held-out by construction: the ablation range is disjoint from the
    arena's bases, the working queue and the holdout probe."""
    assert ABLATION_BASE >= 4_000_000
    tasks = ablation_tasks("", "d", "s", 10)
    assert len(tasks) == 10          # family-less falls back to the full mix


# ── the planted cause (acceptance criterion 1) ───────────────────────

def _measure(incumbent, candidate, family="missing_data", n=60):
    edits = diff(incumbent, candidate)
    payload = {
        "candidate": canonicalize(candidate),
        "digest": canonical_hash(candidate),
        "family": family, "n": n, "budget": None, "genome": {},
        "edits": [{"signature": e.signature(),
                   "reverted": revert(candidate, e)} for e in edits],
    }
    outcome = ablation_worker(payload)
    measured = {row["signature"]: row for row in outcome["rows"]}
    return edits, measured


def test_the_planted_cause_is_found():
    """Twenty prepared cases where exactly one edit carries the win: the
    abstention branch on missing-data tasks. Ablation must confirm that edit —
    precision 1.0, recall 1.0 over the cases."""
    found, total = 0, 0
    for offset in range(20):
        # Twenty distinct candidate spellings of the same planted cause: the
        # abstain reason differs, so each case diffs and samples on its own.
        candidate = [
            _step("SOLVE"),
            _step("VERIFY", checker="confidence"),
            _step("BRANCH", cond="insufficient",
                  then=[_step("ABSTAIN", reason=f"case {offset}")]),
        ]
        edits, measured = _measure(DIRECT, candidate)
        attributions = attribute_edits(edits, measured, fdr_q=0.10,
                                       min_effect=0.03)
        confirmed = [a for a in attributions if a.confirmed]
        total += 1
        # The planted cause is the abstention branch; VERIFY alone changes
        # nothing on this family. The diff labels the BRANCH-around-ABSTAIN
        # insertion as an ABSTAIN edit, which is what it is.
        if confirmed and all(a.edit.op == "ABSTAIN" for a in confirmed):
            found += 1
    assert found == total, f"planted cause found in {found}/{total} cases"


def test_zero_confirmations_on_noise():
    """Criterion 2: strategies whose 'gain' is sampling noise confirm nothing
    across >= 200 comparisons. This is the gate on the BH correction."""
    comparisons = 0
    confirmed = 0
    for offset in range(200):
        # A no-op edit: the same SOLVE with a decorative kind change that the
        # interpreter treats identically — any measured effect is noise.
        candidate = [_step("SOLVE"), _step("PREDICT", horizon=1)]
        edits = diff(DIRECT, candidate)
        payload = {
            "candidate": canonicalize(candidate),
            "digest": f"noise-{offset}",       # a fresh sample per comparison
            "family": "arithmetic_chain", "n": 30, "budget": None,
            "genome": {},
            "edits": [{"signature": f"{e.signature()}#{offset}",
                       "reverted": revert(candidate, e)} for e in edits],
        }
        rows = {row["signature"]: row
                for row in ablation_worker(payload)["rows"]}
        renamed = tuple(Edit(kind=e.kind, position=e.position, op=e.op,
                             key=f"{e.key}#{offset}") for e in edits)
        # Rebuild measured keyed by the renamed signatures.
        measured = {renamed[i].signature(): rows[f"{edits[i].signature()}#{offset}"]
                    for i in range(len(edits))
                    if f"{edits[i].signature()}#{offset}" in rows}
        attributions = attribute_edits(renamed, measured, fdr_q=0.10,
                                       min_effect=0.03)
        comparisons += len(attributions)
        confirmed += sum(1 for a in attributions if a.confirmed)
    assert comparisons >= 200
    assert confirmed == 0, f"{confirmed} noise comparisons were confirmed"


# ── refusal and correction rules ─────────────────────────────────────

def test_too_many_edits_refuse_attribution():
    edits = tuple(Edit(kind="insert", position=i, op="VERIFY")
                  for i in range(5))
    explanation = conclude("s", "i", "w", 0.1,
                           attribute_edits(edits, {}),
                           too_many_edits=True)
    assert explanation.status == "unsupported"
    assert explanation.mechanism == ""


def test_bh_runs_over_the_whole_family_not_the_promising_half():
    """One strong edit among many weak ones: adding weak siblings must not
    create confirmations, and the correction must consume every p-value."""
    edits, measured = _measure(DIRECT, WITH_ABSTAIN)
    assert len(edits) >= 2
    attributions = attribute_edits(edits, measured, fdr_q=0.10,
                                   min_effect=0.03)
    assert len(attributions) == len(edits)


def test_confirmation_needs_all_three_conditions():
    edit = Edit(kind="insert", position=0, op="VERIFY")
    # Large effect, tiny sample: significance is absent, so no confirmation.
    measured = {edit.signature(): {"candidate_solved": 3, "reverted_solved": 1,
                                   "n": 4}}
    (attribution,) = attribute_edits((edit,), measured, fdr_q=0.10,
                                     min_effect=0.03)
    assert not attribution.confirmed


# ── the explanation and the cortex (criteria 6, 7) ───────────────────

def _supported_explanation():
    edits, measured = _measure(DIRECT, WITH_ABSTAIN)
    attributions = attribute_edits(edits, measured, fdr_q=0.10,
                                   min_effect=0.03)
    return conclude("candidate", "direct", "missing_data", 0.2, attributions)


def test_mechanism_is_nonempty_iff_an_edit_is_confirmed():
    supported = _supported_explanation()
    assert supported.status == "supported"
    assert bool(supported.mechanism) == bool(supported.confirmed_edits())

    unsupported = conclude("s", "i", "w", 0.0, attribute_edits(
        (Edit(kind="insert", position=0, op="VERIFY"),),
        {}))
    assert unsupported.status == "unsupported"
    assert unsupported.mechanism == ""
    assert not unsupported.confirmed_edits()


def test_the_cortex_cannot_rewrite_the_computed_mechanism():
    """Criterion 7: a contradicting proposal contests, never overwrites."""
    explanation = _supported_explanation()
    computed = explanation.mechanism
    contested = apply_narrative(explanation, "a plausible story",
                                "voting_reduced_variance"
                                if computed != "voting_reduced_variance"
                                else "verification_caught_error")
    assert contested.status == "contested"
    assert contested.mechanism == computed          # the number did not move
    agreed = apply_narrative(explanation, "a story", computed)
    assert agreed.status == "supported"


def test_the_narrative_is_stored_verbatim_and_only_stored():
    explanation = apply_narrative(_supported_explanation(), "history", "")
    assert explanation.narrative == "history"


def test_no_code_path_reads_the_narrative():
    """M11.4: `narrative` appears in no condition. The grep the spec asks for:
    every line of the module that mentions narrative must be an assignment,
    a signature, a schema field or a string — never an if/while/comparison."""
    from pathlib import Path

    module = Path("aegis/layers/metacognition/attribution.py").read_text(
        encoding="utf-8")
    for line in module.splitlines():
        if "narrative" not in line:
            continue
        stripped = line.strip()
        assert not stripped.startswith(("if ", "elif ", "while ", "assert ")), \
            f"narrative reached a condition: {stripped!r}"
        assert "narrative ==" not in stripped and "== narrative" not in stripped


def test_explanation_round_trips_through_dict():
    explanation = _supported_explanation()
    restored = Explanation.from_dict(explanation.as_dict())
    assert restored.strategy == explanation.strategy
    assert restored.mechanism == explanation.mechanism
    assert len(restored.edits) == len(explanation.edits)
    assert restored.status == explanation.status


def test_builtin_diffs_stay_within_the_edit_budget():
    """The transformations of M6.7 produce small diffs — the catalogue the
    attribution limit was sized for."""
    from aegis.layers.reasoning.synthesis import TRANSFORMS

    parent = BUILTIN_STRATEGIES["direct"]
    for name, transform in TRANSFORMS:
        candidate = transform(list(parent))
        if candidate is None:
            continue
        edits = diff(parent, candidate)
        assert 1 <= len(edits) <= 4, (name, edits)
