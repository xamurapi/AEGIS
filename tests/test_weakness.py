"""The weakness detector (spec M6.6, M6.11).

The named test of the spec is the first one: **zero weaknesses on uniform
noise**. A detector that finds structure in a system failing at random would
send the synthesiser after nothing, and every strategy it produced would be
accepted or rejected by chance. Everything else here is about the three things
that keep that from happening — a base rate, false-discovery control, and
pruning combinations that only repeat their parents.
"""
import pytest

from aegis.layers.reasoning.weakness import (
    EXAMPLES, MIN_SUPPORT, Weakness, WeaknessDetector, labels_of,
)
from aegis.util.quasirandom import hash_index


@pytest.fixture
def detector():
    return WeaknessDetector()


def _row(index, *, solved, family="alpha", **features):
    return {"task": f"t{index}", "family": family, "solved": solved,
            "features": features}


def _uniform(count, fail_every=3):
    """Rows whose failures fall independently of every feature.

    The failure pattern is a hash of the index, so it is fixed across runs and
    spread evenly over every feature combination — which is what "noise" has to
    mean for this test to be about the detector rather than about a seed.
    """
    rows = []
    for index in range(count):
        rows.append(_row(
            index,
            solved=hash_index(fail_every, "noise", index) != 0,
            family=("alpha", "beta", "gamma")[index % 3],
            numeric=bool(index % 2),
            steps=index % 4,
            incomplete=bool(index % 5 == 0),
        ))
    return rows


# ── the named test ───────────────────────────────────────────────────

def test_uniform_noise_has_no_weaknesses(detector):
    """A system failing at random is not weak anywhere in particular."""
    assert detector.scan(_uniform(600)) == []


def test_uniform_noise_at_several_failure_rates_still_has_none(detector):
    for fail_every in (2, 3, 5, 10):
        assert detector.scan(_uniform(600, fail_every)) == [], fail_every


def test_a_system_that_fails_at_everything_has_no_weakness(detector):
    """Failing everywhere is a level, not a weakness. There is nothing here for
    a synthesiser to aim at that is not simply "be better"."""
    rows = [_row(index, solved=False, family=("alpha", "beta")[index % 2])
            for index in range(200)]
    assert detector.scan(rows) == []


def test_a_system_that_fails_at_nothing_has_no_weakness(detector):
    assert detector.scan([_row(index, solved=True) for index in range(200)]) == []


# ── it finds what is really there ────────────────────────────────────

def test_a_genuinely_weak_feature_is_found(detector):
    rows = _uniform(400)
    rows += [_row(1000 + index, solved=False, family="delta", brittle=True)
             for index in range(60)]
    found = detector.scan(rows)
    assert found
    assert any("brittle" in weakness.combo for weakness in found)


def test_the_weakness_carries_its_evidence(detector):
    rows = _uniform(400)
    rows += [_row(1000 + index, solved=False, family="delta", brittle=True)
             for index in range(60)]
    weakness = next(w for w in detector.scan(rows) if "brittle" in w.combo)
    assert weakness.support == 60 and weakness.fails == 60
    assert weakness.fail_rate == 1.0
    assert weakness.base_rate < 1.0 and weakness.excess > 0
    assert 0.0 < weakness.lower <= 1.0
    assert len(weakness.examples) == EXAMPLES


def test_a_weakness_that_belongs_to_one_family_names_it(detector):
    rows = _uniform(400)
    rows += [_row(1000 + index, solved=False, family="delta", brittle=True)
             for index in range(60)]
    weakness = next(w for w in detector.scan(rows) if "brittle" in w.combo)
    assert weakness.family == "delta"


def test_a_weakness_spanning_families_names_none(detector):
    """Which is what tells the arena to judge it on the general set rather than
    on one class."""
    rows = _uniform(400)
    rows += [_row(1000 + index, solved=False, brittle=True,
                  family=("delta", "epsilon")[index % 2]) for index in range(60)]
    weakness = next(w for w in detector.scan(rows) if "brittle" in w.combo)
    assert weakness.family == ""


def test_the_worst_weakness_ranks_first(detector):
    rows = _uniform(600)
    rows += [_row(2000 + index, solved=False, family="delta", small=True)
             for index in range(20)]
    rows += [_row(3000 + index, solved=False, family="epsilon", large=True)
             for index in range(100)]
    found = detector.scan(rows)
    # By rank — volume times excess — so the bigger loss comes first even
    # though both groups fail outright. Which of the co-extensive labels
    # survives pruning ("large" or "family=epsilon") is not the claim; they
    # describe the same 100 attempts.
    assert found[0].support == 100 and found[0].fail_rate == 1.0
    assert found[0].rank == max(weakness.rank for weakness in found)
    assert any(weakness.support == 20 for weakness in found)


# ── the guards ───────────────────────────────────────────────────────

def test_a_group_below_the_support_floor_is_not_reported(detector):
    rows = _uniform(400)
    rows += [_row(1000 + index, solved=False, family="delta", rare=True)
             for index in range(MIN_SUPPORT - 1)]
    assert not any("rare" in weakness.combo for weakness in detector.scan(rows))


def test_too_few_rows_to_say_anything_says_nothing(detector):
    assert detector.scan([_row(0, solved=False)]) == []


def test_a_group_covering_everything_is_not_a_weakness(detector):
    """It has no "rest" to be compared against, and a comparison with itself is
    not a comparison."""
    rows = [_row(index, solved=index % 3 != 0, family="alpha")
            for index in range(120)]
    assert not any(weakness.combo == ("family=alpha",)
                   for weakness in detector.scan(rows))


def test_a_specialisation_that_only_repeats_its_parent_is_dropped(detector):
    """Otherwise the synthesiser is sent after one problem twice and the
    evidence for it is split in half."""
    rows = _uniform(400)
    rows += [_row(1000 + index, solved=False, family="delta", brittle=True,
                  numeric=True) for index in range(60)]
    found = detector.scan(rows)
    combos = {weakness.combo for weakness in found}
    assert ("brittle", "numeric") not in combos or \
        all(len(combo) == 1 for combo in combos if "brittle" in combo)


def test_a_specialisation_is_dropped_by_a_two_label_parent_too():
    """The same pruning one level down, where it is actually reachable.

    At ``max_combo=2`` a specialisation is only ever checked against single
    labels, and sorting a one-tuple cannot reorder it — so a parent lookup keyed
    differently from the group it stores would still find every parent it looks
    for, and the mismatch would never show. At three the check reaches pairs.

    Here ``family=delta AND aa`` fails every one of its 120 attempts, and the
    triple that adds ``bb`` to it fails every one of its 60 — the same rate on
    half the evidence, which is the definition of a specialisation that only
    repeats its parent. Its single-label parents all sit near 0.6, so nothing
    shorter can drop it: only the pair can, and only if the pair can be found.
    """
    rows = [_row(index, solved=False, family="delta", aa=True, bb=True)
            for index in range(60)]
    rows += [_row(100 + index, solved=False, family="delta", aa=True)
             for index in range(60)]
    rows += [_row(200 + index, solved=hash_index(3, "d", index) != 0,
                  family="delta") for index in range(180)]
    rows += [_row(400 + index, solved=hash_index(3, "e", index) != 0,
                  family="alpha", aa=True, bb=True) for index in range(180)]
    rows += [_row(600 + index, solved=hash_index(3, "f", index) != 0,
                  family="beta", bb=True) for index in range(180)]

    found = WeaknessDetector(max_combo=3).scan(rows)
    assert not any(len(weakness.combo) == 3 for weakness in found), (
        "a three-label weakness survived the pair that already explains it: "
        + ", ".join(w.label for w in found if len(w.combo) == 3))


def test_labels_are_ordered_independently_of_how_the_row_was_built():
    """Group keys are built by ``combinations`` over this tuple, so its order is
    the order of every key. Two rows carrying the same labels must produce the
    same keys whatever order the feature mapping happened to be in (§3.1)."""
    one = labels_of({"family": "alpha",
                     "features": {"zeta": True, "alpha_feature": True}})
    two = labels_of({"family": "alpha",
                     "features": {"alpha_feature": True, "zeta": True}})
    assert one == two == tuple(sorted(one))


def test_only_the_recent_window_is_scanned(detector):
    old = [_row(index, solved=False, family="delta", brittle=True)
           for index in range(60)]
    recent = _uniform(400)
    assert not any("brittle" in weakness.combo
                   for weakness in detector.scan(old + recent, window=400))


def test_false_discovery_control_can_be_loosened_and_tightened():
    rows = _uniform(400)
    rows += [_row(1000 + index, solved=False, family="delta", brittle=True)
             for index in range(20)]
    strict = WeaknessDetector(alpha=1e-12).scan(rows)
    loose = WeaknessDetector(alpha=0.5).scan(rows)
    assert len(strict) <= len(loose)


def test_how_many_features_may_be_combined_is_a_limit():
    rows = _uniform(400)
    detector = WeaknessDetector(max_combo=1)
    assert all(len(weakness.combo) == 1
               for weakness in detector.scan(rows + [
                   _row(1000 + i, solved=False, family="delta", a=True, b=True)
                   for i in range(60)]))


# ── labels ───────────────────────────────────────────────────────────

def test_a_true_flag_is_a_label_and_a_false_one_is_not():
    """A label is a thing the task *has*. "not incomplete" is the absence of a
    property, and grouping by absences would double every axis."""
    labels = labels_of({"family": "alpha",
                        "features": {"incomplete": True, "numeric": False}})
    assert "incomplete" in labels and "numeric" not in labels


def test_a_value_becomes_a_named_label():
    labels = labels_of({"family": "alpha", "features": {"steps": 3}})
    assert "steps=3" in labels and "family=alpha" in labels


def test_a_list_of_operations_becomes_one_label_each():
    labels = labels_of({"features": {"ops": ["compute", "verify"]}})
    assert "op:compute" in labels and "op:verify" in labels


def test_the_system_state_is_an_axis_too():
    """The same problem attempted while out of energy is a different situation,
    and a weakness that only appears there is a real one (M6.6)."""
    assert "state=lo|hi" in labels_of({"features": {}, "state": "lo|hi"})


def test_labels_do_not_repeat():
    labels = labels_of({"family": "alpha", "features": {"ops": ["a", "a"]}})
    assert len(labels) == len(set(labels))


def test_a_row_with_nothing_in_it_has_no_labels():
    assert labels_of({}) == ()


# ── reporting ────────────────────────────────────────────────────────

def test_a_weakness_renders_as_data(detector):
    import json

    rows = _uniform(400) + [_row(1000 + i, solved=False, family="delta",
                                 brittle=True) for i in range(60)]
    weakness = detector.scan(rows)[0]
    rendered = json.loads(json.dumps(weakness.as_dict()))
    assert rendered["label"] == weakness.label
    assert rendered["support"] == weakness.support


def test_the_label_reads_as_a_conjunction():
    assert Weakness(combo=("a", "b"), fail_rate=1.0, base_rate=0.0, support=1,
                    fails=1, lower=0.0, excess=1.0, p_value=0.0,
                    rank=1.0).label == "a AND b"


def test_a_weakness_cannot_be_edited_after_it_is_reported(detector):
    """It is handed to the synthesiser, kept on the candidate that came out,
    and read again by the arena cycles later. A caller able to widen its
    excess could get any candidate accepted."""
    rows = _uniform(400) + [_row(1000 + i, solved=False, family="delta",
                                 brittle=True) for i in range(60)]
    weakness = detector.scan(rows)[0]
    with pytest.raises(Exception):
        weakness.excess = 1.0


def test_the_excess_is_measured_against_the_rest_not_against_everything(detector):
    """A large group is part of its own base rate; comparing it with a total it
    dominates hides exactly the biggest weaknesses."""
    rows = [_row(index, solved=index % 10 != 0, family="alpha")
            for index in range(200)]
    rows += [_row(1000 + index, solved=False, family="delta", brittle=True)
             for index in range(150)]
    weakness = next(w for w in detector.scan(rows) if "brittle" in w.combo)
    assert weakness.base_rate == pytest.approx(0.1, abs=0.02)
    assert weakness.excess == pytest.approx(0.9, abs=0.02)


def test_rank_is_volume_times_excess(detector):
    """Not the failure rate alone: a class that fails outright twenty times is
    a smaller loss than one that fails half the time in a thousand."""
    rows = _uniform(600)
    rows += [_row(2000 + index, solved=False, family="delta", small=True)
             for index in range(20)]
    found = detector.scan(rows)
    weakness = next(w for w in found if w.support == 20)
    assert weakness.rank == pytest.approx(weakness.support * weakness.excess)


def test_a_specialisation_that_fails_more_than_its_parent_is_kept(detector):
    """Pruning removes repetition, not detail. A combination that really is
    worse than every part of it is the narrower target worth having."""
    rows = _uniform(400)
    # `brittle` fails 60% overall; `brittle AND numeric` fails outright.
    rows += [_row(1000 + index, solved=index % 5 >= 3, family="delta",
                  brittle=True) for index in range(100)]
    rows += [_row(2000 + index, solved=False, family="delta", brittle=True,
                  numeric=True) for index in range(60)]
    found = detector.scan(rows)
    combos = {weakness.combo for weakness in found}
    assert ("brittle", "numeric") in combos


def test_status_reports_what_was_tested(detector):
    detector.scan(_uniform(400))
    status = detector.status()
    assert status["scans"] == 1 and status["tested"] > 0
