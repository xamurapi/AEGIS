"""Proposing hypotheses, and controlling how often we are wrong (spec M7.4, M7.11).

The named test of the spec is **no discoveries on noise**. It is the one that
decides whether this contour produces knowledge or produces confident nonsense:
a scan over a dozen variables at six lags under three measures performs
hundreds of tests, and at α = 0.05 roughly one in twenty of the useless ones
clears an uncorrected threshold. A system that mined without correction would
never fail to find a "law", and would find one just as readily in a series of
hashes as in its own behaviour.

The second claim is quieter and just as load-bearing: the correction is applied
over **every comparison made**, not over the ones that looked promising.
Filtering first and correcting afterwards yields a number that looks like a
controlled error rate and is not one.
"""
import pytest

from aegis.layers.discovery.datapool import Frame
from aegis.layers.discovery.hypothesis import (
    MAX_JOINT_PREDICTORS, MEASURES, Hypothesis, Scanner, from_formal,
    from_world_model, hypothesis_id, parse_formal,
)
from aegis.util.quasirandom import hash_unit


@pytest.fixture
def scanner():
    return Scanner()


def _noise(count=400, variables=6):
    """Rows whose columns are independent of each other by construction.

    Derived from a hash rather than a generator so the test is the same on
    every run — a "no false discoveries" test that depended on a seed would be
    a test of the seed.
    """
    return Frame.from_rows([
        {"tick": index,
         **{f"v{k}": hash_unit("noise", k, index) for k in range(variables)}}
        for index in range(count)])


def _linked(count=400, slope=3.0, noise=0.05):
    rows = []
    for index in range(count):
        x = hash_unit("x", index)
        rows.append({"tick": index, "x": x,
                     "y": slope * x + noise * (hash_unit("e", index) - 0.5),
                     "z": hash_unit("z", index)})
    return Frame.from_rows(rows)


# ── the named test ───────────────────────────────────────────────────

def test_pure_noise_yields_no_hypotheses(scanner):
    """The test the whole contour stands on (M7.10)."""
    assert scanner.scan(_noise(), "v0") == []
    assert scanner.tested > 50, "the scan has to have actually tested something"


def test_noise_is_tested_and_rejected_rather_than_skipped(scanner):
    scanner.scan(_noise(), "v0")
    assert scanner.rejected == scanner.tested


@pytest.mark.parametrize("variables", [3, 6, 9])
def test_widening_the_search_still_finds_nothing_in_noise(variables):
    """More variables means more tests, which is exactly when an uncorrected
    scan starts producing findings. The correction has to hold as it widens."""
    assert Scanner().scan(_noise(variables=variables), "v0") == []


# ── it does find what is there ───────────────────────────────────────

def test_a_real_relationship_is_found(scanner):
    """The complement of the noise test. A scan that found nothing anywhere
    would pass the test above and be useless."""
    found = scanner.scan(_linked(), "y")
    assert any("x" in item.predictors for item in found)


def test_the_count_of_tests_covers_every_comparison_made(scanner):
    """Not just the surviving ones — otherwise the false-discovery rate is
    computed against a denominator chosen after seeing the answers."""
    scanner.scan(_linked(), "y")
    assert scanner.tested >= len(MEASURES) * 2


def test_every_hypothesis_carries_its_evidence(scanner):
    item = scanner.scan(_linked(), "y")[0]
    assert item.p_value <= 1.0 and item.strength != 0.0
    assert item.target == "y" and item.formal.startswith("y ~ f(")


def test_the_same_relationship_keeps_the_same_identity():
    """A rescan must recognise what it already tested, or the refuted archive
    matches nothing and every scan rediscovers what it just rejected."""
    first = Scanner().scan(_linked(), "y")
    second = Scanner().scan(_linked(), "y")
    assert [item.id for item in first] == [item.id for item in second]


def test_an_identity_is_derived_from_content_not_from_order():
    assert hypothesis_id("y", ["a", "b"], {"a": 1, "b": 0}) == \
        hypothesis_id("y", ["b", "a"], {"b": 0, "a": 1})


def test_a_different_lag_is_a_different_hypothesis():
    assert hypothesis_id("y", ["a"], {"a": 1}) != hypothesis_id("y", ["a"], {"a": 2})


# ── the joint hypothesis ─────────────────────────────────────────────

def test_two_surviving_predictors_are_also_offered_together():
    """A law over two variables is not the sum of two laws over one. Without
    this the symbolic search is never shown both at once and recovers the wrong
    formula while explaining most of the variance."""
    rows = []
    for index in range(400):
        a, b = hash_unit("a", index), hash_unit("b", index)
        rows.append({"tick": index, "a": a, "b": b,
                     "y": 2.5 * a - b * b + 0.02 * (hash_unit("n", index) - 0.5)})
    found = Scanner().scan(Frame.from_rows(rows), "y")
    joint = [item for item in found if len(item.predictors) > 1]
    assert joint, "the survivors were never offered as one hypothesis"
    assert set(joint[0].predictors) == {"a", "b"}
    assert joint[0] is found[0], "the joint hypothesis should be ranked first"


def test_a_single_survivor_produces_no_joint_hypothesis():
    """One predictor at one lag is already its own hypothesis. Emitting a
    "joint" hypothesis over a set of one would spend a second fit proving the
    same thing."""
    rows = [{"tick": index, "a": float(index % 7)} for index in range(200)]
    for row in rows:                      # y depends on a and on nothing else
        row["y"] = 4.0 * row["a"]
    found = Scanner(max_lag=0).scan(Frame.from_rows(rows), "y")
    assert found, "the real relationship was not found at all"
    assert all(len(item.predictors) == 1 for item in found)


def test_the_joint_hypothesis_is_capped():
    rows = []
    for index in range(500):
        parts = [hash_unit(f"p{k}", index) for k in range(6)]
        rows.append({"tick": index,
                     **{f"p{k}": parts[k] for k in range(6)},
                     "y": sum(parts) + 0.01 * (hash_unit("n", index) - 0.5)})
    found = Scanner().scan(Frame.from_rows(rows), "y")
    for item in found:
        assert len(item.predictors) <= MAX_JOINT_PREDICTORS


def test_the_joint_hypothesis_adds_no_test_to_the_family():
    """It is a different question about comparisons already made and corrected,
    not a new comparison — so it must not inflate the denominator."""
    rows = []
    for index in range(400):
        a, b = hash_unit("a", index), hash_unit("b", index)
        rows.append({"tick": index, "a": a, "b": b, "y": 2.0 * a + 3.0 * b})
    scanner = Scanner()
    scanner.scan(Frame.from_rows(rows), "y")
    # Two predictors × (max_lag + 1) lags × three measures.
    assert scanner.tested == 2 * (scanner.max_lag + 1) * len(MEASURES)


# ── shape and guards ─────────────────────────────────────────────────

def test_too_few_rows_are_not_scanned(scanner):
    assert scanner.scan(_noise(count=10), "v0") == []


def test_a_frame_with_only_the_target_has_nothing_to_relate_it_to(scanner):
    frame = Frame.from_rows([{"tick": i, "y": float(i)} for i in range(100)])
    assert scanner.scan(frame, "y") == []


def test_the_tick_column_is_never_a_predictor(scanner):
    """Everything drifts with tick, so tick correlates with everything — and a
    "law" saying reward depends on how long the system has been running is a
    restatement of the drift, not a finding."""
    frame = Frame.from_rows([{"tick": i, "y": float(i), "v": hash_unit("v", i)}
                             for i in range(200)])
    assert all("tick" not in item.predictors for item in scanner.scan(frame, "y"))


def test_a_hypothesis_round_trips_through_a_dict():
    item = Scanner().scan(_linked(), "y")[0]
    assert Hypothesis.from_dict(item.as_dict()).id == item.id


@pytest.mark.parametrize("bad", [None, {}, {"no_id": 1}, "text"])
def test_a_malformed_record_is_not_a_hypothesis(bad):
    assert Hypothesis.from_dict(bad) is None


def test_a_record_with_unusable_numbers_is_not_a_hypothesis():
    assert Hypothesis.from_dict({"id": "x", "prior": "high"}) is None


def test_the_status_reports_the_denominator(scanner):
    scanner.scan(_noise(), "v0")
    assert scanner.status()["tested"] == scanner.tested


# ── the grammar (the cortex path) ────────────────────────────────────

def test_a_well_formed_statement_parses():
    assert parse_formal("y ~ f(a@lag1, b)", ["y", "a", "b"]) == \
        ("y", ("a", "b"), {"a": 1, "b": 0})


@pytest.mark.parametrize("text", [
    "y ~ f(unknown)",              # a variable that does not exist
    "unknown ~ f(a)",              # a target that does not exist
    "y ~ f(y)",                    # the target predicting itself
    "y ~ f(a@lag99)",              # a lag past the configured maximum
    "y ~ f(a, a)",                 # the same predictor twice
    "y ~ f()",                     # nothing to predict with
    "y = 2*a",                     # not the grammar
    "y ~ g(a)",                    # not the grammar
    "",
])
def test_anything_outside_the_grammar_is_refused(text):
    """This string decides what gets fitted and then experimented on. A model
    that could name an arbitrary expression would be choosing the engine's next
    action in prose."""
    assert parse_formal(text, ["y", "a", "b"]) is None


def test_a_hypothesis_can_be_built_from_a_formal_statement():
    item = from_formal("y ~ f(a@lag2)", ["y", "a"], tick=7)
    assert item is not None
    assert item.predictors == ("a",) and item.lags == {"a": 2}
    assert item.origin == "cortex" and item.created_tick == 7


def test_a_statement_outside_the_grammar_builds_no_hypothesis():
    assert from_formal("y ~ f(ghost)", ["y", "a"]) is None


# ── the theory path ──────────────────────────────────────────────────

class _WorldModel:
    def __init__(self, links):
        self._links = links

    def strongest_links(self, limit):
        return list(self._links)[:limit]


def test_a_strong_causal_link_becomes_a_hypothesis():
    """A link the model learned is a claim that has never been tested as such.
    Promoting it is how a belief becomes something that can be refuted."""
    model = _WorldModel([{"cause": "a", "effect": "y", "strength": 0.8}])
    found = from_world_model(model, ["a", "y"], tick=3)
    assert len(found) == 1
    assert found[0].origin == "theory" and found[0].kind == "causal"
    assert found[0].lags == {"a": 1}


def test_a_link_naming_an_undeclared_variable_is_skipped():
    model = _WorldModel([{"cause": "ghost", "effect": "y", "strength": 0.9}])
    assert from_world_model(model, ["a", "y"]) == []


def test_a_link_from_something_to_itself_is_skipped():
    model = _WorldModel([{"cause": "y", "effect": "y", "strength": 0.9}])
    assert from_world_model(model, ["y"]) == []


def test_a_world_model_that_raises_yields_nothing():
    class _Broken:
        def strongest_links(self, limit):
            raise RuntimeError("no links")

    assert from_world_model(_Broken(), ["a", "y"]) == []


def test_the_production_world_model_feeds_the_theory_path(tmp_path):
    """Against the real `PredictiveWorldModel`, not a fake.

    Every earlier test here defined `strongest_links` on its stand-in — and the
    production facade did not have the method, so in the running system this
    source was permanently dead: the AttributeError vanished into the blanket
    `except` and the theory path returned [] forever while this suite passed.
    """
    from aegis.layers.world_model import PredictiveWorldModel

    model = PredictiveWorldModel(store_path=tmp_path / "model.json")
    for _ in range(4):
        model.observe("a", "y", success=True)

    found = from_world_model(model, ["a", "y"], tick=5)
    assert len(found) == 1
    assert found[0].target == "y" and found[0].predictors == ("a",)
    assert found[0].origin == "theory"


def test_an_absent_strongest_links_api_fails_loudly():
    """A model that *raises* is a world-model problem the scan may shrug off;
    a model that lacks the API is a wiring bug in this codebase, and hiding it
    behind an empty list is how the theory source died the first time."""
    class _NoSuchApi:
        causal = object()          # no strongest_links anywhere

    with pytest.raises(AttributeError):
        from_world_model(_NoSuchApi(), ["a", "y"])


def test_the_theory_path_is_capped():
    model = _WorldModel([{"cause": f"c{k}", "effect": "y", "strength": 0.9}
                         for k in range(20)])
    known = ["y"] + [f"c{k}" for k in range(20)]
    assert len(from_world_model(model, known, limit=3)) == 3


# ── the statement, which is what an operator actually reads ──────────
#
# A hypothesis carries a human-readable sentence alongside its formal form, and
# nothing downstream parses it — which is exactly why it needs testing. A
# statement that said "rises with" about a falling relationship, or "at the same
# tick" about a lagged one, would be wrong in the one place a person looks and
# right everywhere a machine does.

def test_a_same_tick_relationship_says_so():
    rows = [{"tick": index, "a": float(index % 7)} for index in range(200)]
    for row in rows:
        row["y"] = 4.0 * row["a"]
    found = Scanner(max_lag=0).scan(Frame.from_rows(rows), "y")
    assert found and "at the same tick" in found[0].statement


def test_a_lagged_relationship_says_how_far_back():
    rows = []
    for index in range(300):
        rows.append({"tick": index, "a": float(index % 11)})
    for index, row in enumerate(rows):
        # y follows a from two ticks earlier, and nothing else.
        row["y"] = 4.0 * rows[max(0, index - 2)]["a"]
    found = Scanner().scan(Frame.from_rows(rows), "y")
    lagged = [item for item in found
              if len(item.predictors) == 1 and item.lags.get("a")]
    assert lagged, "the lag was never found"
    assert "tick(s) earlier" in lagged[0].statement
    assert "at the same tick" not in lagged[0].statement


def test_a_falling_relationship_is_described_as_falling():
    rows = [{"tick": index, "a": float(index % 9)} for index in range(200)]
    for row in rows:
        row["y"] = -3.0 * row["a"]
    found = Scanner(max_lag=0).scan(Frame.from_rows(rows), "y")
    correlational = [item for item in found if item.measure in ("pearson", "spearman")]
    assert correlational, "no correlation survived to describe"
    assert "falls as" in correlational[0].statement


def test_a_rising_relationship_is_described_as_rising():
    rows = [{"tick": index, "a": float(index % 9)} for index in range(200)]
    for row in rows:
        row["y"] = 3.0 * row["a"]
    found = Scanner(max_lag=0).scan(Frame.from_rows(rows), "y")
    correlational = [item for item in found if item.measure in ("pearson", "spearman")]
    assert correlational and "rises with" in correlational[0].statement


def test_a_mutual_information_finding_is_described_as_information():
    """MI is not a direction, so describing it as one would be a lie about the
    only measure that sees a U-shape."""
    from aegis.layers.discovery.hypothesis import _statement

    assert "mutual information" in _statement("y", "a", 0, "mi", 0.42)
    assert "rises with" not in _statement("y", "a", 0, "mi", 0.42)
    assert "mutual information" not in _statement("y", "a", 0, "pearson", 0.42)


# ── the measure that gets reported ───────────────────────────────────

def test_each_measure_dispatches_to_its_own_estimator():
    """Three measures answering three different questions. A dispatch that
    reached the wrong one would report a Spearman number under a Pearson name
    and nothing downstream could tell."""
    from aegis.layers.discovery.statistics import (
        mutual_information, pearson, spearman,
    )

    xs = [float(value) for value in range(1, 41)]
    ys = [x ** 3 for x in xs]
    scanner = Scanner()
    assert scanner._measure("pearson", xs, ys) == pearson(xs, ys)
    assert scanner._measure("spearman", xs, ys) == spearman(xs, ys)
    assert scanner._measure("mi", xs, ys) == mutual_information(xs, ys)
    # Not interchangeable: Spearman sees the cube exactly, Pearson does not.
    assert scanner._measure("spearman", xs, ys)[0] > \
        scanner._measure("pearson", xs, ys)[0]


def test_the_strongest_measure_for_a_pair_is_the_one_reported():
    """Three measures test one relationship. The hypothesis carries the most
    significant of them — keeping the weakest would understate every finding
    and make the rank order meaningless."""
    from aegis.layers.discovery.statistics import (
        mutual_information, pearson, spearman,
    )

    rows = [{"tick": index, "a": float(index % 13)} for index in range(300)]
    for row in rows:
        row["y"] = row["a"] ** 3
    frame = Frame.from_rows(rows)
    found = Scanner(max_lag=0).scan(frame, "y")
    reported = next(item for item in found if item.predictors == ("a",))

    xs, ys = frame.column("a"), frame.column("y")
    best = min(pearson(xs, ys)[1], spearman(xs, ys)[1], mutual_information(xs, ys)[1])
    assert reported.p_value == pytest.approx(best)


# ── the record survives a round trip intact ──────────────────────────

def test_a_hypothesis_is_frozen_once_it_exists():
    """Its identity is derived from its content. A hypothesis whose content
    could be edited afterwards would answer to an id that no longer describes
    it, and the refuted archive would stop matching."""
    item = Scanner().scan(_linked(), "y")[0]
    with pytest.raises(Exception):
        item.target = "something else"


def test_a_round_trip_preserves_the_predictors_and_their_lags():
    """Not just the id. A record that came back with its predictors emptied
    would be a hypothesis about nothing that still had a name."""
    original = Hypothesis(
        id="hyp_x", statement="s", formal="y ~ f(a@lag1, b@lag0)", target="y",
        predictors=("a", "b"), lags={"a": 1, "b": 0}, kind="law", prior=0.7,
        origin="cortex", created_tick=5, measure="pearson", strength=-0.4,
        p_value=0.001)
    restored = Hypothesis.from_dict(original.as_dict())
    assert restored.predictors == ("a", "b")
    assert restored.lags == {"a": 1, "b": 0}
    assert restored.target == "y" and restored.kind == "law"
    assert restored.strength == pytest.approx(-0.4)


def test_a_record_with_no_predictors_round_trips_as_empty():
    restored = Hypothesis.from_dict({"id": "hyp_y", "target": "y"})
    assert restored.predictors == () and restored.lags == {}


def test_a_formal_hypothesis_keeps_the_statement_it_was_given():
    """And falls back to a generated one when there is none — the fallback is
    what an operator reads when a model proposed the hypothesis without
    bothering to describe it."""
    given = from_formal("y ~ f(a)", ["y", "a"], statement="a drives y, we think")
    assert given.statement == "a drives y, we think"

    generated = from_formal("y ~ f(a)", ["y", "a"])
    assert generated.statement and "y" in generated.statement
    assert "a" in generated.statement


def test_a_joint_hypothesis_names_each_variable_once():
    """``best`` is keyed by (name, lag), so a predictor that survives at three
    lags appears three times. Taking the top three as they come produced
    ``y ~ f(a@lag5, a@lag5, a@lag5)``: duplicated predictors, a lag mapping
    collapsed to one entry, and a fit handed the same column three times.
    """
    rows = [{"tick": index, "a": float(index % 11)} for index in range(300)]
    for index, row in enumerate(rows):
        row["y"] = 4.0 * rows[max(0, index - 2)]["a"]
    found = Scanner().scan(Frame.from_rows(rows), "y")
    for item in found:
        assert len(item.predictors) == len(set(item.predictors)), item.formal
        assert len(item.lags) == len(item.predictors), item.formal
