"""Recovering a formula, not a score (spec M7.5, M7.11).

The acceptance test of the whole contour is here: a known law is planted in the
data and the search has to write it down. ``R²`` alone would not prove that —
a large enough expression reproduces anything — so the tests check the *terms*
as well as the fit, and check that BIC actually charges for size.

The other claim is determinism. Genetic programming is the usual tool for
symbolic regression and it is stochastic; §3.1 forbids that outright, and the
reason is not tidiness. A discovery that cannot be rederived from the same data
cannot be replicated, and replication is what turns a supported result into a
law.
"""
import math

import pytest

from aegis.layers.discovery.datapool import Frame
from aegis.layers.discovery.symbolic import (
    MAX_TERMS, Model, build_library, fit, predict, solve_least_squares,
)
from aegis.util.quasirandom import hash_unit


def _frame(law, count=300, noise=0.0, variables=("x1", "x2")):
    rows = []
    for index in range(count):
        values = {name: -5.0 + 10.0 * hash_unit(name, index)
                  for name in variables}
        wobble = noise * (hash_unit("noise", index) - 0.5)
        rows.append({"tick": index, **values, "y": law(values) + wobble})
    return Frame.from_rows(rows)


# ── the acceptance case (M7.10) ──────────────────────────────────────

def test_the_planted_law_is_recovered():
    """``y = 2.5·x₁ − x₂²`` — the spec's own example. The engine has to produce
    the formula, not merely a good fit: the terms are checked as well as R²."""
    frame = _frame(lambda v: 2.5 * v["x1"] - v["x2"] ** 2, noise=0.1)
    model = fit(frame, "y", ["x1", "x2"])

    assert model is not None
    assert model.r2_valid >= 0.9, f"R²_valid was {model.r2_valid}"
    assert set(model.terms) == {"x1", "x2^2"}
    assert model.params[0] == pytest.approx(2.5, abs=0.05)
    assert model.params[1] == pytest.approx(-1.0, abs=0.05)


def test_the_recovered_law_is_written_down_as_a_formula():
    """A model that cannot be read is a score. The spec asks for a mathematical
    model, and a formula is what makes it reusable and publishable (M7.5)."""
    frame = _frame(lambda v: 2.5 * v["x1"] - v["x2"] ** 2)
    model = fit(frame, "y", ["x1", "x2"])
    assert "x1" in model.expr and "x2^2" in model.expr


def test_the_same_data_always_gives_the_same_formula():
    """§3.1. Without this every measured gain in this contour is unfalsifiable."""
    frame = _frame(lambda v: 1.7 * v["x1"] + 0.4 * v["x2"], noise=0.2)
    assert fit(frame, "y", ["x1", "x2"]).expr == \
        fit(frame, "y", ["x1", "x2"]).expr


@pytest.mark.parametrize("law,expected", [
    (lambda v: 3.0 * v["x1"], "x1"),
    (lambda v: 2.0 * v["x1"] ** 2, "x1^2"),
    (lambda v: 1.5 * v["x1"] ** 3, "x1^3"),
    (lambda v: 4.0 * v["x1"] * v["x2"], "x1*x2"),
])
def test_each_shape_in_the_library_can_be_recovered(law, expected):
    model = fit(_frame(law), "y", ["x1", "x2"])
    assert model is not None and expected in model.terms


def test_a_pure_constant_is_explained_by_the_intercept_alone():
    frame = _frame(lambda v: 7.0)
    model = fit(frame, "y", ["x1", "x2"])
    assert model is None or model.intercept == pytest.approx(7.0, abs=0.01)


# ── it does not invent structure ─────────────────────────────────────

def test_noise_is_not_fitted_into_a_law():
    """The complement of the recovery test. A search that always produced a
    formula would produce one here too, and BIC is what stops it."""
    frame = _frame(lambda v: 0.0, noise=10.0)
    model = fit(frame, "y", ["x1", "x2"])
    assert model is None or model.r2_valid < 0.3


def test_a_predictor_that_explains_nothing_is_left_out():
    frame = _frame(lambda v: 3.0 * v["x1"], variables=("x1", "x2"))
    model = fit(frame, "y", ["x1", "x2"])
    assert "x2" not in " ".join(model.terms)


def test_one_variable_appears_only_once_in_a_formula():
    """Fitting ``x`` and ``sqrt(x)`` together describes one relationship with
    two coefficients, and reliably beats the truth on BIC by a hair."""
    frame = _frame(lambda v: 2.0 * v["x1"], noise=0.5)
    model = fit(frame, "y", ["x1", "x2"])
    inputs = [term.split("^")[0].replace("sqrt(", "").replace("log(", "")
              .replace("exp(", "").replace("1/", "").rstrip(")")
              for term in model.terms]
    assert len(inputs) == len(set(inputs))


def test_the_search_is_bounded_by_the_term_limit():
    frame = _frame(lambda v: 2.0 * v["x1"] - v["x2"] ** 2, noise=0.3)
    model = fit(frame, "y", ["x1", "x2"], max_terms=1)
    assert len(model.terms) == 1


def test_the_default_term_limit_is_the_documented_one():
    frame = _frame(lambda v: 2.0 * v["x1"])
    model = fit(frame, "y", ["x1", "x2"])
    assert len(model.terms) <= MAX_TERMS


# ── validation is out of sample, and by position ─────────────────────

def test_validation_is_the_tail_of_the_series_not_a_sample_of_it():
    """For a time series that is the only split that means anything: a model
    validated on rows interleaved with its training data is validated on its own
    neighbourhood, and every autocorrelated series passes."""
    rows = [{"tick": index, "x1": float(index), "x2": 0.0,
             "y": float(index) if index < 200 else 1000.0}
            for index in range(300)]
    model = fit(Frame.from_rows(rows), "y", ["x1", "x2"], valid_fraction=0.3)
    assert model is None or model.r2_valid < 0.9


def test_a_model_reports_how_much_data_stood_behind_each_half():
    model = fit(_frame(lambda v: 2.0 * v["x1"], count=200), "y", ["x1", "x2"])
    assert model.n_train > 0 and model.n_valid > 0
    assert model.n_train > model.n_valid


def test_residuals_are_reported():
    model = fit(_frame(lambda v: 2.0 * v["x1"], noise=1.0), "y", ["x1", "x2"])
    assert model.residual_sd > 0.0
    assert model.residual_mean == pytest.approx(0.0, abs=1e-6)


# ── guards ───────────────────────────────────────────────────────────

def test_too_little_data_produces_no_model():
    assert fit(_frame(lambda v: v["x1"], count=6), "y", ["x1"]) is None


def test_no_predictors_produce_no_model():
    assert fit(_frame(lambda v: v["x1"]), "y", []) is None


def test_an_empty_frame_produces_no_model():
    assert fit(Frame.from_rows([]), "y", ["x1"]) is None


def test_the_target_is_never_its_own_predictor():
    model = fit(_frame(lambda v: 2.0 * v["x1"]), "y", ["x1", "y"])
    assert model is None or "y" not in model.terms


def test_rows_whose_target_is_unusable_are_skipped():
    frame = _frame(lambda v: 2.0 * v["x1"], count=200)
    rows = frame.rows()
    rows[5]["y"] = None
    rows[9]["y"] = float("nan")
    model = fit(Frame.from_rows(rows), "y", ["x1", "x2"])
    assert model is not None and model.r2_valid > 0.9


def test_a_column_of_zeroes_does_not_break_the_search():
    """``1/x`` and ``log(x)`` are undefined there; the term is dropped rather
    than the search aborting on data that merely contains a constant."""
    rows = [{"tick": i, "x1": 0.0, "x2": float(i), "y": 2.0 * i}
            for i in range(200)]
    assert fit(Frame.from_rows(rows), "y", ["x1", "x2"]) is not None


# ── the least-squares core ───────────────────────────────────────────

def test_least_squares_recovers_known_coefficients():
    """``y = 3 + 2x`` through three exact points."""
    design = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    target = [3.0, 5.0, 7.0]
    solution = solve_least_squares(design, target)
    assert solution[0] == pytest.approx(3.0, abs=1e-6)
    assert solution[1] == pytest.approx(2.0, abs=1e-6)


def test_least_squares_survives_a_collinear_column():
    """Two identical columns make the normal equations singular. A search that
    aborted here would abort on any dataset containing a degenerate variable."""
    design = [[1.0, 1.0, 1.0], [1.0, 2.0, 2.0], [1.0, 3.0, 3.0]]
    assert solve_least_squares(design, [1.0, 2.0, 3.0]) is not None


def test_least_squares_on_nothing_is_nothing():
    assert solve_least_squares([], []) is None
    assert solve_least_squares([[]], [1.0]) is None


# ── reapplying a stored model (M7.9) ─────────────────────────────────

def test_a_stored_formula_can_be_applied_to_a_new_observation():
    """Knowledge that cannot be reapplied is a report. This is what lets a
    confirmed discovery become a prior in the world model."""
    frame = _frame(lambda v: 2.5 * v["x1"] - v["x2"] ** 2)
    model = fit(frame, "y", ["x1", "x2"])
    got = predict(model, {"x1": 2.0, "x2": 3.0}, ["x1", "x2"])
    assert got == pytest.approx(2.5 * 2.0 - 9.0, abs=0.05)


def test_applying_a_model_with_no_terms_gives_nothing():
    assert predict(Model(), {"x1": 1.0}) is None
    assert predict(None, {"x1": 1.0}) is None


def test_applying_a_model_to_a_row_missing_its_inputs_gives_nothing():
    frame = _frame(lambda v: 2.0 * v["x1"])
    model = fit(frame, "y", ["x1", "x2"])
    assert predict(model, {"other": 1.0}, ["x1", "x2"]) is None


def test_a_model_serialises_to_data():
    model = fit(_frame(lambda v: 2.0 * v["x1"]), "y", ["x1", "x2"])
    record = model.as_dict()
    assert record["expr"] == model.expr
    assert isinstance(record["terms"], list)


# ── the library ──────────────────────────────────────────────────────

def test_the_library_is_the_same_every_time():
    """The search walks it in order, so its order is part of the determinism."""
    assert [term.name for term in build_library(["a", "b"])] == \
        [term.name for term in build_library(["a", "b"])]


def test_the_library_covers_the_grammar_of_the_spec():
    names = {term.name for term in build_library(["a", "b"])}
    assert {"a", "a^2", "sqrt(a)", "log(a)", "1/a", "a^3", "exp(a)",
            "a*b", "min(a,b)", "max(a,b)"} <= names


def test_a_term_that_cannot_be_evaluated_reports_nothing_rather_than_raising():
    reciprocal = next(term for term in build_library(["a"]) if term.name == "1/a")
    assert reciprocal.value({"a": 0.0}) is None
    assert reciprocal.value({"a": 4.0}) == pytest.approx(0.25)


def test_a_logarithm_of_zero_is_floored_rather_than_infinite():
    logarithm = next(term for term in build_library(["a"]) if term.name == "log(a)")
    value = logarithm.value({"a": 0.0})
    assert value is not None and value < 0 and not math.isinf(value)


# ── the basis terms themselves ───────────────────────────────────────

def test_a_term_is_frozen():
    term = build_library(["a"])[0]
    with pytest.raises(Exception):
        term.name = "something else"


def test_two_libraries_built_the_same_way_hold_the_same_terms():
    """Terms are compared by what they *are* — name, inputs, complexity — not
    by the closure that evaluates them. Two libraries built from the same
    predictors hold different function objects and the same terms, and the
    beam's deduplication depends on that."""
    assert build_library(["a", "b"]) == build_library(["a", "b"])


def test_a_term_that_returns_a_boolean_is_not_a_value():
    """``True`` is an ``int`` in Python and would enter the design matrix as
    1.0, indistinguishable from a real reading of one."""
    from aegis.layers.discovery.symbolic import Term

    assert Term("flag", ("a",), 1, lambda row: True).value({"a": 1.0}) is None


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_term_that_returns_an_unusable_number_is_not_a_value(bad):
    from aegis.layers.discovery.symbolic import Term

    assert Term("x", ("a",), 1, lambda row: bad).value({"a": 1.0}) is None


def test_a_variable_is_never_paired_with_itself():
    """``a*a`` is ``a²``, which the library already has, and ``min(a,a)`` is
    ``a``. Pairing a variable with itself would fill the library with terms
    that duplicate the unary ones under longer names."""
    names = [term.name for term in build_library(["a", "b", "c"])]
    assert "a*a" not in names and "min(a,a)" not in names
    assert names.count("a*b") == 1
    assert "b*a" not in names, "the same pair appeared in both orders"


# ── the formula as it is written down ────────────────────────────────

def test_a_negative_coefficient_is_rendered_as_a_subtraction():
    """The formula is the deliverable. One that printed ``+ -1*x`` for a
    negative coefficient would be right and unreadable; one that printed
    ``+ 1*x`` would be wrong."""
    frame = _frame(lambda v: -3.0 * v["x1"])
    model = fit(frame, "y", ["x1", "x2"])
    assert " - " in model.expr and "+ -" not in model.expr


def test_a_positive_coefficient_is_rendered_as_an_addition():
    frame = _frame(lambda v: 3.0 * v["x1"])
    model = fit(frame, "y", ["x1", "x2"])
    assert " + " in model.expr or model.expr.startswith("0")


def test_the_reported_complexity_counts_the_terms_and_their_shapes():
    """Complexity is what BIC is weighed against when two formulas fit alike,
    so it has to be the sum of the term costs *plus* one per term."""
    from aegis.layers.discovery.symbolic import build_library

    frame = _frame(lambda v: 2.5 * v["x1"] - v["x2"] ** 2)
    model = fit(frame, "y", ["x1", "x2"])
    by_name = {term.name: term for term in build_library(["x1", "x2"])}
    expected = sum(by_name[name].complexity for name in model.terms) + len(model.terms)
    assert model.complexity == expected


def test_the_residual_spread_is_the_population_standard_deviation():
    """Checked against the arithmetic: a spread computed with the wrong
    divisor, or over ``value + mean`` instead of ``value − mean``, still looks
    like a plausible number."""
    frame = _frame(lambda v: 2.0 * v["x1"], count=200, noise=1.0)
    model = fit(frame, "y", ["x1", "x2"])

    # Recompute it from the fit itself. A least-squares fit with an intercept
    # has a mean residual of zero, so the spread is the root mean square of the
    # residuals — and the residuals are recoverable from R² and the target's
    # own variance, which is what makes this checkable without reaching inside.
    assert model.residual_mean == pytest.approx(0.0, abs=1e-9)
    rows = frame.rows()[:model.n_train]
    actual = [row["y"] for row in rows]
    average = sum(actual) / len(actual)
    total = sum((value - average) ** 2 for value in actual)
    rss = (1.0 - model.r2_train) * total
    assert model.residual_sd == pytest.approx((rss / len(actual)) ** 0.5, rel=1e-6)


# ── the guards on what may be fitted ─────────────────────────────────

def test_a_row_whose_predictor_is_a_boolean_is_not_fitted():
    frame = _frame(lambda v: 2.0 * v["x1"], count=200)
    rows = frame.rows()
    rows[3]["x1"] = True
    rows[7]["x2"] = False
    model = fit(Frame.from_rows(rows), "y", ["x1", "x2"])
    assert model is not None and model.n_train + model.n_valid <= 198


def test_a_validation_split_too_small_to_judge_is_no_model():
    """Eight training rows and three validation rows is enough to *fit* and not
    enough to *check*, and a model reported without a check is the failure this
    whole contour exists to avoid."""
    frame = _frame(lambda v: 2.0 * v["x1"], count=11)
    assert fit(frame, "y", ["x1", "x2"], valid_fraction=0.25) is None


def test_a_fit_needs_more_rows_than_it_has_parameters():
    """With as many observations as parameters the fit is exact and says
    nothing. The guard is what stops a two-row dataset producing a law."""
    rows = [{"tick": i, "x1": float(i), "x2": 0.0, "y": float(i)}
            for i in range(3)]
    assert fit(Frame.from_rows(rows), "y", ["x1", "x2"]) is None


def test_no_two_terms_in_a_formula_share_a_variable():
    """Fitting ``x`` and ``sqrt(x)`` together describes one relationship with
    two coefficients, and on any single dataset it beats the truth on BIC by a
    hair. Checked over a law where the temptation is real."""
    frame = _frame(lambda v: 2.0 * v["x1"] + 0.3 * v["x2"], noise=0.4)
    model = fit(frame, "y", ["x1", "x2"], max_terms=3)
    inputs = []
    from aegis.layers.discovery.symbolic import build_library

    by_name = {term.name: term for term in build_library(["x1", "x2"])}
    for name in model.terms:
        inputs.extend(by_name[name].inputs)
    assert len(inputs) == len(set(inputs)), model.terms


def test_a_formula_never_repeats_a_term():
    frame = _frame(lambda v: 2.5 * v["x1"] - v["x2"] ** 2, noise=0.2)
    model = fit(frame, "y", ["x1", "x2"], max_terms=3)
    assert len(model.terms) == len(set(model.terms))


# ── reapplying a formula ─────────────────────────────────────────────

def test_applying_a_formula_uses_the_predictors_it_is_given():
    frame = _frame(lambda v: 2.0 * v["x1"])
    model = fit(frame, "y", ["x1", "x2"])
    assert predict(model, {"x1": 3.0, "x2": 0.0}, ["x1", "x2"]) == \
        pytest.approx(6.0, abs=0.1)


def test_applying_a_formula_can_derive_the_predictors_from_the_row():
    """A stored discovery is reapplied to whatever observation arrives, and the
    caller does not always still know which columns it was fitted on."""
    frame = _frame(lambda v: 2.0 * v["x1"])
    model = fit(frame, "y", ["x1", "x2"])
    assert predict(model, {"x1": 3.0, "x2": 0.0}) == pytest.approx(6.0, abs=0.1)


def test_the_tick_column_is_not_a_predictor_when_they_are_derived():
    """Tick is the index, not a variable. A derivation that included it would
    build a different library from the one the model was fitted against and
    fail to find its own terms."""
    frame = _frame(lambda v: 2.0 * v["x1"])
    model = fit(frame, "y", ["x1", "x2"])
    assert predict(model, {"tick": 5, "x1": 3.0, "x2": 0.0}) == \
        pytest.approx(6.0, abs=0.1)


# ── the internal guards, reached directly ────────────────────────────
#
# Three of these cannot be provoked through `fit`, because `fit` refuses the
# inputs that would reach them long before they are hit. That does not make
# them dead — it makes them the second line, and a second line nobody has ever
# exercised is a second line nobody knows the shape of.

def test_a_design_matrix_needs_more_rows_than_parameters():
    """``len(terms) + 1`` parameters: one per term plus the intercept. At or
    below that the system is exactly determined or under-determined, and the
    "fit" is interpolation reported as a law."""
    from aegis.layers.discovery.symbolic import _design_matrix, build_library

    term = build_library(["a"])[0]
    rows = [{"a": float(index)} for index in range(2)]
    assert _design_matrix(rows, [term]) is None, "two rows, two parameters"

    rows = [{"a": float(index)} for index in range(3)]
    built = _design_matrix(rows, [term])
    assert built is not None and len(built[0]) == 3


def test_a_design_matrix_skips_rows_a_term_cannot_evaluate():
    from aegis.layers.discovery.symbolic import _design_matrix, build_library

    reciprocal = next(term for term in build_library(["a"]) if term.name == "1/a")
    rows = [{"a": 1.0}, {"a": 0.0}, {"a": 2.0}, {"a": 4.0}, {"a": 5.0}]
    design, keep = _design_matrix(rows, [reciprocal])
    assert keep == [0, 2, 3, 4], "the undefined row was not dropped"


def test_a_fit_whose_predictions_are_not_finite_is_refused():
    """Columns near the top of the float range overflow the normal equations:
    the sums of squares reach infinity, the solve divides infinity by infinity,
    and every coefficient comes back NaN.

    The terms themselves stay finite, so nothing upstream catches it — the
    design matrix is built, the solve "succeeds", and only this guard stands
    between a NaN model and an R² computed from NaN, which compares false
    against every threshold and would be recorded as a refutation of a formula
    that was never actually fitted.
    """
    from aegis.layers.discovery.symbolic import Term, _fit_terms

    finite_but_enormous = Term("a", ("a",), 1, lambda row: float(row["a"]))
    rows = [{"a": 1e200 * (1 + index)} for index in range(20)]
    target = [1e200 * (2 + index) for index in range(20)]
    assert _fit_terms(rows, target, [finite_but_enormous]) is None


def test_a_model_whose_validation_predictions_are_not_finite_is_refused():
    """The same guard on the other half, and it is a separate guard because a
    model can be finite on train and not on valid — which is exactly what an
    unstable formula does. Reporting its held-out R² would be reporting
    arithmetic on NaN."""
    from aegis.layers.discovery.symbolic import Term, _build_model

    term = Term("a", ("a",), 1, lambda row: float(row["a"]))
    train_target = [float(index) for index in range(20)]
    predictions = list(train_target)
    valid_rows = [{"a": float(index), "y": float(index)} for index in range(10)]
    # A NaN coefficient is what a degenerate solve leaves behind; it makes every
    # validation prediction NaN while the terms and the rows stay ordinary.
    assert _build_model([term], [0.0, float("nan")], train_target, predictions,
                        0.0, 1.0, valid_rows, "y") is None
    # The same call with a usable coefficient does produce a model, so the test
    # is about the guard rather than about the arguments.
    assert _build_model([term], [0.0, 1.0], train_target, predictions,
                        0.0, 1.0, valid_rows, "y") is not None


# ── the clash rule, on data where breaking it would pay ──────────────

def test_a_formula_never_pairs_a_variable_with_a_transform_of_itself():
    """``y = 2·x + 0.5·√|x|`` is genuinely fitted better by ``x`` and
    ``sqrt(x)`` together than by either alone — which is precisely why the rule
    has to hold here. Two coefficients describing one relationship is an
    overfit that beats the truth on BIC by a hair, every time.
    """
    from aegis.layers.discovery.symbolic import build_library

    rows = []
    for index in range(300):
        x1 = 0.5 + 9.5 * hash_unit("x1", index)
        x2 = -5.0 + 10.0 * hash_unit("x2", index)
        rows.append({"tick": index, "x1": x1, "x2": x2,
                     "y": 2.0 * x1 + 0.5 * math.sqrt(abs(x1))})
    model = fit(Frame.from_rows(rows), "y", ["x1", "x2"], max_terms=3)

    by_name = {term.name: term for term in build_library(["x1", "x2"])}
    used = [name for term in model.terms for name in by_name[term].inputs]
    assert len(used) == len(set(used)), f"{model.terms} share a variable"


def test_a_three_term_law_is_still_found():
    """The beam has to reach size three without being crowded out by subsets it
    has already considered under a different ordering."""
    rows = []
    for index in range(400):
        x1 = -5.0 + 10.0 * hash_unit("p1", index)
        x2 = -5.0 + 10.0 * hash_unit("p2", index)
        x3 = -5.0 + 10.0 * hash_unit("p3", index)
        rows.append({"tick": index, "x1": x1, "x2": x2, "x3": x3,
                     "y": 2.0 * x1 - 1.5 * x2 ** 2 + 3.0 * x3})
    model = fit(Frame.from_rows(rows), "y", ["x1", "x2", "x3"], max_terms=3)
    assert model is not None
    assert set(model.terms) == {"x1", "x2^2", "x3"}, model.terms
    assert model.r2_valid > 0.99


def test_the_terms_of_a_formula_come_back_in_library_order():
    """Subsets are generated with strictly increasing library indices, so a
    formula's terms are always in the library's own order.

    That is what makes the rendered expression canonical: the same term set
    found by two different routes prints identically, and a discovery is
    recognisable by its formula rather than by which prefix happened to reach
    it first.
    """
    from aegis.layers.discovery.symbolic import build_library

    frame = _frame(lambda v: 2.5 * v["x1"] - v["x2"] ** 2, noise=0.1)
    model = fit(frame, "y", ["x1", "x2"], max_terms=3)
    order = {term.name: index
             for index, term in enumerate(build_library(["x1", "x2"]))}
    positions = [order[name] for name in model.terms]
    assert positions == sorted(positions), model.terms


def test_a_pair_is_built_in_library_order_even_when_the_weaker_term_is_first():
    """``y = x₁ + 5x₂``: ``x₂`` dominates, so it is the first prefix the search
    extends, and ``x₁`` sits immediately before it in the library.

    That is the arrangement that catches an off-by-one in where a prefix starts
    extending from. Extending backwards produces the same pair in the reverse
    order — same fit, same BIC, and a formula that prints its terms the wrong
    way round, so the same discovery no longer has the same expression.
    """
    from aegis.layers.discovery.symbolic import build_library

    rows = []
    for index in range(300):
        x1 = -5.0 + 10.0 * hash_unit("x1", index)
        x2 = -5.0 + 10.0 * hash_unit("x2", index)
        rows.append({"tick": index, "x1": x1, "x2": x2, "y": x1 + 5.0 * x2})
    model = fit(Frame.from_rows(rows), "y", ["x1", "x2"], max_terms=2)

    assert set(model.terms) == {"x1", "x2"}, model.terms
    order = {term.name: index
             for index, term in enumerate(build_library(["x1", "x2"]))}
    positions = [order[name] for name in model.terms]
    assert positions == sorted(positions), model.terms


# ── which prefixes are carried forward ───────────────────────────────

def test_the_frontier_keeps_the_best_of_each_variable_set_not_the_worst():
    """A variable set is carried forward by its *best* member.

    Two subsets over the same variables are interchangeable for extension, so
    only one needs to survive — and keeping the worse one would extend the
    weakest route to those variables while discarding the strongest, which is
    the exact opposite of what a beam is for.
    """
    from aegis.layers.discovery.symbolic import _widen, build_library

    library = build_library(["a", "b"])          # index 0 = a, index 1 = b
    candidates = [(1.0, (0,)), (5.0, (1,)), (3.0, (1,))]
    kept = _widen(candidates, library, beam=1)

    assert (1.0, (0,)) in kept, "the top-beam entry was dropped"
    b_entries = [entry for entry in kept if entry[1] == (1,)]
    assert b_entries == [(3.0, (1,))], b_entries


def test_the_frontier_does_not_duplicate_a_variable_set_already_in_the_beam():
    from aegis.layers.discovery.symbolic import _widen, build_library

    library = build_library(["a", "b"])
    candidates = [(1.0, (0,)), (2.0, (2,)), (5.0, (1,))]   # 0 and 2 are both `a`
    kept = _widen(candidates, library, beam=2)
    assert len(kept) == 3, kept


def test_a_variable_set_missing_from_the_beam_is_added_to_it():
    """The reason the two rules are a union. Without the addition a variable
    whose best shape ranks below the beam is never extended, and the law that
    needs it is never reached."""
    from aegis.layers.discovery.symbolic import _widen, build_library

    library = build_library(["a", "b"])
    candidates = [(1.0, (0,)), (2.0, (2,)), (3.0, (4,)), (9.0, (1,))]
    kept = _widen(candidates, library, beam=3)
    assert (9.0, (1,)) in kept, "the only route to `b` was pruned"
