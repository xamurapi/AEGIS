"""From a hypothesis to a formula (spec M7.5).

The spec asks for a *mathematical model*: not "reward correlates with surprise"
but an expression, written down, refittable, and applicable to new data. That is
the difference between the engine reporting a pattern and the engine producing
knowledge.

**The search is a deterministic beam over increasing complexity, not genetic
programming.** Genetic programming is the usual tool and it is the wrong one
here for a reason that has nothing to do with quality: it is stochastic, and
§3.1 forbids that outright. Two runs on identical data would produce different
formulas, and a discovery that cannot be rederived cannot be replicated.

The construction is: build a fixed, ordered library of **basis terms** out of the
predictors — the variable itself, its square, its logarithm, a product with
another variable — then search subsets of that library in order of size,
fitting the coefficients of each subset analytically by least squares. Every
form considered is linear *in its parameters*, which is what makes the fit exact
rather than a search of its own, and the library is what makes the *form*
nonlinear. ``2.5·x₁ − x₂²`` is a two-term subset, and it is found by looking at
two-term subsets.

Selection is by BIC on train, confirmed by R² on valid. BIC and not R², because
R² never falls when a term is added: a search that optimised it would return
the largest expression it was allowed to build, every time.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from aegis.util.stats import bic as bic_score
from aegis.util.stats import mean, r_squared

logger = logging.getLogger("aegis.discovery")

#: Terms in a formula. Three is where the useful expressions are and four
#: multiplies the search by the library size for gains BIC then charges back.
MAX_TERMS = 3

#: Guard for logarithms and reciprocals of values at or below zero.
EPSILON = 1e-9


@dataclass(frozen=True)
class Term:
    """One basis function: a name, and how to evaluate it on a row."""

    name: str
    #: Which predictors it reads — used to keep a formula from using one
    #: variable twice under two disguises.
    inputs: tuple[str, ...]
    complexity: int
    fn: object = field(compare=False, default=None)

    def value(self, row: dict) -> float | None:
        try:
            out = self.fn(row)
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            # ``KeyError`` and ``TypeError`` are the ones that matter outside
            # the fit: a stored formula is reapplied to live observations
            # (M7.9), and an observation with a metric missing or holding a
            # string is normal. A term that raised there would take the tick
            # down with it for a reason that is not an error.
            return None
        if not isinstance(out, (int, float)) or isinstance(out, bool):
            return None
        if out != out or abs(out) == float("inf"):
            return None
        return float(out)


def _safe_log(value: float) -> float:
    return math.log(abs(value)) if abs(value) > EPSILON else math.log(EPSILON)


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < EPSILON:
        raise ZeroDivisionError
    return numerator / denominator


def build_library(predictors) -> list[Term]:
    """The ordered set of basis functions the search may combine.

    Ordered and fixed: the search walks it in this order, so the same data
    yields the same formula. Unary shapes first because a law is more often a
    power or a logarithm of one quantity than a product of two, and the beam
    keeps what it meets first among equals.
    """
    predictors = [str(name) for name in predictors]
    library: list[Term] = []
    for name in predictors:
        library.append(Term(f"{name}", (name,), 1,
                            lambda row, n=name: float(row[n])))
    for name in predictors:
        library.append(Term(f"{name}^2", (name,), 2,
                            lambda row, n=name: float(row[n]) ** 2))
    for name in predictors:
        library.append(Term(f"sqrt({name})", (name,), 2,
                            lambda row, n=name: math.sqrt(abs(float(row[n])))))
    for name in predictors:
        library.append(Term(f"log({name})", (name,), 2,
                            lambda row, n=name: _safe_log(float(row[n]))))
    for name in predictors:
        library.append(Term(f"1/{name}", (name,), 2,
                            lambda row, n=name: _safe_div(1.0, float(row[n]))))
    for name in predictors:
        library.append(Term(f"{name}^3", (name,), 3,
                            lambda row, n=name: float(row[n]) ** 3))
    for name in predictors:
        library.append(Term(f"exp({name})", (name,), 3,
                            lambda row, n=name: math.exp(min(30.0, abs(float(row[n]))))))
    for index, left in enumerate(predictors):
        for right in predictors[index + 1:]:
            library.append(Term(f"{left}*{right}", (left, right), 3,
                                lambda row, a=left, b=right:
                                float(row[a]) * float(row[b])))
            library.append(Term(f"min({left},{right})", (left, right), 3,
                                lambda row, a=left, b=right:
                                min(float(row[a]), float(row[b]))))
            library.append(Term(f"max({left},{right})", (left, right), 3,
                                lambda row, a=left, b=right:
                                max(float(row[a]), float(row[b]))))
    return library


@dataclass
class Model:
    """A fitted formula and everything needed to judge or reuse it."""

    expr: str = ""
    terms: tuple[str, ...] = ()
    params: tuple[float, ...] = ()
    intercept: float = 0.0
    r2_train: float = 0.0
    r2_valid: float = 0.0
    bic: float = float("inf")
    complexity: int = 0
    residual_mean: float = 0.0
    residual_sd: float = 0.0
    n_train: int = 0
    n_valid: int = 0

    def as_dict(self) -> dict:
        return {"expr": self.expr, "terms": list(self.terms),
                "params": [round(value, 6) for value in self.params],
                "intercept": round(self.intercept, 6),
                "r2_train": round(self.r2_train, 4),
                "r2_valid": round(self.r2_valid, 4),
                "bic": round(self.bic, 4), "complexity": self.complexity,
                "residual_mean": round(self.residual_mean, 6),
                "residual_sd": round(self.residual_sd, 6),
                "n_train": self.n_train, "n_valid": self.n_valid}


def solve_least_squares(design: list[list[float]], target: list[float]) -> list[float] | None:
    """Normal equations with partial pivoting and a ridge floor.

    The ridge term is tiny and it is not regularisation in the usual sense: two
    basis terms in this library can be exactly collinear on a particular dataset
    — ``x`` and ``sqrt(x)`` when ``x`` is constant, say — and the normal
    equations are then singular. Without the floor the search would abort on
    data that merely contains a degenerate column.
    """
    if not design or not design[0]:
        return None
    n_params = len(design[0])
    matrix = [[0.0] * (n_params + 1) for _ in range(n_params)]
    for i in range(n_params):
        for j in range(n_params):
            matrix[i][j] = sum(row[i] * row[j] for row in design)
        matrix[i][i] += 1e-10
        matrix[i][n_params] = sum(row[i] * value
                                  for row, value in zip(design, target))

    for column in range(n_params):
        pivot = max(range(column, n_params),
                    key=lambda r: abs(matrix[r][column]))
        if abs(matrix[pivot][column]) < 1e-14:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        for j in range(column, n_params + 1):
            matrix[column][j] /= divisor
        for row in range(n_params):
            if row == column:
                continue
            factor = matrix[row][column]
            if factor == 0.0:
                continue
            for j in range(column, n_params + 1):
                matrix[row][j] -= factor * matrix[column][j]
    return [matrix[i][n_params] for i in range(n_params)]


def _design_matrix(rows, terms) -> tuple[list[list[float]], list[int]] | None:
    """Evaluate the chosen terms over the rows. ``None`` if any term is unusable."""
    design: list[list[float]] = []
    keep: list[int] = []
    for index, row in enumerate(rows):
        values = [term.value(row) for term in terms]
        if any(value is None for value in values):
            continue
        design.append([1.0] + list(values))
        keep.append(index)
    if len(design) <= len(terms) + 1:
        return None
    return design, keep


def _fit_terms(rows, target_values, terms):
    """Fit one subset of terms. Returns (coefficients, predictions, rss, n)."""
    built = _design_matrix(rows, terms)
    if built is None:
        return None
    design, keep = built
    target = [target_values[index] for index in keep]
    coefficients = solve_least_squares(design, target)
    if coefficients is None:
        return None
    predictions = [sum(c * v for c, v in zip(coefficients, row)) for row in design]
    if any(value != value or abs(value) == float("inf") for value in predictions):
        return None
    rss = sum((actual - predicted) ** 2
              for actual, predicted in zip(target, predictions))
    return coefficients, predictions, target, rss


def _render(coefficients, terms) -> str:
    """The formula as text — what makes this a model rather than a score."""
    parts = [f"{coefficients[0]:.4g}"]
    for coefficient, term in zip(coefficients[1:], terms):
        sign = "+" if coefficient >= 0 else "-"
        parts.append(f" {sign} {abs(coefficient):.4g}*{term.name}")
    return "".join(parts)


def fit(frame, target: str, predictors, *, max_terms: int = MAX_TERMS,
        beam: int = 6, valid_fraction: float = 0.3) -> Model | None:
    """Search for the formula that best explains ``target`` (M7.5).

    Split is by position, not by sampling: the last ``valid_fraction`` of the
    rows are held back. For a time series that is the only split that means
    anything — a model validated on rows interleaved with its training data is
    validated on its own neighbourhood, and every autocorrelated series passes.
    """
    rows = [row for row in frame.rows()]
    predictors = [name for name in predictors if name != target]
    if not rows or not predictors:
        return None

    usable = []
    for row in rows:
        value = row.get(target)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value != value or abs(value) == float("inf"):
            continue
        if any(not isinstance(row.get(name), (int, float))
               or isinstance(row.get(name), bool) for name in predictors):
            continue
        usable.append(row)

    split = int(len(usable) * (1.0 - max(0.05, min(0.9, float(valid_fraction)))))
    train_rows, valid_rows = usable[:split], usable[split:]
    if len(train_rows) < 8 or len(valid_rows) < 4:
        return None

    train_target = [float(row[target]) for row in train_rows]
    library = build_library(predictors)

    best: Model | None = None
    #: Subsets kept from the previous size, cheapest BIC first. The beam is what
    #: keeps this from being an exhaustive search over the power set.
    frontier: list[tuple[float, tuple[int, ...]]] = [(float("inf"), ())]

    for size in range(1, max(1, int(max_terms)) + 1):
        candidates: list[tuple[float, tuple[int, ...]]] = []
        seen: set[tuple[int, ...]] = set()
        for _, prefix in frontier:
            for index in range(len(library)):
                if index in prefix:
                    continue
                # Canonical form: a subset is a *set*, so it is identified by
                # its sorted indices and generated once however it was reached.
                #
                # The obvious alternative — only offering indices above the
                # prefix's last — is an offset that has to be exactly right, and
                # when it is not the same subset is fitted twice in two orders.
                # Both fits are the same fit, but their BIC differs in the last
                # bits, so which one won was decided by floating-point noise and
                # the printed formula listed its terms in whichever order that
                # happened to favour. A discovery is supposed to be recognisable
                # by its formula.
                #
                # Sorting also makes the search slightly more complete: a subset
                # is now reachable from any of its members' prefixes, not only
                # from the one that happens to sort first.
                combination = tuple(sorted(prefix + (index,)))
                if combination in seen:
                    continue
                seen.add(combination)
                terms = [library[position] for position in combination]
                # One variable may appear once. Fitting `x` and `sqrt(x)`
                # together describes one relationship with two coefficients and
                # reliably beats the truth on BIC by a hair.
                used: set[str] = set()
                clash = False
                for term in terms:
                    for name in term.inputs:
                        if name in used:
                            clash = True
                        used.add(name)
                if clash:
                    continue

                fitted = _fit_terms(train_rows, train_target, terms)
                if fitted is None:
                    continue
                coefficients, predictions, target_used, rss = fitted
                score = bic_score(rss, len(target_used), len(coefficients))
                candidates.append((score, combination))

                model = _build_model(
                    terms, coefficients, target_used, predictions, rss, score,
                    valid_rows, target)
                if model is not None and (best is None or model.bic < best.bic):
                    best = model

        candidates.sort(key=lambda item: (item[0], item[1]))
        frontier = _widen(candidates, library, max(1, int(beam)))
        if not frontier:
            break

    return best


def _widen(candidates, library, beam: int):
    """Choose the prefixes worth extending — by variables, not by fit alone.

    A plain top-``beam`` cut prunes on how well a subset fits *at its current
    size*, and that is the wrong question. What makes a prefix worth extending
    is **which variables it has committed to**, because that is what its
    extensions can still add.

    The failure is not hypothetical. For ``y = 2x₁ − 1.5x₂² + 3x₃`` the pair
    ``(x₁, x₃)`` is mediocre on its own — it leaves the whole quadratic term
    unexplained — and it is the only route to the exact law. A search that
    ranked it against four hundred better-fitting pairs dropped it, never
    evaluated ``(x₁, x₂², x₃)`` at all, and returned ``x₁³`` in its place: R²
    0.97 where the truth was 1.0, and a formula that is simply not the law.

    So candidates are grouped by their *variable set* and the best of each group
    is kept: two subsets over the same variables are interchangeable for
    extension, subsets over different variables are not.

    The two rules are a union, not a replacement. Taking *only* the best of each
    variable set would be the opposite mistake: at size one that is a single
    shape per variable, and the best shape for a variable on its own is not
    always the shape it takes in the law — ``x`` can lose to ``x³`` alone and
    still be the term the true formula needs. So the plain top-``beam`` is kept
    and the missing variable sets are added to it.
    """
    kept = list(candidates[:beam])
    represented = {
        frozenset(name for position in combination
                  for name in library[position].inputs)
        for _, combination in kept
    }
    best_per_group: dict[frozenset, tuple[float, tuple[int, ...]]] = {}
    for score, combination in candidates:
        variables = frozenset(name for position in combination
                              for name in library[position].inputs)
        if variables in represented:
            continue
        current = best_per_group.get(variables)
        if current is None or (score, combination) < current:
            best_per_group[variables] = (score, combination)
    kept.extend(sorted(best_per_group.values()))
    return kept


def _build_model(terms, coefficients, train_target, predictions, rss, score,
                 valid_rows, target) -> Model | None:
    residuals = [actual - predicted
                 for actual, predicted in zip(train_target, predictions)]
    residual_mean = mean(residuals)
    residual_sd = (sum((value - residual_mean) ** 2 for value in residuals)
                   / len(residuals)) ** 0.5 if residuals else 0.0

    valid_built = _design_matrix(valid_rows, terms)
    if valid_built is None:
        return None
    valid_design, valid_keep = valid_built
    valid_actual = [float(valid_rows[index][target]) for index in valid_keep]
    valid_predicted = [sum(c * v for c, v in zip(coefficients, row))
                       for row in valid_design]
    if any(value != value or abs(value) == float("inf")
           for value in valid_predicted):
        return None

    return Model(
        expr=_render(coefficients, terms),
        terms=tuple(term.name for term in terms),
        params=tuple(coefficients[1:]),
        intercept=coefficients[0],
        r2_train=r_squared(train_target, predictions),
        r2_valid=r_squared(valid_actual, valid_predicted),
        bic=score,
        complexity=sum(term.complexity for term in terms) + len(terms),
        residual_mean=residual_mean,
        residual_sd=residual_sd,
        n_train=len(train_target),
        n_valid=len(valid_actual),
    )


def predict(model: Model, row: dict, predictors=()) -> float | None:
    """Apply a stored formula to a new observation.

    A model that cannot be reapplied is a report, not knowledge (M7.9) — this
    is what lets a confirmed discovery become a prior in the world model or a
    term in the planner's score.
    """
    if model is None or not model.terms:
        return None
    names = list(predictors) or sorted(
        {name for name in row if name != "tick"})
    library = {term.name: term for term in build_library(names)}
    total = float(model.intercept)
    for coefficient, name in zip(model.params, model.terms):
        term = library.get(name)
        if term is None:
            return None
        value = term.value(row)
        if value is None:
            return None
        total += float(coefficient) * value
    return total
