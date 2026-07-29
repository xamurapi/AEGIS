"""Proposing something that could be true, and counting how often we asked
(spec M7.4).

A hypothesis here is a *formal* object — a target, a list of predictors, a lag
per predictor — not a sentence. That is what lets the next stage fit it and the
stage after that preregister it.

The part of this module that matters most is the least interesting to read: the
**count of tests performed**. Benjamini–Hochberg controls the false-discovery
rate over a family of tests, and the family is every pair the scan looked at,
not every pair it liked. An implementation that filtered first and corrected
afterwards would report a controlled error rate while having no control at all,
and would find "laws" in pure noise at exactly the uncorrected rate. So
:meth:`Scanner.scan` counts every comparison it makes, corrects over all of
them, and publishes the count so the claim can be checked.

Three sources, per the spec: the association scan, the cortex under a grammar
that admits only known variables and lags, and theory — a strong causal link in
the world model, or a gene the evolution lineage says tracks fitness.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import aegis.config as cfg
from aegis.layers.discovery.statistics import (
    benjamini_hochberg, mutual_information, pearson, spearman,
)

logger = logging.getLogger("aegis.discovery")

#: Association measures run per pair. All three, because each sees a shape the
#: others miss — and all three count toward the correction.
MEASURES = ("pearson", "spearman", "mi")

#: Rows a pair needs before it is tested at all.
MIN_ROWS = 30

#: Predictors carried into the joint hypothesis. Three, because the symbolic
#: search is over subsets of a library built from these and the library grows
#: with the square of the count once products are included.
MAX_JOINT_PREDICTORS = 3


@dataclass(frozen=True)
class Hypothesis:
    """A statement precise enough to be fitted and then bet on."""

    id: str
    statement: str
    formal: str
    target: str
    predictors: tuple[str, ...]
    lags: dict = field(default_factory=dict)
    kind: str = "association"
    prior: float = 0.5
    origin: str = "scan"
    created_tick: int = 0
    measure: str = ""
    strength: float = 0.0
    p_value: float = 1.0

    def as_dict(self) -> dict:
        return {"id": self.id, "statement": self.statement,
                "formal": self.formal, "target": self.target,
                "predictors": list(self.predictors), "lags": dict(self.lags),
                "kind": self.kind, "prior": round(self.prior, 4),
                "origin": self.origin, "created_tick": self.created_tick,
                "measure": self.measure, "strength": round(self.strength, 6),
                "p_value": round(self.p_value, 8)}

    @classmethod
    def from_dict(cls, data: dict) -> "Hypothesis | None":
        if not isinstance(data, dict) or not data.get("id"):
            return None
        try:
            return cls(
                id=str(data["id"]), statement=str(data.get("statement", "")),
                formal=str(data.get("formal", "")),
                target=str(data.get("target", "")),
                predictors=tuple(str(name) for name in
                                 data.get("predictors", []) or []),
                lags=dict(data.get("lags") or {}),
                kind=str(data.get("kind", "association")),
                prior=float(data.get("prior", 0.5)),
                origin=str(data.get("origin", "scan")),
                created_tick=int(data.get("created_tick", 0)),
                measure=str(data.get("measure", "")),
                strength=float(data.get("strength", 0.0)),
                p_value=float(data.get("p_value", 1.0)))
        except (TypeError, ValueError):
            return None


def hypothesis_id(target: str, predictors, lags: dict) -> str:
    """A stable identity, so the same relationship is one hypothesis forever.

    Derived from the content rather than a counter: a scan rerun on more data
    must recognise a hypothesis it has already tested, or the refuted archive
    would never match anything and every scan would rediscover what it just
    rejected.
    """
    from aegis.util.quasirandom import hash_index

    parts = [str(target)] + [f"{name}@{int(lags.get(name, 0))}"
                             for name in sorted(predictors)]
    material = "|".join(parts)
    return f"hyp_{hash_index(1 << 32, 'hypothesis', material):08x}"


def _statement(target: str, predictor: str, lag: int, measure: str,
               strength: float) -> str:
    when = "at the same tick" if lag == 0 else f"{lag} tick(s) earlier"
    direction = "rises with" if strength >= 0 else "falls as"
    if measure == "mi":
        return (f"{target} carries information about {predictor} {when} "
                f"(mutual information {abs(strength):.3f} nats)")
    return f"{target} {direction} {predictor} {when} ({measure} {strength:+.3f})"


class Scanner:
    """The association scan, with the count that makes its correction honest."""

    def __init__(self, *, alpha: float | None = None,
                 max_lag: int | None = None, min_rows: int = MIN_ROWS):
        self.alpha = float(cfg.DISC_ALPHA if alpha is None else alpha)
        self.max_lag = int(cfg.DISC_MAX_LAG if max_lag is None else max_lag)
        self.min_rows = int(min_rows)
        #: Every comparison ever made by this scanner. The denominator of the
        #: claim "the false-discovery rate is controlled".
        self.tested = 0
        self.scans = 0
        self.rejected = 0

    def scan(self, frame, target: str, predictors=None,
             tick: int = 0) -> list[Hypothesis]:
        """Every predictor, at every lag, under every measure — then correct."""
        self.scans += 1
        names = [name for name in (predictors if predictors is not None
                                   else frame.names)
                 if name != target and name != "tick"]
        if not names or len(frame) < self.min_rows:
            return []

        trials: list[tuple] = []
        for name in sorted(names):
            for lag in range(0, max(0, self.max_lag) + 1):
                lagged = frame.lag(name, lag, as_name="__lagged__") \
                    .numeric(target, "__lagged__") if lag else \
                    frame.numeric(target, name)
                column = "__lagged__" if lag else name
                if len(lagged) < self.min_rows:
                    continue
                xs = lagged.column(column)
                ys = lagged.column(target)
                for measure in MEASURES:
                    strength, p_value = self._measure(measure, xs, ys)
                    trials.append((name, lag, measure, strength, p_value))

        self.tested += len(trials)
        if not trials:
            return []

        keep = benjamini_hochberg([trial[4] for trial in trials], self.alpha)
        self.rejected += sum(1 for flag in keep if not flag)

        # One hypothesis per (predictor, lag): three measures asking about one
        # relationship are three tests but one thing that might be true, and
        # emitting three would spend three fits on it.
        best: dict[tuple[str, int], tuple] = {}
        for trial, significant in zip(trials, keep):
            if not significant:
                continue
            name, lag, measure, strength, p_value = trial
            key = (name, lag)
            if key not in best or p_value < best[key][4]:
                best[key] = trial

        found = []
        for (name, lag), (_, _, measure, strength, p_value) in sorted(best.items()):
            lags = {name: lag}
            found.append(Hypothesis(
                id=hypothesis_id(target, [name], lags),
                statement=_statement(target, name, lag, measure, strength),
                formal=f"{target} ~ f({name}@lag{lag})",
                target=target, predictors=(name,), lags=lags,
                kind="association", prior=min(0.9, abs(strength)),
                origin="scan", created_tick=int(tick), measure=measure,
                strength=strength, p_value=p_value))
        found.sort(key=lambda item: (item.p_value, item.id))

        joint = self._joint(target, best, tick)
        if joint is not None:
            found.insert(0, joint)
        return found

    def _joint(self, target: str, best: dict, tick: int) -> Hypothesis | None:
        """One hypothesis over every predictor that survived, together.

        The pairwise scan answers "which variables matter"; it cannot answer
        "what is the law", because a law over two variables is not the sum of
        two laws over one. A system whose reward is ``2.5·surprise − brier²``
        shows a strong pairwise association with each, and fitting either alone
        recovers most of the variance and the wrong formula.

        So the survivors are also offered as a single multivariate hypothesis,
        ranked first — the symbolic search is what decides which of them earn
        their place, and BIC is what charges for the ones that do not.

        This adds no test to the family: it is a different question about
        comparisons already made and corrected, not a new comparison.
        """
        if len(best) < 2:
            return None
        ranked = sorted(best.items(), key=lambda item: (item[1][4], item[0]))

        # One entry per *variable*, at its most significant lag. ``best`` is
        # keyed by (name, lag), so a predictor that survives at three lags
        # appears three times — and taking the top three as they come produced
        # `y ~ f(a@lag5, a@lag5, a@lag5)`: a duplicated predictor list, a lag
        # mapping that had silently collapsed to one entry, and a fit handed the
        # same column three times over. A joint hypothesis is about which
        # variables act together, and a variable does not act together with
        # itself.
        chosen: list = []
        seen: set[str] = set()
        for (name, lag), trial in ranked:
            if name in seen:
                continue
            seen.add(name)
            chosen.append(((name, lag), trial))
            if len(chosen) >= MAX_JOINT_PREDICTORS:
                break
        if len(chosen) < 2:
            return None

        predictors = tuple(name for (name, _), _ in chosen)
        lags = {name: lag for (name, lag), _ in chosen}
        formal_terms = ", ".join(f"{name}@lag{lags[name]}" for name in predictors)
        return Hypothesis(
            id=hypothesis_id(target, predictors, lags),
            statement=(f"{target} is a function of "
                       f"{' and '.join(predictors)} together"),
            formal=f"{target} ~ f({formal_terms})",
            target=target, predictors=predictors, lags=lags,
            kind="law", prior=min(0.9, max(abs(trial[3]) for _, trial in chosen)),
            origin="scan", created_tick=int(tick), measure="joint",
            strength=max(abs(trial[3]) for _, trial in chosen),
            p_value=min(trial[4] for _, trial in chosen))

    @staticmethod
    def _measure(measure: str, xs, ys) -> tuple[float, float]:
        if measure == "pearson":
            return pearson(xs, ys)
        if measure == "spearman":
            return spearman(xs, ys)
        return mutual_information(xs, ys)

    def status(self) -> dict:
        return {"scans": self.scans, "tested": self.tested,
                "rejected": self.rejected, "alpha": self.alpha,
                "max_lag": self.max_lag}


# ── the cortex path, under a grammar ─────────────────────────────────

#: ``target ~ f(pred@lagN, pred@lagM)`` and nothing else. A grammar rather than
#: free text because this string decides what gets fitted and then experimented
#: on: a model that could name an arbitrary expression would be choosing the
#: engine's next action in prose.
FORMAL = re.compile(
    r"^\s*(?P<target>[A-Za-z_][\w.]*)\s*~\s*f\(\s*(?P<predictors>[^)]*)\)\s*$")
PREDICTOR = re.compile(r"^\s*(?P<name>[A-Za-z_][\w.]*)\s*(?:@lag(?P<lag>\d+))?\s*$")


def parse_formal(text: str, known, max_lag: int | None = None):
    """Parse ``y ~ f(a@lag1, b)`` against the declared variables.

    Returns ``(target, predictors, lags)`` or ``None``. Every name must be a
    variable the pool actually declares — a hypothesis about a variable that
    does not exist cannot be tested, and one about a variable invented by the
    model cannot be trusted.
    """
    max_lag = int(cfg.DISC_MAX_LAG if max_lag is None else max_lag)
    known = {str(name) for name in known}
    match = FORMAL.match(str(text or ""))
    if not match:
        return None
    target = match.group("target")
    if target not in known:
        return None

    predictors: list[str] = []
    lags: dict[str, int] = {}
    for chunk in match.group("predictors").split(","):
        if not chunk.strip():
            continue
        piece = PREDICTOR.match(chunk)
        if not piece:
            return None
        name = piece.group("name")
        if name not in known or name == target or name in lags:
            return None
        lag = int(piece.group("lag") or 0)
        if lag < 0 or lag > max_lag:
            return None
        predictors.append(name)
        lags[name] = lag
    if not predictors:
        return None
    return target, tuple(predictors), lags


def from_formal(text: str, known, *, origin: str = "cortex", tick: int = 0,
                statement: str = "", prior: float = 0.5,
                max_lag: int | None = None) -> Hypothesis | None:
    """A hypothesis from a formal string, or ``None`` if it does not parse."""
    parsed = parse_formal(text, known, max_lag=max_lag)
    if parsed is None:
        return None
    target, predictors, lags = parsed
    return Hypothesis(
        id=hypothesis_id(target, predictors, lags),
        statement=str(statement or f"{target} depends on "
                                   f"{', '.join(predictors)}"),
        formal=f"{target} ~ f({', '.join(f'{name}@lag{lags[name]}' for name in predictors)})",
        target=target, predictors=predictors, lags=lags, kind="causal",
        prior=max(0.0, min(1.0, float(prior))), origin=str(origin),
        created_tick=int(tick))


# ── the theory path ──────────────────────────────────────────────────

def from_world_model(world_model, known, *, tick: int = 0,
                     limit: int = 5) -> list[Hypothesis]:
    """Hypotheses from causal links the world model already believes.

    A link the model learned from its own experience is a claim about the world
    that has never been tested as such. Promoting it to a hypothesis is how a
    belief becomes something that can be refuted.
    """
    known = {str(name) for name in known}
    found: list[Hypothesis] = []
    try:
        links = world_model.strongest_links(limit * 4)
    except Exception:
        return []
    for link in links or []:
        cause = str(link.get("cause", ""))
        effect = str(link.get("effect", ""))
        if cause not in known or effect not in known or cause == effect:
            continue
        lags = {cause: 1}
        found.append(Hypothesis(
            id=hypothesis_id(effect, [cause], lags),
            statement=f"the world model holds that {cause} leads to {effect}",
            formal=f"{effect} ~ f({cause}@lag1)",
            target=effect, predictors=(cause,), lags=lags, kind="causal",
            prior=min(0.9, float(link.get("strength", 0.5))),
            origin="theory", created_tick=int(tick)))
        if len(found) >= limit:
            break
    return found
