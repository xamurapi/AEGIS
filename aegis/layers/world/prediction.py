"""Prediction as a first-class object (spec M1.1, M1.3, M1.5).

The old world model held frequencies and was read once a tick to shade a
confidence number. Nothing it produced was ever written down before the fact,
so nothing could ever be checked afterwards, and "the model is good" was not a
statement anyone could evaluate.

Here a forecast is recorded **before** the action, scored **after** it, and
kept. That ordering is the whole point: a prediction made after the outcome is
not a prediction. What it buys:

* an error signal to learn from;
* calibration — whether "70%" actually means seventy percent;
* surprise, which is the honest measure of how novel a situation was and
  therefore the right thing to point curiosity at.

Two baselines are scored alongside the model on the same events, because a
Brier score alone says nothing. Predicting the long-run average, or a flat
0.5, are both free — a model that cannot beat them has learned nothing, and
the acceptance criterion (§M1.9) is exactly that comparison.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.util.stats import (
    Welford, brier_score, calibration_curve, expected_calibration_error,
    exponential_smooth, safe_log,
)

logger = logging.getLogger("aegis.world.prediction")

#: Weight of each new sample in the smoothed error metrics. Slow enough that a
#: single unlucky tick does not move the reported number, fast enough that a
#: genuine regression shows up within a few hundred ticks.
SMOOTHING_ALPHA = 0.02


@dataclass
class Prediction:
    """What the model expected, written down before the action was taken."""

    id: str
    tick: int
    state: str
    action: str
    p_success: float
    expected_reward: float
    reward_sd: float
    predicted_next: list = field(default_factory=list)      # [(state_key, p)]
    predicted_effects: list = field(default_factory=list)   # from CausalLinks
    confidence: float = 0.0
    horizon: int = 1
    created: float = 0.0
    #: Probability mass the model left on everything outside the listed
    #: successors, spread over how many such states it knew of. Without this,
    #: landing anywhere outside the top-k scores as −log(1e-12) ≈ 27.6, and
    #: the surprise metric measures the floor rather than the model.
    other_mass: float = 0.0
    other_states: int = 0

    def probability_of(self, successor: str) -> float:
        """What this forecast assigned to a particular successor."""
        listed = dict(self.predicted_next)
        if successor in listed:
            return listed[successor]
        if self.other_states > 0 and self.other_mass > 0:
            return self.other_mass / self.other_states
        return 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "tick": self.tick, "state": self.state,
            "action": self.action,
            "p_success": round(self.p_success, 5),
            "expected_reward": round(self.expected_reward, 5),
            "reward_sd": round(self.reward_sd, 5),
            "predicted_next": [[k, round(p, 5)] for k, p in self.predicted_next],
            "predicted_effects": self.predicted_effects,
            "confidence": round(self.confidence, 5),
            "horizon": self.horizon,
            "created": self.created,
            "other_mass": round(self.other_mass, 5),
            "other_states": self.other_states,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Prediction | None:
        try:
            return cls(
                id=str(data["id"]), tick=int(data.get("tick", 0)),
                state=str(data.get("state", "")), action=str(data.get("action", "")),
                p_success=float(data.get("p_success", 0.5)),
                expected_reward=float(data.get("expected_reward", 0.0)),
                reward_sd=float(data.get("reward_sd", 0.0)),
                predicted_next=[(str(k), float(p))
                                for k, p in (data.get("predicted_next") or [])],
                predicted_effects=list(data.get("predicted_effects") or []),
                confidence=float(data.get("confidence", 0.0)),
                horizon=int(data.get("horizon", 1)),
                created=float(data.get("created", 0.0)),
                other_mass=float(data.get("other_mass", 0.0)),
                other_states=int(data.get("other_states", 0)),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class PredictionScore:
    """How wrong one prediction turned out to be."""

    id: str
    tick: int
    brier: float
    reward_error: float
    nll_next: float
    success: bool
    actual_reward: float
    actual_next: str
    baseline_brier_mean: float
    baseline_brier_half: float

    def as_dict(self) -> dict:
        return {
            "id": self.id, "tick": self.tick,
            "brier": round(self.brier, 5),
            "reward_error": round(self.reward_error, 5),
            "nll_next": round(self.nll_next, 5),
            "success": self.success,
            "actual_reward": round(self.actual_reward, 5),
            "actual_next": self.actual_next,
            "baseline_brier_mean": round(self.baseline_brier_mean, 5),
            "baseline_brier_half": round(self.baseline_brier_half, 5),
        }


class PredictionScorer:
    """Keeps open predictions, scores them, and reports how good the model is."""

    def __init__(self, store_path: Path | None = None,
                 max_predictions: int | None = None,
                 bins: int | None = None, window: int | None = None):
        self._store_path = store_path
        self.max_predictions = int(cfg.WM_MAX_PREDICTIONS
                                   if max_predictions is None else max_predictions)
        self.bins = int(cfg.WM_CALIBRATION_BINS if bins is None else bins)
        self.window = int(cfg.WM_SURPRISE_WINDOW if window is None else window)

        self._open: dict[str, Prediction] = {}
        self._seq = 0
        self.scored = 0

        # Smoothed error metrics.
        self.brier: float | None = None
        self.reward_mae: float | None = None
        self.nll_next: float | None = None

        # Baselines, scored on exactly the same events.
        self.baseline_brier_mean: float | None = None
        self.baseline_brier_half: float | None = None
        self._outcome_rate = Welford()

        #: (predicted probability, outcome) pairs for the reliability curve.
        self._calibration: deque = deque(maxlen=2000)
        #: recent −log P(actual next state), for surprise
        self._surprise: deque = deque(maxlen=max(2, self.window))
        self.rows_written = 0

    # ── opening ──────────────────────────────────────────────────────

    def next_id(self, tick: int) -> str:
        self._seq += 1
        return f"pred_{tick:08d}_{self._seq:06d}"

    def open(self, prediction: Prediction) -> Prediction:
        """Record a forecast before the action is taken."""
        prediction.created = CLOCK.now()
        self._open[prediction.id] = prediction
        # Bound the outstanding set: a prediction whose outcome never arrived
        # (the tick failed, the action was deferred) must not accumulate.
        if len(self._open) > 256:
            for stale in sorted(self._open, key=lambda k: self._open[k].tick)[:64]:
                del self._open[stale]
        return prediction

    def pending(self) -> int:
        return len(self._open)

    # ── closing ──────────────────────────────────────────────────────

    def score(self, prediction_id: str, success: bool, reward: float,
              actual_next: str, next_probability: float | None = None
              ) -> PredictionScore | None:
        """Close a forecast against what actually happened.

        Returns None for an unknown id rather than raising: a prediction can
        legitimately go unclosed when the tick that would have closed it failed,
        and the scorer is not the place to notice that.
        """
        prediction = self._open.pop(str(prediction_id), None)
        if prediction is None:
            return None

        # The baselines are computed from the outcome history *before* this
        # event is folded in, so all three forecasters see the same information
        # and the comparison is fair.
        mean_rate = self._outcome_rate.mean if self._outcome_rate.n > 0 else 0.5

        if next_probability is None:
            next_probability = prediction.probability_of(str(actual_next))

        score = PredictionScore(
            id=prediction.id,
            tick=prediction.tick,
            brier=brier_score(prediction.p_success, success),
            reward_error=abs(prediction.expected_reward - float(reward)),
            nll_next=-safe_log(next_probability),
            success=bool(success),
            actual_reward=float(reward),
            actual_next=str(actual_next),
            baseline_brier_mean=brier_score(mean_rate, success),
            baseline_brier_half=brier_score(0.5, success),
        )

        self._outcome_rate.update(1.0 if success else 0.0)
        self.brier = exponential_smooth(self.brier, score.brier, SMOOTHING_ALPHA)
        self.reward_mae = exponential_smooth(self.reward_mae, score.reward_error,
                                             SMOOTHING_ALPHA)
        self.nll_next = exponential_smooth(self.nll_next, score.nll_next,
                                           SMOOTHING_ALPHA)
        self.baseline_brier_mean = exponential_smooth(
            self.baseline_brier_mean, score.baseline_brier_mean, SMOOTHING_ALPHA)
        self.baseline_brier_half = exponential_smooth(
            self.baseline_brier_half, score.baseline_brier_half, SMOOTHING_ALPHA)

        self._calibration.append((prediction.p_success, bool(success)))
        self._surprise.append(score.nll_next)
        self.scored += 1
        self._append(prediction, score)
        return score

    # ── reporting ────────────────────────────────────────────────────

    def surprise(self) -> float:
        """Mean information content of recent outcomes.

        High when the world keeps doing things the model did not expect, which
        is exactly where exploration is worth spending on — this is what makes
        curiosity directed rather than blind.
        """
        return sum(self._surprise) / len(self._surprise) if self._surprise else 0.0

    def ece(self) -> float:
        return expected_calibration_error(list(self._calibration), self.bins)

    def beats_baselines(self) -> bool:
        """Whether the model is better than predicting the average, or 0.5.

        The acceptance question from §M1.9, asked directly. Until there is
        anything to compare, the answer is no — an untrained model has not
        earned the claim.
        """
        if self.brier is None or self.baseline_brier_mean is None \
                or self.baseline_brier_half is None:
            return False
        return (self.brier < self.baseline_brier_mean
                and self.brier < self.baseline_brier_half)

    def calibration(self) -> dict:
        return {
            "brier": round(self.brier, 5) if self.brier is not None else None,
            "ece": round(self.ece(), 5),
            "reward_mae": round(self.reward_mae, 5) if self.reward_mae is not None else None,
            "nll_next": round(self.nll_next, 5) if self.nll_next is not None else None,
            "surprise": round(self.surprise(), 5),
            "scored": self.scored,
            "pending": len(self._open),
            "samples": len(self._calibration),
            "baseline_brier_mean": (round(self.baseline_brier_mean, 5)
                                    if self.baseline_brier_mean is not None else None),
            "baseline_brier_half": (round(self.baseline_brier_half, 5)
                                    if self.baseline_brier_half is not None else None),
            "beats_baselines": self.beats_baselines(),
            "curve": calibration_curve(list(self._calibration), self.bins),
        }

    # ── persistence ──────────────────────────────────────────────────

    def _append(self, prediction: Prediction, score: PredictionScore) -> None:
        """Append the closed forecast to the log. Never raises."""
        if self._store_path is None:
            return
        row = {**prediction.to_dict(), "score": score.as_dict()}
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            with self._store_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.rows_written += 1
        except Exception:
            logger.warning("Failed to append prediction %s", prediction.id,
                           exc_info=True)
            return
        if self.rows_written > self.max_predictions * 2:
            self._truncate()

    def _truncate(self) -> None:
        try:
            with self._store_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
            if len(lines) <= self.max_predictions:
                self.rows_written = len(lines)
                return
            keep = lines[-self.max_predictions:]
            tmp = self._store_path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(keep), encoding="utf-8")
            tmp.replace(self._store_path)
            self.rows_written = len(keep)
        except Exception:
            logger.warning("Failed to truncate the prediction log", exc_info=True)

    def recent(self, limit: int = 20) -> list[dict]:
        """The most recently closed predictions, newest last."""
        if self._store_path is None or not self._store_path.exists():
            return []
        rows: list[dict] = []
        try:
            with self._store_path.open("r", encoding="utf-8") as handle:
                for line in handle.readlines()[-max(1, limit):]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue    # one torn line must not hide the rest
                    if isinstance(row, dict):
                        rows.append(row)
        except Exception:
            logger.warning("Failed to read the prediction log", exc_info=True)
        return rows

    def state_dict(self) -> dict:
        """The smoothed metrics, so calibration survives a restart."""
        return {
            "brier": self.brier, "reward_mae": self.reward_mae,
            "nll_next": self.nll_next,
            "baseline_brier_mean": self.baseline_brier_mean,
            "baseline_brier_half": self.baseline_brier_half,
            "outcome_rate": self._outcome_rate.to_dict(),
            "scored": self.scored,
            "calibration": [[p, o] for p, o in list(self._calibration)[-500:]],
        }

    def load_state(self, data: dict) -> None:
        if not isinstance(data, dict):
            return

        def _float_or_none(key):
            value = data.get(key)
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        self.brier = _float_or_none("brier")
        self.reward_mae = _float_or_none("reward_mae")
        self.nll_next = _float_or_none("nll_next")
        self.baseline_brier_mean = _float_or_none("baseline_brier_mean")
        self.baseline_brier_half = _float_or_none("baseline_brier_half")
        self._outcome_rate = Welford.from_dict(data.get("outcome_rate"))
        try:
            self.scored = int(data.get("scored", 0))
        except (TypeError, ValueError):
            self.scored = 0
        for row in (data.get("calibration") or []):
            try:
                self._calibration.append((float(row[0]), bool(row[1])))
            except (IndexError, TypeError, ValueError):
                continue
