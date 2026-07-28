"""System 1: the world model — now with prediction (spec M1).

    модель мира → прогноз → решение

The development text's first complaint is that the system had no real
understanding of its world. It had a frequency table of causes and effects,
read once a tick to shade a confidence number; nothing it produced was ever
written down in advance, so nothing could be checked afterwards.

This facade keeps that table — as :class:`~aegis.layers.world.causal.CausalLinks`,
byte-for-byte the same behaviour, same file on disk — and puts a predictive
model on top of it:

* :class:`~aegis.layers.world.state.StateEncoder` turns the tick into a
  discrete state;
* :class:`~aegis.layers.world.transition.TransitionModel` learns where each
  action leads from there;
* :class:`~aegis.layers.world.outcome.OutcomeModel` learns what it is worth;
* :class:`~aegis.layers.world.prediction.PredictionScorer` records the forecast
  before the action and scores it after, which is what makes "the model is
  good" a checkable claim rather than an opinion;
* :class:`~aegis.layers.world.simulate.Simulator` looks ahead over all of it,
  deterministically, so the planner (M2) can compare plans instead of guessing.

Every legacy method keeps its name and signature. The old class is still
importable as ``WorldModel``, because thirty call sites and the dashboard use
that name and none of them should have to care that it grew a second half.
"""
from __future__ import annotations

import logging
from pathlib import Path

import aegis.config as cfg
from aegis.layers.world.causal import (
    MAX_CHAINS, MAX_LINKS, MIN_OBSERVATIONS_FOR_PREDICTION, CausalLinks,
)
from aegis.layers.world.outcome import OutcomeModel, OutcomePrediction
from aegis.layers.world.prediction import Prediction, PredictionScorer
from aegis.layers.world.simulate import RolloutResult, Simulator
from aegis.layers.world.state import StateEncoder, StateKey, collect_state_inputs
from aegis.layers.world.transition import TransitionModel
from aegis.store.migrations import read_store, write_store

logger = logging.getLogger("aegis.world_model")


class PredictiveWorldModel:
    """Causal links plus a model that predicts, and knows how wrong it is."""

    def __init__(self, store_path: Path | None = None, telemetry=None):
        self._store_path = Path(store_path) if store_path is not None \
            else (cfg.WORLD_MODEL_DIR / "model.json")
        self._directory = self._store_path.parent
        self.telemetry = telemetry

        # The legacy half, on the legacy file. Untouched.
        self.causal = CausalLinks(store_path=self._store_path)

        # The predictive half.
        self.encoder = StateEncoder()
        self.transitions = TransitionModel()
        self.outcomes = OutcomeModel()
        self.scorer = PredictionScorer(
            store_path=self._directory / "predictions.jsonl")
        self.simulator = Simulator(self.transitions, self.outcomes)

        #: Ticks on which the model had a usable estimate for what it chose —
        #: the coverage metric of §M1.8.
        self._covered = 0
        self._decisions = 0
        self._load_predictive()

    # ── legacy surface (unchanged behaviour) ─────────────────────────
    # Delegation rather than inheritance: the causal table is a separate thing
    # with its own file and its own retention rule, and keeping that boundary
    # visible is what stops the two halves from quietly entangling.

    @property
    def links(self) -> dict:
        return self.causal.links

    @links.setter
    def links(self, value: dict) -> None:
        self.causal.links = value
        self.causal._rebuild_index()

    @property
    def chains(self) -> list:
        return self.causal.chains

    @chains.setter
    def chains(self, value: list) -> None:
        self.causal.chains = value

    @property
    def total_observations(self) -> int:
        return self.causal.total_observations

    @total_observations.setter
    def total_observations(self, value: int) -> None:
        self.causal.total_observations = value

    @property
    def max_links(self) -> int:
        return self.causal.max_links

    @max_links.setter
    def max_links(self, value: int) -> None:
        self.causal.max_links = value

    def _retention_score(self, link: dict) -> float:
        return self.causal._retention_score(link)

    def observe(self, cause: str, effect: str, success: bool = True) -> None:
        self.causal.observe(cause, effect, success)

    def predict(self, cause: str, k: int = 5) -> list[dict]:
        return self.causal.predict(cause, k)

    def explain(self, effect: str, k: int = 5) -> list[dict]:
        return self.causal.explain(effect, k)

    def risks_for(self, tokens: list[str], k: int = 5) -> list[dict]:
        return self.causal.risks_for(tokens, k)

    def build_chain(self, objective: str, constraints: list[str] | None = None) -> dict:
        return self.causal.build_chain(objective, constraints)

    def refine_chain(self, parsed: dict) -> dict | None:
        return self.causal.refine_chain(parsed)

    # ── state ────────────────────────────────────────────────────────

    def encode(self, inputs) -> StateKey:
        return self.encoder.encode(inputs)

    def encode_substrate(self, substrate) -> StateKey:
        return self.encoder.encode(collect_state_inputs(substrate))

    # ── learning ─────────────────────────────────────────────────────

    def observe_transition(self, state, action: str, next_state) -> None:
        self.transitions.observe(state, action, next_state)

    def observe_outcome(self, state, action: str, success: bool,
                        reward: float = 0.0, cost: float = 0.0) -> None:
        self.outcomes.observe(state, action, success, reward, cost)

    # ── prediction ───────────────────────────────────────────────────

    def predict_outcome(self, state, action: str) -> OutcomePrediction:
        """What to expect from an action. Never fails on an unseen pair."""
        return self.outcomes.predict(state, action)

    def predict_next(self, state, action: str, k: int = 3) -> list[tuple[str, float]]:
        return self.transitions.top_next(state, action, k)

    def knows(self, state, action: str) -> float:
        """0..1 — how much the model actually knows about this pair.

        Both halves have to be known for the answer to be useful: knowing where
        an action leads without knowing what it is worth is not knowledge a
        planner can act on, so the weaker of the two is what counts.
        """
        return min(self.transitions.knows(state, action),
                   self.outcomes.knows(state, action))

    def make_prediction(self, state, action: str, tick: int,
                        horizon: int = 1) -> Prediction:
        """Write down a forecast, before the action is taken."""
        state_key = state.key() if isinstance(state, StateKey) else str(state)
        outcome = self.outcomes.predict(state_key, action)
        known = self.knows(state_key, action)
        successors = self.transitions.top_next(state_key, action, cfg.WM_BRANCH)
        # Everything the forecast did NOT list still has to carry a probability,
        # or landing outside the top-k would score as an impossible event and
        # the surprise metric would measure the log floor instead of the model.
        listed_mass = sum(p for _, p in successors)
        known_states = max(len(self.transitions.states()), len(successors))
        prediction = Prediction(
            id=self.scorer.next_id(tick),
            tick=int(tick),
            state=state_key,
            action=str(action),
            p_success=outcome.p_success,
            expected_reward=outcome.expected_reward,
            reward_sd=outcome.reward_sd,
            predicted_next=successors,
            other_mass=max(0.0, 1.0 - listed_mass),
            other_states=max(1, known_states - len(successors)),
            predicted_effects=self.causal.predict(str(action), k=3),
            # Confidence is not the success probability: it is how much the
            # model trusts its own estimate. Thin evidence or a wide reward
            # spread both mean "do not lean on this number".
            confidence=round(known * (1.0 - min(1.0, outcome.reward_sd)), 4),
            horizon=int(horizon),
        )
        self._decisions += 1
        if known >= 0.5:
            self._covered += 1
        return self.scorer.open(prediction)

    def score_prediction(self, prediction_id: str, success: bool, reward: float,
                         actual_next) -> object:
        """Close a forecast against what happened."""
        successor = actual_next.key() if isinstance(actual_next, StateKey) \
            else str(actual_next)
        return self.scorer.score(prediction_id, success, reward, successor)

    # ── simulation ───────────────────────────────────────────────────

    def rollout(self, state, actions: list[str], depth: int | None = None,
                beam: int | None = None) -> RolloutResult:
        return self.simulator.rollout(state, actions, depth, beam)

    def best_sequence(self, state, actions: list[str], depth: int | None = None,
                      beam: int | None = None,
                      discount: float | None = None) -> list[str]:
        return self.simulator.best_sequence(state, actions, depth, beam, discount)

    def evaluate_sequence(self, state, sequence: list[str]) -> float:
        return self.simulator.evaluate(state, sequence)

    # ── model quality ────────────────────────────────────────────────

    def calibration(self) -> dict:
        report = self.scorer.calibration()
        report["coverage"] = self.coverage()
        return report

    def surprise(self) -> float:
        return self.scorer.surprise()

    def coverage(self) -> float:
        """Fraction of decisions the model actually had an estimate for."""
        return round(self._covered / self._decisions, 4) if self._decisions else 0.0

    def apply_genome(self, genome: dict) -> None:
        """Adopt the evolved parameters that shape this model (Appendix C)."""
        mapping = {
            "wm_smoothing": (("transitions", "smoothing"), ("outcomes", "smoothing")),
            "wm_half_life": (("transitions", "half_life"), ("outcomes", "half_life")),
            "explore_bonus": (("simulator", "explore_bonus"),),
            "plan_discount": (("simulator", "discount"),),
        }
        for gene, targets in mapping.items():
            if gene not in (genome or {}):
                continue
            for attribute, field_name in targets:
                try:
                    setattr(getattr(self, attribute), field_name,
                            type(getattr(getattr(self, attribute), field_name))(genome[gene]))
                except (TypeError, ValueError):
                    logger.debug("Ignoring unusable genome value for %s", gene)

    # ── persistence ──────────────────────────────────────────────────

    def _path(self, name: str) -> Path:
        return self._directory / name

    def _load_predictive(self) -> None:
        self.transitions.load(read_store(self._path("transitions.json"),
                                         store="wm_transitions"))
        self.outcomes.load(read_store(self._path("outcomes.json"),
                                      store="wm_outcomes"))
        calibration = read_store(self._path("calibration.json"),
                                 store="wm_calibration")
        self.scorer.load_state(calibration.get("scorer") or {})
        try:
            self._covered = int(calibration.get("covered", 0))
            self._decisions = int(calibration.get("decisions", 0))
        except (TypeError, ValueError):
            self._covered, self._decisions = 0, 0

    def save(self) -> None:
        """Persist both halves. Never raises — a tick must survive a full disk."""
        try:
            self.causal.save()
        except Exception:
            logger.warning("Failed to save the causal links", exc_info=True)
        write_store(self._path("transitions.json"), self.transitions.to_dict())
        write_store(self._path("outcomes.json"), self.outcomes.to_dict())
        write_store(self._path("calibration.json"), {
            "scorer": self.scorer.state_dict(),
            "covered": self._covered,
            "decisions": self._decisions,
        })

    # ── reporting ────────────────────────────────────────────────────

    def publish_metrics(self, tick: int) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            report = self.scorer.calibration()
            for name, value in (
                (M.WM_BRIER, report["brier"]),
                (M.WM_ECE, report["ece"]),
                (M.WM_REWARD_MAE, report["reward_mae"]),
                (M.WM_NLL_NEXT, report["nll_next"]),
                (M.WM_SURPRISE, report["surprise"]),
            ):
                if value is not None:
                    self.telemetry.record(name, value, tick)
            self.telemetry.record(M.WM_COVERAGE, self.coverage(), tick)
            self.telemetry.record(M.WM_STATES, len(self.transitions.states()), tick)
            self.telemetry.record(M.WM_TRANSITIONS, len(self.transitions.pairs), tick)
            self.telemetry.record(M.WM_ROLLOUT_MS, self.simulator.last_elapsed_ms, tick)
        except Exception:
            logger.exception("World-model metric publication failed")

    def status(self) -> dict:
        """The legacy report, plus the predictive half."""
        report = self.causal.status()
        report.update({
            "predictive": {
                "transitions": self.transitions.status(),
                "outcomes": self.outcomes.status(),
                "simulator": self.simulator.status(),
                "calibration": self.calibration(),
                "decisions": self._decisions,
            },
        })
        return report


#: The name thirty call sites, the dashboard and the existing suite already
#: use. It now resolves to the model that also predicts.
WorldModel = PredictiveWorldModel

__all__ = ["MAX_CHAINS", "MAX_LINKS", "MIN_OBSERVATIONS_FOR_PREDICTION",
           "CausalLinks", "PredictiveWorldModel", "WorldModel"]
