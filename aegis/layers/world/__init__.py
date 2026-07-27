"""The predictive world model (spec M1).

    state → prediction → decision → outcome → error → learning

The old model recorded which causes tended to produce which effects and was
read once a tick to shade a confidence number. This package adds the missing
half: a *forecast*, written down before the action, scored after it, and used
to look ahead.
"""
from aegis.layers.world.causal import (
    MAX_CHAINS, MAX_LINKS, MIN_OBSERVATIONS_FOR_PREDICTION, CausalLinks,
)
from aegis.layers.world.outcome import OutcomeEntry, OutcomeModel, OutcomePrediction
from aegis.layers.world.prediction import (
    Prediction, PredictionScore, PredictionScorer,
)
from aegis.layers.world.simulate import RolloutResult, Simulator
from aegis.layers.world.state import (
    FIELDS, LABELS, StateEncoder, StateKey, collect_state_inputs,
)
from aegis.layers.world.transition import TransitionEntry, TransitionModel

__all__ = [
    "CausalLinks", "FIELDS", "LABELS", "MAX_CHAINS", "MAX_LINKS",
    "MIN_OBSERVATIONS_FOR_PREDICTION", "OutcomeEntry", "OutcomeModel",
    "OutcomePrediction", "Prediction", "PredictionScore", "PredictionScorer",
    "RolloutResult", "Simulator", "StateEncoder", "StateKey", "TransitionEntry",
    "TransitionModel", "collect_state_inputs",
]
