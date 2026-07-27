"""Motivation with teeth: value → priority → resource → action (spec M4)."""
from aegis.layers.motivation.priority import Candidate, PriorityScheduler
from aegis.layers.motivation.resources import (
    CONCURRENT, CUMULATIVE, PER_TICK, WINDOWED, Lease, ResourceBudget,
    ResourceCost, ResourceManager,
)
from aegis.layers.motivation.roi import DRIVES, ActivityROI, ROITracker, normalize_cost

__all__ = [
    "ActivityROI", "CONCURRENT", "CUMULATIVE", "Candidate", "DRIVES", "Lease",
    "PER_TICK", "PriorityScheduler", "ROITracker", "ResourceBudget",
    "ResourceCost", "ResourceManager", "WINDOWED", "normalize_cost",
]
