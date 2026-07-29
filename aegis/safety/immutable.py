"""The untouchable set (spec Appendix B).

Seven contours can now change the running system: parametric self-modification,
weight training, source rewriting, the evolution engine, the behaviour policy,
resource reallocation and interventional self-experiments. Each of them asks
here BEFORE changing anything.

Three levels of protection, because "immutable" is too blunt for all of it:

* ``IMMUTABLE_PARAMS`` — no change, ever, by any contour. Ethics, the kill
  switch, the sandbox gate, the control plane.
* ``BOUNDED_PARAMS`` — may move, but never outside a hard ceiling/floor that
  the system itself cannot widen (e.g. a sandbox timeout may be tuned, but
  never past 30 s).
* ``MONOTONIC_PARAMS`` — may move in the safe direction only (a training
  cool-down may grow, never shrink).

A contour that forgets to ask is a defect: ``tests/test_immutable_params.py``
walks every contour and asserts the refusal.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

# ── Level 1: never changeable ────────────────────────────────────────

CATEGORIES: dict[str, tuple[str, ...]] = {
    # 1. Ethics and stopping. The system may not re-weigh its own veto.
    "ethics_and_stop": (
        "ETHICAL_THRESHOLD_AUTO",
        "ETHICAL_THRESHOLD_REVIEW",
        "ethics.axioms",
        "ethics.axiom_hashes",
        "ethics.kill_switch_active",
        "ethics.veto_check",
    ),
    # 2. Self-preservation: the detector must not be able to blunt itself.
    "self_preservation": (
        "self_preservation.thresholds",
        "self_preservation.critical_modules",
        "self_preservation.critical_hashes",
        "self_preservation.lethal_patterns",
        "self_preservation.filter_llm_response",
        "self_preservation.can_stop",
    ),
    # 3. Sandbox: the static gate and the isolation flags of skill execution.
    "sandbox": (
        "sandbox.SAFE_IMPORTS",
        "sandbox.FORBIDDEN_CALLS",
        "sandbox.safety_visitor",
        "sandbox.isolated_interpreter",
    ),
    # 4. Source self-modification stays opt-in and capped.
    "code_self_mod": (
        "CODE_SELF_MOD_ENABLED",
        "CODE_MOD_MAX_PER_SESSION",
        "CODE_MOD_MAX_FILE_CHARS",
        "CODE_MOD_MIN_TICK",
    ),
    # 5. The control plane is how a human stays in charge.
    "control_plane": (
        "API_TOKEN",
        "API_HOST",
        "API_CORS_ORIGINS",
        "config.network_exposure_warning",
    ),
    # 6. Training guards that are not merely bounded but structural.
    "training_guards": (
        "TRAIN_MAX_CHECKPOINTS",
        "training.ethics_gate_required",
    ),
    # 7. Resource floors that keep safety work funded.
    "resource_floors": (
        "RESOURCE_SAFETY_FLOOR",
        "resources.safety_action_shares",
    ),
}

IMMUTABLE_PARAMS: frozenset[str] = frozenset(
    name for names in CATEGORIES.values() for name in names
)

# ── Level 2: changeable within a hard range ──────────────────────────
# The system may tune these; it may not widen the range they live in.

BOUNDED_PARAMS: dict[str, tuple[float, float]] = {
    "SANDBOX_TIMEOUT": (0.5, 30.0),
    "MAX_RISK_CONFIDENCE_PENALTY": (0.0, 0.6),
    "DISC_INTERVENTION_MAX_DELTA": (0.0, 0.5),
    "CHECKPOINT_EVERY_N_TICKS": (1, 200),
    "TRAIN_MIN_DATASET_SIZE": (10, 100_000),
}

# ── Level 3: one-way ratchets ────────────────────────────────────────
# "up"   — the value may only grow (a cool-down may get longer, not shorter)
# "down" — the value may only shrink (a tolerance may get stricter, not laxer)

MONOTONIC_PARAMS: dict[str, str] = {
    "TRAIN_MIN_INTERVAL_SECONDS": "up",
    "TRAIN_VAL_LOSS_THRESHOLD": "down",
}


class ImmutableParameterError(Exception):
    """Raised when a contour tries to change something it must not."""


@dataclass(frozen=True)
class Verdict:
    """Outcome of a change request. ``value`` is what may actually be applied."""

    allowed: bool
    reason: str
    value: float | None = None
    clamped: bool = False


def normalize(name: str) -> str:
    """Strip a contour prefix so ``evolution/SANDBOX_TIMEOUT`` matches the bare
    name. Contours label their proposals (``parametric/x``, ``evolution/y``) and
    a lookup on the raw label would silently miss every protected parameter."""
    text = str(name).strip()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text


#: The last dotted segment of every protected name.
#:
#: The set spells structural things with an owning prefix —
#: ``ethics.kill_switch_active``, ``sandbox.SAFE_IMPORTS`` — because that is
#: what they are called where they live. A contour proposing a change spells it
#: however the proposal arrived, and ``parametric/kill_switch_active``
#: normalises to a bare ``kill_switch_active`` that was in no set at all. The
#: guard said yes.
#:
#: So a bare last segment is protected too. This over-matches slightly — a
#: parameter called ``thresholds`` is refused whoever owns it — and that is the
#: correct direction for a safety boundary: a refusal that has to be argued
#: with costs a conversation, a permission that should not have been given
#: costs the guarantee.
_PROTECTED_SEGMENTS: frozenset[str] = frozenset(
    name.rsplit(".", 1)[-1] for name in IMMUTABLE_PARAMS
)


def category_of(name: str) -> str | None:
    """Which protection category a name belongs to, or None if unprotected."""
    key = normalize(name)
    for category, names in CATEGORIES.items():
        if key in names:
            return category
    # Matched by its last segment — see `_PROTECTED_SEGMENTS`. Reported under
    # the category that owns the full name, so a refusal still says which
    # guarantee it is defending.
    segment = key.rsplit(".", 1)[-1]
    for category, names in CATEGORIES.items():
        if any(candidate.rsplit(".", 1)[-1] == segment for candidate in names):
            return category
    return None


def is_immutable(name: str) -> bool:
    key = normalize(name)
    return key in IMMUTABLE_PARAMS or key.rsplit(".", 1)[-1] in _PROTECTED_SEGMENTS


def assert_mutable(name: str, context: str = "") -> None:
    """Raise unless ``name`` may be changed at all. Call this first, always."""
    if is_immutable(name):
        where = f" (in {context})" if context else ""
        raise ImmutableParameterError(
            f"{normalize(name)!r} is immutable [{category_of(name)}]{where}"
        )


def check_change(name: str, old_value, new_value) -> Verdict:
    """Full verdict for a proposed parameter change.

    Returns rather than raises, because the callers are cognitive contours that
    must degrade gracefully: a refused mutation is a normal outcome to record,
    not an error to propagate out of a tick.
    """
    key = normalize(name)

    if key in IMMUTABLE_PARAMS:
        return Verdict(False, f"{key!r} is immutable [{category_of(key)}]")

    try:
        proposed = float(new_value)
    except (TypeError, ValueError):
        return Verdict(False, f"{key!r}: non-numeric value {new_value!r}")

    direction = MONOTONIC_PARAMS.get(key)
    if direction is not None:
        try:
            current = float(old_value)
        except (TypeError, ValueError):
            return Verdict(False, f"{key!r}: non-numeric current value {old_value!r}")
        if direction == "up" and proposed < current:
            return Verdict(False, f"{key!r} may only increase ({current} -> {proposed})")
        if direction == "down" and proposed > current:
            return Verdict(False, f"{key!r} may only decrease ({current} -> {proposed})")

    bounds = BOUNDED_PARAMS.get(key)
    if bounds is not None:
        lo, hi = bounds
        if proposed < lo or proposed > hi:
            clamped = max(lo, min(hi, proposed))
            return Verdict(
                True,
                f"{key!r} clamped to hard range [{lo}, {hi}]",
                value=clamped,
                clamped=True,
            )

    return Verdict(True, "allowed", value=proposed)


def digest() -> str:
    """Stable hash of the whole protection contract.

    It goes into ``Substrate.state_digest()``: quietly editing this module then
    claiming the system is unchanged is exactly the failure mode worth catching.
    """
    payload = json.dumps(
        {
            "categories": {k: sorted(v) for k, v in sorted(CATEGORIES.items())},
            "bounded": {k: list(v) for k, v in sorted(BOUNDED_PARAMS.items())},
            "monotonic": dict(sorted(MONOTONIC_PARAMS.items())),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def status() -> dict:
    return {
        "immutable_count": len(IMMUTABLE_PARAMS),
        "categories": {k: len(v) for k, v in CATEGORIES.items()},
        "bounded": {k: list(v) for k, v in BOUNDED_PARAMS.items()},
        "monotonic": dict(MONOTONIC_PARAMS),
        "digest": digest(),
    }
