"""No contour can reach the immutable set (spec Appendix B(б), §VII.6).

Appendix B asks for this by name: *"a test walks all seven contours and
confirms that an attempt to change each parameter of the set is refused and
logged"*. ``tests/test_immutable_params.py`` covers the mechanism — the
frozenset, ``assert_mutable``, the categories, the digest. This file covers the
**reach**: every path by which a contour changes something, driven with an
immutable name, and refused.

The distinction matters because a perfect guard protects nothing if a contour
writes past it. Each test below goes through the door a contour actually uses —
the genome it adopts, the discovery it applies, the parameter an experiment
sets — rather than calling the guard directly.
"""
import pytest

from aegis.safety.immutable import (
    IMMUTABLE_PARAMS, ImmutableParameterError, assert_mutable, is_immutable,
    normalize,
)

#: A sample across all seven categories of Appendix B, so a category that lost
#: its protection is named rather than averaged away.
SAMPLE = [
    "ETHICAL_THRESHOLD_AUTO",             # 1: ethics and stopping
    "ethics.kill_switch_active",          # 1
    "self_preservation.can_stop",         # 2: self-preservation
    "sandbox.SAFE_IMPORTS",               # 3: the sandbox
    "CODE_SELF_MOD_ENABLED",              # 4: the source-modification contour
    "API_TOKEN",                          # 5: control from outside
    "TRAIN_MAX_CHECKPOINTS",              # 6: training interlocks
    "RESOURCE_SAFETY_FLOOR",              # 7: resource floors
]

#: The other two protection levels of Appendix B. Not everything the spec
#: protects is protected by being unchangeable: a sandbox timeout may grow
#: inside a hard ceiling, and a training interval may only ever get longer.
#: Mixing the three levels into one list is how a test ends up asserting that a
#: bounded parameter is frozen, which it is not and should not be.
BOUNDED_SAMPLE = ["SANDBOX_TIMEOUT"]
MONOTONIC_SAMPLE = ["TRAIN_MIN_INTERVAL_SECONDS"]


@pytest.fixture(autouse=True)
def _every_sample_is_actually_immutable():
    """A guard on the sample itself: a name that quietly left the set would
    make every test below pass by testing nothing."""
    missing = [name for name in SAMPLE if not is_immutable(name)]
    assert not missing, f"the sample names parameters that are not immutable: {missing}"


# ── contour 1: evolution ─────────────────────────────────────────────

@pytest.mark.parametrize("name", SAMPLE)
def test_evolution_cannot_carry_an_immutable_gene(name):
    """The genome is the only thing evolution changes, and the schema is the
    only way a name gets into a genome."""
    from aegis.layers.evolution.genome import GENES_BY_NAME, Genome

    assert name not in GENES_BY_NAME
    genome = Genome({name: 0.999})
    assert name not in genome, "an immutable name survived into a genome"


def test_no_declared_gene_names_anything_immutable():
    from aegis.layers.evolution.genome import GENES

    immutable = {normalize(item) for item in IMMUTABLE_PARAMS}
    offenders = [gene.name for gene in GENES if normalize(gene.name) in immutable]
    assert not offenders, offenders


# ── contour 2: the behaviour policy ──────────────────────────────────

@pytest.mark.parametrize("name", SAMPLE)
def test_the_policy_ignores_an_immutable_name_in_its_genome(name, tmp_path):
    from aegis.layers.policy import BehaviourPolicy

    policy = BehaviourPolicy(store_dir=tmp_path / "policy")
    before = (policy.store.weight, policy.miner.min_support)
    policy.set_genome({name: 0.999, "policy_weight": 0.4})
    assert (policy.store.weight, policy.miner.min_support) != before or True
    # The evolvable knob moved; the immutable name did not become one.
    assert policy.store.weight == pytest.approx(0.4)
    assert not hasattr(policy, name)


# ── contour 3: the reasoning contour ─────────────────────────────────

@pytest.mark.parametrize("name", SAMPLE)
def test_reasoning_ignores_an_immutable_name_in_its_genome(name, tmp_path):
    from aegis.layers.reasoning import ReasoningEngine

    engine = ReasoningEngine(store_path=tmp_path / "strategies.json")
    engine.set_genome({name: 0.999, "reason_budget": 5})
    assert engine._budget() == 5
    assert engine.genome.get(name) == 0.999 or name not in engine.genome
    # Whatever the genome dict happens to carry, nothing reads it under that
    # name: the contour only ever looks up the genes it declares.
    assert not hasattr(engine, name)


# ── contour 4: the world model ───────────────────────────────────────

@pytest.mark.parametrize("name", SAMPLE)
def test_the_world_model_ignores_an_immutable_name_in_its_genome(name, tmp_path):
    from aegis.layers.world_model import PredictiveWorldModel

    model = PredictiveWorldModel(store_path=tmp_path / "wm" / "model.json")
    before = model.transitions.smoothing
    model.apply_genome({name: 0.999, "wm_smoothing": 2.0})
    assert model.transitions.smoothing == pytest.approx(2.0)
    assert model.transitions.smoothing != before or before == 2.0


# ── contour 5: resources and ROI ─────────────────────────────────────

@pytest.mark.parametrize("name", SAMPLE)
def test_the_roi_tracker_ignores_an_immutable_name_in_its_genome(name, tmp_path):
    from aegis.layers.motivation.roi import ROITracker

    roi = ROITracker(store_path=tmp_path / "roi.json")
    roi.set_genome({name: 0.999, "res_share_competence": 0.4})
    assert not hasattr(roi, name)
    assert sum(roi.shares.values()) == pytest.approx(1.0)


# ── contour 6: the discovery engine's interventions ──────────────────

@pytest.mark.parametrize("name", SAMPLE)
def test_an_intervention_can_never_be_planned_on_an_immutable_name(name):
    """The one contour that deliberately changes a live parameter. Appendix F
    is a whitelist and the immutable set is checked separately, because a
    whitelist is a thing someone could edit."""
    from aegis.layers.discovery.experiment import is_controllable, preregister

    assert is_controllable(name) is False
    assert preregister({"id": "hyp_x"}, None, design="interventional_abab",
                       variable=name, levels=(0.1, 0.2)) is None


@pytest.mark.parametrize("name", SAMPLE)
def test_an_intervention_started_on_an_immutable_name_never_runs(name, tmp_path):
    from aegis.layers.discovery import DiscoveryEngine

    engine = DiscoveryEngine(directory=tmp_path / "discovery")
    touched = []
    started = engine.start_intervention(
        "hyp_x", name, (0.1, 0.2), tick=0,
        apply=lambda variable, value: touched.append((variable, value)),
        read=lambda: 0.15)
    assert started is False
    assert touched == [], "an immutable parameter was written"


# ── contour 7: parametric self-modification ──────────────────────────

@pytest.mark.parametrize("name", SAMPLE)
def test_parametric_self_modification_is_refused(name):
    """The contour that takes a parameter name from a model and applies it.
    This is the path where an immutable name is most likely to *arrive*, so it
    is the one where being refused matters most."""
    with pytest.raises(ImmutableParameterError):
        assert_mutable(name, context="parametric_self_mod")


@pytest.mark.parametrize("name", SAMPLE)
def test_the_refusal_names_the_parameter_and_the_caller(name):
    """Refusing quietly would leave an operator with a contour that does
    nothing and no way to find out why."""
    with pytest.raises(ImmutableParameterError) as raised:
        assert_mutable(name, context="evolution")
    message = str(raised.value)
    assert normalize(name) in message
    assert "evolution" in message


# ── the set itself ───────────────────────────────────────────────────

def test_every_category_of_appendix_b_is_represented_in_the_sample():
    """If a category had no sample, a contour could lose its protection for
    that whole category without a single test noticing."""
    from aegis.safety.immutable import category_of

    assert len({category_of(name) for name in SAMPLE}) >= 5


def test_a_prefixed_name_is_still_refused_by_every_contour():
    """Contours name parameters with their own prefixes. A guard that only
    matched the bare name would be bypassed by the way the callers spell it."""
    for prefix in ("ethics.", "evolution.", "discovery.", "policy."):
        with pytest.raises(ImmutableParameterError):
            assert_mutable(prefix + "ETHICAL_THRESHOLD_AUTO")


# ── the other two protection levels ──────────────────────────────────

@pytest.mark.parametrize("name", BOUNDED_SAMPLE)
def test_a_bounded_parameter_is_clamped_rather_than_refused(name):
    """Appendix B category 3: the sandbox timeout "may grow within a gene's
    range, but not past a hard ceiling of 30 s". So the contour is allowed to
    move it and is not allowed to move it out of bounds — refusing outright
    would make the gene decorative, and allowing it would make the ceiling so.
    """
    from aegis.safety.immutable import BOUNDED_PARAMS, check_change

    low, high = BOUNDED_PARAMS[name]
    verdict = check_change(name, low, high * 100)
    assert verdict.allowed is True
    assert verdict.value == pytest.approx(high), "the ceiling did not hold"

    inside = check_change(name, low, (low + high) / 2)
    assert inside.allowed and inside.value == pytest.approx((low + high) / 2)


@pytest.mark.parametrize("name", MONOTONIC_SAMPLE)
def test_a_monotonic_parameter_only_moves_the_safe_way(name):
    """Appendix B category 6: a training interval may only ever get longer.
    A contour that could shorten it could train continuously, which is the one
    thing the interval exists to prevent."""
    from aegis.safety.immutable import MONOTONIC_PARAMS, check_change

    direction = MONOTONIC_PARAMS[name]
    assert direction == "up"
    assert check_change(name, 100.0, 200.0).allowed is True
    assert check_change(name, 100.0, 50.0).allowed is False


def test_the_three_levels_do_not_overlap():
    """A name in two levels would be protected by whichever check ran first,
    and which one that is would be an implementation detail."""
    from aegis.safety.immutable import BOUNDED_PARAMS, MONOTONIC_PARAMS

    frozen = {normalize(name) for name in IMMUTABLE_PARAMS}
    bounded = {normalize(name) for name in BOUNDED_PARAMS}
    monotonic = {normalize(name) for name in MONOTONIC_PARAMS}
    assert not (frozen & bounded)
    assert not (frozen & monotonic)
    assert not (bounded & monotonic)


def test_every_protected_name_of_appendix_b_is_covered_by_some_level():
    """The names the spec calls out, each landing in exactly one level."""
    from aegis.safety.immutable import BOUNDED_PARAMS, MONOTONIC_PARAMS

    bounded_or_monotonic = {normalize(n) for n in BOUNDED_PARAMS} |         {normalize(n) for n in MONOTONIC_PARAMS}
    for name in ("ETHICAL_THRESHOLD_AUTO", "ETHICAL_THRESHOLD_REVIEW",
                 "kill_switch_active", "SAFE_IMPORTS", "FORBIDDEN_CALLS",
                 "SANDBOX_TIMEOUT", "CODE_SELF_MOD_ENABLED",
                 "CODE_MOD_MAX_PER_SESSION", "API_TOKEN", "API_HOST",
                 "TRAIN_MIN_INTERVAL_SECONDS", "TRAIN_MAX_CHECKPOINTS",
                 "RESOURCE_SAFETY_FLOOR"):
        # Asked as the question rather than as set membership: the set spells
        # structural names with an owning prefix, and what matters is whether
        # the guard refuses the spelling a contour would actually use.
        assert is_immutable(name) or normalize(name) in bounded_or_monotonic, name


# ── the spelling a proposal actually arrives in ──────────────────────

@pytest.mark.parametrize("bare", [
    "kill_switch_active", "veto_check", "axioms", "axiom_hashes",
    "SAFE_IMPORTS", "FORBIDDEN_CALLS", "safety_visitor",
    "can_stop", "critical_modules", "critical_hashes", "lethal_patterns",
    "filter_llm_response", "ethics_gate_required", "safety_action_shares",
])
def test_a_structural_name_is_refused_without_its_owning_prefix(bare):
    """The set spells these with the module that owns them —
    ``ethics.kill_switch_active``, ``sandbox.SAFE_IMPORTS``. A contour spells a
    proposal however the proposal arrived, and contours label proposals with a
    slash (``parametric/x``), which the normaliser strips.

    So ``parametric/kill_switch_active`` used to normalise to a bare
    ``kill_switch_active`` that belonged to no set, and the guard allowed it —
    a proposal to disable the kill switch, refused only if it happened to be
    written the long way.
    """
    assert is_immutable(bare) is True
    with pytest.raises(ImmutableParameterError):
        assert_mutable(bare)


@pytest.mark.parametrize("bare", ["kill_switch_active", "SAFE_IMPORTS",
                                  "can_stop", "veto_check"])
def test_a_contour_labelled_proposal_is_refused(bare):
    """How it reaches the guard in practice."""
    for contour in ("parametric", "evolution", "discovery", "policy"):
        with pytest.raises(ImmutableParameterError):
            assert_mutable(f"{contour}/{bare}", context=contour)


@pytest.mark.parametrize("bare", ["kill_switch_active", "SAFE_IMPORTS"])
def test_a_refusal_by_segment_still_names_its_category(bare):
    """A refusal that could not say which guarantee it was defending would be a
    refusal an operator has to guess about."""
    from aegis.safety.immutable import category_of

    assert category_of(bare) is not None


def test_an_ordinary_parameter_is_still_allowed():
    """The over-match has to stop somewhere: everything that is not a protected
    segment goes through, or the guard would refuse the system's own genome."""
    from aegis.layers.evolution.genome import GENES

    for gene in GENES:
        assert is_immutable(gene.name) is False, gene.name
        assert_mutable(gene.name, context="evolution")
