"""The fence around a self-experiment (spec M7.6, §3.3, M7.11).

This is the only contour in the system that deliberately makes the running
system worse for a while in order to learn something. Everything here is about
the conditions under which it is allowed to, and — more importantly — the
conditions under which it must stop immediately rather than at a convenient
boundary.

The named test of the spec is the last section: an intervention interrupted by
``health=critical`` in the middle of a series restores the parameter it changed.
A series that waited for a block boundary to react to a critical health reading
would be a series that keeps making things worse for up to a hundred ticks
after it has been told to stop.
"""
import pytest

from aegis.layers.discovery.experiment import (
    CONTROLLABLE, MIN_BLOCKS, Intervention, is_controllable, preregister,
)
from aegis.safety.immutable import IMMUTABLE_PARAMS


class _Model:
    expr = "y ~ 2x"
    r2_valid = 0.9


class _Hypothesis:
    id = "hyp_intervention"


class _Knob:
    """Stands in for whatever actually holds the parameter."""

    def __init__(self, value=0.15):
        self.value = value
        self.writes = []

    def apply(self, name, value):
        self.writes.append((name, value))
        self.value = value

    def read(self):
        return self.value


def _series(knob, *, variable="explore_bonus", levels=(0.10, 0.20),
            block_ticks=4, tick=0):
    prereg = preregister(_Hypothesis(), _Model(), design="interventional_abab",
                         tick=tick, variable=variable, levels=levels,
                         block_ticks=block_ticks)
    return Intervention(prereg, apply=knob.apply, read=knob.read,
                        block_ticks=block_ticks)


# ── what may be touched at all ───────────────────────────────────────

def test_only_whitelisted_variables_are_controllable():
    assert is_controllable("explore_bonus") is True
    assert is_controllable("plan_beam") is True


def test_a_variable_nobody_listed_is_not_controllable():
    """A whitelist, not a blacklist. Anything not named is refused, so a new
    parameter is safe by default rather than dangerous by default."""
    assert is_controllable("something_invented_today") is False


@pytest.mark.parametrize("name", sorted(IMMUTABLE_PARAMS)[:12])
def test_no_immutable_parameter_is_ever_controllable(name):
    """Checked separately from the whitelist, because a whitelist is a thing
    someone could edit and the immutable set is what must hold even then."""
    assert is_controllable(name) is False


def test_the_whitelist_and_the_immutable_set_do_not_intersect():
    assert not (CONTROLLABLE & set(IMMUTABLE_PARAMS))


def test_a_series_on_an_uncontrolled_variable_never_starts():
    knob = _Knob()
    prereg = preregister(_Hypothesis(), _Model(), design="interventional_abab",
                         variable="explore_bonus", levels=(0.1, 0.2))
    prereg.variable = "ETHICAL_THRESHOLD_AUTO"       # after freezing
    series = Intervention(prereg, apply=knob.apply, read=knob.read)
    assert series.start(0) is False
    assert knob.writes == []


def test_a_series_whose_plan_was_altered_never_starts():
    knob = _Knob()
    series = _series(knob)
    series.prereg.levels = (0.0, 0.9)
    assert series.start(0) is False
    assert "frozen" in series.abort_reason


def test_a_series_needs_two_levels():
    knob = _Knob()
    series = _series(knob)
    series.levels = [0.1]
    assert series.start(0) is False


def test_interventions_can_be_switched_off_entirely(monkeypatch):
    import aegis.config as cfg

    monkeypatch.setattr(cfg, "DISC_INTERVENTION_ENABLED", 0, raising=False)
    knob = _Knob()
    series = _series(knob)
    assert series.start(0) is False
    assert knob.writes == []


def test_a_series_cannot_be_started_twice():
    knob = _Knob()
    series = _series(knob)
    assert series.start(0) is True
    assert series.start(1) is False


# ── the ABAB schedule ────────────────────────────────────────────────

def test_blocks_alternate_between_the_two_levels():
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    levels = [series.level_for(tick) for tick in (0, 4, 8, 12)]
    assert levels == [0.10, 0.20, 0.10, 0.20]


def test_the_parameter_is_written_once_per_block_not_once_per_tick():
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    for tick in range(8):
        series.step(tick, reward=1.0)
    assert len(knob.writes) == 2


def test_observations_are_split_between_the_two_arms():
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    for tick in range(8):
        series.step(tick, reward=float(tick))
    assert len(series.samples[0]) == 4 and len(series.samples[1]) == 4


def test_a_series_runs_at_least_the_minimum_number_of_blocks():
    knob = _Knob()
    series = _series(knob, block_ticks=2)
    series.start(0)
    finished_at = None
    for tick in range(200):
        outcome = series.step(tick, reward=1.0)
        if outcome.get("state") == "finished":
            finished_at = tick
            break
    assert finished_at is not None
    assert len(series.blocks) >= MIN_BLOCKS


def test_a_step_on_an_inactive_series_does_nothing():
    knob = _Knob()
    series = _series(knob)
    assert series.step(0, reward=1.0) == {"state": "inactive"}


# ── it always gives the parameter back ───────────────────────────────

def test_the_original_value_is_restored_when_the_series_finishes():
    knob = _Knob(value=0.15)
    series = _series(knob, block_ticks=2)
    series.start(0)
    for tick in range(200):
        if series.step(tick, reward=1.0).get("state") == "finished":
            break
    assert knob.value == 0.15


def test_the_original_value_is_restored_when_the_series_aborts():
    knob = _Knob(value=0.15)
    series = _series(knob, block_ticks=4)
    series.start(0)
    series.step(0, reward=1.0)
    assert knob.value == 0.10                      # the experiment is running
    series.abort("because")
    assert knob.value == 0.15


def test_restoring_twice_is_harmless():
    knob = _Knob(value=0.15)
    series = _series(knob)
    series.start(0)
    series.step(0, reward=1.0)
    assert series.restore() is True
    assert series.restore() is False
    assert knob.value == 0.15


def test_a_restore_that_fails_is_reported_rather_than_raised():
    class _Broken(_Knob):
        def apply(self, name, value):
            raise RuntimeError("the genome is locked")

    knob = _Broken()
    series = _series(knob)
    series.original = 0.15
    series.variable = "explore_bonus"
    assert series.restore() is False


# ── the named test: it stops immediately, not at a boundary ──────────

def test_a_critical_health_reading_aborts_mid_block_and_restores():
    """The spec's own scenario (M7.10). A series that waited for a block
    boundary would keep making things worse for up to a hundred ticks after
    being told to stop."""
    knob = _Knob(value=0.15)
    series = _series(knob, block_ticks=50)
    series.start(0)
    series.step(0, reward=1.0)
    assert knob.value == 0.10

    outcome = series.step(1, reward=1.0, health="critical")
    assert outcome["state"] == "aborted"
    assert knob.value == 0.15
    assert not series.active


def test_the_kill_switch_aborts_a_series():
    knob = _Knob(value=0.15)
    series = _series(knob, block_ticks=50)
    series.start(0)
    series.step(0, reward=1.0)
    outcome = series.step(1, reward=1.0, kill_switch=True)
    assert outcome["state"] == "aborted"
    assert knob.value == 0.15


def test_a_tick_that_should_abort_contributes_no_observation():
    """Otherwise the abort condition becomes part of the result it was meant to
    prevent."""
    knob = _Knob()
    series = _series(knob, block_ticks=50)
    series.start(0)
    series.step(0, reward=1.0)
    before = len(series.samples[0])
    series.step(1, reward=99.0, health="critical")
    assert len(series.samples[0]) == before


def test_a_sustained_collapse_in_reward_aborts_the_series():
    """``baseline − 2σ`` for half a block. The system is allowed to experiment,
    not to keep experimenting while the experiment is visibly harming it."""
    knob = _Knob(value=0.15)
    series = _series(knob, block_ticks=8)
    series.start(0)
    for tick in range(12):                    # establish a baseline near 1.0
        series.step(tick, reward=1.0 + 0.01 * (tick % 3))
    outcome = {}
    for tick in range(12, 40):
        outcome = series.step(tick, reward=-50.0)
        if outcome.get("state") == "aborted":
            break
    assert outcome.get("state") == "aborted"
    assert "baseline" in series.abort_reason
    assert knob.value == 0.15


def test_a_single_bad_tick_does_not_abort_a_series():
    """Noise is not a collapse. A series that aborted on one reading would
    never complete on any real metric."""
    knob = _Knob()
    series = _series(knob, block_ticks=20)
    series.start(0)
    for tick in range(12):
        series.step(tick, reward=1.0 + 0.01 * (tick % 3))
    assert series.step(12, reward=-50.0).get("state") == "running"


# ── the analysis, exactly as preregistered ───────────────────────────

def test_a_real_difference_between_the_arms_is_supported():
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    for tick in range(32):
        arm = series.block_index(tick) % 2
        series.step(tick, reward=(5.0 if arm else 1.0) + 0.01 * (tick % 3))
    result = series.analyse()
    assert result["status"] == "supported"
    assert result["effect_size"] > 0


def test_no_difference_between_the_arms_is_refuted():
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    for tick in range(32):
        series.step(tick, reward=1.0 + 0.01 * (tick % 5))
    assert series.analyse()["status"] == "refuted"


def test_an_aborted_series_yields_no_verdict():
    """Half a series is not a result. Analysing one would let the abort
    condition select which half was measured."""
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    for tick in range(12):
        series.step(tick, reward=float(tick))
    series.abort("health")
    assert series.analyse()["status"] == "invalid"


def test_too_few_observations_yield_no_verdict():
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    series.step(0, reward=1.0)
    assert series.analyse()["status"] == "pending"


def test_an_altered_plan_invalidates_the_analysis():
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    for tick in range(32):
        series.step(tick, reward=float(tick % 5))
    series.prereg.analysis = "mann_whitney"       # after the fact
    assert series.analyse()["status"] == "invalid"


def test_the_preregistered_analysis_is_the_one_that_runs():
    knob = _Knob()
    prereg = preregister(_Hypothesis(), _Model(), design="interventional_abab",
                         variable="explore_bonus", levels=(0.1, 0.2),
                         analysis="mann_whitney", block_ticks=4)
    series = Intervention(prereg, apply=knob.apply, read=knob.read, block_ticks=4)
    series.start(0)
    for tick in range(32):
        arm = series.block_index(tick) % 2
        series.step(tick, reward=5.0 if arm else 1.0)
    assert series.analyse()["analysis"] == "mann_whitney"


def test_the_status_describes_the_series():
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    series.step(0, reward=1.0)
    status = series.status()
    assert status["variable"] == "explore_bonus" and status["active"] is True
    assert status["blocks"] == 1


# ── a refused start says so, and stays refused ───────────────────────
#
# `start()` returning False is only half the contract. The object also has to
# *remember* that it was refused and why: a series that reported a refusal and
# left itself in a startable state would be restartable by the next caller, and
# the reason is what an operator reads to find out which gate said no.

@pytest.mark.parametrize("break_it,expected", [
    (lambda s: setattr(s.prereg, "levels", (0.0, 0.9)), "frozen"),
    (lambda s: setattr(s, "variable", "ETHICAL_THRESHOLD_AUTO"), "controllable"),
    (lambda s: setattr(s, "levels", [0.1]), "two levels"),
])
def test_a_refused_start_records_that_it_was_refused(break_it, expected):
    knob = _Knob()
    series = _series(knob)
    break_it(series)
    assert series.start(0) is False
    assert series.aborted is True, "a refused series did not mark itself aborted"
    assert expected in series.abort_reason
    assert series.active is False


def test_a_start_refused_by_configuration_records_it(monkeypatch):
    import aegis.config as cfg

    monkeypatch.setattr(cfg, "DISC_INTERVENTION_ENABLED", 0, raising=False)
    knob = _Knob()
    series = _series(knob)
    assert series.start(0) is False
    assert series.aborted is True
    assert "configuration" in series.abort_reason


def test_an_aborted_series_is_no_longer_active():
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    series.step(0, reward=1.0)
    series.abort("because")
    assert series.active is False and series.aborted is True


# ── the block schedule, counted from where it started ────────────────

def test_blocks_are_counted_from_the_tick_the_series_started_on():
    """A series that started at tick 500 is in its first block at tick 500, not
    its hundred-and-twenty-fifth. Counting from zero would put a mid-run series
    straight past its own completion."""
    knob = _Knob()
    series = _series(knob, block_ticks=4, tick=500)
    series.start(500)
    assert series.block_index(500) == 0
    assert series.block_index(503) == 0
    assert series.block_index(504) == 1
    assert series.block_index(511) == 2


def test_a_tick_before_the_start_is_still_the_first_block():
    knob = _Knob()
    series = _series(knob, block_ticks=4, tick=100)
    series.start(100)
    assert series.block_index(50) == 0


def test_each_block_counts_its_own_observations():
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    for tick in range(4):
        series.step(tick, reward=1.0)
    assert series.blocks[0]["n"] == 4


def test_a_series_finishes_on_a_block_boundary_of_the_first_arm():
    """It ends where it began — on an A block — so both arms have had the same
    number of blocks and the comparison is balanced."""
    knob = _Knob()
    series = _series(knob, block_ticks=2)
    series.start(0)
    finished_block = None
    for tick in range(200):
        outcome = series.step(tick, reward=1.0)
        if outcome.get("state") == "finished":
            finished_block = series.block_index(tick)
            break
    assert finished_block is not None
    assert finished_block % 2 == 0
    assert finished_block >= MIN_BLOCKS * 2


def test_a_tick_below_the_baseline_says_so_without_aborting():
    """One bad reading is reported and not counted; it is the *run* of them
    that stops the series."""
    knob = _Knob()
    series = _series(knob, block_ticks=20)
    series.start(0)
    for tick in range(12):
        series.step(tick, reward=1.0 + 0.01 * (tick % 3))
    outcome = series.step(12, reward=-50.0)
    assert outcome["state"] == "running"
    assert outcome.get("below_baseline") is True


# ── the analysis reports what it actually measured ───────────────────

def test_the_reported_count_is_every_observation_in_both_arms():
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    for tick in range(32):
        arm = series.block_index(tick) % 2
        series.step(tick, reward=(5.0 if arm else 1.0) + 0.01 * (tick % 3))
    result = series.analyse()
    assert result["n"] == result["n_a"] + result["n_b"]
    assert result["n"] == len(series.samples[0]) + len(series.samples[1])


def test_one_arm_with_too_little_data_is_pending(monkeypatch):
    """Both arms need observations. A comparison against an arm of one is not a
    weak comparison, it is not a comparison."""
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    for tick in range(5):                     # four in arm A, one in arm B
        series.step(tick, reward=float(tick))
    result = series.analyse()
    assert result["status"] == "pending"
    assert result["n"] == 5


def test_the_rank_test_and_the_t_test_give_different_numbers():
    """Which test ran has to be visible in the result, not only in its label.
    Two arms of constant values are exactly where they disagree: Welch's t is
    undefined and reports nothing, the rank test reports a separation."""
    from aegis.util.stats import compare_samples, mann_whitney_u

    arm_a = [1.0, 1.0, 1.0, 1.0]
    arm_b = [2.0, 2.0, 2.0, 2.0]
    assert mann_whitney_u(arm_b, arm_a).p_value < 0.05
    assert compare_samples(arm_b, arm_a) == mann_whitney_u(arm_b, arm_a)

    knob = _Knob()
    prereg = preregister(_Hypothesis(), _Model(), design="interventional_abab",
                         variable="explore_bonus", levels=(0.1, 0.2),
                         analysis="mann_whitney", block_ticks=4)
    series = Intervention(prereg, apply=knob.apply, read=knob.read, block_ticks=4)
    series.start(0)
    for tick in range(32):
        arm = series.block_index(tick) % 2
        series.step(tick, reward=2.0 if arm else 1.0)
    result = series.analyse()
    assert result["analysis"] == "mann_whitney"
    expected = mann_whitney_u(series.samples[1], series.samples[0])
    assert result["p_value"] == pytest.approx(round(expected.p_value, 8))


def test_a_finished_series_is_no_longer_active():
    """It has already given the parameter back. A series still marked active
    would be stepped again and would re-apply the experimental level to a
    system that had finished experimenting."""
    knob = _Knob()
    series = _series(knob, block_ticks=2)
    series.start(0)
    for tick in range(200):
        if series.step(tick, reward=1.0).get("state") == "finished":
            break
    assert series.active is False
    assert series.step(500, reward=1.0) == {"state": "inactive"}


def test_an_aborted_series_still_reports_how_much_it_had_collected():
    """Invalid, not empty. How far a series got before it was stopped is what
    says whether the abort cost a nearly-complete experiment or an idea."""
    knob = _Knob()
    series = _series(knob, block_ticks=4)
    series.start(0)
    for tick in range(12):
        series.step(tick, reward=float(tick))
    collected = len(series.samples[0]) + len(series.samples[1])
    series.abort("health")
    result = series.analyse()
    assert result["status"] == "invalid"
    assert result["n"] == collected


def test_the_preregistered_test_is_the_one_that_actually_runs():
    """Checked on arms that *have* variance, which is where the two tests
    disagree. On two constant arms Welch's t is undefined and falls through to
    the rank test, so a comparison there cannot tell which one ran — and a
    dispatch that ignored the plan would pass unnoticed.
    """
    from aegis.util.stats import compare_samples, mann_whitney_u

    def _run(analysis):
        knob = _Knob()
        prereg = preregister(_Hypothesis(), _Model(),
                             design="interventional_abab",
                             variable="explore_bonus", levels=(0.1, 0.2),
                             analysis=analysis, block_ticks=4)
        series = Intervention(prereg, apply=knob.apply, read=knob.read,
                              block_ticks=4)
        series.start(0)
        for tick in range(32):
            arm = series.block_index(tick) % 2
            series.step(tick, reward=(5.0 if arm else 1.0) + 0.4 * (tick % 5))
        return series

    ranked = _run("mann_whitney")
    welch = _run("welch_t")
    arm_a, arm_b = ranked.samples[0], ranked.samples[1]

    # The two tests genuinely disagree on this data, so the comparison below is
    # about which one ran rather than about arithmetic that happens to match.
    assert mann_whitney_u(arm_b, arm_a).p_value != \
        pytest.approx(compare_samples(arm_b, arm_a).p_value)

    assert ranked.analyse()["p_value"] == pytest.approx(
        round(mann_whitney_u(arm_b, arm_a).p_value, 8))
    assert welch.analyse()["p_value"] == pytest.approx(
        round(compare_samples(arm_b, arm_a).p_value, 8))
