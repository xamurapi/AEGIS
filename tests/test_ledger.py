"""The register of what is known (spec M7.8, M7.11).

Two rules are doing the work.

**A refutation is permanent.** It is knowledge in its own right — a relationship
that looked real is not — and it is the anti-rediscovery mechanism. Without it
the scan proposes the same appealing pattern every thousand ticks, spends an
experiment on it every time, and never learns that it has already answered the
question.

**Replication means a different window.** Re-analysing the same rows twice is
arithmetic performed twice. A ledger that counted it as replication would
promote a single result to a law by reading it three times.
"""
import pytest

from aegis.layers.discovery.ledger import MAX_ENTRIES, OPEN, STATUSES, Discovery, Ledger


class _Hypothesis:
    def __init__(self, identifier="hyp_1", target="y", predictors=("x",)):
        self.id = identifier
        self._target = target
        self._predictors = predictors

    def as_dict(self):
        return {"id": self.id, "target": self._target,
                "predictors": list(self._predictors)}


class _Model:
    def as_dict(self):
        return {"expr": "0.5 + 2*x", "r2_valid": 0.95}


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.json", law_reps=3)


def _supported(effect=1.0, p=0.001):
    return {"status": "supported", "effect_size": effect, "p_value": p}


# ── opening a record ─────────────────────────────────────────────────

def test_a_hypothesis_becomes_a_proposed_discovery(ledger):
    record = ledger.propose(_Hypothesis(), _Model(), tick=10)
    assert record.status == "proposed"
    assert record.first_tick == 10
    assert record.formula == "0.5 + 2*x"


def test_proposing_the_same_hypothesis_twice_reuses_the_record(ledger):
    first = ledger.propose(_Hypothesis(), _Model(), tick=1)
    second = ledger.propose(_Hypothesis(), _Model(), tick=2)
    assert first is second
    assert first.first_tick == 1


def test_a_hypothesis_with_no_identity_is_refused(ledger):
    assert ledger.propose({}, _Model()) is None
    assert ledger.rejected == 1


def test_a_plain_mapping_is_accepted_as_a_hypothesis(ledger):
    assert ledger.propose({"id": "hyp_x", "target": "y"}) is not None


# ── the status ladder ────────────────────────────────────────────────

def test_a_supported_experiment_moves_a_proposal_to_supported(ledger):
    ledger.propose(_Hypothesis(), _Model())
    record = ledger.record_result("hyp_1", _supported(), tick=20, window=(0, 100))
    assert record.status == "supported"
    assert record.effect_size == 1.0


def test_a_second_window_makes_it_replicated(ledger):
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", _supported(), tick=20, window=(0, 100))
    record = ledger.record_result("hyp_1", _supported(), tick=40, window=(200, 300))
    assert record.status == "replicated"
    assert record.replications == 1


def test_enough_stable_replications_make_a_law(ledger):
    ledger.propose(_Hypothesis(), _Model())
    for index, window in enumerate([(0, 100), (200, 300), (400, 500)]):
        ledger.record_result("hyp_1", _supported(), tick=20 * index, window=window)
    assert ledger.get("hyp_1").status == "law"


def test_a_swinging_effect_is_not_a_law(ledger):
    """Significant every time but with a size that swings is not one law; it is
    several different effects sharing a name."""
    ledger.propose(_Hypothesis(), _Model())
    for effect, window in [(1.0, (0, 100)), (1.0, (200, 300)), (0.01, (400, 500))]:
        ledger.record_result("hyp_1", _supported(effect), tick=1, window=window)
    assert ledger.get("hyp_1").status == "replicated"


def test_a_refuting_experiment_refutes_it(ledger):
    ledger.propose(_Hypothesis(), _Model())
    record = ledger.record_result("hyp_1", {"status": "refuted"}, tick=5)
    assert record.status == "refuted"


def test_an_invalid_experiment_is_recorded_as_invalid(ledger):
    ledger.propose(_Hypothesis(), _Model())
    record = ledger.record_result("hyp_1", {"status": "invalid",
                                            "reason": "the plan changed"}, tick=5)
    assert record.status == "invalid"


def test_a_pending_experiment_changes_nothing(ledger):
    ledger.propose(_Hypothesis(), _Model())
    record = ledger.record_result("hyp_1", {"status": "pending"}, tick=5)
    assert record.status == "proposed"


def test_a_result_for_an_unknown_discovery_is_refused(ledger):
    assert ledger.record_result("nobody", _supported()) is None


def test_a_result_that_is_not_a_mapping_is_refused(ledger):
    ledger.propose(_Hypothesis(), _Model())
    assert ledger.record_result("hyp_1", "supported!") is None


# ── the two rules ────────────────────────────────────────────────────

def test_a_refuted_hypothesis_is_never_reopened(ledger):
    """The anti-rediscovery gate, and the reason refutations are kept."""
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", {"status": "refuted"}, tick=5)
    assert ledger.propose(_Hypothesis(), _Model(), tick=999) is None
    assert ledger.is_refuted("hyp_1") is True


def test_an_overlapping_window_is_not_a_replication(ledger):
    """Re-analysing the same rows is arithmetic performed twice."""
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", _supported(), tick=20, window=(0, 100))
    record = ledger.record_result("hyp_1", _supported(), tick=30, window=(50, 150))
    assert record.status == "supported"
    assert record.replications == 0


def test_a_touching_window_still_overlaps(ledger):
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", _supported(), tick=20, window=(0, 100))
    record = ledger.record_result("hyp_1", _supported(), tick=30, window=(100, 200))
    assert record.replications == 0


def test_a_result_with_no_window_does_not_count_as_replication(ledger):
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", _supported(), tick=20, window=(0, 100))
    record = ledger.record_result("hyp_1", _supported(), tick=30)
    assert record.replications == 1        # no window given: taken at its word


# ── application (M7.9) ───────────────────────────────────────────────

def test_a_supported_discovery_can_be_marked_as_applied(ledger):
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", _supported(), window=(0, 100))
    assert ledger.note_application("hyp_1", "world_model", tick=7) is True
    assert ledger.get("hyp_1").applications == ["world_model"]


def test_an_unproven_discovery_cannot_be_applied(ledger):
    """Applying something the experiments have not supported is exactly the
    mistake the ladder exists to prevent."""
    ledger.propose(_Hypothesis(), _Model())
    assert ledger.note_application("hyp_1", "world_model") is False


def test_the_same_application_is_not_recorded_twice(ledger):
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", _supported(), window=(0, 100))
    ledger.note_application("hyp_1", "policy")
    ledger.note_application("hyp_1", "policy")
    assert ledger.get("hyp_1").applications == ["policy"]


def test_an_applied_discovery_can_be_sent_back_for_re_testing(ledger):
    """A discovery whose application made things worse is not knowledge yet,
    whatever the experiment said."""
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", _supported(), window=(0, 100))
    assert ledger.retest("hyp_1", "the metric fell", tick=9) is True
    assert ledger.get("hyp_1").status == "proposed"


def test_a_refuted_discovery_is_not_re_tested(ledger):
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", {"status": "refuted"})
    assert ledger.retest("hyp_1", "why not") is False


def test_re_testing_something_unknown_is_refused(ledger):
    assert ledger.retest("nobody", "reason") is False


# ── reading ──────────────────────────────────────────────────────────

def test_discoveries_can_be_read_by_status(ledger):
    for index in range(3):
        ledger.propose(_Hypothesis(f"hyp_{index}"), _Model())
    ledger.record_result("hyp_0", _supported(), window=(0, 10))
    assert [record.id for record in ledger.by_status("supported")] == ["hyp_0"]
    assert len(ledger.by_status("proposed")) == 2


def test_the_counts_cover_every_declared_status(ledger):
    counts = ledger.counts()
    assert set(counts) == set(STATUSES)


def test_open_statuses_are_the_ones_still_workable():
    assert set(OPEN) <= set(STATUSES)
    assert "refuted" not in OPEN


def test_laws_are_reported_separately(ledger):
    ledger.propose(_Hypothesis(), _Model())
    for window in [(0, 100), (200, 300), (400, 500)]:
        ledger.record_result("hyp_1", _supported(), window=window)
    assert [record.id for record in ledger.laws()] == ["hyp_1"]


# ── persistence ──────────────────────────────────────────────────────

def test_a_ledger_reloads_what_it_recorded(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.propose(_Hypothesis(), _Model(), tick=3)
    ledger.record_result("hyp_1", _supported(), tick=5, window=(0, 100))
    ledger.note_application("hyp_1", "planner")
    assert ledger.save() is True

    reloaded = Ledger(tmp_path / "ledger.json")
    record = reloaded.get("hyp_1")
    assert record.status == "supported"
    assert record.applications == ["planner"]
    assert record.windows == [[0, 100]]


def test_a_refutation_survives_a_restart(tmp_path):
    """The anti-rediscovery guard is worthless if it does not persist."""
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", {"status": "refuted"})
    ledger.save()

    assert Ledger(tmp_path / "ledger.json").is_refuted("hyp_1") is True


def test_a_missing_store_loads_empty(tmp_path):
    assert Ledger(tmp_path / "absent.json").counts()["proposed"] == 0


def test_a_corrupt_store_does_not_stop_the_ledger(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{broken", encoding="utf-8")
    assert Ledger(path).entries == {}


@pytest.mark.parametrize("bad", [None, {}, "text", {"no_id": 1}])
def test_a_malformed_entry_is_not_a_discovery(bad):
    assert Discovery.from_dict(bad) is None


def test_an_entry_with_an_unknown_status_falls_back_to_proposed():
    record = Discovery.from_dict({"id": "x", "status": "brilliant"})
    assert record.status == "proposed"


def test_an_entry_with_unusable_numbers_is_not_a_discovery():
    assert Discovery.from_dict({"id": "x", "p_value": "small"}) is None


# ── the cap ──────────────────────────────────────────────────────────

def test_the_ledger_is_capped(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", max_entries=10)
    for index in range(30):
        ledger.propose(_Hypothesis(f"hyp_{index:03d}"), _Model(), tick=index)
    assert len(ledger.entries) <= 10


def test_refutations_and_laws_are_never_dropped_for_age(tmp_path):
    """Dropping a refutation would let the scan rediscover it, which is the one
    thing the permanent record exists to prevent."""
    ledger = Ledger(tmp_path / "ledger.json", max_entries=5)
    ledger.propose(_Hypothesis("hyp_refuted"), _Model(), tick=0)
    ledger.record_result("hyp_refuted", {"status": "refuted"}, tick=0)
    for index in range(40):
        ledger.propose(_Hypothesis(f"hyp_{index:03d}"), _Model(), tick=index + 1)
    assert ledger.is_refuted("hyp_refuted") is True


def test_the_default_cap_is_the_documented_one(tmp_path):
    assert Ledger(tmp_path / "ledger.json").max_entries == MAX_ENTRIES


def test_the_status_summarises_the_register(ledger):
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", _supported(), window=(0, 10))
    ledger.note_application("hyp_1", "planner")
    status = ledger.status()
    assert status["total"] == 1 and status["supported"] == 1
    assert status["applied"] == 1


# ── the cap, exactly ─────────────────────────────────────────────────

def test_nothing_is_dropped_while_under_the_cap(tmp_path):
    """Trimming is what happens when there is too much, and only then. A
    ledger that dropped entries while it had room would lose findings for no
    reason at all."""
    ledger = Ledger(tmp_path / "ledger.json", max_entries=10)
    for index in range(10):
        ledger.propose(_Hypothesis(f"hyp_{index:03d}"), _Model(), tick=index)
    assert len(ledger.entries) == 10


def test_trimming_drops_exactly_the_overflow(tmp_path):
    """Not more. The oldest workable entries go until the ledger fits, and one
    off in the count is the difference between keeping the cap and emptying the
    register."""
    ledger = Ledger(tmp_path / "ledger.json", max_entries=10)
    for index in range(15):
        ledger.propose(_Hypothesis(f"hyp_{index:03d}"), _Model(), tick=index)
    assert len(ledger.entries) == 10
    # The five oldest went; the five newest stayed.
    assert ledger.get("hyp_014") is not None
    assert ledger.get("hyp_000") is None


def test_the_default_path_is_under_the_discovery_directory(tmp_path, monkeypatch):
    """Built by joining, not by string arithmetic. A ledger that could not
    construct its own default path would fail only in the one configuration
    nobody tests: the real one."""
    import aegis.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path, raising=False)
    ledger = Ledger()
    assert ledger.path.name == "ledger.json"
    assert ledger.path.parent.name == "discovery"


# ── stability, which decides what becomes a law ──────────────────────

def test_a_first_measurement_is_not_yet_unstable(ledger):
    """One reading cannot swing. Calling it unstable would make the first
    confirmation of every discovery impossible."""
    record = Discovery(id="hyp_one", effect_size=1.0)
    assert Ledger._stable(record) is True


def test_an_effect_that_halves_is_still_the_same_effect(ledger):
    record = Discovery(id="hyp_x", effect_size=0.6,
                       history=[{"effect": 1.0}])
    assert Ledger._stable(record) is True


def test_a_small_effect_that_swings_is_not_stable():
    """The ratio is what matters, not the product: two effects of 0.4 and 0.3
    are stable at 0.75 of each other, and a check that multiplied them would
    call every small effect unstable and every large one stable."""
    swinging = Discovery(id="hyp_a", effect_size=0.1, history=[{"effect": 1.0}])
    assert Ledger._stable(swinging) is False

    steady = Discovery(id="hyp_b", effect_size=0.3, history=[{"effect": 0.4}])
    assert Ledger._stable(steady) is True


def test_an_effect_of_nothing_is_not_a_stable_effect():
    record = Discovery(id="hyp_zero", effect_size=0.0, history=[{"effect": 0.0}])
    assert Ledger._stable(record) is False


def test_the_law_transition_says_how_many_confirmations_it_took(ledger):
    """The count in the record is what an operator audits a law by."""
    ledger.propose(_Hypothesis(), _Model())
    for window in [(0, 100), (200, 300), (400, 500)]:
        ledger.record_result("hyp_1", _supported(), window=window)
    record = ledger.get("hyp_1")
    assert record.status == "law"
    promotion = [entry for entry in record.history if entry.get("to") == "law"][0]
    assert "3 confirmations" in promotion["reason"]


# ── malformed records ────────────────────────────────────────────────

def test_a_window_that_is_not_a_pair_is_dropped_on_load():
    """A window is what replication is judged against. One that survived
    loading in the wrong shape would make `overlaps` compare against garbage."""
    record = Discovery.from_dict({
        "id": "hyp_x",
        "windows": [[0, 100], "not a window", [1, 2, 3], [200, 300], {}],
    })
    assert record.windows == [[0, 100], [200, 300]]


def test_a_confidence_interval_is_only_taken_when_it_is_a_pair(ledger):
    ledger.propose(_Hypothesis(), _Model())
    ledger.record_result("hyp_1", {**_supported(), "ci": [0.1, 0.9]},
                         window=(0, 100))
    assert ledger.get("hyp_1").ci == (0.1, 0.9)

    ledger.record_result("hyp_1", {**_supported(), "ci": [0.1, 0.5, 0.9]},
                         window=(200, 300))
    assert ledger.get("hyp_1").ci == (0.1, 0.9), "a three-element ci was taken"

    ledger.record_result("hyp_1", {**_supported(), "ci": "wide"},
                         window=(400, 500))
    assert ledger.get("hyp_1").ci == (0.1, 0.9), "a non-sequence ci was taken"
