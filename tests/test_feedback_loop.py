"""Unit tests for System 5: FeedbackLoop (situation->result->cause->experience)."""
from aegis.layers.feedback_loop import FeedbackLoop


def _fb(tmp_path):
    return FeedbackLoop(store_path=tmp_path / "exp.jsonl")


def test_record_situation_returns_id(tmp_path):
    fb = _fb(tmp_path)
    eid = fb.record_situation("low energy", "rest", {"tick": 1})
    assert eid.startswith("exp_")
    assert eid in fb._open


def test_record_result_closes_experience(tmp_path):
    fb = _fb(tmp_path)
    eid = fb.record_situation("situation", "decision", {"tick": 1})
    exp = fb.record_result(eid, success=True, metric=0.8)
    assert exp is not None
    assert exp["success"] is True
    assert exp["metric"] == 0.8
    assert eid not in fb._open  # no longer open
    assert fb.resolved == 1
    assert fb.successes == 1


def test_record_result_unknown_id_returns_none(tmp_path):
    fb = _fb(tmp_path)
    assert fb.record_result("nonexistent", success=True, metric=0.5) is None


def test_cause_inference(tmp_path):
    fb = _fb(tmp_path)
    e1 = fb.record_situation("s", "d")
    exp = fb.record_result(e1, success=False, metric=0.1)
    assert "failed hard" in exp["cause"]
    e2 = fb.record_situation("s", "d")
    exp2 = fb.record_result(e2, success=True, metric=0.9)
    assert "high verified metric" in exp2["cause"]


def test_explicit_cause_preserved(tmp_path):
    fb = _fb(tmp_path)
    eid = fb.record_situation("s", "d")
    exp = fb.record_result(eid, success=True, metric=0.7, cause="custom reason")
    assert exp["cause"] == "custom reason"


def test_success_rate(tmp_path):
    fb = _fb(tmp_path)
    for i in range(4):
        eid = fb.record_situation("s", "d")
        fb.record_result(eid, success=(i < 3), metric=0.5)
    assert fb.success_rate() == 0.75


def test_export_examples_as_training_rows(tmp_path):
    fb = _fb(tmp_path)
    eid = fb.record_situation("open company", "consult_legal", {"tick": 1})
    fb.record_result(eid, success=True, metric=0.9, cause="matched")
    rows = fb.export_examples()
    assert rows
    assert "Situation" in rows[0]["prompt"]
    assert "success" in rows[0]["completion"]
    assert rows[0]["success"] is True


def test_open_experiences_are_bounded(tmp_path, monkeypatch):
    import aegis.layers.feedback_loop as fbmod
    monkeypatch.setattr(fbmod, "MAX_OPEN", 5)
    fb = _fb(tmp_path)
    for i in range(20):
        fb.record_situation(f"s{i}", "d")
    assert len(fb._open) <= 5


def test_persistence_appends_to_jsonl(tmp_path):
    p = tmp_path / "exp.jsonl"
    fb = FeedbackLoop(store_path=p)
    eid = fb.record_situation("s", "d")
    fb.record_result(eid, success=True, metric=0.7)
    assert p.exists()
    content = p.read_text(encoding="utf-8").strip()
    assert content  # one JSON line written
    assert content.count("\n") == 0  # exactly one row


def test_export_survives_restart(tmp_path):
    p = tmp_path / "exp.jsonl"
    fb = FeedbackLoop(store_path=p)
    eid = fb.record_situation("s", "d")
    fb.record_result(eid, success=True, metric=0.7)
    fb2 = FeedbackLoop(store_path=p)  # fresh instance, same file
    rows = fb2.export_examples()
    assert len(rows) == 1


def test_status_shape(tmp_path):
    fb = _fb(tmp_path)
    eid = fb.record_situation("s", "d")
    fb.record_result(eid, success=True, metric=0.6)
    st = fb.status()
    assert st["resolved"] == 1
    assert st["success_rate"] == 1.0
    assert isinstance(st["recent"], list)


# ── mutation-hardening tests ──────────────────────────────────────────

def test_default_store_path_is_used(tmp_path, monkeypatch):
    import aegis.layers.feedback_loop as fbmod
    monkeypatch.setattr(fbmod, "FEEDBACK_DIR", tmp_path)
    fb = FeedbackLoop()
    assert fb._store_path == tmp_path / "experiences.jsonl"


def test_open_experiences_bounded_to_exactly_max(tmp_path, monkeypatch):
    # Kills the `> MAX_OPEN` boundary and `len - MAX_OPEN` slice-count mutants.
    import aegis.layers.feedback_loop as fbmod
    monkeypatch.setattr(fbmod, "MAX_OPEN", 5)
    fb = _fb(tmp_path)
    for i in range(20):
        fb.record_situation(f"s{i}", "d")
    assert len(fb._open) == 5   # exactly the cap, not 0 and not 20


def test_latency_is_nonnegative_and_small(tmp_path):
    # Kills the `time.time() - opened` -> `time.time() + opened` mutant, which
    # would produce an astronomically large latency.
    fb = _fb(tmp_path)
    eid = fb.record_situation("s", "d")
    exp = fb.record_result(eid, success=True, metric=0.5)
    assert 0.0 <= exp["latency_s"] < 10.0


def test_recent_list_capped_at_50(tmp_path):
    # Kills the `len(recent) > 50` boundary mutant.
    fb = _fb(tmp_path)
    for i in range(60):
        eid = fb.record_situation(f"s{i}", "d")
        fb.record_result(eid, success=True, metric=0.5)
    assert len(fb.recent) == 50


def test_cause_weak_margin_when_success_but_low_metric(tmp_path):
    # Kills the `success and metric >= 0.7` And->Or mutant: a success with a
    # LOW metric must be "weak margin", not "high verified metric".
    fb = _fb(tmp_path)
    eid = fb.record_situation("s", "d")
    exp = fb.record_result(eid, success=True, metric=0.5)  # success, but <0.7
    assert "weak margin" in exp["cause"]
    assert "high verified" not in exp["cause"]


def test_jsonl_log_is_truncated_when_over_budget(tmp_path, monkeypatch):
    # Kills the `len(lines) <= MAX*2` boundary and `MAX*2` Mult->Div mutants in
    # _truncate_if_needed.
    import aegis.layers.feedback_loop as fbmod
    MAX_ROWS = 2
    monkeypatch.setattr(fbmod, "MAX_JSONL_ROWS", MAX_ROWS)  # threshold = 4 rows
    p = tmp_path / "exp.jsonl"
    fb = FeedbackLoop(store_path=p)
    for i in range(6):   # exceeds 2*MAX -> truncation fires, then re-accumulates
        eid = fb.record_situation(f"s{i}", "d")
        fb.record_result(eid, success=True, metric=0.5)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Correct: bounded strictly between MAX and 2*MAX (settles at 3 here).
    # `> MAX*2` (never truncates) would grow to 6; `MAX/2` would clamp to 2.
    assert MAX_ROWS < len(lines) <= 2 * MAX_ROWS
