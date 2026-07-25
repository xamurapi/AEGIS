"""Unit tests for System 1: WorldModel (causal cause->effect model)."""
from aegis.layers.world_model import WorldModel


def _wm(tmp_path):
    return WorldModel(store_path=tmp_path / "wm.json")


def test_observe_records_link_and_strength(tmp_path):
    wm = _wm(tmp_path)
    for _ in range(3):
        wm.observe("open_company", "legal_review", success=True)
    preds = wm.predict("open_company")
    assert preds
    assert preds[0]["effect"] == "legal_review"
    # 3/3 successes, Laplace-smoothed (3+1)/(3+2) = 0.8
    assert preds[0]["strength"] == 0.8


def test_prediction_needs_minimum_observations(tmp_path):
    wm = _wm(tmp_path)
    wm.observe("a", "b", success=True)  # only one observation
    assert wm.predict("a") == []  # below MIN_OBSERVATIONS_FOR_PREDICTION


def test_predict_ranks_stronger_effect_first(tmp_path):
    wm = _wm(tmp_path)
    for _ in range(4):
        wm.observe("deploy", "success", success=True)
    for _ in range(4):
        wm.observe("deploy", "rollback", success=False)
    preds = wm.predict("deploy")
    assert preds[0]["effect"] == "success"
    assert preds[0]["strength"] > preds[1]["strength"]


def test_explain_finds_causes(tmp_path):
    wm = _wm(tmp_path)
    for _ in range(2):
        wm.observe("cause_x", "effect_y", success=True)
    causes = wm.explain("effect_y")
    assert causes and causes[0]["cause"] == "cause_x"


def test_explain_excludes_low_observation_links(tmp_path):
    # Kills the And->Or mutant in explain's filter: obs=1 (<MIN) must be excluded.
    wm = _wm(tmp_path)
    wm.observe("c", "eff", success=True)  # single observation
    assert wm.explain("eff") == []


def test_explain_sorted_strongest_cause_first(tmp_path):
    # Kills the explain sort-direction mutant.
    wm = _wm(tmp_path)
    for _ in range(4):
        wm.observe("strong_cause", "eff", success=True)   # high strength
    for _ in range(2):
        wm.observe("weak_cause", "eff", success=True)
        wm.observe("weak_cause", "eff", success=False)    # lower strength
    causes = wm.explain("eff")
    assert causes[0]["cause"] == "strong_cause"
    assert causes[0]["strength"] > causes[1]["strength"]


def test_risks_sorted_highest_failure_first(tmp_path):
    # Kills the risks sort-direction mutant.
    wm = _wm(tmp_path)
    for _ in range(3):
        wm.observe("risky total_loss", "gone", success=False)  # 0/3 -> fail 0.8
    # "risky mild": 1 success / 3 fail -> (1+1)/(4+2)=0.333 -> fail 0.667 (<0.8)
    wm.observe("risky mild", "dip", success=True)
    for _ in range(3):
        wm.observe("risky mild", "dip", success=False)
    risks = wm.risks_for(["risky"])
    assert len(risks) >= 2
    assert risks[0]["failure_rate"] >= risks[1]["failure_rate"]
    assert risks[0]["cause"] == "risky total_loss"


def test_risks_surface_weak_links(tmp_path):
    wm = _wm(tmp_path)
    for _ in range(5):
        wm.observe("risky_action", "failure", success=False)
    risks = wm.risks_for(["risky"])
    assert risks
    assert risks[0]["failure_rate"] > 0.5


def test_build_chain_structure(tmp_path):
    wm = _wm(tmp_path)
    for _ in range(3):
        wm.observe("expand_knowledge", "new_concept", success=True)
    chain = wm.build_chain("expand_knowledge", constraints=["preserve_safety"])
    assert chain["objective"] == "expand_knowledge"
    assert "preserve_safety" in chain["constraints"]
    assert chain["plan"]
    assert chain["expected_result"] != "unknown — no causal data yet"
    assert 0.0 <= chain["confidence"] <= 1.0


def test_build_chain_without_data_is_safe(tmp_path):
    wm = _wm(tmp_path)
    chain = wm.build_chain("unknown_objective")
    assert chain["plan"] == []
    assert chain["confidence"] == 0.0


def test_refine_chain_validates_shape(tmp_path):
    wm = _wm(tmp_path)
    assert wm.refine_chain({"no_objective": 1}) is None
    good = wm.refine_chain({
        "objective": "launch",
        "constraints": ["budget"],
        "plan": ["step1", "step2"],
        "risks": ["market"],
        "expected_result": "revenue",
        "confidence": 0.7,
    })
    assert good is not None
    assert good["source"] == "llm"
    assert good["plan"][0]["action"] == "step1"


def test_pruning_bounds_links(tmp_path, monkeypatch):
    import aegis.layers.world_model as wmmod
    monkeypatch.setattr(wmmod, "MAX_LINKS", 10)
    wm = _wm(tmp_path)
    for i in range(50):
        wm.observe(f"cause_{i}", f"effect_{i}", success=True)
    total = sum(len(v) for v in wm.links.values())
    # Exactly at the cap, not below: the `total - MAX_LINKS` slice must drop only
    # the overflow (kills the Sub->Add mutant, which would drop everything to 0).
    assert total == 10


def test_persistence_round_trip(tmp_path):
    p = tmp_path / "wm.json"
    wm = WorldModel(store_path=p)
    for _ in range(3):
        wm.observe("x", "y", success=True)
    wm.save()
    wm2 = WorldModel(store_path=p)
    assert wm2.total_observations == 3
    assert wm2.predict("x")


def test_status_shape(tmp_path):
    wm = _wm(tmp_path)
    for _ in range(2):
        wm.observe("a", "b", success=True)
    st = wm.status()
    assert st["links"] == 1
    assert st["total_observations"] == 2
    assert isinstance(st["strongest_links"], list)


def test_corrupt_store_starts_empty(tmp_path):
    p = tmp_path / "wm.json"
    p.write_text("{ not json", encoding="utf-8")
    wm = WorldModel(store_path=p)  # must not raise
    assert wm.total_observations == 0


# ── mutation-hardening tests (pin exact values, boundaries, directions) ──

def test_observe_defaults_to_success(tmp_path):
    # Kills the `success: bool = True` default-const mutant.
    wm = _wm(tmp_path)
    wm.observe("a", "b")  # no success arg -> should count as success
    wm.observe("a", "b")
    # 2 successes / 2 obs -> Laplace (2+1)/(2+2) = 0.75, NOT 0.25
    assert wm.predict("a")[0]["strength"] == 0.75


def test_strength_is_laplace_smoothed_exactly(tmp_path):
    # Kills Add/Sub and Div/Mult mutants in _strength.
    wm = _wm(tmp_path)
    wm.observe("a", "b", success=True)
    wm.observe("a", "b", success=False)
    # 1 success / 2 obs -> (1+1)/(2+2) = 0.5
    assert wm.predict("a")[0]["strength"] == 0.5


def test_min_observations_boundary(tmp_path):
    # Exactly MIN_OBSERVATIONS_FOR_PREDICTION (2) must qualify; 1 must not.
    wm = _wm(tmp_path)
    wm.observe("one", "x", success=True)
    assert wm.predict("one") == []          # 1 obs -> below threshold
    wm.observe("one", "x", success=True)
    assert wm.predict("one")                # 2 obs -> at threshold, qualifies


def test_predict_sort_is_strongest_first_exact(tmp_path):
    wm = _wm(tmp_path)
    for _ in range(2):                       # weak: 2/2 wait -> make it weak
        wm.observe("d", "weak", success=False)
    for _ in range(2):
        wm.observe("d", "strong", success=True)
    preds = wm.predict("d")
    # strong (3/4=0.75) must rank strictly before weak (1/4=0.25)
    assert preds[0]["effect"] == "strong"
    assert preds[0]["strength"] > preds[1]["strength"]


def test_risks_threshold_excludes_strong_links(tmp_path):
    # A strong link (s >= 0.5) must NOT appear as a risk (kills the s<0.5 flip).
    wm = _wm(tmp_path)
    for _ in range(4):
        wm.observe("safe_action", "ok", success=True)  # s high
    assert wm.risks_for(["safe"]) == []


def test_risks_include_only_matching_token(tmp_path):
    # Kills the And->Or / any() mutants in the token filter.
    wm = _wm(tmp_path)
    for _ in range(3):
        wm.observe("alpha_task", "fail", success=False)
    for _ in range(3):
        wm.observe("beta_task", "fail", success=False)
    risks = wm.risks_for(["alpha"])
    assert risks
    assert all("alpha" in r["cause"] for r in risks)


def test_build_chain_confidence_is_mean_of_steps(tmp_path):
    wm = _wm(tmp_path)
    # Two solid links under the same objective token.
    for _ in range(4):
        wm.observe("grow revenue", "hire", success=True)   # 5/6
    chain = wm.build_chain("grow revenue")
    assert chain["plan"]
    # confidence equals the mean of the plan-step confidences
    mean = round(sum(s["confidence"] for s in chain["plan"]) / len(chain["plan"]), 3)
    assert chain["confidence"] == mean
    assert chain["confidence"] > 0


def test_chains_are_bounded(tmp_path, monkeypatch):
    # Kills the `len(chains) > MAX_CHAINS` boundary mutant.
    import aegis.layers.world_model as wmmod
    monkeypatch.setattr(wmmod, "MAX_CHAINS", 3)
    wm = _wm(tmp_path)
    for i in range(10):
        wm.build_chain(f"objective_{i}")
    assert len(wm.chains) <= 3


def test_refine_chain_confidence_clamped(tmp_path):
    # Kills clamp mutants (min/max) in refine_chain confidence.
    wm = _wm(tmp_path)
    hi = wm.refine_chain({"objective": "o", "confidence": 5.0})
    assert hi["confidence"] == 1.0
    lo = wm.refine_chain({"objective": "o", "confidence": -3.0})
    assert lo["confidence"] == 0.0


def test_default_store_path_is_used(tmp_path, monkeypatch):
    import aegis.layers.world_model as wmmod
    monkeypatch.setattr(wmmod, "WORLD_MODEL_DIR", tmp_path)
    wm = WorldModel()
    assert wm._store_path == tmp_path / "model.json"


def test_refine_chain_also_bounds_chain_list(tmp_path, monkeypatch):
    # Kills the `len(chains) > MAX_CHAINS` boundary mutant in refine_chain.
    import aegis.layers.world_model as wmmod
    monkeypatch.setattr(wmmod, "MAX_CHAINS", 3)
    wm = _wm(tmp_path)
    for i in range(10):
        wm.refine_chain({"objective": f"o{i}", "confidence": 0.5})
    assert len(wm.chains) <= 3


def test_plan_steps_sorted_and_confidence_is_true_mean(tmp_path):
    # Two links under the same objective token, different strengths. Kills the
    # plan sort-direction mutant AND the confidence sum/len -> sum*len mutant
    # (which are equivalent only when there is a single step).
    wm = _wm(tmp_path)
    # "grow sales": 3/3 successes -> (3+1)/(3+2) = 0.8
    for _ in range(3):
        wm.observe("grow sales", "hire", success=True)
    # "grow team": 1 success, 3 fail -> (1+1)/(4+2) = 0.333 (excluded, <0.5)
    #   use 2 success / 2 fail -> (2+1)/(4+2) = 0.5 (included)
    for _ in range(2):
        wm.observe("grow team", "train", success=True)
    for _ in range(2):
        wm.observe("grow team", "train", success=False)
    chain = wm.build_chain("grow the business")
    confs = [s["confidence"] for s in chain["plan"]]
    assert len(confs) >= 2
    assert confs == sorted(confs, reverse=True)          # strongest first
    expected_mean = round(sum(confs) / len(confs), 3)
    assert chain["confidence"] == expected_mean
    # sum*len would give a different (larger) number for >=2 steps
    assert chain["confidence"] != round(sum(confs) * len(confs), 3)


def test_plan_excludes_high_strength_low_observation_link(tmp_path):
    # A single successful observation: strength 0.667 (>=0.5) but obs=1 (<MIN).
    # Kills the And->Or mutant in the plan filter — it must be EXCLUDED.
    wm = _wm(tmp_path)
    wm.observe("launch product", "ship", success=True)  # obs=1 only
    chain = wm.build_chain("launch product")
    assert chain["plan"] == []


def test_risk_failure_rate_is_one_minus_strength_exactly(tmp_path):
    # Kills the `1 - s` -> `1 + s` mutant.
    wm = _wm(tmp_path)
    for _ in range(3):
        wm.observe("bad move", "loss", success=False)  # 0 succ/3 -> (0+1)/(3+2)=0.2
    risk = wm.risks_for(["bad"])[0]
    assert risk["failure_rate"] == round(1 - 0.2, 3)     # 0.8, not 1.2


def test_status_strongest_links_ordered_by_observations(tmp_path):
    # Kills the status sort-direction mutant.
    wm = _wm(tmp_path)
    for _ in range(5):
        wm.observe("frequent", "e", success=True)
    for _ in range(2):
        wm.observe("rare", "e", success=True)
    strongest = wm.status()["strongest_links"]
    assert strongest[0]["cause"] == "frequent"           # most observed first


def test_objective_token_length_filter(tmp_path):
    # Kills the `len(t) > 2` -> `len(t) <= 2` mutant. With the correct filter a
    # two-letter objective yields NO tokens, so no token filtering is applied
    # and an unrelated strong link still becomes a plan step. The mutant would
    # instead keep "go" as a token and exclude the unrelated link.
    wm = _wm(tmp_path)
    for _ in range(3):
        wm.observe("swim daily", "fit", success=True)
    chain = wm.build_chain("go")   # single 2-letter word
    assert chain["plan"]           # unrelated link is included -> non-empty
