"""pytest-bdd step definitions for tests/features/metacognition.feature.

Executable Gherkin over the real facade: the engine, the arena, the ablation
worker and the skeleton catalogue all run for real — nothing here stubs the
machinery whose behaviour the feature file describes.
"""
import asyncio

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

import aegis.config as cfg
from aegis.layers.metacognition import MetaCognition
from aegis.layers.metacognition.distance import distance
from aegis.layers.reasoning import ReasoningEngine
from aegis.layers.reasoning.weakness import Weakness

scenarios("features/metacognition.feature")


def _weakness(family="missing_data"):
    return Weakness(combo=(f"family={family}",), fail_rate=0.9, base_rate=0.3,
                    support=40, fails=36, lower=0.7, excess=0.6,
                    p_value=0.001, rank=24.0, family=family, examples=())


@given("a reasoning engine with metacognition enabled", target_fixture="ctx")
def _engine(tmp_path):
    engine = ReasoningEngine(store_path=tmp_path / "strategies.json")
    meta = MetaCognition(reasoning=engine, store_dir=tmp_path / "meta",
                         enabled=True)
    return {"engine": engine, "meta": meta}


# ── attribution ──────────────────────────────────────────────────────

@given("an accepted strategy whose win is carried by its abstention branch")
def _planted(ctx):
    ctx["engine"].library.admit(
        "planted",
        [{"op": "SOLVE"},
         {"op": "VERIFY", "checker": "confidence"},
         {"op": "BRANCH", "cond": "insufficient",
          "then": [{"op": "ABSTAIN", "reason": "guess"}]}],
        origin="synth", parent="direct", status="trial",
        weakness="family=missing_data", family="missing_data",
        incumbent="direct")
    ctx["meta"].on_reflect(1)


@given("an accepted strategy whose extra step changes nothing")
def _inert(ctx):
    ctx["engine"].library.admit(
        "inert",
        [{"op": "PREDICT", "horizon": 1}, {"op": "SOLVE"}],
        origin="synth", parent="direct", status="trial",
        weakness="family=arithmetic_chain", family="arithmetic_chain",
        incumbent="direct")
    ctx["meta"].on_reflect(1)


@given("a cortex that names a contradicting mechanism")
def _contrary_cortex(ctx):
    class _Contrary:
        def role_available(self, role):
            return True

        async def structured(self, role, messages, schema):
            return {"narrative": "a plausible story",
                    "mechanism": "voting_reduced_variance"}

    ctx["meta"].cortex = _Contrary()


@when("the strategy is attributed")
def _attribute(ctx):
    ctx["report"] = asyncio.run(ctx["meta"].attribute(tick=1))
    assert ctx["report"] is not None


@then(parsers.parse("the explanation should be {status}"))
def _status(ctx, status):
    assert ctx["report"]["status"] == status


@then(parsers.parse('the mechanism should be "{mechanism}"'))
def _mechanism(ctx, mechanism):
    assert ctx["report"]["mechanism"] == mechanism


@then("the mechanism should be empty")
def _mechanism_empty(ctx):
    assert ctx["report"]["mechanism"] == ""


@then("at least one edit should be confirmed by ablation")
def _confirmed(ctx):
    assert any(edit["confirmed"] for edit in ctx["report"]["edits"])


# ── invention ────────────────────────────────────────────────────────

@given("a weak class whose incumbent is expensive")
def _expensive_incumbent(ctx):
    engine = ctx["engine"]
    engine.library.admit(
        "costly",
        [{"op": "LLM_STEP", "template": "write an expression", "role": "fast"},
         {"op": "COMPUTE", "expr": "$last"},
         {"op": "VERIFY", "checker": "type"}],
        origin="synth", parent="direct")
    for _ in range(5):
        engine.library.note_result("costly", "missing_data", solved=True)
    engine.found = [_weakness()]
    ctx["prior_archive"] = [list(entry["steps"])
                            for entry in ctx["meta"].archive.entries]


@when("invention proposes far candidates and the arena judges them")
def _invent_and_judge(ctx):
    engine, meta = ctx["engine"], ctx["meta"]
    ctx["issued"] = meta.invent(tick=1)
    assert ctx["issued"], "invention proposed nothing"
    ctx["verdicts"] = []
    while engine.pending_candidates():
        ctx["verdicts"].append(engine.evaluate_candidate(tick=1))
    meta.on_reflect(1)


@then("at least one far candidate should be accepted")
def _far_accepted(ctx):
    accepted = [v for v in ctx["verdicts"]
                if v["accepted"] and str(v.get("transform", "")).startswith("skeleton:")]
    assert accepted, [v["reasons"] for v in ctx["verdicts"]]
    assert ctx["meta"].far_accepted >= 1


@then("every accepted far candidate should be far from the prior archive")
def _far_by_measure(ctx):
    accepted = [v for v in ctx["verdicts"]
                if v["accepted"] and str(v.get("transform", "")).startswith("skeleton:")]
    for verdict in accepted:
        strategy = ctx["engine"].library.get(verdict["candidate"])
        nearest = min(distance(strategy.steps, steps)
                      for steps in ctx["prior_archive"])
        assert nearest >= cfg.META_FAR, (verdict["candidate"], nearest)


@given("a skeleton that failed its class three times", target_fixture="ctx")
def _failed_skeleton(ctx):
    engine, meta = ctx["engine"], ctx["meta"]
    engine.found = [_weakness()]
    issued = meta.invent(tick=1)
    assert issued
    first = issued[0]["transform"].split(":", 1)[1]
    for _ in range(int(cfg.META_RETIRE_AFTER)):
        meta.skeletons.note_failure(first, _weakness().combo)
    engine.candidates.clear()
    ctx["retired_skeleton"] = first
    return ctx


@when("invention proposes far candidates again")
def _invent_again(ctx):
    ctx["issued"] = ctx["meta"].invent(tick=2)


@then("that skeleton should not be among the proposals")
def _not_reinvented(ctx):
    retired = f"skeleton:{ctx['retired_skeleton']}"
    assert all(row["transform"] != retired for row in ctx["issued"])
