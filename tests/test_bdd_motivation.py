"""pytest-bdd step definitions for tests/features/motivation.feature.

Executable Gherkin over the real resource manager, priority scheduler and ROI
tracker (M4). The chain the spec asks for is goal → value → priority → resource
→ action, and the scenarios here are about the two links that were missing
before this contour existed: priority as a number, and resource as something
that can actually run out.
"""
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aegis.layers.motivation.priority import Candidate, PriorityScheduler
from aegis.layers.motivation.resources import ResourceCost, ResourceManager
from aegis.layers.motivation.roi import ROITracker

scenarios("features/motivation.feature")


@given("a resource manager with a full budget", target_fixture="ctx")
def _full(tmp_path):
    return {"resources": ResourceManager(
        store_path=tmp_path / "budgets.json",
        limits={"llm_tokens": 100_000, "llm_calls": 500})}


@given("a resource manager with no token budget at all", target_fixture="ctx")
def _empty(tmp_path):
    return {"resources": ResourceManager(store_path=tmp_path / "budgets.json",
                                         limits={"llm_tokens": 0})}


@given("a resource manager with almost no budget left", target_fixture="ctx")
def _almost_empty(tmp_path):
    # The floor is a *share* of the budget, so the budget has to be big enough
    # for "ordinary work is out" and "safety work is not" to be different
    # states at all: at a limit of 1000 the 15% floor is 150, and one request
    # of 500 clears neither.
    resources = ResourceManager(store_path=tmp_path / "budgets.json",
                                limits={"llm_tokens": 10_000})
    lease = resources.reserve(ResourceCost(llm_tokens=8_400), "soak_it_up")
    assert lease is not None
    return {"resources": resources}


@given("a priority scheduler", target_fixture="ctx")
def _scheduler(tmp_path):
    resources = ResourceManager(store_path=tmp_path / "budgets.json")
    return {"resources": resources,
            "priority": PriorityScheduler(resources=resources),
            "candidates": []}


@given("a ROI tracker", target_fixture="ctx")
def _roi(tmp_path):
    return {"roi": ROITracker(store_path=tmp_path / "roi.json")}


# ── leases ───────────────────────────────────────────────────────────

@when(parsers.parse("a lease is requested for {tokens:d} tokens"))
def _request(ctx, tokens):
    ctx["lease"] = ctx["resources"].reserve(ResourceCost(llm_tokens=tokens),
                                            "a_purpose")


@when("the lease is released")
def _release(ctx):
    ctx["resources"].release(ctx["lease"])


@when(parsers.parse("only {tokens:d} tokens are actually used"))
def _commit(ctx, tokens):
    ctx["resources"].commit(ctx["lease"], ResourceCost(llm_tokens=tokens))


@then("the lease should be granted")
def _granted(ctx):
    assert ctx["lease"] is not None


@then("the lease should be refused")
def _refused(ctx):
    """Refusal is a normal outcome, not an error: the action is deferred and
    the tick carries on."""
    assert ctx["lease"] is None


@then("the tokens should be counted as reserved")
def _reserved(ctx):
    assert ctx["resources"].spent("llm_tokens") >= 1000


@then("the reserved tokens should be back to zero")
def _returned(ctx):
    assert ctx["resources"].spent("llm_tokens") == 0


@then(parsers.parse("{tokens:d} tokens should be recorded as spent"))
def _spent(ctx, tokens):
    """The reservation was an estimate. Charging the estimate would let an
    action that reserved generously permanently shrink the allowance."""
    assert ctx["resources"].spent("llm_tokens") == tokens


# ── the safety floor ─────────────────────────────────────────────────

@then("ordinary work should not be affordable")
def _not_affordable(ctx):
    assert ctx["resources"].can_afford(ResourceCost(llm_tokens=500)) is False


@then("safety-critical work should still be affordable")
def _floor_holds(ctx):
    """A health check that could be starved by ordinary work is not a health
    check, it is a suggestion."""
    assert ctx["resources"].can_afford(ResourceCost(llm_tokens=500),
                                       safety_critical=True) is True


# ── priority and aging ───────────────────────────────────────────────

@when(parsers.parse("a candidate has been waiting {ticks:d} ticks"))
def _waiting(ctx, ticks):
    ctx["resources"].waiting["patient"] = ticks
    ctx["candidates"].append(Candidate(objective="patient", value=0.5))


@when("an identical candidate has just arrived")
def _fresh(ctx):
    ctx["candidates"].append(Candidate(objective="fresh", value=0.5))


@when("a candidate of high value has just arrived")
def _valuable(ctx):
    ctx["candidates"].append(Candidate(objective="valuable", value=1.0))


@when(parsers.parse("a candidate of no value has been waiting {ticks:d} ticks"))
def _worthless(ctx, ticks):
    ctx["resources"].waiting["worthless"] = ticks
    ctx["candidates"].append(Candidate(objective="worthless", value=0.0))


@then("the long-waiting candidate should be ordered first")
def _aged_first(ctx):
    ordered = ctx["priority"].order(ctx["candidates"])
    assert ordered[0].objective == "patient"


@then("the valuable candidate should be ordered first")
def _value_wins(ctx):
    """Aging is capped, so anti-starvation cannot become a different
    starvation in which nothing valuable ever runs."""
    ordered = ctx["priority"].order(ctx["candidates"])
    assert ordered[0].objective == "valuable"


# ── ROI ──────────────────────────────────────────────────────────────

@when("an activity spends repeatedly and returns nothing")
def _no_return(ctx):
    for _ in range(20):
        ctx["roi"].record("knowledge_work", ResourceCost(llm_tokens=1000), 0.0,
                          drive="knowledge")


@when("one activity returns well and another returns nothing")
def _mixed(ctx):
    for _ in range(20):
        ctx["roi"].record("paying", ResourceCost(llm_tokens=1000), 5.0,
                          drive="competence")
        ctx["roi"].record("not_paying", ResourceCost(llm_tokens=1000), 0.0,
                          drive="knowledge")


@when("the budget is reallocated")
def _reallocate(ctx):
    ctx["shares"] = ctx["roi"].reallocate(tick=100)


@then("its share should be above zero")
def _floor(ctx):
    """An activity funded at zero can never update its own ROI, so a
    reallocation that could reach zero is a one-way door."""
    assert ctx["roi"].share("knowledge") > 0.0


@then("the paying activity should hold the larger share")
def _larger(ctx):
    assert ctx["roi"].share("competence") > ctx["roi"].share("knowledge")
