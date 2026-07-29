"""Every metric of Appendix G actually reaches the telemetry (spec §M10, §3.5).

A metric that is declared and never written is worse than one that is missing:
the dashboard shows a panel, the Discovery Engine registers the variable and
counts it toward the family of tests it corrects over — and the series behind
all of it is empty. Nothing errors. The system simply has a blind spot shaped
exactly like the thing it believes it is watching.

So this walks the declared list against a **real substrate**, really ticked,
with its telemetry read back afterwards. What is stubbed is deliberate and
narrow: the *producers* that are expensive (the benchmark, the sandboxed
environment) and the ones that are legitimately non-reproducible (the network,
the model, the host's sensors). The *publication* — the code that decides which
name each number is written under, which is the only thing this contract is
about — is entirely real.

Two contours publish on a cadence longer than any run short enough to be a test
(a generation every 250 ticks, a hypothesis scan every 1000). Their publication
is called directly. That is not a weakening: the question is whether the metric
is ever written, not whether a short run happens to contain a generation.
"""
import asyncio

import pytest

from aegis.clock import frozen
from aegis.layers.substrate import Substrate
from aegis.telemetry.metrics import REQUIRED_METRICS

#: Enough ticks for every per-tick and periodic publication to come round.
TICKS = 30
SECONDS_PER_TICK = 3.0

#: A canned benchmark report. Its shape is what the publication reads; its
#: numbers are arbitrary, because this test is about names.
CANNED_REPORT = {
    "score": 0.5,
    "per_kind": {"calc": {"passed": 3, "total": 4},
                 "roman": {"passed": 2, "total": 2}},
}


def _cheap(substrate: Substrate) -> Substrate:
    """Silence the network, the model, the host — and the two slow producers."""
    async def _no_agents():
        return []

    async def _no_learning(*args, **kwargs):
        return {"success": False}

    substrate.agent_system.run_due_agents = _no_agents
    substrate.external_learning.learn_from_source = _no_learning
    substrate.llm.enabled = False
    substrate.sensors.read_all = lambda: {"pinned": True}

    # The benchmark runs sandboxed subprocesses per task and the environment
    # runs one per step. Both are producers; the publication that reads their
    # output is what is under test, so it is left alone and given something to
    # read.
    substrate.evaluator.run = lambda *a, **k: dict(CANNED_REPORT)
    substrate.evaluator.last_report = dict(CANNED_REPORT)
    substrate._last_benchmark_score = CANNED_REPORT["score"]
    substrate.environment.step = lambda *a, **k: {
        "reward": 0.25, "solved": True, "task": "canned", "kind": "calc"}

    # A generation evaluates ten variants, each running a benchmark in another
    # process — eighty seconds of work that the tick correctly detaches and
    # `cancel_background_tasks` correctly waits for. Its *publication* is what
    # this test is about, and that reads counters rather than results.
    substrate.evolution.run_generation = lambda *a, **k: {"generation": 1}
    return substrate


def _publish_the_slow_contours(substrate: Substrate) -> None:
    """Contours whose cadence is longer than this run, asked directly."""
    # `aegis.reason.win_rate` is per strategy and exists only once a strategy
    # has a record, so the contour has to have thought about something. Four
    # problems is enough for one strategy to be used.
    substrate.reasoning.solve(4)
    tick = substrate.tick_count
    for contour in (substrate.evolution, substrate.discovery,
                    substrate.reasoning, substrate.policy, substrate.resources,
                    substrate.roi, substrate.world_model, substrate.planner,
                    substrate.llm.cortex):
        contour.publish_metrics(tick)


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """The set of metric names one real run wrote."""
    import importlib

    from tests.conftest import _STATE_DIRS

    root = tmp_path_factory.mktemp("metrics_contract")
    restore = []
    for module_name, constant, subdir in _STATE_DIRS:
        module = importlib.import_module(module_name)
        if not hasattr(module, constant):
            continue
        target = root / subdir
        target.mkdir(parents=True, exist_ok=True)
        restore.append((module, constant, getattr(module, constant)))
        setattr(module, constant, target)

    import aegis.layers.substrate as substrate_mod
    from aegis.layers.state_backup import StateBackup

    restore.append((substrate_mod, "StateBackup", substrate_mod.StateBackup))
    substrate_mod.StateBackup = lambda *a, **k: StateBackup(
        backup_dir=root / "backups")

    names: set[str] = set()
    try:
        with frozen() as clock:
            substrate = _cheap(Substrate())

            async def _drive():
                for _ in range(TICKS):
                    await substrate.tick()
                    clock.advance(SECONDS_PER_TICK)
                # Inside the same loop: the conftest routes `asyncio.run`
                # through one shared loop, so a second call from a running loop
                # raises rather than nesting.
                _publish_the_slow_contours(substrate)
                await substrate.cancel_background_tasks()

            asyncio.run(_drive())
            substrate.telemetry.flush()
            names = set(substrate.telemetry.metrics())
    finally:
        for module, constant, value in restore:
            setattr(module, constant, value)

    return names


def _base(name: str) -> str:
    """``aegis.res.spent{kind}`` and ``aegis.res.spent`` are one metric.

    Tags are values of a metric, not metrics of their own — the store writes one
    series per base name and carries the tags on the rows.
    """
    return name.split("{", 1)[0]


# ── the contract ─────────────────────────────────────────────────────

def test_the_declared_list_is_a_list_of_distinct_names():
    """The list *is* the contract, so it has to be one."""
    assert len(REQUIRED_METRICS) == len(set(REQUIRED_METRICS))
    assert all(name.startswith("aegis.") for name in REQUIRED_METRICS)


def test_the_run_produced_telemetry_at_all(published):
    """A guard on the test itself. Every case below asserts membership in this
    set, and an empty set would make all of them fail for one reason while
    reading as fifty-six separate findings."""
    assert len(published) > 20, sorted(published)


@pytest.mark.parametrize("metric", sorted(REQUIRED_METRICS))
def test_the_metric_reaches_the_telemetry(metric, published):
    """One case per name of Appendix G, so a failure names the metric."""
    assert _base(metric) in {_base(name) for name in published}, (
        f"{metric} is declared in Appendix G and was never written"
    )


def test_every_contour_of_appendix_g_is_represented(published):
    prefixes = {name.split(".")[1] for name in published if name.count(".") >= 2}
    assert {"tick", "reward", "bench", "wm", "plan", "policy", "res", "evo",
            "reason", "disc", "cortex", "mem", "graph", "health"} <= prefixes, \
        sorted(prefixes)


def test_nothing_is_written_under_a_name_outside_the_contract(published):
    """The other direction, and the one that keeps the list honest. A metric
    written but never declared is invisible to the dashboard, to this contract
    and to the Discovery Engine's variable list — it costs disk and answers to
    nobody."""
    declared = {_base(name) for name in REQUIRED_METRICS}
    written = {_base(name) for name in published}
    assert written <= declared, sorted(written - declared)
