"""Telemetry is wired into the tick, not merely available (spec §3.5, M9.2).

The store existed and was unit-tested before this, but nothing in the package
ever constructed one — so the system had no history of itself and the discovery
engine (M7) would have had no data source at all. These tests assert the wiring,
which is the part that was missing.
"""
import asyncio

import pytest

from aegis.telemetry import metrics as M
from aegis.layers.substrate import Substrate


@pytest.fixture
def substrate(isolated_state):
    s = Substrate()
    s.llm.enabled = False
    return s


def _run(coro):
    return asyncio.run(coro)


# ── the substrate owns a telemetry store ─────────────────────────────

def test_substrate_has_a_telemetry_store(substrate):
    assert substrate.telemetry is not None


def test_telemetry_appears_in_full_status(substrate):
    assert "telemetry" in substrate.full_status()
    assert "records_written" in substrate.full_status()["telemetry"]


# ── a tick actually publishes ────────────────────────────────────────

def test_a_tick_records_its_duration(substrate):
    _run(substrate.tick())
    substrate.telemetry.flush()
    assert len(substrate.telemetry.series(M.TICK_DURATION_MS)) >= 1


def test_a_tick_records_every_phase(substrate):
    _run(substrate.tick())
    substrate.telemetry.flush()
    phases = {row["tags"].get("phase")
              for row in substrate.telemetry.series(M.TICK_PHASE_MS).rows()}
    assert phases == {"perceive", "evaluate", "decide", "act", "reflect"}


def test_a_tick_records_reward_and_memory_sizes(substrate):
    _run(substrate.tick())
    substrate.telemetry.flush()
    for metric in (M.REWARD_VALUE, M.REWARD_ENV_ROLLING,
                   M.MEM_SEMANTIC, M.MEM_EPISODIC, M.GRAPH_NODES):
        assert len(substrate.telemetry.series(metric)) >= 1, metric


def test_a_tick_records_health(substrate):
    _run(substrate.tick())
    substrate.telemetry.flush()
    codes = substrate.telemetry.series(M.HEALTH_STATUS_CODE).values
    assert codes and codes[-1] in (0, 1, 2, 3)


def test_series_grows_with_ticks(substrate):
    for _ in range(3):
        _run(substrate.tick())
    substrate.telemetry.flush()
    assert len(substrate.telemetry.series(M.TICK_DURATION_MS)) == 3


def test_recorded_ticks_are_numbered(substrate):
    for _ in range(3):
        _run(substrate.tick())
    substrate.telemetry.flush()
    assert substrate.telemetry.series(M.TICK_DURATION_MS).ticks == [1, 2, 3]


# ── failure containment ──────────────────────────────────────────────

def test_a_broken_telemetry_store_cannot_break_a_tick(substrate):
    class Exploding:
        def record(self, *a, **k):
            raise RuntimeError("disk on fire")

        def flush(self):
            raise RuntimeError("disk on fire")

        def status(self):
            return {}

    substrate.telemetry = Exploding()
    _run(substrate.tick())              # must not raise
    assert substrate.tick_count == 1


def test_checkpoint_flushes_buffered_metrics(substrate, isolated_state):
    _run(substrate.tick())
    substrate._save_checkpoint()
    # Reading with the buffer excluded proves the rows reached disk.
    on_disk = substrate.telemetry.series(M.TICK_DURATION_MS, include_buffer=False)
    assert len(on_disk) >= 1


# ── the name contract ────────────────────────────────────────────────

def test_metric_names_follow_the_canonical_format():
    assert all(name.startswith("aegis.") and name.count(".") >= 2
               for name in M.REQUIRED_METRICS)


def test_required_metric_names_are_unique():
    assert len(set(M.REQUIRED_METRICS)) == len(M.REQUIRED_METRICS)


def test_health_code_maps_known_and_unknown_statuses():
    assert M.health_code("healthy") == 0
    assert M.health_code("warning") == 1
    assert M.health_code("critical") == 2
    assert M.health_code("something-else") == 3


def test_health_monitor_exposes_its_last_status(substrate):
    substrate.health.check()
    assert substrate.health.last_status in ("healthy", "warning", "critical")
