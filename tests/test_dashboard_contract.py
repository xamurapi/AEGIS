"""The dashboard reads `full_status()` — this pins that contract.

The dashboard is plain JS reading dotted paths out of the status payload. When a
layer renames a field, JS does not fail: it silently renders `undefined` (or
nothing at all), and the operator sees a broken panel with no error anywhere.
Two such drifts were live when this test was written — the whole event log
printed `undefined` for every event kind, and both event-bus counters rendered
blank (audit R3-13).

Anything the dashboard reads must therefore exist in the real payload.
"""
import re
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).parent.parent / "aegis" / "dashboard" / "index.html"

# Fields that are legitimately absent from a freshly-constructed Substrate and
# only appear once the corresponding subsystem has run at least once.
OPTIONAL = {
    "llm.lifetime_tokens",       # derived only after the first LLM call
    "llm.total_tokens",
}


@pytest.fixture(scope="module")
def status():
    from aegis.layers.substrate import Substrate
    return Substrate().full_status()


def _dashboard_reads():
    """Every status field the dashboard dereferences.

    Covers both direct reads (`d.emotions.mood`) and the far more common aliased
    form the script uses (`const em = d.emotions; ... em.mood`).
    """
    html = DASHBOARD.read_text(encoding="utf-8")
    reads = set(re.findall(r"\bd\.([a-z_]+)\.([a-zA-Z_][a-zA-Z0-9_]*)", html))

    # Aliases are short (`const a = d.autobiography`) and collide with lambda
    # parameters elsewhere in the file, so only scan from the assignment up to
    # the next one — the block where the alias is actually in scope.
    lines = html.splitlines()
    assign = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*d\.([a-z_]+)\s*;?\s*$")
    points = [(i, m.group(1), m.group(2))
              for i, line in enumerate(lines)
              for m in [assign.search(line.strip())] if m]

    for idx, (start, var, section) in enumerate(points):
        end = points[idx + 1][0] if idx + 1 < len(points) else len(lines)
        block = "\n".join(lines[start + 1:end])
        for key in re.findall(rf"(?<![\w.]){re.escape(var)}\.([a-zA-Z_][a-zA-Z0-9_]*)", block):
            reads.add((section, key))
    return sorted(reads)


def test_dashboard_reads_are_discoverable():
    """Guard the guard: if the regexes stop matching, this test is worthless."""
    assert len(_dashboard_reads()) > 100


@pytest.mark.parametrize("section,key", _dashboard_reads())
def test_every_dashboard_field_exists_in_full_status(status, section, key):
    if f"{section}.{key}" in OPTIONAL:
        pytest.skip("populated only after the subsystem runs")
    assert section in status, f"dashboard reads d.{section}, absent from full_status()"
    payload = status[section]
    if isinstance(payload, dict):
        assert key in payload, (
            f"dashboard reads d.{section}.{key}, which full_status() does not "
            f"provide — the panel renders 'undefined' with no error")


def test_event_history_rows_carry_the_field_the_log_renders(status):
    """The event log prints `ev.type`; the bus must actually serialize it."""
    from aegis.event_bus import EventBus, Event, Layer
    import asyncio

    bus = EventBus()
    asyncio.run(bus.publish(Event(source=Layer.INTROSPECTION, target=None,
                                  event_type="decision", payload={"x": 1})))
    row = bus.get_history(1)[0]
    assert "type" in row and row["type"] == "decision"
    assert "source" in row and "payload" in row


def test_event_bus_counter_names_match_what_the_dashboard_shows(status):
    assert {"total_events", "blocked_events"} <= set(status["event_bus"])
