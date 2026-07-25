"""Extended tests for the async EventBus — publish, veto, subscribers, capping."""
import asyncio
import pytest
from aegis.event_bus import EventBus, Event, Layer


def _ev(event_type="ping", target=None, payload=None, clearance=1.0):
    return Event(
        source=Layer.SUBSTRATE,
        target=target,
        event_type=event_type,
        payload=payload or {},
        ethical_clearance=clearance,
    )


def test_event_defaults_populate_id_and_timestamp():
    ev = _ev()
    assert isinstance(ev.id, str) and len(ev.id) == 12
    assert ev.timestamp > 0
    # Two events get distinct ids
    assert _ev().id != _ev().id


def test_publish_broadcast_records_history_and_stats():
    bus = EventBus()
    ok = asyncio.run(bus.publish(_ev(target=None)))
    assert ok is True
    hist = bus.get_history()
    assert len(hist) == 1
    assert hist[0]["target"] == "broadcast"
    stats = bus.stats()
    assert stats["total_events"] == 1
    assert stats["blocked_events"] == 0


def test_publish_targeted_records_target_name():
    bus = EventBus()
    asyncio.run(bus.publish(_ev(target=Layer.MEMORY)))
    assert bus.get_history()[0]["target"] == "memory"


def test_sync_subscriber_receives_event():
    bus = EventBus()
    received = []
    bus.subscribe("ping", lambda e: received.append(e))
    asyncio.run(bus.publish(_ev("ping")))
    assert len(received) == 1


def test_async_subscriber_is_awaited():
    bus = EventBus()
    seen = []

    async def cb(e):
        seen.append(e.event_type)

    bus.subscribe("ping", cb)
    asyncio.run(bus.publish(_ev("ping")))
    assert seen == ["ping"]


def test_global_subscriber_receives_all():
    bus = EventBus()
    seen = []
    bus.subscribe_all(lambda e: seen.append(e.event_type))
    asyncio.run(bus.publish(_ev("a")))
    asyncio.run(bus.publish(_ev("b")))
    assert seen == ["a", "b"]


def test_subscriber_exception_is_swallowed():
    bus = EventBus()

    def boom(e):
        raise RuntimeError("subscriber failure")

    bus.subscribe("ping", boom)
    # Publish must still succeed despite the callback raising.
    assert asyncio.run(bus.publish(_ev("ping"))) is True


def test_async_global_subscriber_is_awaited():
    bus = EventBus()
    seen = []

    async def cb(e):
        seen.append(e.event_type)

    bus.subscribe_all(cb)
    asyncio.run(bus.publish(_ev("ping")))
    assert seen == ["ping"]


def test_global_subscriber_exception_is_swallowed():
    bus = EventBus()

    def boom(e):
        raise ValueError("global failure")

    bus.subscribe_all(boom)
    assert asyncio.run(bus.publish(_ev("ping"))) is True


def test_async_subscriber_exception_is_swallowed():
    bus = EventBus()

    async def boom(e):
        raise RuntimeError("async subscriber failure")

    bus.subscribe("ping", boom)
    assert asyncio.run(bus.publish(_ev("ping"))) is True


def test_veto_blocks_event():
    bus = EventBus()
    bus.set_veto(lambda e: False)
    ok = asyncio.run(bus.publish(_ev("ping")))
    assert ok is False
    assert bus.stats()["blocked_events"] == 1
    assert bus.stats()["total_events"] == 0
    assert len(bus.get_blocked()) == 1
    assert bus.get_blocked()[0]["blocked"] is True


def test_veto_allows_event():
    bus = EventBus()
    bus.set_veto(lambda e: True)
    assert asyncio.run(bus.publish(_ev("ping"))) is True
    assert bus.stats()["total_events"] == 1


def test_history_is_capped_at_max():
    bus = EventBus()
    bus._max_history = 10
    for _ in range(15):
        asyncio.run(bus.publish(_ev("ping")))
    assert len(bus._history) == 10
    # Running totals keep counting past the cap.
    assert bus.stats()["total_events"] == 15


def test_blocked_list_is_capped_at_max():
    bus = EventBus()
    bus._max_history = 10
    bus.set_veto(lambda e: False)
    for _ in range(15):
        asyncio.run(bus.publish(_ev("ping")))
    assert len(bus._blocked) == 10
    assert bus.stats()["blocked_events"] == 15


def test_get_history_and_blocked_respect_limit():
    bus = EventBus()
    for _ in range(5):
        asyncio.run(bus.publish(_ev("ping")))
    assert len(bus.get_history(limit=2)) == 2


def test_subscriber_count_in_stats():
    bus = EventBus()
    bus.subscribe("a", lambda e: None)
    bus.subscribe("a", lambda e: None)
    bus.subscribe("b", lambda e: None)
    assert bus.stats()["subscriber_count"] == 3
