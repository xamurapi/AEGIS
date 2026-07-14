"""AEGIS Event Bus — inter-layer typed message system (INT-001..004)."""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("aegis.event_bus")


class Layer(str, Enum):
    SUBSTRATE = "substrate"
    MEMORY = "memory"
    INTROSPECTION = "introspection"
    SELF_MODIFICATION = "self_modification"
    GOAL_ENGINE = "goal_engine"
    WORLD_INTERFACE = "world_interface"
    ETHICS_CORE = "ethics_core"
    SYSTEM = "system"
    DASHBOARD = "dashboard"


@dataclass
class Event:
    source: Layer
    target: Layer | None
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    ethical_clearance: float = 1.0
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self._global_subscribers: list[Callable] = []
        self._history: list[dict] = []
        self._blocked: list[dict] = []
        self._veto_fn: Callable | None = None
        self._max_history = 500
        # Running lifetime counters — the capped lists above must not be used
        # as totals (they plateau at _max_history).
        self._total_events = 0
        self._total_blocked = 0

    def set_veto(self, fn: Callable):
        self._veto_fn = fn

    def subscribe(self, event_type: str, callback: Callable):
        self._subscribers.setdefault(event_type, []).append(callback)

    def subscribe_all(self, callback: Callable):
        self._global_subscribers.append(callback)

    async def publish(self, event: Event) -> bool:
        record = {
            "id": event.id,
            "time": event.timestamp,
            "source": event.source.value,
            "target": event.target.value if event.target else "broadcast",
            "type": event.event_type,
            "payload": event.payload,
            "ethical_clearance": event.ethical_clearance,
        }

        if self._veto_fn:
            allowed = self._veto_fn(event)
            if not allowed:
                record["blocked"] = True
                self._total_blocked += 1
                self._blocked.append(record)
                if len(self._blocked) > self._max_history:
                    self._blocked = self._blocked[-self._max_history:]
                return False

        self._total_events += 1
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        for cb in self._subscribers.get(event.event_type, []):
            try:
                result = cb(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Event subscriber for %r failed", event.event_type)

        for cb in self._global_subscribers:
            try:
                result = cb(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Global event subscriber failed for %r", event.event_type)

        return True

    def get_history(self, limit: int = 50) -> list[dict]:
        return self._history[-limit:]

    def get_blocked(self, limit: int = 20) -> list[dict]:
        return self._blocked[-limit:]

    def stats(self) -> dict:
        return {
            "total_events": self._total_events,
            "blocked_events": self._total_blocked,
            "subscriber_count": sum(len(v) for v in self._subscribers.values()),
        }
