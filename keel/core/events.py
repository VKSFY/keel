"""Event bus and the @event decorator.

Events are plain dataclasses. The bus stores per-type queues that are
cleared at the start of each frame, so events emitted during one
system's run remain readable by later systems within the same frame.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterator


def event(cls: type) -> type:
    """Register a class as an event type, dataclass-decorating it if needed."""
    if not dataclasses.is_dataclass(cls):
        cls = dataclass(cls)
    cls.__keel_event__ = True
    return cls


class EventBus:
    """Per-frame queues of emitted events, keyed by event type."""

    __slots__ = ("_queues",)

    def __init__(self) -> None:
        self._queues: dict[type, list[Any]] = {}

    def emit(self, event_instance: Any) -> None:
        """Append an event instance to its type-specific queue."""
        et = type(event_instance)
        q = self._queues.get(et)
        if q is None:
            q = []
            self._queues[et] = q
        q.append(event_instance)

    def read(self, event_type: type) -> Iterator[Any]:
        """Iterate over all events of the given type queued this frame."""
        return iter(self._queues.get(event_type, ()))

    def count(self, event_type: type) -> int:
        """Return the number of events of `event_type` queued this frame."""
        return len(self._queues.get(event_type, ()))

    def clear(self) -> None:
        """Drop every queued event (called by World.tick at the start of each frame)."""
        self._queues.clear()
