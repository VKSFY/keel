"""Phase enum and the per-tick system scheduler.

Systems are plain functions registered with a Phase. The scheduler
invokes them in phase order each tick, passing (world, dt) plus any
additional resources resolved from parameter type annotations.
"""
from __future__ import annotations

import inspect
import typing
from enum import IntEnum
from typing import Any, Callable


class Phase(IntEnum):
    """Order in which systems run within a tick."""
    PRE_UPDATE = 0
    UPDATE = 1
    POST_UPDATE = 2
    PRE_RENDER = 3
    RENDER = 4
    POST_RENDER = 5


class _SystemEntry:
    """Internal record of a registered system: its function and its resource types."""

    __slots__ = ("fn", "resource_types")

    def __init__(self, fn: Callable, resource_types: list[type]) -> None:
        self.fn = fn
        self.resource_types = resource_types


class Scheduler:
    """Per-world scheduler that owns systems grouped by Phase."""

    __slots__ = ("_systems", "_profiler")

    def __init__(self) -> None:
        self._systems: dict[Phase, list[_SystemEntry]] = {p: [] for p in Phase}
        self._profiler: Any = None

    def attach_profiler(self, profiler: Any) -> None:
        """Wrap every system call in `profiler.begin(name)` / `profiler.end(name)`."""
        self._profiler = profiler

    def detach_profiler(self) -> None:
        """Stop forwarding system timings to a profiler."""
        self._profiler = None

    def register(self, phase: Phase, fn: Callable) -> None:
        """Register a function to run in the given phase, inspecting its signature for resources."""
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        try:
            resolved = typing.get_type_hints(fn)
        except Exception:
            resolved = {}
        resource_types: list[type] = []
        for i, p in enumerate(params):
            if i < 2:
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            ann = resolved.get(p.name, p.annotation)
            if ann is inspect.Parameter.empty:
                raise TypeError(
                    f"System {fn.__qualname__!r}: parameter {p.name!r} "
                    "needs a type annotation for resource injection"
                )
            if not isinstance(ann, type):
                raise TypeError(
                    f"System {fn.__qualname__!r}: parameter {p.name!r} annotation "
                    f"{ann!r} could not be resolved to a type — define resource types "
                    "at module level so they are visible to typing.get_type_hints"
                )
            resource_types.append(ann)
        self._systems[phase].append(_SystemEntry(fn, resource_types))

    def systems(self, phase: Phase) -> list[Callable]:
        """Return the registered functions for a phase, in registration order."""
        return [e.fn for e in self._systems[phase]]

    def clear(self) -> None:
        """Remove all registered systems from every phase."""
        for entries in self._systems.values():
            entries.clear()

    def _invoke(self, entry: _SystemEntry, world: Any, dt: float) -> None:
        """Invoke one system, wrapped in profiler timing if a profiler is attached."""
        prof = self._profiler
        if prof is not None:
            name = entry.fn.__name__
            prof.begin(name)
            try:
                if entry.resource_types:
                    resources = [world.get_resource(t) for t in entry.resource_types]
                    entry.fn(world, dt, *resources)
                else:
                    entry.fn(world, dt)
            finally:
                prof.end(name)
        else:
            if entry.resource_types:
                resources = [world.get_resource(t) for t in entry.resource_types]
                entry.fn(world, dt, *resources)
            else:
                entry.fn(world, dt)

    def run(self, world: Any, dt: float) -> None:
        """Invoke every system once, in (Phase order, registration order)."""
        for phase in Phase:
            for entry in self._systems[phase]:
                self._invoke(entry, world, dt)

    def tick(self, world: Any, dt: float) -> None:
        """Run every system once for one tick — alias for run()."""
        self.run(world, dt)

    def tick_phases(self, world: Any, dt: float, phases) -> None:
        """Invoke systems only for the listed phases, in the given order."""
        for phase in phases:
            for entry in self._systems[phase]:
                self._invoke(entry, world, dt)

    def tick_simulation(self, world: Any, dt: float) -> None:
        """Run only the simulation phases: PRE_UPDATE, UPDATE, POST_UPDATE."""
        self.tick_phases(world, dt, (Phase.PRE_UPDATE, Phase.UPDATE, Phase.POST_UPDATE))

    def tick_render(self, world: Any, dt: float) -> None:
        """Run only the render phases: PRE_RENDER, RENDER, POST_RENDER."""
        self.tick_phases(world, dt, (Phase.PRE_RENDER, Phase.RENDER, Phase.POST_RENDER))
