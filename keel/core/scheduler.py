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

    __slots__ = ("_systems", "_profiler", "_phase_of", "_deps")

    def __init__(self) -> None:
        self._systems: dict[Phase, list[_SystemEntry]] = {p: [] for p in Phase}
        self._profiler: Any = None
        # Per-system ordering metadata. _phase_of maps fn → Phase so cross-phase
        # `after=` dependencies can be rejected loudly. _deps maps fn → list of
        # fns that must run before it within the same phase; consumed by
        # _topo_sort() after each registration.
        self._phase_of: dict[Callable, Phase] = {}
        self._deps: dict[Callable, list[Callable]] = {}

    def attach_profiler(self, profiler: Any) -> None:
        """Wrap every system call in `profiler.begin(name)` / `profiler.end(name)`."""
        self._profiler = profiler

    def detach_profiler(self) -> None:
        """Stop forwarding system timings to a profiler."""
        self._profiler = None

    def register(
        self,
        phase: Phase,
        fn: Callable,
        after: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> None:
        """Register `fn` to run in `phase`. Inspects the signature for resources.

        `after` declares ordering dependencies: every listed system must run
        before `fn` within the same phase. Independent systems still run in
        registration order. Raises ValueError if any listed system is not yet
        registered in this same phase, or if the new dependency closes a cycle.
        """
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

        # Normalize and validate `after` before mutating any scheduler state.
        if after is None:
            deps: list[Callable] = []
        elif callable(after):
            deps = [after]
        else:
            deps = list(after)
        for dep in deps:
            dep_phase = self._phase_of.get(dep)
            if dep_phase is None:
                raise ValueError(
                    f"after: {getattr(dep, '__qualname__', dep)!r} is not a "
                    "registered system in any phase"
                )
            if dep_phase != phase:
                raise ValueError(
                    f"after: {getattr(dep, '__qualname__', dep)!r} is in "
                    f"phase {dep_phase.name}, cannot be a dependency for a "
                    f"system in phase {phase.name}"
                )

        self._systems[phase].append(_SystemEntry(fn, resource_types))
        self._phase_of[fn] = phase
        self._deps[fn] = deps

        # Re-sort this phase so the ordering invariants hold for tick().
        try:
            self._systems[phase] = self._topo_sort_phase(phase)
        except ValueError:
            # Roll back this registration so the scheduler stays consistent.
            self._systems[phase] = [
                e for e in self._systems[phase] if e.fn is not fn
            ]
            self._phase_of.pop(fn, None)
            self._deps.pop(fn, None)
            raise

    def _topo_sort_phase(self, phase: Phase) -> list[_SystemEntry]:
        """Topo-sort `phase`'s entries by `_deps`. Stable on registration order."""
        remaining = list(self._systems[phase])
        emitted: list[_SystemEntry] = []
        emitted_fns: set[Callable] = set()
        while remaining:
            for i, entry in enumerate(remaining):
                deps = self._deps.get(entry.fn, ())
                if all(d in emitted_fns for d in deps):
                    emitted.append(entry)
                    emitted_fns.add(entry.fn)
                    del remaining[i]
                    break
            else:
                names = [e.fn.__qualname__ for e in remaining]
                raise ValueError(
                    f"after: cycle detected among systems {names} in phase "
                    f"{phase.name}"
                )
        return emitted

    def systems(self, phase: Phase) -> list[Callable]:
        """Return the registered functions for a phase, in registration order."""
        return [e.fn for e in self._systems[phase]]

    def clear(self) -> None:
        """Remove all registered systems from every phase."""
        for entries in self._systems.values():
            entries.clear()
        self._phase_of.clear()
        self._deps.clear()

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
