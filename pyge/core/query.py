"""Query DSL: Without, Optional markers and the QueryResult iterator.

Queries are not cached between frames. Each call scans archetypes by
intersecting per-component archetype sets, then yields one tuple of
column views per matching archetype (not per entity).
"""
from __future__ import annotations

from typing import Any, Iterator

import numpy as np


class Without:
    """Query marker: exclude entities that have the wrapped component."""

    __slots__ = ("type",)

    def __init__(self, t: type) -> None:
        self.type = t

    def __class_getitem__(cls, t: type) -> "Without":
        return cls(t)

    def __repr__(self) -> str:
        return f"Without[{self.type.__name__}]"


class Optional:
    """Query marker: yield the column if present, else None."""

    __slots__ = ("type",)

    def __init__(self, t: type) -> None:
        self.type = t

    def __class_getitem__(cls, t: type) -> "Optional":
        return cls(t)

    def __repr__(self) -> str:
        return f"Optional[{self.type.__name__}]"


class QueryResult:
    """Iterable view over archetypes matching a query — yields per-archetype column tuples."""

    __slots__ = ("_world", "_required", "_without", "_optional", "_slots")

    def __init__(
        self,
        world: Any,
        required: list[type],
        without: list[type],
        optional: list[type],
        slots: list[tuple[str, type]],
    ) -> None:
        self._world = world
        self._required = required
        self._without = without
        self._optional = optional
        self._slots = slots

    def __iter__(self) -> Iterator[tuple]:
        return self._iter_archetypes()

    def _matching_archetypes(self) -> Iterator[Any]:
        registry = self._world.archetypes
        if not self._required:
            candidates: list[Any] | set[Any] = registry.all_archetypes()
        else:
            sets = [registry.archetypes_with(t) for t in self._required]
            sets.sort(key=len)
            base = sets[0]
            if len(sets) == 1:
                candidates = list(base)
            else:
                rest = sets[1:]
                candidates = [a for a in base if all(a in s for s in rest)]
        without = self._without
        for arch in candidates:
            if arch.length == 0:
                continue
            if without and any(t in arch.component_types for t in without):
                continue
            yield arch

    def _iter_archetypes(self) -> Iterator[tuple]:
        for arch in self._matching_archetypes():
            n = arch.length
            cols = arch.columns
            comp_types = arch.component_types
            out: list[Any] = []
            for kind, ct in self._slots:
                if kind == "req":
                    out.append(cols[ct][:n])
                elif ct in comp_types:
                    out.append(cols[ct][:n])
                else:
                    out.append(None)
            yield tuple(out)

    def archetypes(self) -> list[Any]:
        """Return the list of archetypes matched by this query (after Without filtering)."""
        return list(self._matching_archetypes())

    def entities(self) -> Iterator[int]:
        """Iterate over every entity ID matched by this query."""
        for arch in self._matching_archetypes():
            for eid in arch.entities[: arch.length]:
                yield eid

    def count(self) -> int:
        """Return the total number of matching entities across all matching archetypes."""
        return sum(arch.length for arch in self._matching_archetypes())


def build_query(world: Any, args: tuple) -> QueryResult:
    """Construct a QueryResult from positional query arguments."""
    required: list[type] = []
    without: list[type] = []
    optional: list[type] = []
    slots: list[tuple[str, type]] = []
    for a in args:
        if isinstance(a, Without):
            without.append(a.type)
        elif isinstance(a, Optional):
            optional.append(a.type)
            slots.append(("opt", a.type))
        elif isinstance(a, type):
            required.append(a)
            slots.append(("req", a))
        else:
            raise TypeError(f"Unsupported query argument: {a!r}")
    return QueryResult(world, required, without, optional, slots)
