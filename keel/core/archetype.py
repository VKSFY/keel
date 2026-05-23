"""Archetype storage and component registration.

An Archetype owns the storage for every entity sharing the same set of
component types. Each component type maps to one column: a structured
numpy array if all of the component's fields are numpy-supported, or a
plain Python list otherwise. Rows are appended on spawn and removed via
swap-remove on despawn / migration.
"""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from typing import Any

import numpy as np


_PYTHON_TO_NUMPY: dict[type, np.dtype] = {
    float: np.dtype(np.float64),
    int: np.dtype(np.int64),
    bool: np.dtype(np.bool_),
}

_NAMED_NUMPY: dict[str, np.dtype] = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "bool": np.dtype(np.bool_),
}

_INITIAL_CAPACITY = 8


def _resolve_numpy_dtype(t: Any) -> np.dtype | None:
    """Map a Python or numpy type annotation to a numpy dtype, or None if unsupported."""
    if t in _PYTHON_TO_NUMPY:
        return _PYTHON_TO_NUMPY[t]
    if isinstance(t, np.dtype):
        return t
    if isinstance(t, type) and issubclass(t, np.generic):
        return np.dtype(t)
    if isinstance(t, str) and t in _NAMED_NUMPY:
        return _NAMED_NUMPY[t]
    return None


@dataclass
class ComponentMeta:
    """Metadata for a registered component type."""
    cls: type
    field_names: list[str]
    numpy_dtype: np.dtype | None
    is_numpy: bool


def component(cls: type) -> type:
    """Register a class as an ECS component, attaching numpy-dtype metadata."""
    if not dataclasses.is_dataclass(cls):
        cls = dataclass(cls)
    hints = typing.get_type_hints(cls)
    field_names = [f.name for f in dataclasses.fields(cls)]

    numpy_pairs: list[tuple[str, np.dtype]] = []
    is_numpy = True
    for name in field_names:
        np_dt = _resolve_numpy_dtype(hints.get(name))
        if np_dt is None:
            is_numpy = False
            break
        numpy_pairs.append((name, np_dt))

    if is_numpy and numpy_pairs:
        numpy_dtype: np.dtype | None = np.dtype(numpy_pairs)
    else:
        numpy_dtype = None
        is_numpy = False

    setattr(
        cls,
        "__keel_component__",
        ComponentMeta(
            cls=cls,
            field_names=field_names,
            numpy_dtype=numpy_dtype,
            is_numpy=is_numpy,
        ),
    )
    return cls


def get_component_meta(cls: type) -> ComponentMeta:
    """Return the ComponentMeta attached to a registered component class."""
    meta = getattr(cls, "__keel_component__", None)
    if meta is None:
        raise TypeError(f"{cls!r} is not a registered @keel.component")
    return meta


def _write_numpy_row(arr: np.ndarray, row: int, inst: Any, meta: ComponentMeta) -> None:
    """Write a component instance into a structured-array row."""
    arr[row] = tuple(getattr(inst, name) for name in meta.field_names)


def _read_numpy_row(arr: np.ndarray, row: int, meta: ComponentMeta) -> Any:
    """Reconstruct a component instance from a structured-array row."""
    rec = arr[row]
    kwargs = {}
    for name in meta.field_names:
        v = rec[name]
        kwargs[name] = v.item() if isinstance(v, np.generic) else v
    return meta.cls(**kwargs)


class Archetype:
    """Storage for entities sharing the same component set."""

    __slots__ = ("component_types", "entities", "length", "capacity", "columns", "_metas")

    def __init__(self, component_types: frozenset[type]) -> None:
        self.component_types: frozenset[type] = component_types
        self.entities: list[int] = []
        self.length: int = 0
        self.capacity: int = _INITIAL_CAPACITY
        self.columns: dict[type, Any] = {}
        self._metas: dict[type, ComponentMeta] = {}
        for ct in component_types:
            meta = get_component_meta(ct)
            self._metas[ct] = meta
            if meta.is_numpy:
                self.columns[ct] = np.zeros(self.capacity, dtype=meta.numpy_dtype)
            else:
                self.columns[ct] = []

    def __len__(self) -> int:
        return self.length

    def _ensure_capacity(self, needed: int) -> None:
        """Grow numpy columns so they can hold at least `needed` rows."""
        if needed <= self.capacity:
            return
        new_cap = self.capacity
        while new_cap < needed:
            new_cap *= 2
        for ct, col in self.columns.items():
            if isinstance(col, np.ndarray):
                grown = np.zeros(new_cap, dtype=col.dtype)
                grown[: self.length] = col[: self.length]
                self.columns[ct] = grown
        self.capacity = new_cap

    def add_row(self, entity_id: int, components: dict[type, Any]) -> int:
        """Append an entity row, returning its row index."""
        self._ensure_capacity(self.length + 1)
        row = self.length
        for ct, inst in components.items():
            col = self.columns[ct]
            if isinstance(col, np.ndarray):
                _write_numpy_row(col, row, inst, self._metas[ct])
            else:
                col.append(inst)
        self.entities.append(entity_id)
        self.length += 1
        return row

    def swap_remove(self, row: int) -> int | None:
        """Remove a row by swapping with the last; return the entity that moved into `row`, or None."""
        last = self.length - 1
        moved_entity: int | None = None
        if row != last:
            for col in self.columns.values():
                col[row] = col[last]
            moved_entity = self.entities[last]
            self.entities[row] = moved_entity
        for col in self.columns.values():
            if isinstance(col, list):
                col.pop()
        self.entities.pop()
        self.length -= 1
        return moved_entity

    def get_component(self, row: int, ct: type) -> Any:
        """Reconstruct a component instance for the given row."""
        col = self.columns[ct]
        if isinstance(col, np.ndarray):
            return _read_numpy_row(col, row, self._metas[ct])
        return col[row]

    def write_component(self, row: int, ct: type, inst: Any) -> None:
        """Overwrite a single component on an existing row."""
        col = self.columns[ct]
        if isinstance(col, np.ndarray):
            _write_numpy_row(col, row, inst, self._metas[ct])
        else:
            col[row] = inst


class ArchetypeRegistry:
    """Indexes archetypes by their component set and by individual component types."""

    __slots__ = ("_by_set", "_by_component")

    def __init__(self) -> None:
        self._by_set: dict[frozenset[type], Archetype] = {}
        self._by_component: dict[type, set[Archetype]] = {}

    def get_or_create(self, component_types: frozenset[type]) -> Archetype:
        """Return the archetype for this component set, creating and indexing it on miss."""
        arch = self._by_set.get(component_types)
        if arch is None:
            arch = Archetype(component_types)
            self._by_set[component_types] = arch
            for ct in component_types:
                self._by_component.setdefault(ct, set()).add(arch)
        return arch

    def archetypes_with(self, ct: type) -> set[Archetype]:
        """Return all archetypes containing the given component type."""
        return self._by_component.get(ct, set())

    def all_archetypes(self) -> list[Archetype]:
        """Return every registered archetype as a list."""
        return list(self._by_set.values())

    def __len__(self) -> int:
        return len(self._by_set)
