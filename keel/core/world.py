"""World, CommandBuffer, and the entity-ID allocator.

All structural changes (spawn, despawn, add/remove component) flow
through the CommandBuffer and apply only on flush — never mid-system.
The World holds a buffer, an archetype registry, an entity allocator,
a scheduler, an event bus, and a resource map.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np

from .archetype import Archetype, ArchetypeRegistry, get_component_meta
from .events import EventBus
from .query import QueryResult, build_query
from .scheduler import Phase, Scheduler


NULL_ENTITY: int = 0


class EntityAllocator:
    """Allocates fresh entity IDs and recycles freed ones."""

    __slots__ = ("_next", "_free")

    def __init__(self) -> None:
        self._next: int = 1
        self._free: deque[int] = deque()

    def allocate(self) -> int:
        """Return a fresh ID, preferring recycled IDs from the free list."""
        if self._free:
            return self._free.popleft()
        eid = self._next
        self._next += 1
        return eid

    def free(self, eid: int) -> None:
        """Return an entity ID to the free list."""
        self._free.append(eid)

    def free_count(self) -> int:
        """Return how many IDs are currently waiting to be recycled."""
        return len(self._free)


@dataclass
class _Spawn:
    entity: int
    components: dict[type, Any]


@dataclass
class _Despawn:
    entity: int


@dataclass
class _AddComponent:
    entity: int
    component: Any
    component_type: type


@dataclass
class _RemoveComponent:
    entity: int
    component_type: type


class CommandBuffer:
    """Buffers structural changes (spawn, despawn, add, remove) for deferred application."""

    __slots__ = ("_commands",)

    def __init__(self) -> None:
        self._commands: list = []

    def spawn(self, entity: int, components: dict[type, Any]) -> None:
        """Queue a spawn command for an already-allocated entity ID."""
        self._commands.append(_Spawn(entity, components))

    def despawn(self, entity: int) -> None:
        """Queue a despawn command."""
        self._commands.append(_Despawn(entity))

    def add_component(self, entity: int, component: Any) -> None:
        """Queue an add-component command."""
        self._commands.append(_AddComponent(entity, component, type(component)))

    def remove_component(self, entity: int, component_type: type) -> None:
        """Queue a remove-component command."""
        self._commands.append(_RemoveComponent(entity, component_type))

    def take(self) -> list:
        """Atomically drain and return every queued command."""
        cmds = self._commands
        self._commands = []
        return cmds

    def __len__(self) -> int:
        return len(self._commands)


class World:
    """The ECS world: entities, archetypes, systems, events, and resources."""

    def __init__(self) -> None:
        self.archetypes: ArchetypeRegistry = ArchetypeRegistry()
        self.entities: EntityAllocator = EntityAllocator()
        self.commands: CommandBuffer = CommandBuffer()
        self.scheduler: Scheduler = Scheduler()
        self.events: EventBus = EventBus()
        self._location: dict[int, tuple[Archetype, int]] = {}
        self._resources: dict[type, Any] = {}

    def __repr__(self) -> str:
        entities = len(self._location)
        archetypes = len(self.archetypes.all_archetypes())
        return f"<World entities={entities} archetypes={archetypes}>"

    # ----------------------------------------------------------------------
    # Entity lifecycle — spawn / despawn / structural component changes.
    # Every method here just enqueues a CommandBuffer command; the change
    # only becomes visible to queries after world.flush() (called by the
    # main loop at end-of-frame, or manually by you).
    # ----------------------------------------------------------------------

    def spawn(self, *components: Any) -> int:
        """Allocate an entity id and queue a deferred spawn.

        The new entity is NOT visible to queries until the next world.flush()
        (the main loop calls flush at end-of-frame). If you spawn and need to
        query the entity in the same code path, call world.flush() yourself.
        """
        eid = self.entities.allocate()
        comp_map: dict[type, Any] = {}
        for c in components:
            ct = type(c)
            if ct in comp_map:
                raise ValueError(f"Duplicate component {ct.__name__} in spawn()")
            comp_map[ct] = c
        self.commands.spawn(eid, comp_map)
        return eid

    def despawn(self, entity: int) -> None:
        """Queue a deferred despawn for the given entity."""
        self.commands.despawn(entity)

    def add_component(self, entity: int, component: Any) -> None:
        """Queue a deferred add-component for the given entity."""
        self.commands.add_component(entity, component)

    def remove_component(self, entity: int, component_type: type) -> None:
        """Queue a deferred remove-component for the given entity."""
        self.commands.remove_component(entity, component_type)

    # ----------------------------------------------------------------------
    # Entity inspection — synchronous reads + writes against entities that
    # are already in an archetype (i.e. were spawned and then flushed).
    # ----------------------------------------------------------------------

    def is_alive(self, entity: int) -> bool:
        """Return True if `entity` has been flushed and is currently in an archetype."""
        return entity in self._location

    def has_component(self, entity: int, component_type: type) -> bool:
        """Return True if `entity` is alive and currently has `component_type`."""
        loc = self._location.get(entity)
        return loc is not None and component_type in loc[0].component_types

    def get_component(self, entity: int, component_type: type) -> Any:
        """Reconstruct and return a component instance for `entity`, or None if absent."""
        loc = self._location.get(entity)
        if loc is None:
            return None
        arch, row = loc
        if component_type not in arch.component_types:
            return None
        return arch.get_component(row, component_type)

    def location_of(self, entity: int) -> tuple[Archetype, int] | None:
        """Return the (archetype, row) location of `entity`, or None if not alive."""
        return self._location.get(entity)

    def get(self, entity: int, component_type: type) -> dict[str, Any] | None:
        """Return the entity's component fields as Python scalars, or None if absent."""
        loc = self._location.get(entity)
        if loc is None or component_type not in loc[0].component_types:
            return None
        arch, row = loc
        col = arch.columns[component_type]
        if isinstance(col, np.ndarray):
            rec = col[row]
            out = {}
            for name in rec.dtype.names:
                v = rec[name]
                out[name] = v.item() if isinstance(v, np.generic) else v
            return out
        meta = get_component_meta(component_type)
        inst = col[row]
        return {name: getattr(inst, name) for name in meta.field_names}

    def set(self, entity: int, component_type: type, **fields: Any) -> bool:
        """Write component fields in place. Returns False if the entity lacks the component.

        Raises ValueError if any keyword names a field that does not exist on
        the component — the bad name, the component, and the entity id are
        included in the message.
        """
        loc = self._location.get(entity)
        if loc is None or component_type not in loc[0].component_types:
            return False
        if not fields:
            return True
        meta = get_component_meta(component_type)
        known = set(meta.field_names)
        unknown = [name for name in fields if name not in known]
        if unknown:
            raise ValueError(
                f"{component_type.__name__} has no field(s) {unknown!r} "
                f"on entity {entity}; known fields: {meta.field_names}"
            )
        arch, row = loc
        col = arch.columns[component_type]
        if isinstance(col, np.ndarray):
            for name, value in fields.items():
                col[name][row] = value
        else:
            inst = col[row]
            for name, value in fields.items():
                setattr(inst, name, value)
        return True

    # ----------------------------------------------------------------------
    # Query API — iterate component column views by required / Without /
    # Optional component types. See keel/core/query.py for the markers.
    # ----------------------------------------------------------------------

    def query(self, *args: Any) -> QueryResult:
        """Build a query over component types and Without[]/Optional[] markers."""
        return build_query(self, args)

    def query_one(self, component_type: type) -> dict[str, Any] | None:
        """Return the first entity's component fields as plain Python scalars.

        For singleton components (GameState, Camera2D, one-of-a-kind config
        rows). Returns a plain Python dict with Python scalars instead of the
        numpy array views you get from `world.query()`, so reading a field
        does not require `[0]` indexing and works with `int()` / `bool()` /
        comparison operators directly.

            gs = world.query_one(GameState)
            if gs is not None:
                print(gs["score"])  # plain int, not numpy.int64

        This is read-only: mutating the returned dict does **not** write back
        to the ECS. To write fields, use `world.set(entity_id, GameState, ...)`
        or iterate with `world.query(GameState)` and mutate the views in place.
        Returns None if no entity has the component.
        """
        for arch in self.archetypes.archetypes_with(component_type):
            if arch.length == 0:
                continue
            col = arch.columns[component_type]
            if isinstance(col, np.ndarray):
                rec = col[0]
                out: dict[str, Any] = {}
                for name in rec.dtype.names:
                    v = rec[name]
                    out[name] = v.item() if isinstance(v, np.generic) else v
                return out
            meta = get_component_meta(component_type)
            inst = col[0]
            return {name: getattr(inst, name) for name in meta.field_names}
        return None

    # ----------------------------------------------------------------------
    # Resources — singleton objects keyed by their type, injected into
    # systems via parameter type annotations.
    # ----------------------------------------------------------------------

    def insert_resource(self, resource: Any, *, type_: type | None = None) -> None:
        """Register `resource` as a singleton injectable by its type (or by `type_` override)."""
        key = type_ if type_ is not None else type(resource)
        self._resources[key] = resource

    def get_resource(self, type_: type) -> Any:
        """Return the registered resource of type `type_`, or None."""
        return self._resources.get(type_)

    def has_resource(self, type_: type) -> bool:
        """Return True if a resource of the given type is registered."""
        return type_ in self._resources

    def remove_resource(self, type_: type) -> Any:
        """Remove and return the registered resource of type `type_`, or None."""
        return self._resources.pop(type_, None)

    # ----------------------------------------------------------------------
    # Events — queued per-type each frame, drained by world.events.clear().
    # ----------------------------------------------------------------------

    def emit(self, event_instance: Any) -> None:
        """Queue an event for systems to read this frame."""
        self.events.emit(event_instance)

    def read_events(self, event_type: type) -> Iterator[Any]:
        """Iterate over events of `event_type` queued this frame."""
        return self.events.read(event_type)

    # ----------------------------------------------------------------------
    # Systems — phase-ordered functions the scheduler runs each tick.
    # ----------------------------------------------------------------------

    def system(
        self,
        phase: Phase,
        after: Callable | list[Callable] | tuple[Callable, ...] | None = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator: register the wrapped function as a system in `phase`.

        `after` is an optional system fn (or list of fns) that must run before
        this one within the same phase. See Scheduler.register for details.
        """
        def decorator(fn: Callable) -> Callable:
            self.scheduler.register(phase, fn, after=after)
            return fn
        return decorator

    def tick(self, dt: float) -> None:
        """Run one frame: clear events, run all systems in phase order, then flush commands."""
        self.events.clear()
        self.scheduler.run(self, dt)
        self.flush()

    # ----------------------------------------------------------------------
    # Flush — apply every buffered structural change. The main loop calls
    # this at end-of-frame; tests / inline code call it explicitly.
    # ----------------------------------------------------------------------

    def flush(self) -> None:
        """Apply every queued structural change in the order it was buffered."""
        cmds = self.commands.take()
        for cmd in cmds:
            t = type(cmd)
            if t is _Spawn:
                self._do_spawn(cmd.entity, cmd.components)
            elif t is _Despawn:
                self._do_despawn(cmd.entity)
            elif t is _AddComponent:
                self._do_add(cmd.entity, cmd.component, cmd.component_type)
            elif t is _RemoveComponent:
                self._do_remove(cmd.entity, cmd.component_type)

    # ----------------------------------------------------------------------
    # Internal: how each buffered command actually lands in the archetypes.
    # ----------------------------------------------------------------------

    def _do_spawn(self, eid: int, components: dict[type, Any]) -> None:
        cset = frozenset(components.keys())
        arch = self.archetypes.get_or_create(cset)
        row = arch.add_row(eid, components)
        self._location[eid] = (arch, row)

    def _do_despawn(self, eid: int) -> None:
        loc = self._location.pop(eid, None)
        if loc is None:
            return
        arch, row = loc
        moved = arch.swap_remove(row)
        if moved is not None:
            self._location[moved] = (arch, row)
        self.entities.free(eid)

    def _do_add(self, eid: int, component: Any, ct: type) -> None:
        loc = self._location.get(eid)
        if loc is None:
            return
        old_arch, old_row = loc
        if ct in old_arch.component_types:
            old_arch.write_component(old_row, ct, component)
            return
        new_set = old_arch.component_types | {ct}
        new_arch = self.archetypes.get_or_create(new_set)
        comps: dict[type, Any] = {
            c: old_arch.get_component(old_row, c) for c in old_arch.component_types
        }
        comps[ct] = component
        new_row = new_arch.add_row(eid, comps)
        moved = old_arch.swap_remove(old_row)
        if moved is not None:
            self._location[moved] = (old_arch, old_row)
        self._location[eid] = (new_arch, new_row)

    def _do_remove(self, eid: int, ct: type) -> None:
        loc = self._location.get(eid)
        if loc is None:
            return
        old_arch, old_row = loc
        if ct not in old_arch.component_types:
            return
        new_set = old_arch.component_types - {ct}
        new_arch = self.archetypes.get_or_create(new_set)
        comps: dict[type, Any] = {
            c: old_arch.get_component(old_row, c) for c in new_set
        }
        new_row = new_arch.add_row(eid, comps)
        moved = old_arch.swap_remove(old_row)
        if moved is not None:
            self._location[moved] = (old_arch, old_row)
        self._location[eid] = (new_arch, new_row)
