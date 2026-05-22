"""ECS core package: archetypes, world, queries, scheduler, events."""
from .archetype import (
    Archetype,
    ArchetypeRegistry,
    ComponentMeta,
    component,
    get_component_meta,
)
from .events import EventBus, event
from .query import Optional, QueryResult, Without, build_query
from .scheduler import Phase, Scheduler
from .world import CommandBuffer, EntityAllocator, NULL_ENTITY, World

__all__ = [
    "Archetype",
    "ArchetypeRegistry",
    "ComponentMeta",
    "component",
    "get_component_meta",
    "EventBus",
    "event",
    "Optional",
    "QueryResult",
    "Without",
    "build_query",
    "Phase",
    "Scheduler",
    "CommandBuffer",
    "EntityAllocator",
    "NULL_ENTITY",
    "World",
]
