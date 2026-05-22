"""Tests for the PyGE ECS core."""
from __future__ import annotations

import time
from dataclasses import make_dataclass

import numpy as np
import pytest

from unittest.mock import patch

import pyge
from pyge import Optional, Phase, Without, World, component, event


# --- Component / event fixtures used across tests ----------------------

@component
class Position:
    x: float = 0.0
    y: float = 0.0


@component
class Velocity:
    x: float = 0.0
    y: float = 0.0


@component
class Health:
    hp: int = 100


@component
class Tag:
    flag: bool = False


@component
class Name:
    """A component with a non-numpy field — falls back to list storage."""
    label: str = ""


@event
class HitEvent:
    target: int
    damage: int


@event
class TickEvent:
    n: int


class Config:
    """A plain resource type — module-level so typing.get_type_hints can resolve it."""

    def __init__(self, gravity: float = 9.8) -> None:
        self.gravity = gravity


# --- Component registration --------------------------------------------

def test_numeric_component_gets_numpy_dtype():
    meta = Position.__pyge_component__
    assert meta.is_numpy is True
    assert meta.numpy_dtype is not None
    assert meta.numpy_dtype.names == ("x", "y")
    assert meta.numpy_dtype["x"] == np.float64


def test_string_component_falls_back_to_list_storage():
    meta = Name.__pyge_component__
    assert meta.is_numpy is False
    assert meta.numpy_dtype is None


# --- Spawn / despawn / lifecycle ---------------------------------------

def test_spawn_is_deferred_until_flush():
    world = World()
    e = world.spawn(Position(x=1.0, y=2.0), Velocity(x=3.0, y=4.0))
    assert e != 0
    assert not world.is_alive(e)  # spawn is buffered
    world.flush()
    assert world.is_alive(e)
    pos = world.get_component(e, Position)
    vel = world.get_component(e, Velocity)
    assert (pos.x, pos.y) == (1.0, 2.0)
    assert (vel.x, vel.y) == (3.0, 4.0)


def test_despawn_recycles_entity_id():
    world = World()
    e1 = world.spawn(Position())
    world.flush()
    assert world.is_alive(e1)
    world.despawn(e1)
    assert world.is_alive(e1)  # despawn is also buffered
    world.flush()
    assert not world.is_alive(e1)
    e2 = world.spawn(Position())
    assert e2 == e1, "freed ID should be recycled by the next allocation"


def test_duplicate_component_in_spawn_raises():
    world = World()
    with pytest.raises(ValueError):
        world.spawn(Position(), Position())


# --- Add / remove component --------------------------------------------

def test_add_component_is_deferred():
    world = World()
    e = world.spawn(Position())
    world.flush()
    assert not world.has_component(e, Velocity)
    world.add_component(e, Velocity(x=5.0))
    assert not world.has_component(e, Velocity), "must not be visible before flush"
    world.flush()
    assert world.has_component(e, Velocity)
    assert world.get_component(e, Velocity).x == 5.0


def test_remove_component_is_deferred():
    world = World()
    e = world.spawn(Position(), Velocity(x=7.0))
    world.flush()
    assert world.has_component(e, Velocity)
    world.remove_component(e, Velocity)
    assert world.has_component(e, Velocity), "must not be removed before flush"
    world.flush()
    assert not world.has_component(e, Velocity)
    assert world.has_component(e, Position)


def test_add_existing_component_overwrites():
    world = World()
    e = world.spawn(Position(x=1.0))
    world.flush()
    world.add_component(e, Position(x=42.0, y=99.0))
    world.flush()
    p = world.get_component(e, Position)
    assert p.x == 42.0 and p.y == 99.0


# --- Migration ---------------------------------------------------------

def test_migration_preserves_other_entities():
    world = World()
    entities = [world.spawn(Position(x=float(i), y=float(i * 10))) for i in range(5)]
    world.flush()

    world.add_component(entities[2], Velocity(x=99.0, y=88.0))
    world.flush()

    for i, e in enumerate(entities):
        p = world.get_component(e, Position)
        assert p.x == float(i), f"entity {i} pos.x corrupted: {p.x}"
        assert p.y == float(i * 10), f"entity {i} pos.y corrupted: {p.y}"

    assert world.has_component(entities[2], Velocity)
    v = world.get_component(entities[2], Velocity)
    assert (v.x, v.y) == (99.0, 88.0)


def test_migration_round_trip_back_to_original_archetype():
    world = World()
    e = world.spawn(Position(x=2.0, y=20.0))
    world.flush()
    arch_before, _ = world.location_of(e)

    world.add_component(e, Velocity(x=1.0))
    world.flush()
    assert world.has_component(e, Velocity)

    world.remove_component(e, Velocity)
    world.flush()
    arch_after, _ = world.location_of(e)
    assert arch_after is arch_before
    p = world.get_component(e, Position)
    assert (p.x, p.y) == (2.0, 20.0)


def test_swap_remove_updates_location_for_moved_entity():
    world = World()
    a = world.spawn(Position(x=1.0))
    b = world.spawn(Position(x=2.0))
    c = world.spawn(Position(x=3.0))
    world.flush()
    world.despawn(a)
    world.flush()
    # a is gone; b and c should still be readable with correct values.
    assert not world.is_alive(a)
    assert world.get_component(b, Position).x == 2.0
    assert world.get_component(c, Position).x == 3.0


# --- Queries -----------------------------------------------------------

def test_query_intersects_required_components():
    world = World()
    a = world.spawn(Position(), Velocity())
    b = world.spawn(Position())
    c = world.spawn(Position(), Velocity(), Health())
    world.flush()

    matched: list[int] = []
    for arch in world.query(Position, Velocity).archetypes():
        matched.extend(arch.entities[: arch.length])
    assert set(matched) == {a, c}
    assert b not in matched


def test_query_with_without_filter():
    world = World()
    a = world.spawn(Position(), Velocity(x=1.0))
    b = world.spawn(Position(), Health(hp=50))
    c = world.spawn(Position())
    world.flush()

    only_pos = list(world.query(Position).entities())
    assert set(only_pos) == {a, b, c}

    pos_no_vel = list(world.query(Position, Without[Velocity]).entities())
    assert set(pos_no_vel) == {b, c}


def test_query_with_optional_yields_none_when_absent():
    world = World()
    a = world.spawn(Position(x=1.0), Velocity(x=10.0))
    b = world.spawn(Position(x=2.0))
    world.flush()

    seen: dict[float, float | None] = {}
    for pos, vel in world.query(Position, Optional[Velocity]):
        for i in range(len(pos)):
            x = float(pos["x"][i])
            seen[x] = float(vel["x"][i]) if vel is not None else None
    assert seen == {1.0: 10.0, 2.0: None}
    _ = (a, b)  # entities used implicitly via query


def test_query_yields_views_that_mutate_storage():
    world = World()
    e = world.spawn(Position(x=1.0, y=2.0), Velocity(x=10.0, y=20.0))
    world.flush()

    @world.system(Phase.UPDATE)
    def move(world: World, dt: float) -> None:
        for pos, vel in world.query(Position, Velocity):
            pos["x"] += vel["x"] * dt
            pos["y"] += vel["y"] * dt

    world.tick(0.5)
    p = world.get_component(e, Position)
    assert p.x == pytest.approx(1.0 + 10.0 * 0.5)
    assert p.y == pytest.approx(2.0 + 20.0 * 0.5)


def test_query_skips_empty_archetypes():
    world = World()
    e = world.spawn(Position())
    world.flush()
    world.despawn(e)
    world.flush()
    # Archetype still exists in registry but has no rows; query yields nothing.
    assert list(world.query(Position).entities()) == []


# --- Object-component storage ------------------------------------------

def test_object_component_storage_round_trip():
    world = World()
    e = world.spawn(Position(x=1.0), Name(label="hero"))
    world.flush()
    n = world.get_component(e, Name)
    assert n.label == "hero"
    # Pos should still be a structured-array column on the same archetype.
    arch, _ = world.location_of(e)
    assert isinstance(arch.columns[Position], np.ndarray)
    assert isinstance(arch.columns[Name], list)


# --- Events ------------------------------------------------------------

def test_emit_and_read_events():
    world = World()
    world.emit(HitEvent(target=1, damage=10))
    world.emit(HitEvent(target=2, damage=20))

    hits = list(world.read_events(HitEvent))
    assert [(h.target, h.damage) for h in hits] == [(1, 10), (2, 20)]
    # Reads are non-destructive within a frame.
    assert len(list(world.read_events(HitEvent))) == 2


def test_events_cleared_at_start_of_frame():
    world = World()
    log: list[int] = []

    @world.system(Phase.UPDATE)
    def reader(world: World, dt: float) -> None:
        for e in world.read_events(TickEvent):
            log.append(e.n)

    # Event emitted before tick — tick clears events first, so reader sees nothing.
    world.emit(TickEvent(n=1))
    world.tick(0.016)
    assert log == []


def test_event_visible_within_same_frame_across_phases():
    world = World()
    log: list[int] = []

    @world.system(Phase.PRE_UPDATE)
    def emitter(world: World, dt: float) -> None:
        world.emit(TickEvent(n=99))

    @world.system(Phase.UPDATE)
    def reader(world: World, dt: float) -> None:
        for e in world.read_events(TickEvent):
            log.append(e.n)

    world.tick(0.016)
    assert log == [99]
    log.clear()
    world.tick(0.016)
    # Each tick re-runs both: emit then read again, no leftover from prior frame.
    assert log == [99]


# --- Scheduler ---------------------------------------------------------

def test_systems_run_in_phase_order_then_registration_order():
    world = World()
    log: list[str] = []

    @world.system(Phase.RENDER)
    def s_render(world: World, dt: float) -> None:
        log.append("render")

    @world.system(Phase.UPDATE)
    def s_update_a(world: World, dt: float) -> None:
        log.append("update_a")

    @world.system(Phase.UPDATE)
    def s_update_b(world: World, dt: float) -> None:
        log.append("update_b")

    @world.system(Phase.PRE_UPDATE)
    def s_pre(world: World, dt: float) -> None:
        log.append("pre")

    @world.system(Phase.POST_RENDER)
    def s_post(world: World, dt: float) -> None:
        log.append("post")

    world.tick(0.016)
    assert log == ["pre", "update_a", "update_b", "render", "post"]


def test_resource_injection_by_annotation():
    world = World()
    world.insert_resource(Config(gravity=9.8))

    seen: list[float] = []

    @world.system(Phase.UPDATE)
    def gravity_system(world: World, dt: float, config: Config) -> None:
        seen.append(config.gravity)

    world.tick(0.016)
    assert seen == [9.8]


def test_unannotated_system_param_raises():
    world = World()
    with pytest.raises(TypeError):
        @world.system(Phase.UPDATE)
        def bad(world, dt, missing_annotation):  # noqa: ANN001
            pass


def test_tick_flushes_commands_at_end_of_frame():
    world = World()

    @world.system(Phase.UPDATE)
    def spawner(world: World, dt: float) -> None:
        world.spawn(Position(x=1.0))

    world.tick(0.016)
    # End-of-frame flush should have committed the spawn.
    assert world.query(Position).count() == 1


# --- Benchmarks --------------------------------------------------------

def test_benchmark_iteration_100k_under_16ms():
    world = World()
    n = 100_000
    for _ in range(n):
        world.spawn(Position(x=1.0, y=2.0), Velocity(x=0.5, y=0.25))
    world.flush()

    # Warmup so JIT-style caches are primed (numpy itself has no JIT, but
    # the first run pays for object alloc inside the iterator).
    for pos, vel in world.query(Position, Velocity):
        pos["x"] += vel["x"]

    start = time.perf_counter()
    for pos, vel in world.query(Position, Velocity):
        pos["x"] += vel["x"] * 0.016
        pos["y"] += vel["y"] * 0.016
    elapsed = time.perf_counter() - start
    assert elapsed < 0.016, (
        f"100k entity iteration took {elapsed * 1000:.3f}ms, expected <16ms"
    )


def test_benchmark_archetype_lookup_50_archetypes_under_0_1ms():
    world = World()
    extra_components: list[type] = []
    for i in range(50):
        cls = make_dataclass(f"_BenchC{i}", [("v", float, 0.0)])
        cls = component(cls)
        extra_components.append(cls)

    for ct in extra_components:
        world.spawn(Position(), ct())
    world.flush()
    # Sanity: 50 archetypes containing Position.
    assert len(world.archetypes.archetypes_with(Position)) == 50

    target = extra_components[25]

    # Warmup
    for _ in world.query(Position, target):
        pass

    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        for _ in world.query(Position, target):
            pass
    elapsed = time.perf_counter() - start
    per_query = elapsed / iterations
    assert per_query < 1e-4, (
        f"archetype lookup took {per_query * 1000:.4f}ms per query, expected <0.1ms"
    )


def test_has_resource_false_before_insert_true_after():
    """World.has_resource is a one-liner over the _resources dict."""
    class _DummyResource:
        pass
    world = World()
    assert world.has_resource(_DummyResource) is False
    world.insert_resource(_DummyResource())
    assert world.has_resource(_DummyResource) is True


def test_app_and_world_share_same_scheduler():
    """App._scheduler must alias World.scheduler so @app.system and @world.system
    register into the one scheduler the run loop actually drives."""
    # Build an App without opening a real GLFW window or wiring callbacks.
    with patch("pyge.Window"), patch("pyge.wire_callbacks", return_value=[]):
        app = pyge.App()
    assert app._scheduler is app.world.scheduler


def test_world_system_runs_under_app_tick():
    """A system registered via @world.system fires when scheduler.tick is called
    on the shared scheduler — proving the two decorators share state."""
    with patch("pyge.Window"), patch("pyge.wire_callbacks", return_value=[]):
        app = pyge.App()

    log: list[str] = []

    @app.world.system(Phase.UPDATE)
    def world_side_system(world, dt):
        log.append("world")

    @app.system(Phase.UPDATE)
    def app_side_system(world, dt):
        log.append("app")

    app._scheduler.tick(app.world, 0.016)
    assert log == ["world", "app"]


# --- world.get / world.set --------------------------------------------------

def test_world_get_returns_field_dict_for_numpy_component():
    w = World()
    eid = w.spawn(Position(x=10.0, y=20.0), Health(hp=42))
    w.flush()
    pos = w.get(eid, Position)
    assert isinstance(pos, dict)
    assert pos["x"] == 10.0 and pos["y"] == 20.0
    # values are plain Python scalars, not numpy generics
    assert not isinstance(pos["x"], np.generic)


def test_world_get_returns_none_for_missing_component():
    w = World()
    eid = w.spawn(Position())
    w.flush()
    assert w.get(eid, Health) is None


def test_world_get_returns_none_for_despawned_entity():
    w = World()
    eid = w.spawn(Position(), Health())
    w.flush()
    w.despawn(eid)
    w.flush()
    assert w.get(eid, Position) is None


def test_world_set_updates_field_immediately():
    w = World()
    eid = w.spawn(Position(x=0.0, y=0.0))
    w.flush()
    assert w.set(eid, Position, x=100.0) is True
    pos = w.get(eid, Position)
    assert pos is not None and pos["x"] == 100.0


def test_world_set_returns_false_for_missing_component():
    w = World()
    eid = w.spawn(Position())
    w.flush()
    assert w.set(eid, Health, hp=10) is False


def test_world_set_multiple_fields_in_one_call():
    w = World()
    eid = w.spawn(Position())
    w.flush()
    assert w.set(eid, Position, x=5.0, y=-3.0) is True
    pos = w.get(eid, Position)
    assert pos is not None
    assert pos["x"] == 5.0
    assert pos["y"] == -3.0


def test_world_set_bool_field():
    w = World()
    eid = w.spawn(Tag(flag=False))
    w.flush()
    w.set(eid, Tag, flag=True)
    assert w.get(eid, Tag) == {"flag": True}


def test_world_set_get_roundtrip():
    w = World()
    eid = w.spawn(Position(x=1.0, y=2.0))
    w.flush()
    w.set(eid, Position, x=9.5)
    pos = w.get(eid, Position)
    assert pos is not None and pos["x"] == 9.5 and pos["y"] == 2.0


def test_world_set_on_list_backed_component():
    """Components with non-numpy fields use list storage; world.set must still work."""
    w = World()
    eid = w.spawn(Name(label="alpha"))
    w.flush()
    assert w.set(eid, Name, label="omega") is True
    assert w.get(eid, Name) == {"label": "omega"}
