"""Headless tests for the Phase 6 physics bridges (pymunk + pybullet)."""
from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pymunk
import pytest

import keel
from keel import (
    Collider2D,
    Collider3D,
    CollisionEvent2D,
    CollisionEvent3D,
    Phase,
    Physics2D,
    Physics3D,
    RigidBody2D,
    RigidBody3D,
    Scheduler,
    Transform2D,
    Transform3D,
    World,
    setup_physics_2d,
    setup_physics_3d,
)
from keel.physics.components2d import (
    BODY_TYPE_DYNAMIC as B2_DYNAMIC,
    BODY_TYPE_KINEMATIC as B2_KINEMATIC,
    BODY_TYPE_STATIC as B2_STATIC,
    SHAPE_TYPE_BOX as S2_BOX,
    SHAPE_TYPE_CIRCLE as S2_CIRCLE,
)
from keel.physics.components3d import (
    BODY_TYPE_DYNAMIC as B3_DYNAMIC,
    BODY_TYPE_STATIC as B3_STATIC,
    SHAPE_TYPE_BOX as S3_BOX,
    SHAPE_TYPE_SPHERE as S3_SPHERE,
)


# --- Fakes ---------------------------------------------------------------

def _fake_app() -> Any:
    """Minimal app shape: world + scheduler + system decorator + shutdown hooks."""
    sched = Scheduler()
    world = World()
    hooks: list = []
    app = SimpleNamespace(
        world=world,
        _scheduler=sched,
        scheduler=sched,
        _shutdown_hooks=hooks,
    )

    def system(phase: Phase):
        def deco(fn):
            sched.register(phase, fn)
            return fn
        return deco

    app.system = system
    app.add_shutdown_hook = lambda fn: hooks.append(fn)
    return app


# --- Physics2D -----------------------------------------------------------

def test_physics2d_init_sets_gravity():
    phys = Physics2D(gravity_x=0.0, gravity_y=-100.0)
    assert phys._space.gravity.x == 0.0
    assert phys._space.gravity.y == -100.0


def test_physics2d_sync_creates_body_and_shape():
    phys = Physics2D()
    world = World()
    eid = world.spawn(
        Transform2D(x=10.0, y=20.0),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=5.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    assert eid in phys._bodies
    assert eid in phys._shapes
    body = phys._bodies[eid]
    assert pytest.approx(body.position.x) == 10.0
    assert pytest.approx(body.position.y) == 20.0


def test_physics2d_dynamic_count_matches_entity_count():
    phys = Physics2D()
    world = World()
    ids = []
    for i in range(5):
        ids.append(world.spawn(
            Transform2D(x=float(i * 10)),
            RigidBody2D(mass=1.0),
            Collider2D(shape_type=S2_CIRCLE, radius=2.0),
        ))
    world.flush()
    phys.sync_to_physics(world)
    assert len(phys._bodies) == 5
    assert all(eid in phys._bodies for eid in ids)


def test_physics2d_sync_from_writes_transform():
    phys = Physics2D(gravity_y=-100.0)
    world = World()
    eid = world.spawn(
        Transform2D(x=0.0, y=100.0),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=1.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    for _ in range(60):
        phys.step(1.0 / 60.0)
    phys.sync_from_physics(world)
    t = world.get_component(eid, Transform2D)
    assert t.y < 100.0  # gravity pulled it down


def test_physics2d_static_body_transform_not_overwritten():
    phys = Physics2D(gravity_y=-100.0)
    world = World()
    eid = world.spawn(
        Transform2D(x=10.0, y=20.0),
        RigidBody2D(body_type=B2_STATIC),
        Collider2D(shape_type=S2_BOX, width=10.0, height=10.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    for _ in range(30):
        phys.step(1.0 / 60.0)
    phys.sync_from_physics(world)
    t = world.get_component(eid, Transform2D)
    assert t.x == 10.0
    assert t.y == 20.0


def test_physics2d_despawned_entity_removed_from_space():
    phys = Physics2D()
    world = World()
    eid = world.spawn(
        Transform2D(),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=2.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    assert eid in phys._bodies
    world.despawn(eid)
    world.flush()
    phys.sync_to_physics(world)
    assert eid not in phys._bodies
    assert eid not in phys._shapes


def test_physics2d_apply_impulse_no_op_on_missing_entity():
    phys = Physics2D()
    phys.apply_impulse(99999, 1.0, 0.0)  # must not raise


def test_physics2d_apply_impulse_changes_velocity():
    phys = Physics2D()
    world = World()
    eid = world.spawn(
        Transform2D(),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=1.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    assert phys._bodies[eid].velocity.x == 0.0
    phys.apply_impulse(eid, 50.0, 0.0)
    assert phys._bodies[eid].velocity.x > 0.0


def test_physics2d_set_velocity_mirrors_to_ecs_when_world_attached():
    """Phase 7 fix 3: phys.set_velocity also writes RigidBody2D fields when world is set."""
    world = World()
    phys = Physics2D(world=world)
    eid = world.spawn(
        Transform2D(),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=2.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    phys.set_velocity(eid, 7.5, -3.25)
    rb = world.get_component(eid, RigidBody2D)
    assert rb.vel_x == 7.5
    assert rb.vel_y == -3.25
    # The pymunk body got it too.
    assert phys._bodies[eid].velocity.x == 7.5
    assert phys._bodies[eid].velocity.y == -3.25


def test_physics2d_set_velocity_no_world_only_touches_body():
    """Without a world, set_velocity touches only the pymunk body (back-compat)."""
    phys = Physics2D()  # no world
    world = World()
    eid = world.spawn(
        Transform2D(),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=2.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    phys.set_velocity(eid, 1.0, 2.0)
    assert phys._bodies[eid].velocity.x == 1.0
    rb = world.get_component(eid, RigidBody2D)
    # ECS field unchanged because no world was passed to Physics2D.
    assert rb.vel_x == 0.0


def test_physics2d_set_position_teleports_body_and_ecs():
    """Phase 7 fix 10: set_position moves the body AND writes Transform2D."""
    world = World()
    phys = Physics2D(world=world)
    eid = world.spawn(
        Transform2D(x=0.0, y=0.0),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=2.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    phys.set_position(eid, 100.0, 50.0)
    t = world.get_component(eid, Transform2D)
    assert t.x == 100.0
    assert t.y == 50.0
    body_pos = phys._bodies[eid].position
    assert body_pos.x == 100.0
    assert body_pos.y == 50.0


def test_physics2d_set_position_no_op_for_unknown_entity():
    phys = Physics2D()
    phys.set_position(99999, 1.0, 2.0)  # must not raise


def test_setup_physics_2d_passes_world_to_bridge():
    """The setup helper should attach the world so set_velocity / set_position auto-sync ECS."""
    app = _fake_app()
    phys = setup_physics_2d(app)
    try:
        assert phys.world is app.world
    finally:
        phys.cleanup()


def test_physics2d_raycast_empty_world():
    phys = Physics2D()
    assert phys.raycast_2d((0, 0), (100, 0)) == []


def test_physics2d_raycast_hits_static_in_path():
    phys = Physics2D()
    world = World()
    target = world.spawn(
        Transform2D(x=50.0, y=0.0),
        RigidBody2D(body_type=B2_STATIC),
        Collider2D(shape_type=S2_CIRCLE, radius=10.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    hits = phys.raycast_2d((0, 0), (100, 0))
    assert len(hits) >= 1
    assert hits[0]["entity_id"] == target
    assert "point" in hits[0] and "normal" in hits[0] and "alpha" in hits[0]


def test_physics2d_collision_event_emitted_on_contact():
    phys = Physics2D(gravity_y=-200.0)
    world = World()
    floor = world.spawn(
        Transform2D(x=0.0, y=0.0),
        RigidBody2D(body_type=B2_STATIC),
        Collider2D(shape_type=S2_BOX, width=200.0, height=10.0),
    )
    ball = world.spawn(
        Transform2D(x=0.0, y=30.0),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=5.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    for _ in range(120):  # ~2 seconds
        phys.step(1.0 / 60.0)
    phys._emit_collisions(world)
    events = list(world.read_events(CollisionEvent2D))
    assert len(events) >= 1
    e = events[0]
    assert {e.entity_a, e.entity_b} == {floor, ball}


def test_physics2d_static_body_creates_pymunk_static():
    phys = Physics2D()
    world = World()
    eid = world.spawn(
        Transform2D(),
        RigidBody2D(body_type=B2_STATIC),
        Collider2D(shape_type=S2_BOX, width=10.0, height=10.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    assert phys._bodies[eid].body_type == pymunk.Body.STATIC


def test_physics2d_kinematic_body_creates_pymunk_kinematic():
    phys = Physics2D()
    world = World()
    eid = world.spawn(
        Transform2D(),
        RigidBody2D(body_type=B2_KINEMATIC),
        Collider2D(shape_type=S2_CIRCLE, radius=2.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    assert phys._bodies[eid].body_type == pymunk.Body.KINEMATIC


def test_physics2d_sensor_flag_passes_through():
    phys = Physics2D()
    world = World()
    eid = world.spawn(
        Transform2D(),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=2.0, sensor=True),
    )
    world.flush()
    phys.sync_to_physics(world)
    assert phys._shapes[eid].sensor is True


def test_physics2d_circle_shape_creates_pymunk_circle():
    phys = Physics2D()
    world = World()
    eid = world.spawn(
        Transform2D(),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=4.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    assert isinstance(phys._shapes[eid], pymunk.Circle)


def test_physics2d_box_shape_creates_pymunk_poly():
    phys = Physics2D()
    world = World()
    eid = world.spawn(
        Transform2D(),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_BOX, width=10.0, height=4.0),
    )
    world.flush()
    phys.sync_to_physics(world)
    assert isinstance(phys._shapes[eid], pymunk.Poly)


def test_physics2d_warns_on_collider_without_rigidbody():
    phys = Physics2D()
    world = World()
    # Collider2D but no RigidBody2D.
    world.spawn(Transform2D(), Collider2D(shape_type=S2_CIRCLE))
    world.flush()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        phys.sync_to_physics(world)
    assert any(
        issubclass(warning.category, RuntimeWarning) and "Collider2D" in str(warning.message)
        for warning in w
    )


def test_physics2d_handler_registered_once():
    """The pymunk collision handler must not be re-registered on every tick."""
    phys = Physics2D()
    world = World()
    world.spawn(
        Transform2D(x=0.0, y=20.0),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=2.0),
    )
    world.flush()
    initial_handlers = dict(phys._space._handlers) if hasattr(phys._space, "_handlers") else None
    for _ in range(5):
        phys.sync_to_physics(world)
        phys.step(1.0 / 60.0)
    if initial_handlers is not None:
        assert dict(phys._space._handlers) == initial_handlers


# --- setup_physics_2d -----------------------------------------------------

def test_setup_physics_2d_inserts_resource():
    app = _fake_app()
    phys = setup_physics_2d(app)
    try:
        assert app.world.get_resource(Physics2D) is phys
    finally:
        phys.cleanup()


def test_setup_physics_2d_registers_post_update_system():
    app = _fake_app()
    phys = setup_physics_2d(app)
    try:
        post = app._scheduler.systems(Phase.POST_UPDATE)
        assert len(post) == 1
    finally:
        phys.cleanup()


def test_setup_physics_2d_idempotent():
    app = _fake_app()
    a = setup_physics_2d(app)
    b = setup_physics_2d(app)
    try:
        assert a is b
        assert len(app._scheduler.systems(Phase.POST_UPDATE)) == 1
    finally:
        a.cleanup()


def test_setup_physics_2d_registers_shutdown_hook():
    app = _fake_app()
    phys = setup_physics_2d(app)
    try:
        assert phys.cleanup in app._shutdown_hooks
    finally:
        phys.cleanup()


# --- Physics3D ------------------------------------------------------------

def test_physics3d_init_uses_direct_mode():
    phys = Physics3D()
    try:
        assert phys.connected is True
        assert phys.client_id >= 0
    finally:
        phys.disconnect()


def test_physics3d_sync_creates_body():
    phys = Physics3D(gravity_y=0.0)
    try:
        world = World()
        eid = world.spawn(
            Transform3D(x=1.0, y=2.0, z=3.0),
            RigidBody3D(mass=1.0),
            Collider3D(shape_type=S3_SPHERE, radius=0.5),
        )
        world.flush()
        phys.sync_to_physics(world)
        assert eid in phys._bodies
    finally:
        phys.disconnect()


def test_physics3d_sync_from_updates_transform():
    phys = Physics3D(gravity_y=-9.81)
    try:
        world = World()
        eid = world.spawn(
            Transform3D(x=0.0, y=10.0, z=0.0),
            RigidBody3D(mass=1.0),
            Collider3D(shape_type=S3_SPHERE, radius=0.5),
        )
        world.flush()
        phys.sync_to_physics(world)
        for _ in range(60):
            phys.step(1.0 / 60.0)
        phys.sync_from_physics(world)
        t = world.get_component(eid, Transform3D)
        assert t.y < 10.0  # gravity pulled it down
    finally:
        phys.disconnect()


def test_physics3d_static_body_transform_not_overwritten():
    phys = Physics3D(gravity_y=-9.81)
    try:
        world = World()
        eid = world.spawn(
            Transform3D(x=5.0, y=5.0, z=5.0),
            RigidBody3D(body_type=B3_STATIC),
            Collider3D(shape_type=S3_BOX, size_x=1.0, size_y=1.0, size_z=1.0),
        )
        world.flush()
        phys.sync_to_physics(world)
        for _ in range(30):
            phys.step(1.0 / 60.0)
        phys.sync_from_physics(world)
        t = world.get_component(eid, Transform3D)
        assert (t.x, t.y, t.z) == (5.0, 5.0, 5.0)
    finally:
        phys.disconnect()


def test_physics3d_apply_impulse_no_error():
    phys = Physics3D(gravity_y=0.0)
    try:
        world = World()
        eid = world.spawn(
            Transform3D(),
            RigidBody3D(mass=1.0),
            Collider3D(shape_type=S3_SPHERE, radius=0.5),
        )
        world.flush()
        phys.sync_to_physics(world)
        # Should not raise even if the body just spawned this tick.
        phys.apply_impulse(eid, 1.0, 0.0, 0.0)
        # No-op on missing entity must also not raise.
        phys.apply_impulse(99999, 1.0, 0.0, 0.0)
    finally:
        phys.disconnect()


def test_physics3d_set_velocity_mirrors_to_ecs_when_world_attached():
    """Phase-7 parity: phys.set_velocity also writes RigidBody3D fields."""
    world = World()
    phys = Physics3D(gravity_y=0.0, world=world)
    try:
        eid = world.spawn(
            Transform3D(),
            RigidBody3D(mass=1.0),
            Collider3D(shape_type=S3_SPHERE, radius=0.5),
        )
        world.flush()
        phys.sync_to_physics(world)
        phys.set_velocity(eid, 1.5, -2.5, 3.25)
        rb = world.get_component(eid, RigidBody3D)
        assert rb.vel_x == 1.5
        assert rb.vel_y == -2.5
        assert rb.vel_z == 3.25
        # And pymunk^Wpybullet body got it too.
        lin, _ang = phys._p.getBaseVelocity(
            phys._bodies[eid], physicsClientId=phys.client_id
        )
        assert lin[0] == 1.5 and lin[1] == -2.5 and lin[2] == 3.25
    finally:
        phys.disconnect()


def test_physics3d_set_velocity_no_op_when_world_is_none():
    """Without a world, set_velocity touches only the pybullet body."""
    phys = Physics3D(gravity_y=0.0)  # no world
    world = World()
    try:
        eid = world.spawn(
            Transform3D(),
            RigidBody3D(mass=1.0),
            Collider3D(shape_type=S3_SPHERE, radius=0.5),
        )
        world.flush()
        phys.sync_to_physics(world)
        phys.set_velocity(eid, 9.0, 8.0, 7.0)
        lin, _ = phys._p.getBaseVelocity(
            phys._bodies[eid], physicsClientId=phys.client_id
        )
        assert lin[0] == 9.0
        # ECS field stays at the default since no world was passed in.
        rb = world.get_component(eid, RigidBody3D)
        assert rb.vel_x == 0.0
    finally:
        phys.disconnect()


def test_physics3d_set_position_updates_transform3d():
    """set_position teleports the body AND writes Transform3D."""
    world = World()
    phys = Physics3D(gravity_y=0.0, world=world)
    try:
        eid = world.spawn(
            Transform3D(x=0.0, y=0.0, z=0.0),
            RigidBody3D(mass=1.0),
            Collider3D(shape_type=S3_SPHERE, radius=0.5),
        )
        world.flush()
        phys.sync_to_physics(world)
        phys.set_position(eid, 4.0, 5.0, 6.0)
        t = world.get_component(eid, Transform3D)
        assert t.x == 4.0 and t.y == 5.0 and t.z == 6.0
        pos, _ = phys._p.getBasePositionAndOrientation(
            phys._bodies[eid], physicsClientId=phys.client_id
        )
        assert pos[0] == 4.0 and pos[1] == 5.0 and pos[2] == 6.0
    finally:
        phys.disconnect()


def test_physics3d_set_position_no_op_for_unknown_entity():
    phys = Physics3D()
    try:
        phys.set_position(99999, 1.0, 2.0, 3.0)  # must not raise
    finally:
        phys.disconnect()


def test_physics3d_raycast_empty_world():
    phys = Physics3D()
    try:
        assert phys.raycast_3d((0, 0, 0), (10, 0, 0)) == []
    finally:
        phys.disconnect()


def test_physics3d_sphere_collision_shape():
    phys = Physics3D()
    try:
        world = World()
        world.spawn(
            Transform3D(),
            RigidBody3D(mass=1.0),
            Collider3D(shape_type=S3_SPHERE, radius=0.5),
        )
        world.flush()
        phys.sync_to_physics(world)
        # All three internal dicts should have one entry each.
        assert len(phys._collision_shapes) == 1
        assert len(phys._bodies) == 1
        assert len(phys._body_types) == 1
    finally:
        phys.disconnect()


def test_physics3d_box_collision_shape():
    phys = Physics3D()
    try:
        world = World()
        world.spawn(
            Transform3D(),
            RigidBody3D(mass=1.0),
            Collider3D(shape_type=S3_BOX, size_x=1.0, size_y=2.0, size_z=3.0),
        )
        world.flush()
        phys.sync_to_physics(world)
        assert len(phys._collision_shapes) == 1
    finally:
        phys.disconnect()


def test_physics3d_disconnect_idempotent():
    phys = Physics3D()
    phys.disconnect()
    assert phys.connected is False
    phys.disconnect()  # second call must not raise
    assert phys.connected is False


def test_physics3d_warns_on_collider_without_rigidbody():
    phys = Physics3D()
    try:
        world = World()
        world.spawn(Transform3D(), Collider3D(shape_type=S3_SPHERE))
        world.flush()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            phys.sync_to_physics(world)
        assert any(
            issubclass(warning.category, RuntimeWarning) and "Collider3D" in str(warning.message)
            for warning in w
        )
    finally:
        phys.disconnect()


# --- setup_physics_3d -----------------------------------------------------

def test_setup_physics_3d_inserts_resource():
    app = _fake_app()
    phys = setup_physics_3d(app)
    try:
        assert app.world.get_resource(Physics3D) is phys
    finally:
        phys.disconnect()


def test_setup_physics_3d_registers_post_update_system():
    app = _fake_app()
    phys = setup_physics_3d(app)
    try:
        post = app._scheduler.systems(Phase.POST_UPDATE)
        assert len(post) == 1
    finally:
        phys.disconnect()


def test_setup_physics_3d_idempotent():
    app = _fake_app()
    a = setup_physics_3d(app)
    b = setup_physics_3d(app)
    try:
        assert a is b
        assert len(app._scheduler.systems(Phase.POST_UPDATE)) == 1
    finally:
        a.disconnect()


def test_setup_physics_3d_registers_shutdown_hook():
    app = _fake_app()
    phys = setup_physics_3d(app)
    try:
        assert phys.disconnect in app._shutdown_hooks
    finally:
        phys.disconnect()


# --- 2D + 3D coexistence + general --------------------------------------

def test_2d_and_3d_physics_coexist_on_same_app():
    app = _fake_app()
    p2 = setup_physics_2d(app)
    p3 = setup_physics_3d(app)
    try:
        assert app.world.get_resource(Physics2D) is p2
        assert app.world.get_resource(Physics3D) is p3
        # Both should be registered at POST_UPDATE.
        post = app._scheduler.systems(Phase.POST_UPDATE)
        assert len(post) == 2
    finally:
        p2.cleanup()
        p3.disconnect()


def test_collision_event_3d_is_registered_event():
    assert getattr(CollisionEvent3D, "__keel_event__", False) is True


def test_collision_event_2d_is_registered_event():
    assert getattr(CollisionEvent2D, "__keel_event__", False) is True


def test_physics_components_registered():
    for cls in (RigidBody2D, Collider2D, RigidBody3D, Collider3D):
        assert getattr(cls, "__keel_component__", None) is not None


# --- Public re-exports ---------------------------------------------------

def test_top_level_re_exports():
    assert keel.Physics2D is Physics2D
    assert keel.Physics3D is Physics3D
    assert keel.RigidBody2D is RigidBody2D
    assert keel.RigidBody3D is RigidBody3D
    assert keel.Collider2D is Collider2D
    assert keel.Collider3D is Collider3D
    assert keel.CollisionEvent2D is CollisionEvent2D
    assert keel.CollisionEvent3D is CollisionEvent3D
    assert keel.setup_physics_2d is setup_physics_2d
    assert keel.setup_physics_3d is setup_physics_3d


def test_kinematic_kinematic_no_collision_event_documented():
    """Document the pymunk limitation: KINEMATIC-vs-KINEMATIC emits no events.

    This is intentional pymunk behavior — use DYNAMIC for entities that must
    collide with each other. Keel also surfaces a one-time UserWarning when a
    second kinematic body joins the space.
    """
    from keel.physics.components2d import BODY_TYPE_KINEMATIC

    phys = Physics2D()
    world = World()
    a = world.spawn(
        Transform2D(x=80.0, y=120.0),
        RigidBody2D(mass=1.0, body_type=BODY_TYPE_KINEMATIC, vel_x=200.0),
        Collider2D(shape_type=S2_CIRCLE, radius=10.0),
    )
    b = world.spawn(
        Transform2D(x=240.0, y=120.0),
        RigidBody2D(mass=1.0, body_type=BODY_TYPE_KINEMATIC, vel_x=-200.0),
        Collider2D(shape_type=S2_CIRCLE, radius=10.0),
    )
    world.flush()

    # Second kinematic body should trigger the one-time warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        phys.sync_to_physics(world)
    assert any("kinematic" in str(w.message).lower() for w in caught), (
        "Physics2D should warn the first time a second kinematic body joins it."
    )

    # Step long enough for the two bodies to overlap and pass through.
    for _ in range(60):
        phys.step(1.0 / 60.0)
    phys._emit_collisions(world)
    events = list(world.read_events(CollisionEvent2D))
    assert events == [], (
        "pymunk does not generate CollisionEvent2D between two kinematic "
        f"bodies, but got {len(events)} event(s) — engine behavior changed."
    )


# ---------------------------------------------------------------------------
# BodyType / ShapeType IntEnum exports (v0.1.4)
# ---------------------------------------------------------------------------
#
# These enums are IntEnums: existing call-sites passing raw integers must
# keep working, and the symbolic names must be reachable via both
# `keel.physics` and the top-level `keel` namespace.

def test_body_type_int_values():
    assert keel.BodyType.DYNAMIC == 0
    assert keel.BodyType.STATIC == 1
    assert keel.BodyType.KINEMATIC == 2
    # IntEnum identity: compares equal to plain ints (backwards compat).
    assert int(keel.BodyType.STATIC) == 1


def test_shape_type_2d_int_values():
    assert keel.ShapeType2D.CIRCLE == 0
    assert keel.ShapeType2D.BOX == 1
    assert keel.ShapeType2D.SEGMENT == 2


def test_shape_type_3d_int_values():
    assert keel.ShapeType3D.SPHERE == 0
    assert keel.ShapeType3D.BOX == 1
    assert keel.ShapeType3D.CAPSULE == 2
    assert keel.ShapeType3D.MESH == 3


def test_rigid_body_2d_accepts_body_type_enum():
    body = RigidBody2D(body_type=keel.BodyType.STATIC)
    assert body.body_type == 1


def test_collider_2d_accepts_shape_type_enum():
    col = Collider2D(shape_type=keel.ShapeType2D.CIRCLE, radius=20.0)
    assert col.shape_type == 0
    assert col.radius == 20.0


def test_enums_visible_from_top_level_keel():
    # The top-level package re-exports each enum.
    assert keel.BodyType is not None
    assert keel.ShapeType2D is not None
    assert keel.ShapeType3D is not None
    # And from keel.physics.
    from keel.physics import BodyType, ShapeType2D, ShapeType3D
    assert BodyType is keel.BodyType
    assert ShapeType2D is keel.ShapeType2D
    assert ShapeType3D is keel.ShapeType3D

