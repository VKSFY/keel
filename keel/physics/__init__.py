"""Physics bridges for Keel — pymunk for 2D, pybullet for 3D.

Both bridges are wired the same way: a system at Phase.POST_UPDATE that
runs sync_to_physics → step → sync_from_physics → _emit_collisions, with
the bridge object inserted as a world resource so other systems can read or
inject control inputs.
"""
from __future__ import annotations

from typing import Any

from ..core import Phase
from .components2d import (
    BODY_TYPE_DYNAMIC,
    BODY_TYPE_KINEMATIC,
    BODY_TYPE_STATIC,
    Collider2D,
    CollisionEvent2D,
    RigidBody2D,
    SHAPE_TYPE_BOX as SHAPE_TYPE_2D_BOX,
    SHAPE_TYPE_CIRCLE as SHAPE_TYPE_2D_CIRCLE,
    SHAPE_TYPE_SEGMENT as SHAPE_TYPE_2D_SEGMENT,
)
from .components3d import (
    Collider3D,
    CollisionEvent3D,
    RigidBody3D,
    SHAPE_TYPE_BOX as SHAPE_TYPE_3D_BOX,
    SHAPE_TYPE_CAPSULE as SHAPE_TYPE_3D_CAPSULE,
    SHAPE_TYPE_MESH as SHAPE_TYPE_3D_MESH,
    SHAPE_TYPE_SPHERE as SHAPE_TYPE_3D_SPHERE,
)
from .enums import BodyType, ShapeType2D, ShapeType3D
from .material import PhysicsMaterial2D
from .physics2d import Physics2D
# Physics3D imports pybullet lazily and PYBULLET_AVAILABLE reflects the
# import outcome. Importing Physics3D itself never raises — Physics3D
# .__init__ does, with a clear message.
from .physics3d import PYBULLET_AVAILABLE as PYBULLET_AVAILABLE
from .physics3d import Physics3D


__all__ = [
    "BODY_TYPE_DYNAMIC",
    "BODY_TYPE_KINEMATIC",
    "BODY_TYPE_STATIC",
    "BodyType",
    "Collider2D",
    "Collider3D",
    "CollisionEvent2D",
    "CollisionEvent3D",
    "PYBULLET_AVAILABLE",
    "Physics2D",
    "Physics3D",
    "PhysicsMaterial2D",
    "RigidBody2D",
    "RigidBody3D",
    "SHAPE_TYPE_2D_BOX",
    "SHAPE_TYPE_2D_CIRCLE",
    "SHAPE_TYPE_2D_SEGMENT",
    "SHAPE_TYPE_3D_BOX",
    "SHAPE_TYPE_3D_CAPSULE",
    "SHAPE_TYPE_3D_MESH",
    "SHAPE_TYPE_3D_SPHERE",
    "ShapeType2D",
    "ShapeType3D",
    "apply_material",
    "setup_physics_2d",
    "setup_physics_3d",
]


def apply_material(world: Any, entity_id: int, material: PhysicsMaterial2D) -> None:
    """Apply a PhysicsMaterial2D to an entity before it is synced to physics.

    Call immediately after `world.spawn(...)` + `world.flush()`. The material's
    friction and elasticity override the Collider2D component's values when
    Physics2D builds the pymunk shape on the next sync_to_physics.

    `world` is accepted for forward compatibility (future per-world material
    isolation); the current implementation uses a process-level dict.
    """
    from .physics2d import set_material
    _ = world  # reserved for future per-world routing
    set_material(entity_id, material)


def _register_shutdown_hook(app: Any, hook) -> None:
    """Best-effort: hand the cleanup callable to App if it exposes a hook list."""
    add_hook = getattr(app, "add_shutdown_hook", None)
    if callable(add_hook):
        add_hook(hook)


def setup_physics_2d(
    app: Any,
    gravity_x: float = 0.0,
    gravity_y: float = -980.0,
) -> Physics2D:
    """Create + register the pymunk bridge on `app`. Idempotent — second call is a no-op."""
    existing = getattr(app, "_keel_physics_2d", None)
    if existing is not None:
        return existing

    phys = Physics2D(gravity_x=gravity_x, gravity_y=gravity_y, world=app.world)
    app.world.insert_resource(phys, type_=Physics2D)

    @app.system(Phase.POST_UPDATE)
    def physics_2d_system(world: Any, dt: float, phys: Physics2D) -> None:
        phys.sync_to_physics(world)
        phys.step(dt)
        phys.sync_from_physics(world)
        phys._emit_collisions(world)

    _register_shutdown_hook(app, phys.cleanup)
    app._keel_physics_2d = phys
    return phys


def setup_physics_3d(
    app: Any,
    gravity_y: float = -9.81,
) -> Physics3D:
    """Create + register the pybullet bridge on `app`. Idempotent — second call is a no-op.

    Raises ImportError with install guidance when pybullet is not installed.
    """
    if not PYBULLET_AVAILABLE:
        raise ImportError(
            "3D physics requires pybullet. "
            "Install it with: pip install keelpy[physics3d]"
        )
    existing = getattr(app, "_keel_physics_3d", None)
    if existing is not None:
        return existing

    phys = Physics3D(gravity_x=0.0, gravity_y=gravity_y, gravity_z=0.0, world=app.world)
    app.world.insert_resource(phys, type_=Physics3D)

    @app.system(Phase.POST_UPDATE)
    def physics_3d_system(world: Any, dt: float, phys: Physics3D) -> None:
        phys.sync_to_physics(world)
        phys.step(dt)
        phys.sync_from_physics(world)
        phys._emit_collisions(world)

    _register_shutdown_hook(app, phys.disconnect)
    app._keel_physics_3d = phys
    return phys
