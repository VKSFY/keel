"""2D physics components and the CollisionEvent2D type.

These are pure ECS components — no pymunk imports here. The Physics2D bridge
in physics2d.py reads the data, owns the actual pymunk bodies, and writes
results back into Transform2D each tick.
"""
from __future__ import annotations

from ..core import component, event


# Body type IDs (matching pymunk semantics).
BODY_TYPE_DYNAMIC: int = 0
BODY_TYPE_STATIC: int = 1
BODY_TYPE_KINEMATIC: int = 2

# Shape type IDs.
SHAPE_TYPE_CIRCLE: int = 0
SHAPE_TYPE_BOX: int = 1
SHAPE_TYPE_SEGMENT: int = 2


@component
class RigidBody2D:
    """Rigid-body parameters. Position/rotation come from Transform2D on the same entity.

    body_type values:
      0 = DYNAMIC    — affected by forces; collides with anything; emits
                       CollisionEvent2D against any other body.
      1 = STATIC     — immovable; emits CollisionEvent2D when a dynamic body
                       touches it.
      2 = KINEMATIC  — moved manually via Physics2D.set_velocity /
                       set_position. **Does NOT emit CollisionEvent2D against
                       another kinematic body** (pymunk callback behavior).
                       Use DYNAMIC for entities that must detect collisions
                       with each other.
    """
    mass: float = 1.0
    moment: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    ang_vel: float = 0.0
    damping: float = 0.0
    body_type: int = 0


@component
class Collider2D:
    """Collision shape parameters. shape_type chooses which size fields apply."""
    shape_type: int = 0
    width: float = 32.0
    height: float = 32.0
    radius: float = 16.0
    friction: float = 0.5
    elasticity: float = 0.3
    sensor: bool = False
    category_bits: int = 1
    mask_bits: int = 0xFFFF


@event
class CollisionEvent2D:
    """Emitted by Physics2D when two collidable shapes touch this tick."""
    entity_a: int
    entity_b: int
    normal_x: float
    normal_y: float
    impulse: float
