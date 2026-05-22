"""3D physics components and the CollisionEvent3D type.

Pure ECS data — no pybullet imports. The Physics3D bridge owns the pybullet
bodies; ECS only describes them.
"""
from __future__ import annotations

from ..core import component, event


# Body type IDs (mirror the 2D constants for consistency).
BODY_TYPE_DYNAMIC: int = 0
BODY_TYPE_STATIC: int = 1
BODY_TYPE_KINEMATIC: int = 2

# Shape type IDs.
SHAPE_TYPE_SPHERE: int = 0
SHAPE_TYPE_BOX: int = 1
SHAPE_TYPE_CAPSULE: int = 2
SHAPE_TYPE_MESH: int = 3


@component
class RigidBody3D:
    """3D rigid-body parameters. Position/rotation come from Transform3D on the same entity."""
    mass: float = 1.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_z: float = 0.0
    ang_vel_x: float = 0.0
    ang_vel_y: float = 0.0
    ang_vel_z: float = 0.0
    damping: float = 0.0
    ang_damping: float = 0.0
    body_type: int = 0


@component
class Collider3D:
    """3D collision shape parameters. Selected fields depend on shape_type."""
    shape_type: int = 0
    size_x: float = 1.0
    size_y: float = 1.0
    size_z: float = 1.0
    radius: float = 0.5
    friction: float = 0.5
    restitution: float = 0.3
    mesh_id: int = 0


@event
class CollisionEvent3D:
    """Emitted by Physics3D when two bodies report a contact this tick."""
    entity_a: int
    entity_b: int
    contact_x: float
    contact_y: float
    contact_z: float
    normal_x: float
    normal_y: float
    normal_z: float
