"""Named constants for physics body and shape types.

These are IntEnum so existing code that passes raw ints (body_type=1,
shape_type=0) keeps working — IntEnum members compare equal to their
integer values:

    BodyType.STATIC == 1     # True
    int(BodyType.STATIC)     # 1

Prefer the enum form in new code for readability:

    RigidBody2D(body_type=keel.BodyType.STATIC)
    Collider2D(shape_type=keel.ShapeType2D.CIRCLE, radius=20.0)
"""
from __future__ import annotations

from enum import IntEnum


class BodyType(IntEnum):
    """Rigid-body simulation mode. Shared by 2D and 3D physics bridges."""
    DYNAMIC = 0
    STATIC = 1
    KINEMATIC = 2


class ShapeType2D(IntEnum):
    """Collider2D shape selector."""
    CIRCLE = 0
    BOX = 1
    SEGMENT = 2


class ShapeType3D(IntEnum):
    """Collider3D shape selector."""
    SPHERE = 0
    BOX = 1
    CAPSULE = 2
    MESH = 3
