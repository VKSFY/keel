"""Transform3D — 3D transform with Euler-angle rotation, per-axis scale, and parenting."""
from __future__ import annotations

from ..core import component


@component
class Transform3D:
    """3D transform. parent=0 means no parent (root). Rotations are in radians."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    parent: int = 0
