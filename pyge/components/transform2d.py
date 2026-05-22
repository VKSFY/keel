"""Transform2D — position, rotation (radians), and per-axis scale."""
from __future__ import annotations

from ..core import component


@component
class Transform2D:
    """2D transform: world-space position, rotation in radians, per-axis scale."""
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
