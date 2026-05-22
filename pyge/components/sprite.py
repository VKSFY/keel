"""Sprite — textured quad with tint, world-space size, and per-axis flip."""
from __future__ import annotations

from ..core import component


@component
class Sprite:
    """A textured quad. texture_id indexes into the active TextureAtlas."""
    texture_id: int = 0
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0
    width: float = 64.0
    height: float = 64.0
    flip_x: bool = False
    flip_y: bool = False
