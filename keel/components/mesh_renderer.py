"""MeshRenderer — pairs an entity with a mesh ID, material ID, and visibility flags."""
from __future__ import annotations

from ..core import component


@component
class MeshRenderer:
    """Per-entity reference to a Mesh + Material plus shadow / visibility flags."""
    mesh_id: int = 0
    material_id: int = 0
    cast_shadows: bool = True
    receive_shadows: bool = True
    visible: bool = True
