"""Light components and the AmbientLight resource.

DirectionalLight: world-space direction (sun-like). One per scene; the
renderer takes the first found and ignores the rest.

PointLight: position comes from a co-located Transform3D on the same entity.
Up to 8 point lights per frame are uploaded — extras (sorted by distance to
the camera) are silently dropped.

AmbientLight is NOT a component — it's a world resource. If unset the
renderer falls back to vec3(0.1, 0.1, 0.1).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core import component


MAX_POINT_LIGHTS: int = 8


@component
class DirectionalLight:
    """Sun-style light. dir_x/y/z is the direction the light is travelling (will be normalized)."""
    dir_x: float = -0.577
    dir_y: float = -0.577
    dir_z: float = -0.577
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    intensity: float = 1.0


@component
class PointLight:
    """Point light. Position comes from the entity's Transform3D — none stored here."""
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    intensity: float = 1.0
    radius: float = 10.0


@dataclass
class AmbientLight:
    """Scene-wide ambient term. Insert as a world resource; renderer reads it once per frame."""
    r: float = 0.1
    g: float = 0.1
    b: float = 0.1
