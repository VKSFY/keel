"""Camera2D component and orthographic camera-matrix builder.

The camera produces a single 4x4 float32 matrix that maps world coordinates
into normalized device coordinates: world (cx + w/(2z), cy + h/(2z)) -> NDC
(1, 1) at zoom z. Y is up. Origin is at the center of the framebuffer.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core import component


@component
class Camera2D:
    """Orthographic 2D camera. One entity should carry this — the renderer reads it each frame."""
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0
    rotation: float = 0.0


def _read_camera(camera: Any) -> tuple[float, float, float, float]:
    """Pull (x, y, zoom, rotation) from a Camera2D instance or a structured-array record."""
    if hasattr(camera, "x") and not isinstance(camera, np.void):
        return (
            float(camera.x),
            float(camera.y),
            float(camera.zoom),
            float(camera.rotation),
        )
    return (
        float(camera["x"]),
        float(camera["y"]),
        float(camera["zoom"]),
        float(camera["rotation"]),
    )


def build_camera_matrix(
    camera: Any, viewport_width: int, viewport_height: int
) -> np.ndarray:
    """Return a (4, 4) float32 ortho-projection × view matrix for the given camera and viewport."""
    cx, cy, zoom, rotation = _read_camera(camera)
    if viewport_width <= 0 or viewport_height <= 0:
        raise ValueError("viewport dimensions must be positive")

    theta = -rotation
    c = math.cos(theta)
    s = math.sin(theta)
    z = zoom
    w = float(viewport_width)
    h = float(viewport_height)

    # Combined P * S(z) * R(-rot) * T(-cx,-cy) folded into one 4x4.
    m = np.array(
        [
            [2.0 * z * c / w, -2.0 * z * s / w, 0.0, 2.0 * z * (-c * cx + s * cy) / w],
            [2.0 * z * s / h,  2.0 * z * c / h, 0.0, 2.0 * z * (-s * cx - c * cy) / h],
            [0.0,              0.0,            -1.0, 0.0],
            [0.0,              0.0,             0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return m


def default_camera_matrix(viewport_width: int, viewport_height: int) -> np.ndarray:
    """Camera matrix used when no entity carries Camera2D — centered on the framebuffer
    so spawning entities in pixel coordinates "just works" (world (0,0) is bottom-left,
    (width/2, height/2) is screen center). Spawn an explicit Camera2D to override."""
    cam = Camera2D(x=float(viewport_width) * 0.5, y=float(viewport_height) * 0.5)
    return build_camera_matrix(cam, viewport_width, viewport_height)
