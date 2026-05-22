"""Frustum culling via Gribb/Hartmann plane extraction.

Six planes (left, right, bottom, top, near, far) are produced from the
view-projection matrix as row-vector combinations: row3 ± row{0,1,2}. Each
plane is normalized so the homogeneous distance term `d` is in world units,
which lets us do a cheap signed-distance test against a BoundingSphere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoundingSphere:
    """World-space bounding sphere — center plus radius."""
    center_x: float
    center_y: float
    center_z: float
    radius: float


PLANE_LEFT: int = 0
PLANE_RIGHT: int = 1
PLANE_BOTTOM: int = 2
PLANE_TOP: int = 3
PLANE_NEAR: int = 4
PLANE_FAR: int = 5


class FrustumCuller:
    """Pre-allocated 6×4 plane table, refilled once per frame from the VP matrix."""

    __slots__ = ("_planes",)

    def __init__(self) -> None:
        self._planes: np.ndarray = np.zeros((6, 4), dtype=np.float32)

    def update(self, view_proj: np.ndarray) -> None:
        """Extract 6 normalized frustum planes from VP. Call once per frame."""
        m = np.asarray(view_proj, dtype=np.float32)
        if m.shape != (4, 4):
            raise ValueError(f"view_proj must be (4, 4), got {m.shape}")

        # Gribb/Hartmann (row-vector form): row3 + row_i is "inside" half-space
        # for the negative side, row3 - row_i for the positive side.
        self._planes[PLANE_LEFT]   = m[3] + m[0]
        self._planes[PLANE_RIGHT]  = m[3] - m[0]
        self._planes[PLANE_BOTTOM] = m[3] + m[1]
        self._planes[PLANE_TOP]    = m[3] - m[1]
        self._planes[PLANE_NEAR]   = m[3] + m[2]
        self._planes[PLANE_FAR]    = m[3] - m[2]

        # Normalize so plane.xyz is a unit normal — distance test then yields world units.
        for i in range(6):
            n = float(np.linalg.norm(self._planes[i, :3]))
            if n > 0.0:
                self._planes[i] /= n

    def is_visible(self, sphere: BoundingSphere) -> bool:
        """Return False if `sphere` is entirely outside any one frustum plane."""
        cx = float(sphere.center_x)
        cy = float(sphere.center_y)
        cz = float(sphere.center_z)
        r = float(sphere.radius)
        planes = self._planes
        for i in range(6):
            dist = (
                planes[i, 0] * cx
                + planes[i, 1] * cy
                + planes[i, 2] * cz
                + planes[i, 3]
            )
            if dist < -r:
                return False
        return True

    @property
    def planes(self) -> np.ndarray:
        """The 6×4 plane table — one row per plane in (a, b, c, d) form."""
        return self._planes
