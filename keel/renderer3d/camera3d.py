"""Camera3D component + perspective projection / Euler-angle view-matrix builders."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core import component


@component
class Camera3D:
    """Perspective camera. fov is in radians; pitch/yaw/roll apply in YXZ order."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    fov: float = 1.0472  # ~60 degrees
    near: float = 0.1
    far: float = 1000.0


def _read_camera(camera: Any) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Pull (x, y, z, pitch, yaw, roll, fov, near, far) from a Camera3D or structured record."""
    if hasattr(camera, "x") and not isinstance(camera, np.void):
        return (
            float(camera.x), float(camera.y), float(camera.z),
            float(camera.pitch), float(camera.yaw), float(camera.roll),
            float(camera.fov), float(camera.near), float(camera.far),
        )
    return (
        float(camera["x"]), float(camera["y"]), float(camera["z"]),
        float(camera["pitch"]), float(camera["yaw"]), float(camera["roll"]),
        float(camera["fov"]), float(camera["near"]), float(camera["far"]),
    )


def build_projection_matrix(camera: Any, width: int, height: int) -> np.ndarray:
    """Standard right-handed perspective projection. Returns (4, 4) float32."""
    if width <= 0 or height <= 0:
        raise ValueError(
            f"viewport dimensions must be positive, got ({width}, {height})"
        )
    _, _, _, _, _, _, fov, near, far = _read_camera(camera)
    if fov <= 0.0 or fov >= math.pi:
        raise ValueError(f"fov must be in (0, pi), got {fov}")
    if near <= 0.0 or far <= near:
        raise ValueError(f"near must be >0 and far must be > near, got near={near} far={far}")
    aspect = float(width) / float(height)
    f = 1.0 / math.tan(fov * 0.5)
    nf = near - far
    m = np.array(
        [
            [f / aspect, 0.0, 0.0,                       0.0],
            [0.0,        f,   0.0,                       0.0],
            [0.0,        0.0, (far + near) / nf,         (2.0 * far * near) / nf],
            [0.0,        0.0, -1.0,                      0.0],
        ],
        dtype=np.float32,
    )
    return m


def _euler_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Compose R = Ry(yaw) * Rx(pitch) * Rz(roll). All angles in radians."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)

    rx = np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, cp,  -sp, 0.0],
         [0.0, sp,  cp,  0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    ry = np.array(
        [[cy,  0.0, sy,  0.0],
         [0.0, 1.0, 0.0, 0.0],
         [-sy, 0.0, cy,  0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rz = np.array(
        [[cr,  -sr, 0.0, 0.0],
         [sr,  cr,  0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return ry @ rx @ rz


def build_view_matrix(camera: Any) -> np.ndarray:
    """View matrix = inverse(translate * rotate). Returns (4, 4) float32."""
    cx, cy, cz, pitch, yaw, roll, _, _, _ = _read_camera(camera)

    rotation = _euler_rotation_matrix(pitch, yaw, roll)
    rotation_inv = rotation.T  # rotation matrices are orthogonal

    translation_inv = np.array(
        [[1.0, 0.0, 0.0, -cx],
         [0.0, 1.0, 0.0, -cy],
         [0.0, 0.0, 1.0, -cz],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return (rotation_inv @ translation_inv).astype(np.float32)
