"""Model-matrix builder + parent-chain world-matrix resolver for Transform3D.

resolve_world_matrix walks the parent chain via Transform3D.parent (entity ID).
Multiplies bottom-up so a child sits in its parent's frame. Cycles and
chains deeper than _MAX_DEPTH log a RuntimeWarning and stop instead of
infinite-looping.
"""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from ..components.transform3d import Transform3D


_MAX_DEPTH: int = 32


def _read_transform(t: Any) -> tuple[float, ...]:
    """Pull all 10 fields from a Transform3D instance or numpy structured record."""
    if hasattr(t, "x") and not isinstance(t, np.void):
        return (
            float(t.x), float(t.y), float(t.z),
            float(t.rot_x), float(t.rot_y), float(t.rot_z),
            float(t.scale_x), float(t.scale_y), float(t.scale_z),
            int(t.parent),
        )
    return (
        float(t["x"]), float(t["y"]), float(t["z"]),
        float(t["rot_x"]), float(t["rot_y"]), float(t["rot_z"]),
        float(t["scale_x"]), float(t["scale_y"]), float(t["scale_z"]),
        int(t["parent"]),
    )


def build_model_matrix(transform: Any) -> np.ndarray:
    """Compose T(x,y,z) * Ry(rot_y) * Rx(rot_x) * Rz(rot_z) * S(sx,sy,sz)."""
    x, y, z, rx, ry, rz, sx, sy, sz, _ = _read_transform(transform)

    cx, sxn = math.cos(rx), math.sin(rx)
    cy, syn = math.cos(ry), math.sin(ry)
    cz, szn = math.cos(rz), math.sin(rz)

    rot_x = np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, cx,  -sxn, 0.0],
         [0.0, sxn, cx,   0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rot_y = np.array(
        [[cy,  0.0, syn, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [-syn, 0.0, cy, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rot_z = np.array(
        [[cz,  -szn, 0.0, 0.0],
         [szn, cz,   0.0, 0.0],
         [0.0, 0.0,  1.0, 0.0],
         [0.0, 0.0,  0.0, 1.0]],
        dtype=np.float32,
    )
    scale = np.array(
        [[sx,  0.0, 0.0, 0.0],
         [0.0, sy,  0.0, 0.0],
         [0.0, 0.0, sz,  0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    translation = np.array(
        [[1.0, 0.0, 0.0, x],
         [0.0, 1.0, 0.0, y],
         [0.0, 0.0, 1.0, z],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return (translation @ rot_y @ rot_x @ rot_z @ scale).astype(np.float32)


# Reusable per-call stack of (transform, entity_id) tuples — sized to MAX_DEPTH so
# the walker doesn't reallocate when the chain stays shallow.
_chain_stack: list[Any] = [None] * _MAX_DEPTH


def resolve_world_matrix(entity: int, world: Any) -> np.ndarray:
    """Resolve the world-space matrix for `entity` by walking its Transform3D.parent chain."""
    chain_len = 0
    visited: set[int] = set()
    current = int(entity)

    while True:
        if current == 0:
            break
        if current in visited:
            warnings.warn(
                f"resolve_world_matrix: cycle detected at entity {current} (chain start {entity})",
                RuntimeWarning,
                stacklevel=2,
            )
            break
        if chain_len >= _MAX_DEPTH:
            warnings.warn(
                f"resolve_world_matrix: chain deeper than {_MAX_DEPTH} starting at {entity} — stopping",
                RuntimeWarning,
                stacklevel=2,
            )
            break
        visited.add(current)
        t = world.get_component(current, Transform3D)
        if t is None:
            break
        _chain_stack[chain_len] = t
        chain_len += 1
        parent = int(t.parent)
        if parent == 0:
            break
        current = parent

    if chain_len == 0:
        return np.eye(4, dtype=np.float32)

    # Multiply top-down: world = root.local @ ... @ child.local
    result = build_model_matrix(_chain_stack[chain_len - 1])
    for i in range(chain_len - 2, -1, -1):
        result = result @ build_model_matrix(_chain_stack[i])
    # Clear stack slots so we don't hold transforms across frames.
    for i in range(chain_len):
        _chain_stack[i] = None
    return result.astype(np.float32)
