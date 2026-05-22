"""Mesh data, GPU-side MeshBuffer, OBJ loader, and primitive generators.

CPU mesh layout:  vertices is (N, 8) float32 — pos.xyz | normal.xyz | uv.xy
                  indices is (M,) uint32, triangle list

The OBJ loader supports `v` / `vn` / `vt` / `f` lines, triangulates n-gons
via fan, generates flat normals when none are provided, and ignores
material / group / smoothing / comment directives.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import moderngl
import numpy as np


VERTEX_FLOATS: int = 8
VERTEX_FORMAT: str = "3f 3f 2f"
VERTEX_ATTRS: tuple[str, ...] = ("in_position", "in_normal", "in_uv")


@dataclass
class Mesh:
    """CPU-side mesh: interleaved vertex array + index array."""
    vertices: np.ndarray
    indices: np.ndarray
    name: str = ""


class MeshBuffer:
    """GPU-side mesh: VBO + EBO + a per-shader VAO cache built on first render()."""

    __slots__ = ("ctx", "vbo", "ebo", "index_count", "_vao", "_shader")

    def __init__(
        self,
        ctx: moderngl.Context,
        vertices: np.ndarray,
        indices: np.ndarray,
        shader: Optional[moderngl.Program] = None,
    ) -> None:
        self.ctx: moderngl.Context = ctx
        verts = np.ascontiguousarray(vertices.astype(np.float32, copy=False))
        idx = np.ascontiguousarray(indices.astype(np.uint32, copy=False))
        self.vbo: moderngl.Buffer = ctx.buffer(verts.tobytes())
        self.ebo: moderngl.Buffer = ctx.buffer(idx.tobytes())
        self.index_count: int = int(idx.shape[0])
        self._vao: Optional[moderngl.VertexArray] = None
        self._shader: Optional[moderngl.Program] = None
        if shader is not None:
            self.bind(shader)

    def bind(self, shader: moderngl.Program) -> None:
        """(Re)build the VAO against `shader` if it changed since the last bind."""
        if self._vao is not None and self._shader is shader:
            return
        if self._vao is not None:
            try:
                self._vao.release()
            except Exception:
                pass
        self._vao = self.ctx.vertex_array(
            shader,
            [(self.vbo, VERTEX_FORMAT, *VERTEX_ATTRS)],
            self.ebo,
        )
        self._shader = shader

    def render(self, instances: int = 1) -> None:
        """Issue one indexed-triangles draw call. `bind(shader)` must have been called first."""
        if self._vao is None:
            raise RuntimeError("MeshBuffer.render called before bind(shader)")
        self._vao.render(mode=moderngl.TRIANGLES, instances=instances)

    def release(self) -> None:
        """Release every owned GL object."""
        for obj in (self._vao, self.vbo, self.ebo):
            if obj is None:
                continue
            try:
                obj.release()
            except Exception:
                pass
        self._vao = None


class MeshRegistry:
    """ID-keyed store of MeshBuffers, all bound to a single 3D shader."""

    __slots__ = ("ctx", "shader", "_buffers")

    def __init__(self, ctx: moderngl.Context, shader: moderngl.Program) -> None:
        self.ctx: moderngl.Context = ctx
        self.shader: moderngl.Program = shader
        self._buffers: list[MeshBuffer] = []

    def add(self, mesh: Mesh) -> int:
        """Upload `mesh` to the GPU and return its mesh_id."""
        buf = MeshBuffer(self.ctx, mesh.vertices, mesh.indices, shader=self.shader)
        self._buffers.append(buf)
        return len(self._buffers) - 1

    def get_buffer(self, mesh_id: int) -> MeshBuffer:
        """Return the MeshBuffer for `mesh_id`. Raises IndexError on bad IDs."""
        return self._buffers[mesh_id]

    def count(self) -> int:
        """Number of meshes registered."""
        return len(self._buffers)

    def release(self) -> None:
        """Release every uploaded mesh."""
        for buf in self._buffers:
            buf.release()
        self._buffers.clear()


# --- OBJ loader -----------------------------------------------------------

class OBJLoader:
    """Minimal OBJ parser. Triangulates fans, generates flat normals when missing."""

    @staticmethod
    def load(path: str) -> Mesh:
        """Read `path` from disk and return a Mesh."""
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        mesh = OBJLoader.load_from_string(source)
        mesh.name = os.path.basename(path)
        return mesh

    @staticmethod
    def load_from_string(source: str) -> Mesh:
        """Parse OBJ text and return a Mesh."""
        positions: list[list[float]] = []
        normals: list[list[float]] = []
        uvs: list[list[float]] = []
        face_lines: list[list[str]] = []

        for raw in source.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cmd = parts[0]
            if cmd == "v" and len(parts) >= 4:
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif cmd == "vn" and len(parts) >= 4:
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif cmd == "vt" and len(parts) >= 2:
                u = float(parts[1])
                v = float(parts[2]) if len(parts) > 2 else 0.0
                uvs.append([u, v])
            elif cmd == "f":
                face_lines.append(parts[1:])
            # mtllib, usemtl, g, o, s, etc. are intentionally ignored.

        triangles: list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = []
        for face in face_lines:
            verts: list[tuple[int, int, int]] = []
            for token in face:
                idx = token.split("/")
                p = int(idx[0]) - 1 if idx[0] else -1
                t = int(idx[1]) - 1 if len(idx) > 1 and idx[1] else -1
                n = int(idx[2]) - 1 if len(idx) > 2 and idx[2] else -1
                verts.append((p, t, n))
            if len(verts) < 3:
                continue
            for i in range(1, len(verts) - 1):
                triangles.append((verts[0], verts[i], verts[i + 1]))

        n_vertices = len(triangles) * 3
        out = np.zeros((max(n_vertices, 0), VERTEX_FLOATS), dtype=np.float32)
        indices = np.arange(max(n_vertices, 0), dtype=np.uint32)

        for tri_i, tri in enumerate(triangles):
            need_flat = any(n_idx < 0 or n_idx >= len(normals) for _, _, n_idx in tri)
            flat_normal: Optional[np.ndarray] = None
            if need_flat:
                p0 = np.asarray(positions[tri[0][0]], dtype=np.float32)
                p1 = np.asarray(positions[tri[1][0]], dtype=np.float32)
                p2 = np.asarray(positions[tri[2][0]], dtype=np.float32)
                normal = np.cross(p1 - p0, p2 - p0)
                length = float(np.linalg.norm(normal))
                if length > 0.0:
                    normal = normal / length
                else:
                    normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                flat_normal = normal.astype(np.float32)

            for v_i, (p_idx, t_idx, n_idx) in enumerate(tri):
                row = tri_i * 3 + v_i
                if 0 <= p_idx < len(positions):
                    out[row, 0:3] = positions[p_idx]
                if 0 <= n_idx < len(normals):
                    out[row, 3:6] = normals[n_idx]
                elif flat_normal is not None:
                    out[row, 3:6] = flat_normal
                if 0 <= t_idx < len(uvs):
                    out[row, 6:8] = uvs[t_idx]

        return Mesh(vertices=out, indices=indices)


# --- Primitive generators -------------------------------------------------

def make_cube() -> Mesh:
    """Unit cube centered at the origin: 24 verts, 36 indices, per-face flat normals."""
    faces = [
        # +X
        ([( 0.5, -0.5, -0.5), ( 0.5, -0.5,  0.5), ( 0.5,  0.5,  0.5), ( 0.5,  0.5, -0.5)], ( 1.0,  0.0,  0.0)),
        # -X
        ([(-0.5, -0.5,  0.5), (-0.5, -0.5, -0.5), (-0.5,  0.5, -0.5), (-0.5,  0.5,  0.5)], (-1.0,  0.0,  0.0)),
        # +Y
        ([(-0.5,  0.5,  0.5), ( 0.5,  0.5,  0.5), ( 0.5,  0.5, -0.5), (-0.5,  0.5, -0.5)], ( 0.0,  1.0,  0.0)),
        # -Y
        ([(-0.5, -0.5, -0.5), ( 0.5, -0.5, -0.5), ( 0.5, -0.5,  0.5), (-0.5, -0.5,  0.5)], ( 0.0, -1.0,  0.0)),
        # +Z
        ([( 0.5, -0.5,  0.5), (-0.5, -0.5,  0.5), (-0.5,  0.5,  0.5), ( 0.5,  0.5,  0.5)], ( 0.0,  0.0,  1.0)),
        # -Z
        ([(-0.5, -0.5, -0.5), ( 0.5, -0.5, -0.5), ( 0.5,  0.5, -0.5), (-0.5,  0.5, -0.5)], ( 0.0,  0.0, -1.0)),
    ]
    quad_uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    verts: list[list[float]] = []
    indices: list[int] = []
    for verts_xyz, normal in faces:
        base = len(verts)
        for v, uv in zip(verts_xyz, quad_uvs):
            verts.append([v[0], v[1], v[2], normal[0], normal[1], normal[2], uv[0], uv[1]])
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    return Mesh(
        vertices=np.asarray(verts, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
        name="cube",
    )


def make_plane(width: float = 1.0, depth: float = 1.0) -> Mesh:
    """Y-up plane on the XZ axes, normal = +Y."""
    hw = float(width) * 0.5
    hd = float(depth) * 0.5
    vertices = np.array(
        [
            [-hw, 0.0, -hd, 0.0, 1.0, 0.0, 0.0, 0.0],
            [ hw, 0.0, -hd, 0.0, 1.0, 0.0, 1.0, 0.0],
            [ hw, 0.0,  hd, 0.0, 1.0, 0.0, 1.0, 1.0],
            [-hw, 0.0,  hd, 0.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return Mesh(vertices=vertices, indices=indices, name="plane")


def make_sphere(subdivisions: int = 2) -> Mesh:
    """UV sphere of radius 0.5. `subdivisions` scales segment counts: 1 → coarse, 4 → smoother."""
    sub = max(1, int(subdivisions))
    h_segments = 4 * sub + 4
    v_segments = 2 * sub + 2

    verts: list[list[float]] = []
    indices: list[int] = []

    for v in range(v_segments + 1):
        phi = math.pi * v / v_segments
        for h in range(h_segments + 1):
            theta = 2.0 * math.pi * h / h_segments
            sx = math.sin(phi) * math.cos(theta)
            sy = math.cos(phi)
            sz = math.sin(phi) * math.sin(theta)
            u = h / h_segments
            tv = v / v_segments
            verts.append([sx * 0.5, sy * 0.5, sz * 0.5, sx, sy, sz, u, tv])

    row_stride = h_segments + 1
    for v in range(v_segments):
        for h in range(h_segments):
            i0 = v * row_stride + h
            i1 = i0 + 1
            i2 = i0 + row_stride
            i3 = i2 + 1
            indices.extend([i0, i2, i1, i1, i2, i3])

    return Mesh(
        vertices=np.asarray(verts, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
        name="sphere",
    )
