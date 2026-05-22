"""Tilemap — static, chunk-baked tile renderer sharing the sprite shader.

Tiles are a 2D int32 array. ID 0 is empty; IDs 1..N map to texture index
(N-1) in the bound TextureAtlas. The world is partitioned into 16x16-tile
chunks; each chunk pre-bakes one VAO + instance buffer at load time, so the
per-frame cost is only `len(chunks)` draw calls.
"""
from __future__ import annotations

from typing import Any

import moderngl
import numpy as np

from .batch2d import (
    FLOATS_PER_INSTANCE,
    INSTANCE_ATTRS,
    INSTANCE_FORMAT,
    QUAD_VERTS,
)
from .texture import TextureAtlas


CHUNK_SIZE: int = 16


class _TilemapChunk:
    __slots__ = ("vao", "instance_buf", "count")

    def __init__(
        self,
        vao: moderngl.VertexArray,
        instance_buf: moderngl.Buffer,
        count: int,
    ) -> None:
        self.vao = vao
        self.instance_buf = instance_buf
        self.count = count

    def release(self) -> None:
        for obj in (self.vao, self.instance_buf):
            try:
                obj.release()
            except Exception:
                pass


class Tilemap:
    """Static chunk-baked tilemap renderer."""

    def __init__(
        self,
        ctx: moderngl.Context,
        shader: moderngl.Program,
        atlas: TextureAtlas,
        tile_width: int = 32,
        tile_height: int = 32,
    ) -> None:
        self.ctx: moderngl.Context = ctx
        self.shader: moderngl.Program = shader
        self.atlas: TextureAtlas = atlas
        self.tile_width: int = int(tile_width)
        self.tile_height: int = int(tile_height)
        self.chunks: list[_TilemapChunk] = []
        self.build_count: int = 0

        self._quad_buf: moderngl.Buffer = ctx.buffer(QUAD_VERTS.tobytes())

        try:
            shader["u_textures"].value = list(range(16))
        except KeyError:
            for i in range(16):
                try:
                    shader[f"u_textures[{i}]"].value = i
                except KeyError:
                    pass

    def load(self, tile_data: np.ndarray) -> None:
        """Replace all chunks with VAOs baked from `tile_data` (shape (rows, cols), int32)."""
        if tile_data.ndim != 2:
            raise ValueError(f"tile_data must be 2-D, got shape {tile_data.shape}")
        td = np.ascontiguousarray(tile_data, dtype=np.int32)
        for chunk in self.chunks:
            chunk.release()
        self.chunks = []
        self._build_chunks(td)
        self.build_count += 1

    def _build_chunks(self, tile_data: np.ndarray) -> None:
        rows, cols = tile_data.shape
        tw = float(self.tile_width)
        th = float(self.tile_height)

        for cy in range(0, rows, CHUNK_SIZE):
            for cx in range(0, cols, CHUNK_SIZE):
                y_end = min(cy + CHUNK_SIZE, rows)
                x_end = min(cx + CHUNK_SIZE, cols)
                sub = tile_data[cy:y_end, cx:x_end]
                nonzero = np.argwhere(sub != 0)
                if nonzero.size == 0:
                    continue

                local_y = nonzero[:, 0]
                local_x = nonzero[:, 1]
                ids = sub[local_y, local_x]
                count = ids.shape[0]

                world_x = (cx + local_x).astype(np.float32) * tw
                world_y = (cy + local_y).astype(np.float32) * th
                tex_units = (ids - 1).astype(np.float32)

                inst = np.empty((count, FLOATS_PER_INSTANCE), dtype=np.float32)
                inst[:, 0] = world_x
                inst[:, 1] = world_y
                inst[:, 2] = 0.0
                inst[:, 3] = tw
                inst[:, 4] = th
                inst[:, 5] = 1.0
                inst[:, 6] = 1.0
                inst[:, 7] = 1.0
                inst[:, 8] = 1.0
                inst[:, 9] = 0.0
                inst[:, 10] = 0.0
                inst[:, 11] = 1.0
                inst[:, 12] = 1.0
                inst[:, 13] = tex_units

                inst_buf = self.ctx.buffer(inst.tobytes())
                vao = self.ctx.vertex_array(
                    self.shader,
                    [
                        (self._quad_buf, "2f", "in_position"),
                        (inst_buf, INSTANCE_FORMAT, *INSTANCE_ATTRS),
                    ],
                )
                self.chunks.append(_TilemapChunk(vao, inst_buf, count))

    def render(self, camera_matrix: np.ndarray) -> None:
        """Bind the atlas, set u_camera, and draw every baked chunk."""
        if not self.chunks:
            return
        cam = camera_matrix.astype(np.float32, copy=False)
        self.shader["u_camera"].write(np.ascontiguousarray(cam.T).tobytes())
        self.atlas.bind_all(self.ctx)
        for chunk in self.chunks:
            chunk.vao.render(mode=moderngl.TRIANGLE_STRIP, instances=chunk.count)

    def release(self) -> None:
        """Release every baked chunk and the shared quad buffer."""
        for chunk in self.chunks:
            chunk.release()
        self.chunks = []
        try:
            self._quad_buf.release()
        except Exception:
            pass
