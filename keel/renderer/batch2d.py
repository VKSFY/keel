"""SpriteBatch2D — instanced 2D sprite renderer.

Owns one static unit-quad VBO and one dynamic instance VBO. Each frame the
batcher concatenates the (Transform2D, Sprite) views from every matching
archetype, groups them by texture_id, fills a numpy float32 array of
per-instance data, uploads it via buffer.write(), and issues one
glDrawArraysInstanced call per texture group.
"""
from __future__ import annotations

from typing import Any, Iterable

import moderngl
import numpy as np


# ---------------------------------------------------------------------------
# Per-instance vertex buffer layout
# ---------------------------------------------------------------------------
#
# Each sprite contributes ONE row of 14 float32 values to the dynamic
# instance VBO. The GPU reads it once per instance (not per vertex) and the
# vertex shader stretches the unit quad accordingly. Field order matters —
# the shader pulls in_offset / in_rotation / etc. by attribute *order*, not
# by name.
#
#     index   field        contents
#     ------  -----------  ------------------------------------------------
#       0,1   in_offset    world position of the sprite center (x, y)
#       2     in_rotation  rotation in radians (CCW)
#       3,4   in_scale     full width, height of the sprite in world units
#       5..8  in_tint      RGBA tint multiplied into the sampled texel
#       9..12 in_uv_rect   sub-rect inside the atlas: (u0, v0, u1, v1)
#       13    in_tex_unit  which atlas slot to sample (int, packed as float)
#
# INSTANCE_FORMAT spells this out for moderngl. The final "1f/i" makes
# in_tex_unit an integer-typed attribute on the GPU (declared as
# `in int in_tex_unit;` in GLSL).
FLOATS_PER_INSTANCE: int = 14
INSTANCE_FORMAT: str = "2f 1f 2f 4f 4f 1f/i"
INSTANCE_ATTRS: tuple[str, ...] = (
    "in_offset",
    "in_rotation",
    "in_scale",
    "in_tint",
    "in_uv_rect",
    "in_tex_unit",
)
QUAD_VERTS: np.ndarray = np.array(
    [
        -0.5, -0.5,
         0.5, -0.5,
        -0.5,  0.5,
         0.5,  0.5,
    ],
    dtype="f4",
)
INITIAL_CAPACITY: int = 4096


class SpriteBatch2D:
    """Per-frame instanced sprite renderer driven by world.query(Transform2D, Sprite)."""

    def __init__(self, ctx: moderngl.Context, shader: moderngl.Program) -> None:
        self.ctx: moderngl.Context = ctx
        self.shader: moderngl.Program = shader
        self.capacity: int = INITIAL_CAPACITY

        self.quad_buf: moderngl.Buffer = ctx.buffer(QUAD_VERTS.tobytes())
        self.instance_buf: moderngl.Buffer = ctx.buffer(
            reserve=self.capacity * FLOATS_PER_INSTANCE * 4, dynamic=True
        )
        self.vao: moderngl.VertexArray = self._build_vao()

        # Tell the shader's sampler array to read from texture units 0..15.
        try:
            shader["u_textures"].value = list(range(16))
        except KeyError:
            for i in range(16):
                try:
                    shader[f"u_textures[{i}]"].value = i
                except KeyError:
                    pass

    def _build_vao(self) -> moderngl.VertexArray:
        return self.ctx.vertex_array(
            self.shader,
            [
                (self.quad_buf, "2f", "in_position"),
                (self.instance_buf, INSTANCE_FORMAT, *INSTANCE_ATTRS),
            ],
        )

    def _ensure_capacity(self, count: int) -> None:
        if count <= self.capacity:
            return
        new_cap = self.capacity
        while new_cap < count:
            new_cap *= 2
        # Replace the dynamic instance buffer and re-bind the VAO.
        self.vao.release()
        self.instance_buf.release()
        self.instance_buf = self.ctx.buffer(
            reserve=new_cap * FLOATS_PER_INSTANCE * 4, dynamic=True
        )
        self.capacity = new_cap
        self.vao = self._build_vao()

    @staticmethod
    def _pack_group(
        transforms: np.ndarray,
        sprites: np.ndarray,
        texture_id: int,
        out: np.ndarray,
    ) -> None:
        """Vectorized fill of `out` (shape (N, 14), float32) from one texture group."""
        out[:, 0] = transforms["x"]
        out[:, 1] = transforms["y"]
        out[:, 2] = transforms["rotation"]

        scale_x = transforms["scale_x"] * sprites["width"]
        scale_y = transforms["scale_y"] * sprites["height"]
        scale_x = np.where(sprites["flip_x"], -scale_x, scale_x)
        scale_y = np.where(sprites["flip_y"], -scale_y, scale_y)
        out[:, 3] = scale_x
        out[:, 4] = scale_y

        out[:, 5] = sprites["r"]
        out[:, 6] = sprites["g"]
        out[:, 7] = sprites["b"]
        out[:, 8] = sprites["a"]

        out[:, 9] = 0.0
        out[:, 10] = 0.0
        out[:, 11] = 1.0
        out[:, 12] = 1.0

        out[:, 13] = float(texture_id)

    def render(self, query_results: Iterable[tuple], camera_matrix: np.ndarray) -> None:
        """Draw every (Transform2D, Sprite) entity in `query_results` once, grouped by texture_id."""
        transforms_list: list[np.ndarray] = []
        sprites_list: list[np.ndarray] = []
        for transforms, sprites in query_results:
            if len(transforms) == 0:
                continue
            transforms_list.append(transforms)
            sprites_list.append(sprites)
        if not transforms_list:
            return

        all_transforms = np.concatenate(transforms_list)
        all_sprites = np.concatenate(sprites_list)

        cam = camera_matrix.astype(np.float32, copy=False)
        # u_camera is column-major in GL; numpy is row-major. Tobytes of the row-major
        # matrix matches GLSL's mat4 reading column-major IF we transpose first.
        self.shader["u_camera"].write(np.ascontiguousarray(cam.T).tobytes())

        tex_ids = all_sprites["texture_id"]
        unique_ids = np.unique(tex_ids)

        for tid in unique_ids:
            mask = tex_ids == tid
            group_t = all_transforms[mask]
            group_s = all_sprites[mask]
            count = int(group_t.shape[0])
            if count == 0:
                continue

            self._ensure_capacity(count)
            inst = np.empty((count, FLOATS_PER_INSTANCE), dtype=np.float32)
            self._pack_group(group_t, group_s, int(tid), inst)
            self.instance_buf.write(inst.tobytes())
            self.vao.render(mode=moderngl.TRIANGLE_STRIP, instances=count)

    def release(self) -> None:
        """Release all owned GL resources."""
        for obj in (self.vao, self.instance_buf, self.quad_buf):
            try:
                obj.release()
            except Exception:
                pass
