"""TextBatch — one draw call per (font, color) pair, screen-space coords.

Transform2D.x/y are interpreted as pixels with the origin at the top-left
and y growing downward; text is UI-space and does not move with the camera.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import moderngl
import numpy as np

from ..components import Transform2D
from ..components.text_label import TextLabel
from ..core import World
from .font import Font, FontRegistry


_log = logging.getLogger(__name__)


# numpy structured arrays can't hold variable-length strings, so label text
# lives in a side dict keyed by entity id (v0.1 limitation).
_LABEL_TEXT: dict[int, str] = {}


def set_text(entity_id: int, text: str) -> None:
    """Set the text content for a TextLabel entity."""
    _LABEL_TEXT[int(entity_id)] = str(text)


def get_text(entity_id: int) -> str:
    """Return the text content for `entity_id`, or "" if unset."""
    return _LABEL_TEXT.get(int(entity_id), "")


def clear_text(entity_id: int) -> None:
    """Forget any text content for `entity_id`."""
    _LABEL_TEXT.pop(int(entity_id), None)


def _reset_label_text() -> None:
    """Test helper — wipe the entire side dict."""
    _LABEL_TEXT.clear()


# ---------------------------------------------------------------------------
# Per-glyph vertex layout
# ---------------------------------------------------------------------------
#
# Each glyph is a quad split into two triangles. We emit it as 6 plain
# triangle vertices (no element-array indices), so the shader sees:
#
#     v0 ─── v1          v0 = top-left
#      │  ╲   │           v1 = top-right
#      │   ╲  │           v2 = bottom-right
#      │    ╲ │           v3 = bottom-left
#     v3 ─── v2          triangle 1 = (v0, v1, v2)
#                        triangle 2 = (v0, v2, v3)
#
# Per vertex we ship 4 floats: screen_x, screen_y, atlas_u, atlas_v.
# Why 6 vertices and not 4 + an index buffer? Glyphs are independent quads,
# so a single draw call across N glyphs needs ZERO shared vertices — using
# an index buffer would save nothing and add upload + bind overhead.
_FLOATS_PER_VERTEX = 4
_VERTICES_PER_GLYPH = 6
_FLOATS_PER_GLYPH = _FLOATS_PER_VERTEX * _VERTICES_PER_GLYPH
_INITIAL_CAPACITY_GLYPHS = 4096
_TAB_WIDTH_IN_SPACES = 4


def _orthographic_screen_projection(width: int, height: int) -> np.ndarray:
    """Ortho projection mapping [0, w] x [0, h] (y-down) to NDC."""
    w = max(1, int(width))
    h = max(1, int(height))
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / w
    m[1, 1] = -2.0 / h
    m[2, 2] = -1.0
    m[0, 3] = -1.0
    m[1, 3] = 1.0
    m[3, 3] = 1.0
    return m


class TextBatch:
    """Renders every TextLabel entity each frame into a dynamic vertex buffer.

    One draw call per (font, color) pair. Vertices live in a pre-allocated
    numpy array large enough for 4096 glyphs by default; the buffer doubles
    on overflow rather than reallocating each frame.
    """

    def __init__(self, ctx: moderngl.Context, shader: moderngl.Program) -> None:
        self._ctx = ctx
        self._shader = shader
        self._capacity_glyphs = _INITIAL_CAPACITY_GLYPHS
        self._cpu_buffer = np.zeros(
            self._capacity_glyphs * _FLOATS_PER_GLYPH, dtype=np.float32
        )
        self._gpu_buffer = ctx.buffer(
            reserve=self._capacity_glyphs * _FLOATS_PER_GLYPH * 4, dynamic=True
        )
        self._vao = ctx.vertex_array(
            self._shader,
            [(self._gpu_buffer, "2f 2f", "in_position", "in_uv")],
        )
        self._missing_transform_warned = False
        self.last_glyph_count = 0
        self.last_draw_calls = 0

    @property
    def capacity_glyphs(self) -> int:
        """Current vertex-buffer capacity in glyph quads."""
        return self._capacity_glyphs

    def _ensure_capacity(self, glyph_count: int) -> None:
        """Grow the CPU and GPU buffers (doubling) to fit `glyph_count`."""
        if glyph_count <= self._capacity_glyphs:
            return
        new_cap = self._capacity_glyphs
        while new_cap < glyph_count:
            new_cap *= 2
        self._cpu_buffer = np.zeros(new_cap * _FLOATS_PER_GLYPH, dtype=np.float32)
        for obj in (self._vao, self._gpu_buffer):
            try:
                obj.release()
            except Exception:
                pass
        self._gpu_buffer = self._ctx.buffer(
            reserve=new_cap * _FLOATS_PER_GLYPH * 4, dynamic=True
        )
        self._vao = self._ctx.vertex_array(
            self._shader,
            [(self._gpu_buffer, "2f 2f", "in_position", "in_uv")],
        )
        self._capacity_glyphs = new_cap

    def render(
        self,
        world: World,
        font_registry: FontRegistry,
        viewport_width: int,
        viewport_height: int,
    ) -> None:
        """Draw every visible TextLabel entity in screen space."""
        # One vertex-float list per (font_id, rgba) tuple → one draw call each.
        groups: dict[tuple[int, float, float, float, float], list[np.ndarray]] = {}
        total_glyphs = 0

        for arch in world.query(TextLabel).archetypes():
            n = arch.length
            labels = arch.columns[TextLabel][:n]
            transforms_col = arch.columns.get(Transform2D)
            entity_ids = arch.entities[:n]

            if transforms_col is None:
                if not self._missing_transform_warned:
                    _log.warning("TextLabel without Transform2D; skipping render.")
                    self._missing_transform_warned = True
                continue

            transforms = transforms_col[:n]

            for i in range(n):
                if not bool(labels["visible"][i]):
                    continue

                eid = int(entity_ids[i])
                text = _LABEL_TEXT.get(eid, "")
                if not text:
                    continue

                font_id = int(labels["font_id"][i])
                font = font_registry.get_by_id(font_id)
                if font is None:
                    continue

                scale = float(labels["scale"][i])
                r = float(labels["r"][i])
                g = float(labels["g"][i])
                b = float(labels["b"][i])
                a = float(labels["a"][i])

                origin_x = float(transforms["x"][i])
                origin_y = float(transforms["y"][i])

                verts, glyph_count = _layout_text_quads(
                    text, font, origin_x, origin_y, scale
                )
                if glyph_count == 0:
                    continue

                key = (font_id, r, g, b, a)
                groups.setdefault(key, []).append(verts)
                total_glyphs += glyph_count

        self.last_glyph_count = total_glyphs
        self.last_draw_calls = 0

        if total_glyphs == 0:
            return

        self._ensure_capacity(total_glyphs)

        proj = _orthographic_screen_projection(viewport_width, viewport_height)
        proj_bytes = np.ascontiguousarray(proj.T).tobytes()
        try:
            self._shader["u_projection"].write(proj_bytes)
        except KeyError:
            pass

        # Alpha blend on for the draw, off on exit (no state leak into the
        # next frame's PRE_UPDATE).
        ctx = self._ctx
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        try:
            ctx.blend_equation = moderngl.FUNC_ADD
        except Exception:
            pass

        try:
            for (font_id, r, g, b, a), verts_list in groups.items():
                font = font_registry.get_by_id(font_id)
                if font is None:
                    continue

                # Concatenate the group's vertex floats into the pre-allocated
                # CPU buffer in place — no per-frame allocation.
                offset = 0
                for verts in verts_list:
                    end = offset + verts.size
                    self._cpu_buffer[offset:end] = verts
                    offset = end

                self._gpu_buffer.write(self._cpu_buffer[:offset].tobytes())
                try:
                    self._shader["u_color"].value = (r, g, b, a)
                except KeyError:
                    pass

                font.atlas.texture.use(location=0)
                try:
                    self._shader["u_texture"].value = 0
                except KeyError:
                    pass

                vertex_count = offset // _FLOATS_PER_VERTEX
                self._vao.render(mode=moderngl.TRIANGLES, vertices=vertex_count)
                self.last_draw_calls += 1
        finally:
            ctx.disable(moderngl.BLEND)

    def release(self) -> None:
        """Release every owned GL object."""
        for obj in (self._vao, self._gpu_buffer):
            try:
                obj.release()
            except Exception:
                pass


def _layout_text_quads(
    text: str,
    font: Font,
    origin_x: float,
    origin_y: float,
    scale: float,
) -> tuple[np.ndarray, int]:
    """Lay out `text` into a packed vertex array. origin_y is the baseline."""
    if not text:
        return (np.zeros(0, dtype=np.float32), 0)

    line_height = font.line_height
    space_advance = font.space_advance

    # First pass: count glyphs so we can allocate once.
    glyph_count = 0
    for ch in text:
        if ch in ("\n", "\t", " "):
            continue
        info = font.get_glyph(ch)
        if info is None or info.width == 0 or info.height == 0:
            continue
        glyph_count += 1

    if glyph_count == 0:
        return (np.zeros(0, dtype=np.float32), 0)

    out = np.empty(glyph_count * _FLOATS_PER_GLYPH, dtype=np.float32)
    pen_x = origin_x
    pen_y = origin_y
    idx = 0

    for ch in text:
        if ch == "\n":
            pen_x = origin_x
            pen_y += line_height * scale
            continue
        if ch == "\t":
            pen_x += space_advance * _TAB_WIDTH_IN_SPACES * scale
            continue
        if ch == " ":
            pen_x += space_advance * scale
            continue

        info = font.get_glyph(ch)
        if info is None:
            # Missing glyph: advance by space width per the engine contract.
            pen_x += space_advance * scale
            continue
        if info.width == 0 or info.height == 0:
            pen_x += info.advance * scale
            continue

        x0 = pen_x + info.bearing_x * scale
        y0 = pen_y - info.bearing_y * scale
        x1 = x0 + info.width * scale
        y1 = y0 + info.height * scale

        u0 = info.uv_x
        v0 = info.uv_y
        u1 = u0 + info.uv_w
        v1 = v0 + info.uv_h

        base = idx * _FLOATS_PER_GLYPH
        # Tri 1: (x0,y0), (x1,y0), (x1,y1)
        out[base + 0] = x0; out[base + 1] = y0; out[base + 2] = u0; out[base + 3] = v0
        out[base + 4] = x1; out[base + 5] = y0; out[base + 6] = u1; out[base + 7] = v0
        out[base + 8] = x1; out[base + 9] = y1; out[base + 10] = u1; out[base + 11] = v1
        # Tri 2: (x0,y0), (x1,y1), (x0,y1)
        out[base + 12] = x0; out[base + 13] = y0; out[base + 14] = u0; out[base + 15] = v0
        out[base + 16] = x1; out[base + 17] = y1; out[base + 18] = u1; out[base + 19] = v1
        out[base + 20] = x0; out[base + 21] = y1; out[base + 22] = u0; out[base + 23] = v1

        idx += 1
        pen_x += info.advance * scale

    return (out, glyph_count)
