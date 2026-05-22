"""DebugDraw2D — pymunk shape outline overlay drawn as GL line segments.

Per-frame work:
  1. Walk every Transform2D + Collider2D + RigidBody2D entity.
  2. Tessellate each shape into 2D line segments (32-segment circles, 4
     segments for a box).
  3. Group segments by color (one color per body-type / sensor flag).
  4. For each group: write the float32 buffer once and draw a single
     glLines call (per quality rule 4).
"""
from __future__ import annotations

import math
from typing import Any

import moderngl
import numpy as np

from ..components.transform2d import Transform2D
from ..core import Phase
from ..physics.components2d import (
    BODY_TYPE_DYNAMIC,
    BODY_TYPE_KINEMATIC,
    BODY_TYPE_STATIC,
    SHAPE_TYPE_BOX,
    SHAPE_TYPE_CIRCLE,
    SHAPE_TYPE_SEGMENT,
    Collider2D,
    RigidBody2D,
)


CIRCLE_SEGMENTS: int = 32
INITIAL_VERTEX_CAPACITY: int = 1024 * 64


COLOR_DYNAMIC: tuple[float, float, float] = (0.20, 0.85, 0.30)
COLOR_STATIC: tuple[float, float, float] = (0.55, 0.55, 0.55)
COLOR_KINEMATIC: tuple[float, float, float] = (0.30, 0.55, 1.00)
COLOR_SENSOR: tuple[float, float, float] = (1.00, 0.95, 0.20)


_LINE_VERT_SRC: str = """\
#version 330 core
in vec2 in_position;
uniform mat4 u_camera;
void main() {
    gl_Position = u_camera * vec4(in_position, 0.0, 1.0);
}
"""

_LINE_FRAG_SRC: str = """\
#version 330 core
uniform vec3 u_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(u_color, 1.0);
}
"""


def _color_for(body_type: int, sensor: bool) -> tuple[float, float, float]:
    if sensor:
        return COLOR_SENSOR
    if body_type == BODY_TYPE_STATIC:
        return COLOR_STATIC
    if body_type == BODY_TYPE_KINEMATIC:
        return COLOR_KINEMATIC
    return COLOR_DYNAMIC


def _circle_lines(cx: float, cy: float, r: float, segments: int = CIRCLE_SEGMENTS) -> list[float]:
    """Return interleaved (x, y) pairs for a closed circle outline as line-list segments."""
    pts: list[tuple[float, float]] = []
    for i in range(segments):
        theta = (2.0 * math.pi) * (i / segments)
        pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    out: list[float] = []
    for i in range(segments):
        a = pts[i]
        b = pts[(i + 1) % segments]
        out.extend([a[0], a[1], b[0], b[1]])
    return out


def _box_lines(
    cx: float,
    cy: float,
    half_w: float,
    half_h: float,
    rotation: float,
) -> list[float]:
    """Return interleaved (x, y) pairs for a (possibly rotated) rectangle outline."""
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    locals_ = (
        (-half_w, -half_h),
        ( half_w, -half_h),
        ( half_w,  half_h),
        (-half_w,  half_h),
    )
    pts = []
    for lx, ly in locals_:
        pts.append((cx + cos_r * lx - sin_r * ly, cy + sin_r * lx + cos_r * ly))
    out: list[float] = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        out.extend([a[0], a[1], b[0], b[1]])
    return out


def _segment_lines(cx: float, cy: float, length: float, rotation: float) -> list[float]:
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    half = length * 0.5
    a = (cx - cos_r * half, cy - sin_r * half)
    b = (cx + cos_r * half, cy + sin_r * half)
    return [a[0], a[1], b[0], b[1]]


class DebugDraw2D:
    """GL-line collision-shape overlay. Toggle with `visible`; render at POST_RENDER."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx: moderngl.Context = ctx
        self._visible: bool = False
        self._capacity: int = INITIAL_VERTEX_CAPACITY
        self.shader: moderngl.Program = ctx.program(
            vertex_shader=_LINE_VERT_SRC,
            fragment_shader=_LINE_FRAG_SRC,
        )
        self._buffer: moderngl.Buffer = ctx.buffer(
            reserve=self._capacity * 2 * 4, dynamic=True
        )
        self._vao: moderngl.VertexArray = ctx.vertex_array(
            self.shader,
            [(self._buffer, "2f", "in_position")],
        )
        self.last_line_count: int = 0

    @property
    def visible(self) -> bool:
        """Whether the overlay is currently drawn each frame."""
        return self._visible

    def toggle(self) -> None:
        """Flip the visibility flag."""
        self._visible = not self._visible

    def set_visible(self, visible: bool) -> None:
        """Set the visibility flag explicitly."""
        self._visible = bool(visible)

    def _ensure_capacity(self, vertex_count: int) -> None:
        if vertex_count <= self._capacity:
            return
        new_cap = self._capacity
        while new_cap < vertex_count:
            new_cap *= 2
        self._vao.release()
        self._buffer.release()
        self._buffer = self.ctx.buffer(reserve=new_cap * 2 * 4, dynamic=True)
        self._vao = self.ctx.vertex_array(
            self.shader,
            [(self._buffer, "2f", "in_position")],
        )
        self._capacity = new_cap

    def render(self, world: Any, camera_matrix: np.ndarray) -> None:
        """Draw every collider outline grouped by color. No-op if hidden."""
        if not self._visible:
            self.last_line_count = 0
            return

        groups: dict[tuple[float, float, float], list[float]] = {
            COLOR_DYNAMIC: [],
            COLOR_STATIC: [],
            COLOR_KINEMATIC: [],
            COLOR_SENSOR: [],
        }

        for arch in world.query(Transform2D, Collider2D, RigidBody2D).archetypes():
            n = arch.length
            transforms = arch.columns[Transform2D][:n]
            colliders = arch.columns[Collider2D][:n]
            rbs = arch.columns[RigidBody2D][:n]
            for i in range(n):
                body_type = int(rbs["body_type"][i])
                sensor = bool(colliders["sensor"][i])
                color = _color_for(body_type, sensor)
                shape_type = int(colliders["shape_type"][i])
                cx = float(transforms["x"][i])
                cy = float(transforms["y"][i])
                rotation = float(transforms["rotation"][i])

                if shape_type == SHAPE_TYPE_CIRCLE:
                    groups[color].extend(_circle_lines(cx, cy, float(colliders["radius"][i])))
                elif shape_type == SHAPE_TYPE_BOX:
                    groups[color].extend(
                        _box_lines(
                            cx, cy,
                            float(colliders["width"][i]) * 0.5,
                            float(colliders["height"][i]) * 0.5,
                            rotation,
                        )
                    )
                elif shape_type == SHAPE_TYPE_SEGMENT:
                    groups[color].extend(
                        _segment_lines(cx, cy, float(colliders["width"][i]), rotation)
                    )

        total_floats = sum(len(v) for v in groups.values())
        total_lines = total_floats // 4
        self.last_line_count = total_lines
        if total_floats == 0:
            return

        cam = np.ascontiguousarray(camera_matrix.astype(np.float32, copy=False).T).tobytes()
        try:
            self.shader["u_camera"].write(cam)
        except KeyError:
            pass

        for color, floats in groups.items():
            if not floats:
                continue
            arr = np.asarray(floats, dtype=np.float32)
            self._ensure_capacity(arr.size // 2)
            self._buffer.write(arr.tobytes())
            try:
                self.shader["u_color"].value = color
            except KeyError:
                pass
            self._vao.render(mode=moderngl.LINES, vertices=arr.size // 2)

    def release(self) -> None:
        """Release every owned GL object."""
        for obj in (self._vao, self._buffer, self.shader):
            try:
                obj.release()
            except Exception:
                pass


def setup_debug_draw(app: Any) -> DebugDraw2D:
    """Create a DebugDraw2D, register the POST_RENDER + F3-toggle systems. Idempotent."""
    existing = getattr(app, "_pyge_debug_draw", None)
    if existing is not None:
        return existing

    debug = DebugDraw2D(app.ctx)
    app._pyge_debug_draw = debug

    import glfw

    from ..renderer.camera2d import default_camera_matrix, build_camera_matrix
    from ..renderer.camera2d import Camera2D as _Camera2D

    # See pyge.tools.inspector for why we poll is_key_down with edge detection
    # instead of reading KeyEvent: input events can be missed at high refresh.
    _was = [False]  # one-element list as a mutable closure cell

    @app.system(Phase.PRE_UPDATE)
    def debug_draw_input_system(world: Any, dt: float) -> None:
        is_down = app.input.is_key_down(glfw.KEY_F3)
        if is_down and not _was[0]:
            debug.toggle()
        _was[0] = is_down

    @app.system(Phase.POST_RENDER)
    def debug_draw_render_system(world: Any, dt: float) -> None:
        if not debug.visible:
            return
        try:
            viewport_w, viewport_h = app.window.get_size()
        except Exception:
            viewport_w, viewport_h = 800, 600
        cam_matrix = None
        for (cams,) in world.query(_Camera2D):
            if len(cams) > 0:
                cam_matrix = build_camera_matrix(cams[0], viewport_w, viewport_h)
                break
        if cam_matrix is None:
            cam_matrix = default_camera_matrix(viewport_w, viewport_h)
        debug.render(world, cam_matrix)

    return debug
