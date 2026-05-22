"""ShaderCache and bundled GLSL sources for the 2D renderer.

The sprite shader is shared by SpriteBatch2D and Tilemap. It expects:
  * a unit-quad vertex stream (vec2 in_position in [-0.5, 0.5])
  * 14 floats of per-instance data (offset, rotation, scale, tint, uv_rect, tex_unit)
  * a mat4 u_camera uniform
  * a sampler2D u_textures[16] uniform array bound to texture units 0..15
"""
from __future__ import annotations

from typing import Any


SPRITE_VERT_SRC: str = """\
#version 330 core

in vec2 in_position;
in vec2 in_offset;
in float in_rotation;
in vec2 in_scale;
in vec4 in_tint;
in vec4 in_uv_rect;
in float in_tex_unit;

uniform mat4 u_camera;

out vec2 v_uv;
out vec4 v_tint;
flat out int v_tex_unit;

void main() {
    vec2 local = in_position * in_scale;
    float c = cos(in_rotation);
    float s = sin(in_rotation);
    vec2 rotated = vec2(c * local.x - s * local.y,
                        s * local.x + c * local.y);
    vec2 world_pos = rotated + in_offset;
    gl_Position = u_camera * vec4(world_pos, 0.0, 1.0);

    vec2 unit_uv = in_position + vec2(0.5);
    v_uv = in_uv_rect.xy + unit_uv * in_uv_rect.zw;
    v_tint = in_tint;
    v_tex_unit = int(in_tex_unit);
}
"""


SPRITE_FRAG_SRC: str = """\
#version 330 core

in vec2 v_uv;
in vec4 v_tint;
flat in int v_tex_unit;

uniform sampler2D u_textures[16];

out vec4 frag_color;

void main() {
    vec4 sampled;
    int unit = v_tex_unit;
    if      (unit ==  0) sampled = texture(u_textures[ 0], v_uv);
    else if (unit ==  1) sampled = texture(u_textures[ 1], v_uv);
    else if (unit ==  2) sampled = texture(u_textures[ 2], v_uv);
    else if (unit ==  3) sampled = texture(u_textures[ 3], v_uv);
    else if (unit ==  4) sampled = texture(u_textures[ 4], v_uv);
    else if (unit ==  5) sampled = texture(u_textures[ 5], v_uv);
    else if (unit ==  6) sampled = texture(u_textures[ 6], v_uv);
    else if (unit ==  7) sampled = texture(u_textures[ 7], v_uv);
    else if (unit ==  8) sampled = texture(u_textures[ 8], v_uv);
    else if (unit ==  9) sampled = texture(u_textures[ 9], v_uv);
    else if (unit == 10) sampled = texture(u_textures[10], v_uv);
    else if (unit == 11) sampled = texture(u_textures[11], v_uv);
    else if (unit == 12) sampled = texture(u_textures[12], v_uv);
    else if (unit == 13) sampled = texture(u_textures[13], v_uv);
    else if (unit == 14) sampled = texture(u_textures[14], v_uv);
    else                 sampled = texture(u_textures[15], v_uv);

    vec4 col = sampled * v_tint;
    if (col.a < 0.01) discard;
    frag_color = col;
}
"""


_SHADER_SOURCES: dict[str, tuple[str, str]] = {
    "sprite": (SPRITE_VERT_SRC, SPRITE_FRAG_SRC),
}


class ShaderCache:
    """Compile-once cache of moderngl Programs keyed by source name."""

    __slots__ = ("_programs",)

    def __init__(self) -> None:
        self._programs: dict[str, Any] = {}

    def get(self, ctx: Any, name: str) -> Any:
        """Return the cached program for `name`, compiling it on first access."""
        cached = self._programs.get(name)
        if cached is not None:
            return cached
        try:
            vert, frag = _SHADER_SOURCES[name]
        except KeyError as e:
            raise KeyError(f"Unknown shader: {name!r} (known: {sorted(_SHADER_SOURCES)})") from e
        prog = ctx.program(vertex_shader=vert, fragment_shader=frag)
        self._programs[name] = prog
        return prog

    def __contains__(self, name: str) -> bool:
        return name in self._programs

    def clear(self) -> None:
        """Release every cached program."""
        for prog in self._programs.values():
            try:
                prog.release()
            except Exception:
                pass
        self._programs.clear()
