"""3D shader source + ShaderCache3D.

Bundled shader name: ``"pbr_lite"``. The vertex shader transforms positions
through the standard model/view/projection chain and forwards a world-space
position + normal + uv. The fragment shader implements Lambert diffuse plus
Blinn-Phong specular, with roughness controlling specular sharpness and
metallic blending the dielectric (white) vs metallic (albedo) F0 tint. Up to
8 point lights are supported via uniform arrays.

OpenGL 3.3 Core. No texture sampling — that is deliberately out of v1 scope.
"""
from __future__ import annotations

from typing import Any


PBR_LITE_VERT_SRC: str = """\
#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world_pos = world.xyz;
    // mat3(u_model) is correct for rotation + uniform scale; nonuniform scale
    // would require the inverse-transpose, which v1 doesn't bother with.
    v_normal = normalize(mat3(u_model) * in_normal);
    v_uv = in_uv;
    gl_Position = u_projection * u_view * world;
}
"""


PBR_LITE_FRAG_SRC: str = """\
#version 330 core

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;

uniform vec3  u_albedo;
uniform float u_roughness;
uniform float u_metallic;
uniform vec3  u_emissive;

uniform vec3  u_ambient;

uniform vec3  u_dir_light_dir;
uniform vec3  u_dir_light_color;
uniform float u_dir_light_intensity;

uniform vec3  u_point_light_pos[8];
uniform vec3  u_point_light_color[8];
uniform float u_point_light_intensity[8];
uniform float u_point_light_radius[8];
uniform int   u_num_point_lights;

uniform vec3  u_camera_pos;

out vec4 frag_color;

vec3 shade(vec3 N, vec3 V, vec3 L, vec3 light_color, float intensity) {
    // L points FROM the surface TOWARDS the light.
    float ndl = max(dot(N, L), 0.0);
    vec3 diffuse = u_albedo * ndl;

    vec3 H = normalize(L + V);
    float ndh = max(dot(N, H), 0.0);
    float spec_power = mix(2.0, 128.0, 1.0 - clamp(u_roughness, 0.0, 1.0));
    float spec = pow(ndh, spec_power);

    // Schlick-style F0: dielectrics reflect ~4% white, metals reflect their albedo.
    vec3 spec_color = mix(vec3(0.04), u_albedo, clamp(u_metallic, 0.0, 1.0));
    vec3 specular = spec_color * spec;

    vec3 lit = diffuse * (1.0 - clamp(u_metallic, 0.0, 1.0)) + specular;
    return lit * light_color * intensity;
}

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_camera_pos - v_world_pos);

    // Ambient.
    vec3 color = u_ambient * u_albedo;

    // Directional light: u_dir_light_dir is the direction the light TRAVELS,
    // so the L used in `shade` is the negation (surface -> light).
    vec3 dir_to_dir_light = -normalize(u_dir_light_dir);
    color += shade(N, V, dir_to_dir_light, u_dir_light_color, u_dir_light_intensity);

    // Point lights.
    int n = u_num_point_lights;
    if (n > 8) n = 8;
    for (int i = 0; i < n; i++) {
        vec3 to_light = u_point_light_pos[i] - v_world_pos;
        float dist = length(to_light);
        vec3 L = (dist > 0.0) ? to_light / dist : vec3(0.0, 1.0, 0.0);

        float radius = max(u_point_light_radius[i], 1e-4);
        float falloff = max(0.0, 1.0 - dist / radius);
        falloff *= falloff;

        color += shade(N, V, L, u_point_light_color[i], u_point_light_intensity[i] * falloff);
    }

    color += u_emissive;

    // v_uv is currently unused for shading (no texture sampling in v1) but must
    // stay live so the linker doesn't strip in_uv from the VAO. A near-zero UV
    // tint accomplishes that without a visible effect; v2 will replace this
    // with real texture lookups.
    color += vec3(v_uv, 0.0) * 1e-7;

    frag_color = vec4(color, 1.0);
}
"""


_SHADER_SOURCES: dict[str, tuple[str, str]] = {
    "pbr_lite": (PBR_LITE_VERT_SRC, PBR_LITE_FRAG_SRC),
}


class ShaderCache3D:
    """Compile-once cache of moderngl Programs keyed by source name."""

    __slots__ = ("_programs",)

    def __init__(self) -> None:
        self._programs: dict[str, Any] = {}

    def get(self, ctx: Any, name: str) -> Any:
        """Return the compiled Program for `name`, building it on first access."""
        cached = self._programs.get(name)
        if cached is not None:
            return cached
        try:
            vert, frag = _SHADER_SOURCES[name]
        except KeyError as e:
            raise KeyError(f"Unknown 3D shader: {name!r} (known: {sorted(_SHADER_SOURCES)})") from e
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
