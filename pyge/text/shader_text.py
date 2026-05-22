"""GLSL source for the text-batch shader. R8 atlas, color from u_color uniform."""
from __future__ import annotations


TEXT_VERT_SRC = """\
#version 330 core

in vec2 in_position;
in vec2 in_uv;

uniform mat4 u_projection;

out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = u_projection * vec4(in_position, 0.0, 1.0);
}
"""


TEXT_FRAG_SRC = """\
#version 330 core

in vec2 v_uv;

uniform sampler2D u_texture;
uniform vec4 u_color;

out vec4 f_color;

void main() {
    float alpha = texture(u_texture, v_uv).r;
    float out_a = u_color.a * alpha;
    if (out_a < 0.01) {
        discard;
    }
    f_color = vec4(u_color.rgb, out_a);
}
"""
