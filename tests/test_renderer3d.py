"""Headless GL tests for the Phase 5 3D renderer.

Reuses the same hidden-window pattern as test_renderer2d. Skips cleanly if a
GL context can't be created.
"""
from __future__ import annotations

import math
import sys
import warnings
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import glfw
import moderngl
import numpy as np
import pytest

import pyge
from pyge import (
    Camera3D,
    DirectionalLight,
    MeshRenderer,
    Phase,
    PointLight,
    Scheduler,
    Transform3D,
    World,
)
from pyge.components.sprite import Sprite as _Sprite  # noqa: F401  - keeps dataclass registered
from pyge.renderer import setup_renderer_2d
from pyge.renderer3d import (
    AmbientLight,
    BoundingSphere,
    FrustumCuller,
    MAX_POINT_LIGHTS,
    Material,
    MaterialRegistry,
    MeshBuffer,
    MeshRegistry,
    OBJLoader,
    Renderer3D,
    ShaderCache3D,
    build_model_matrix,
    build_projection_matrix,
    build_view_matrix,
    make_cube,
    make_plane,
    make_sphere,
    resolve_world_matrix,
    setup_renderer_3d,
)


# --- GL fixture ------------------------------------------------------------

@pytest.fixture(scope="module")
def gl_ctx():
    """Hidden 1x1 GLFW window + OpenGL 3.3 Core context, shared across the module."""
    if not glfw.init():
        pytest.skip("GLFW init failed (no display?)")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    if sys.platform == "darwin":
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    win = glfw.create_window(1, 1, "pyge3d-test", None, None)
    if not win:
        glfw.terminate()
        pytest.skip("Could not create offscreen GLFW window")
    glfw.make_context_current(win)
    try:
        ctx = moderngl.create_context()
    except Exception as e:
        glfw.destroy_window(win)
        glfw.terminate()
        pytest.skip(f"ModernGL context unavailable: {e}")
    yield ctx
    try:
        ctx.release()
    except Exception:
        pass
    glfw.destroy_window(win)
    glfw.terminate()


def _fake_app(ctx, viewport=(800, 600)):
    sched = Scheduler()
    world = World()
    app = SimpleNamespace(
        ctx=ctx,
        world=world,
        _scheduler=sched,
        scheduler=sched,
        window=SimpleNamespace(get_size=lambda: viewport),
    )

    def system(phase: Phase):
        def deco(fn):
            sched.register(phase, fn)
            return fn
        return deco

    app.system = system
    return app


# --- OBJ / Mesh -----------------------------------------------------------

def test_obj_loader_parses_positions_normals_uvs():
    src = """
v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
vt 0 0
vt 1 0
vt 0 1
f 1/1/1 2/2/1 3/3/1
"""
    mesh = OBJLoader.load_from_string(src)
    assert mesh.vertices.shape == (3, 8)
    assert mesh.indices.shape == (3,)
    np.testing.assert_allclose(mesh.vertices[0, 0:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(mesh.vertices[1, 0:3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(mesh.vertices[2, 0:3], [0.0, 1.0, 0.0])
    for row in range(3):
        np.testing.assert_allclose(mesh.vertices[row, 3:6], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(mesh.vertices[0, 6:8], [0.0, 0.0])
    np.testing.assert_allclose(mesh.vertices[1, 6:8], [1.0, 0.0])
    np.testing.assert_allclose(mesh.vertices[2, 6:8], [0.0, 1.0])


def test_obj_loader_generates_flat_normals_when_missing():
    src = """
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""
    mesh = OBJLoader.load_from_string(src)
    assert mesh.vertices.shape == (3, 8)
    # Triangle in XY plane → flat normal points along +Z.
    np.testing.assert_allclose(mesh.vertices[0, 3:6], [0.0, 0.0, 1.0], atol=1e-6)


def test_obj_loader_zero_uv_when_missing():
    src = """
v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
f 1//1 2//1 3//1
"""
    mesh = OBJLoader.load_from_string(src)
    assert mesh.vertices.shape == (3, 8)
    for row in range(3):
        np.testing.assert_allclose(mesh.vertices[row, 6:8], [0.0, 0.0])


def test_obj_loader_triangulates_quads():
    src = """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""
    mesh = OBJLoader.load_from_string(src)
    # Quad → 2 triangles → 6 vertices, 6 indices.
    assert mesh.vertices.shape == (6, 8)
    assert mesh.indices.shape == (6,)


def test_obj_loader_ignores_unsupported_directives():
    src = """
mtllib test.mtl
o cube_obj
g group
s 1
usemtl Material
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""
    mesh = OBJLoader.load_from_string(src)
    assert mesh.vertices.shape == (3, 8)


def test_make_cube_has_36_indices():
    cube = make_cube()
    assert cube.indices.shape == (36,)
    assert cube.vertices.shape == (24, 8)


def test_make_plane_has_4_vertices_6_indices():
    p = make_plane(width=2.0, depth=4.0)
    assert p.vertices.shape == (4, 8)
    assert p.indices.shape == (6,)


def test_make_sphere_returns_a_mesh():
    s = make_sphere(subdivisions=1)
    assert s.vertices.ndim == 2 and s.vertices.shape[1] == 8
    assert s.vertices.shape[0] >= 6
    assert s.indices.ndim == 1


def test_mesh_registry_add_returns_int_id(gl_ctx):
    cache = ShaderCache3D()
    shader = cache.get(gl_ctx, "pbr_lite")
    reg = MeshRegistry(gl_ctx, shader)
    cube_id = reg.add(make_cube())
    plane_id = reg.add(make_plane())
    assert cube_id == 0
    assert plane_id == 1
    assert isinstance(reg.get_buffer(cube_id), MeshBuffer)
    assert reg.count() == 2


# --- Material -------------------------------------------------------------

def test_material_default_id_is_stable():
    reg = MaterialRegistry()
    a = reg.default_id()
    b = reg.default_id()
    assert a == b
    assert isinstance(reg.get(a), Material)


def test_material_registry_round_trip():
    reg = MaterialRegistry()
    mid = reg.add(Material(albedo_r=0.5, roughness=0.7, metallic=1.0))
    got = reg.get(mid)
    assert got.albedo_r == 0.5
    assert got.roughness == 0.7
    assert got.metallic == 1.0


# --- Transform3D / model + world matrices ---------------------------------

def test_build_model_matrix_identity_for_default_transform():
    M = build_model_matrix(Transform3D())
    np.testing.assert_allclose(M, np.eye(4, dtype=np.float32), atol=1e-6)


def test_build_model_matrix_translation_lands_in_column_3():
    M = build_model_matrix(Transform3D(x=1.0))
    assert M[0, 3] == 1.0


def test_build_model_matrix_scale_lands_on_diagonal():
    M = build_model_matrix(Transform3D(scale_x=2.0))
    assert M[0, 0] == 2.0


def test_resolve_world_matrix_no_parent_matches_local():
    world = World()
    e = world.spawn(Transform3D(x=5.0, y=2.0))
    world.flush()
    M = resolve_world_matrix(e, world)
    np.testing.assert_allclose(M, build_model_matrix(Transform3D(x=5.0, y=2.0)), atol=1e-6)


def test_resolve_world_matrix_chains_parent_and_child():
    world = World()
    parent = world.spawn(Transform3D(x=10.0))
    world.flush()
    child = world.spawn(Transform3D(x=5.0, parent=parent))
    world.flush()
    M = resolve_world_matrix(child, world)
    # parent_world @ child_local = T(10) @ T(5) = T(15) on the X axis.
    assert pytest.approx(M[0, 3], abs=1e-5) == 15.0


def test_resolve_world_matrix_warns_on_cycle():
    world = World()
    a = world.spawn(Transform3D())
    world.flush()
    b = world.spawn(Transform3D(parent=a))
    world.flush()
    # Make a's parent be b, creating a cycle a -> b -> a.
    world.add_component(a, Transform3D(parent=b))
    world.flush()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        M = resolve_world_matrix(a, world)
    assert M.shape == (4, 4)
    assert any(
        issubclass(warning.category, RuntimeWarning) and "cycle" in str(warning.message).lower()
        for warning in w
    )


# --- Camera matrices ------------------------------------------------------

def test_build_projection_matrix_returns_4x4_float32():
    P = build_projection_matrix(Camera3D(), 800, 600)
    assert P.shape == (4, 4)
    assert P.dtype == np.float32


def test_build_projection_matrix_fov_90_aspect_1():
    P = build_projection_matrix(Camera3D(fov=math.pi / 2.0), 100, 100)
    np.testing.assert_allclose(P[0, 0], 1.0, atol=1e-6)
    assert pytest.approx(P[0, 0], abs=1e-6) == P[1, 1]


def test_build_projection_matrix_rejects_zero_viewport():
    with pytest.raises(ValueError):
        build_projection_matrix(Camera3D(), 0, 600)


def test_build_view_matrix_returns_4x4_float32():
    V = build_view_matrix(Camera3D(z=2.0))
    assert V.shape == (4, 4)
    assert V.dtype == np.float32


def test_build_view_matrix_identity_for_default_camera():
    V = build_view_matrix(Camera3D())
    np.testing.assert_allclose(V, np.eye(4, dtype=np.float32), atol=1e-6)


def test_build_view_matrix_translates_camera_to_origin():
    V = build_view_matrix(Camera3D(x=10.0, y=0.0, z=5.0))
    eye = V @ np.array([10.0, 0.0, 5.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(eye[:3], [0.0, 0.0, 0.0], atol=1e-5)


# --- Frustum --------------------------------------------------------------

def test_frustum_culler_extracts_six_planes():
    culler = FrustumCuller()
    culler.update(np.eye(4, dtype=np.float32))
    assert culler.planes.shape == (6, 4)


def test_frustum_culler_visible_in_front_of_camera():
    cam = Camera3D(z=10.0)
    P = build_projection_matrix(cam, 800, 600)
    V = build_view_matrix(cam)
    culler = FrustumCuller()
    culler.update(P @ V)
    assert culler.is_visible(BoundingSphere(0.0, 0.0, 0.0, 1.0))


def test_frustum_culler_invisible_behind_camera():
    cam = Camera3D(z=10.0)
    P = build_projection_matrix(cam, 800, 600)
    V = build_view_matrix(cam)
    culler = FrustumCuller()
    culler.update(P @ V)
    assert not culler.is_visible(BoundingSphere(0.0, 0.0, 100.0, 1.0))


def test_frustum_culler_radius_can_save_a_marginal_sphere():
    cam = Camera3D(z=10.0)
    P = build_projection_matrix(cam, 800, 600)
    V = build_view_matrix(cam)
    culler = FrustumCuller()
    culler.update(P @ V)
    # Just barely behind the camera but with a huge radius — must still be visible.
    assert culler.is_visible(BoundingSphere(0.0, 0.0, 11.0, 5.0))


# --- Shader ---------------------------------------------------------------

def test_shader3d_compiles(gl_ctx):
    cache = ShaderCache3D()
    prog = cache.get(gl_ctx, "pbr_lite")
    assert isinstance(prog, moderngl.Program)


def test_shader3d_returns_same_program_object(gl_ctx):
    cache = ShaderCache3D()
    a = cache.get(gl_ctx, "pbr_lite")
    b = cache.get(gl_ctx, "pbr_lite")
    assert a is b


def test_shader3d_has_required_uniforms(gl_ctx):
    cache = ShaderCache3D()
    prog = cache.get(gl_ctx, "pbr_lite")
    expected = [
        "u_model", "u_view", "u_projection",
        "u_albedo", "u_roughness", "u_metallic", "u_emissive",
        "u_ambient",
        "u_dir_light_dir", "u_dir_light_color", "u_dir_light_intensity",
        "u_num_point_lights",
    ]
    program_members = set(prog)
    for name in expected:
        assert name in program_members, f"missing uniform {name}"


def test_shader3d_unknown_name_raises(gl_ctx):
    cache = ShaderCache3D()
    with pytest.raises(KeyError):
        cache.get(gl_ctx, "totally_not_a_shader")


# --- Renderer3D -----------------------------------------------------------

def test_setup_renderer_3d_inserts_resources(gl_ctx):
    app = _fake_app(gl_ctx)
    setup = setup_renderer_3d(app)
    assert app.world.get_resource(MeshRegistry) is setup.mesh_registry
    assert app.world.get_resource(MaterialRegistry) is setup.material_registry
    assert app.world.get_resource(Renderer3D) is setup.renderer3d
    render_systems = app._scheduler.systems(Phase.RENDER)
    assert len(render_systems) == 1


def test_setup_renderer_3d_is_idempotent(gl_ctx):
    app = _fake_app(gl_ctx)
    s1 = setup_renderer_3d(app)
    s2 = setup_renderer_3d(app)
    assert s1 is s2
    assert len(app._scheduler.systems(Phase.RENDER)) == 1


def test_renderer_render_with_no_entities(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_renderer_3d(app)
    app._scheduler.tick_render(app.world, 0.016)
    renderer = app.world.get_resource(Renderer3D)
    assert renderer.last_draw_calls == 0


def test_renderer_render_one_cube_issues_draw_call(gl_ctx):
    app = _fake_app(gl_ctx)
    setup = setup_renderer_3d(app)
    cube_id = setup.mesh_registry.add(make_cube())
    app.world.spawn(
        Camera3D(z=5.0),
    )
    app.world.spawn(
        Transform3D(z=0.0),
        MeshRenderer(mesh_id=cube_id, material_id=setup.material_registry.default_id()),
    )
    app.world.flush()
    app._scheduler.tick_render(app.world, 0.016)
    renderer = setup.renderer3d
    assert renderer.last_draw_calls == 1


def test_renderer_skips_invisible_meshes(gl_ctx):
    app = _fake_app(gl_ctx)
    setup = setup_renderer_3d(app)
    cube_id = setup.mesh_registry.add(make_cube())
    app.world.spawn(Camera3D(z=5.0))
    app.world.spawn(
        Transform3D(),
        MeshRenderer(mesh_id=cube_id, visible=False),
    )
    app.world.flush()
    app._scheduler.tick_render(app.world, 0.016)
    assert setup.renderer3d.last_draw_calls == 0


def test_renderer_culls_meshes_behind_camera(gl_ctx):
    app = _fake_app(gl_ctx)
    setup = setup_renderer_3d(app)
    cube_id = setup.mesh_registry.add(make_cube())
    app.world.spawn(Camera3D(z=5.0))
    # Mesh at z=100 is far behind the camera (camera at z=5 looking down -z).
    app.world.spawn(
        Transform3D(z=100.0),
        MeshRenderer(mesh_id=cube_id),
    )
    app.world.flush()
    app._scheduler.tick_render(app.world, 0.016)
    renderer = setup.renderer3d
    assert renderer.last_draw_calls == 0
    assert renderer.last_culled == 1


def test_renderer_toggles_depth_test(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_renderer_3d(app)
    with patch.object(gl_ctx, "enable", wraps=gl_ctx.enable) as enable, \
         patch.object(gl_ctx, "disable", wraps=gl_ctx.disable) as disable:
        app._scheduler.tick_render(app.world, 0.016)
    enable_args = [c.args[0] for c in enable.call_args_list if c.args]
    disable_args = [c.args[0] for c in disable.call_args_list if c.args]
    assert moderngl.DEPTH_TEST in enable_args
    assert moderngl.DEPTH_TEST in disable_args


def test_renderer_clears_when_no_2d(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_renderer_3d(app)
    with patch.object(gl_ctx, "clear", wraps=gl_ctx.clear) as clear:
        app._scheduler.tick_render(app.world, 0.016)
    assert clear.call_count >= 1


def test_renderer_point_light_limit(gl_ctx):
    app = _fake_app(gl_ctx)
    setup = setup_renderer_3d(app)
    cube_id = setup.mesh_registry.add(make_cube())
    app.world.spawn(Camera3D(z=10.0))
    app.world.spawn(
        Transform3D(),
        MeshRenderer(mesh_id=cube_id),
    )
    for i in range(10):
        app.world.spawn(
            Transform3D(x=float(i), z=0.0),
            PointLight(intensity=1.0, radius=5.0),
        )
    app.world.flush()
    app._scheduler.tick_render(app.world, 0.016)
    renderer = setup.renderer3d
    assert renderer.last_point_lights == MAX_POINT_LIGHTS == 8
    # Uniform also clamped to 8.
    assert renderer.shader["u_num_point_lights"].value == 8


def test_renderer_uses_ambient_resource_when_set(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_renderer_3d(app)
    app.world.insert_resource(AmbientLight(0.4, 0.5, 0.6))
    app._scheduler.tick_render(app.world, 0.016)
    renderer = app.world.get_resource(Renderer3D)
    val = renderer.shader["u_ambient"].value
    np.testing.assert_allclose(val, (0.4, 0.5, 0.6), atol=1e-5)


def test_renderer_finds_directional_light(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_renderer_3d(app)
    app.world.spawn(DirectionalLight(dir_x=1.0, dir_y=0.0, dir_z=0.0, intensity=2.0))
    app.world.flush()
    app._scheduler.tick_render(app.world, 0.016)
    renderer = app.world.get_resource(Renderer3D)
    assert pytest.approx(renderer.shader["u_dir_light_intensity"].value, abs=1e-6) == 2.0


# --- 2D + 3D coexistence --------------------------------------------------

def test_2d_and_3d_coexistence_clears_once_and_toggles_depth(gl_ctx):
    app = _fake_app(gl_ctx, viewport=(64, 64))
    setup_renderer_2d(app)
    setup_renderer_3d(app)

    setup3 = app._pyge_renderer_3d
    cube_id = setup3.mesh_registry.add(make_cube())
    app.world.spawn(Camera3D(z=5.0))
    app.world.spawn(
        Transform3D(),
        MeshRenderer(mesh_id=cube_id, material_id=setup3.material_registry.default_id()),
    )
    app.world.flush()

    with patch.object(gl_ctx, "clear", wraps=gl_ctx.clear) as clear, \
         patch.object(gl_ctx, "enable", wraps=gl_ctx.enable) as enable, \
         patch.object(gl_ctx, "disable", wraps=gl_ctx.disable) as disable:
        app._scheduler.tick_render(app.world, 0.016)

    # 2D system clears once; 3D system detected 2D resource and skipped its own clear.
    assert clear.call_count == 1
    enabled = [c.args[0] for c in enable.call_args_list if c.args]
    disabled = [c.args[0] for c in disable.call_args_list if c.args]
    assert moderngl.DEPTH_TEST in enabled
    assert moderngl.DEPTH_TEST in disabled


# --- Top-level re-exports -------------------------------------------------

def test_top_level_re_exports():
    assert pyge.Transform3D is Transform3D
    assert pyge.MeshRenderer is MeshRenderer
    assert pyge.Camera3D is Camera3D
    assert pyge.DirectionalLight is DirectionalLight
    assert pyge.PointLight is PointLight
    assert pyge.Renderer3D is Renderer3D
    assert pyge.setup_renderer_3d is setup_renderer_3d
