"""Headless GL tests for the Phase 3 2D renderer.

Uses a single hidden 1x1 GLFW window per module to drive a real ModernGL
context. Skips cleanly if a context can't be created (e.g. no GPU/display).
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import glfw
import moderngl
import numpy as np
import pytest

import pyge
from pyge import Phase, Scheduler, Sprite, Transform2D, World
from pyge.renderer import (
    Camera2D,
    MAX_TEXTURE_UNITS,
    ShaderCache,
    SpriteBatch2D,
    TextureAtlas,
    Tilemap,
    TilemapSetup,
    build_camera_matrix,
    default_camera_matrix,
    setup_renderer_2d,
    setup_tilemap,
)


# --- GL fixture ------------------------------------------------------------

@pytest.fixture(scope="module")
def gl_ctx():
    """Hidden 1x1 GLFW window + OpenGL 3.3 Core context shared across the module."""
    if not glfw.init():
        pytest.skip("GLFW init failed (no display?)")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    if sys.platform == "darwin":
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    win = glfw.create_window(1, 1, "pyge-test", None, None)
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
    """A minimal app-shape compatible with setup_renderer_2d (no real Window)."""
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


def _fake_pil_image(size=(4, 4)):
    """A MagicMock image that satisfies Image.open(...).convert('RGBA').tobytes() / .size."""
    img = MagicMock()
    img.size = size
    img.tobytes.return_value = b"\xff" * (size[0] * size[1] * 4)
    img.convert.return_value = img
    img.__enter__ = MagicMock(return_value=img)
    img.__exit__ = MagicMock(return_value=False)
    return img


# --- TextureAtlas ----------------------------------------------------------

def test_atlas_load_returns_dense_ids(gl_ctx):
    atlas = TextureAtlas(gl_ctx)
    with patch("PIL.Image.open", return_value=_fake_pil_image()):
        a = atlas.load("a.png")
        b = atlas.load("b.png")
    assert a == 0
    assert b == 1
    assert atlas.texture_count() == 2


def test_atlas_caches_by_path(gl_ctx):
    atlas = TextureAtlas(gl_ctx)
    with patch("PIL.Image.open", return_value=_fake_pil_image()) as mock_open:
        first = atlas.load("hero.png")
        second = atlas.load("hero.png")
        third = atlas.load("hero.png")
    assert first == second == third
    assert mock_open.call_count == 1, "second load must not re-open the file"


def test_atlas_rejects_more_than_max_units(gl_ctx):
    atlas = TextureAtlas(gl_ctx)
    for i in range(MAX_TEXTURE_UNITS):
        atlas.add_texture(f"t{i}", gl_ctx.texture((1, 1), 4, b"\xff\xff\xff\xff"))
    assert atlas.texture_count() == MAX_TEXTURE_UNITS
    with pytest.raises(RuntimeError):
        atlas.add_texture("overflow", gl_ctx.texture((1, 1), 4, b"\xff\xff\xff\xff"))


# --- ShaderCache -----------------------------------------------------------

def test_shader_cache_compiles_sprite_program(gl_ctx):
    cache = ShaderCache()
    prog = cache.get(gl_ctx, "sprite")
    assert isinstance(prog, moderngl.Program)


def test_shader_cache_returns_same_program_object(gl_ctx):
    cache = ShaderCache()
    a = cache.get(gl_ctx, "sprite")
    b = cache.get(gl_ctx, "sprite")
    assert a is b


def test_shader_cache_unknown_shader_raises(gl_ctx):
    cache = ShaderCache()
    with pytest.raises(KeyError):
        cache.get(gl_ctx, "does_not_exist")


# --- SpriteBatch2D ---------------------------------------------------------

def test_sprite_batch_init(gl_ctx):
    cache = ShaderCache()
    prog = cache.get(gl_ctx, "sprite")
    batch = SpriteBatch2D(gl_ctx, prog)
    assert batch.capacity == 4096
    assert batch.instance_buf is not None
    assert batch.vao is not None


def test_sprite_batch_buffer_grows_past_initial_capacity(gl_ctx):
    cache = ShaderCache()
    prog = cache.get(gl_ctx, "sprite")
    batch = SpriteBatch2D(gl_ctx, prog)
    initial = batch.capacity
    batch._ensure_capacity(initial + 1)
    assert batch.capacity > initial
    # Doubles by default.
    assert batch.capacity >= initial * 2


def test_sprite_batch_render_with_no_entities_is_a_noop(gl_ctx):
    cache = ShaderCache()
    prog = cache.get(gl_ctx, "sprite")
    batch = SpriteBatch2D(gl_ctx, prog)
    cam = default_camera_matrix(800, 600)
    batch.render([], cam)  # nothing to draw, must not raise


def test_sprite_batch_render_grows_buffer_for_huge_group(gl_ctx):
    """A single texture group >4096 sprites must trigger growth and still render."""
    cache = ShaderCache()
    prog = cache.get(gl_ctx, "sprite")
    batch = SpriteBatch2D(gl_ctx, prog)

    # Need an actual texture bound at unit 0 for the shader to sample.
    tex = gl_ctx.texture((1, 1), 4, b"\xff\xff\xff\xff")
    tex.use(location=0)

    n = 5000
    transforms = np.zeros(
        n,
        dtype=[
            ("x", "f8"), ("y", "f8"),
            ("rotation", "f8"),
            ("scale_x", "f8"), ("scale_y", "f8"),
        ],
    )
    transforms["scale_x"] = 1.0
    transforms["scale_y"] = 1.0
    sprites = np.zeros(
        n,
        dtype=[
            ("texture_id", "i8"),
            ("r", "f8"), ("g", "f8"), ("b", "f8"), ("a", "f8"),
            ("width", "f8"), ("height", "f8"),
            ("flip_x", "?"), ("flip_y", "?"),
        ],
    )
    sprites["r"] = 1.0
    sprites["g"] = 1.0
    sprites["b"] = 1.0
    sprites["a"] = 1.0
    sprites["width"] = 32.0
    sprites["height"] = 32.0

    cam = default_camera_matrix(800, 600)
    batch.render([(transforms, sprites)], cam)
    assert batch.capacity >= n


# --- Camera matrix ---------------------------------------------------------

def test_build_camera_matrix_returns_4x4_float32():
    m = build_camera_matrix(Camera2D(), 800, 600)
    assert m.shape == (4, 4)
    assert m.dtype == np.float32


def test_build_camera_matrix_maps_corners_to_ndc():
    m = build_camera_matrix(Camera2D(), 800, 600)

    def project(x, y):
        v = np.array([x, y, 0.0, 1.0], dtype=np.float32)
        return (m @ v)[:2]

    np.testing.assert_allclose(project(0.0, 0.0), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(project(400.0, 300.0), [1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(project(-400.0, -300.0), [-1.0, -1.0], atol=1e-5)


def test_build_camera_matrix_applies_translation():
    cam = Camera2D(x=100.0, y=50.0)
    m = build_camera_matrix(cam, 800, 600)
    v = np.array([100.0, 50.0, 0.0, 1.0], dtype=np.float32)
    out = m @ v
    np.testing.assert_allclose(out[:2], [0.0, 0.0], atol=1e-5)


def test_build_camera_matrix_applies_zoom():
    cam = Camera2D(zoom=2.0)
    m = build_camera_matrix(cam, 800, 600)
    # At zoom 2, world (200, 150) (=quarter screen) should reach NDC (1, 1).
    v = np.array([200.0, 150.0, 0.0, 1.0], dtype=np.float32)
    out = m @ v
    np.testing.assert_allclose(out[:2], [1.0, 1.0], atol=1e-5)


def test_build_camera_matrix_accepts_structured_record():
    """The renderer pulls Camera2D out of a numpy structured array — must work directly."""
    rec = np.zeros(
        1,
        dtype=[("x", "f8"), ("y", "f8"), ("zoom", "f8"), ("rotation", "f8")],
    )
    rec["zoom"] = 1.0
    m = build_camera_matrix(rec[0], 800, 600)
    np.testing.assert_allclose(
        (m @ np.array([400.0, 300.0, 0.0, 1.0], dtype=np.float32))[:2],
        [1.0, 1.0],
        atol=1e-5,
    )


def test_build_camera_matrix_rejects_zero_viewport():
    with pytest.raises(ValueError):
        build_camera_matrix(Camera2D(), 0, 600)


# --- Tilemap ---------------------------------------------------------------

def test_tilemap_load_builds_chunks(gl_ctx):
    cache = ShaderCache()
    prog = cache.get(gl_ctx, "sprite")
    atlas = TextureAtlas(gl_ctx)
    atlas.add_texture("dummy", gl_ctx.texture((1, 1), 4, b"\xff\xff\xff\xff"))

    tm = Tilemap(gl_ctx, prog, atlas, tile_width=32, tile_height=32)

    tile_data = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.int32,
    )
    tm.load(tile_data)
    assert tm.build_count == 1
    assert len(tm.chunks) == 1  # all five non-zero tiles fit in one 16x16 chunk
    assert tm.chunks[0].count == 5


def test_tilemap_load_partitions_into_multiple_chunks(gl_ctx):
    cache = ShaderCache()
    prog = cache.get(gl_ctx, "sprite")
    atlas = TextureAtlas(gl_ctx)
    atlas.add_texture("dummy", gl_ctx.texture((1, 1), 4, b"\xff\xff\xff\xff"))

    tm = Tilemap(gl_ctx, prog, atlas, tile_width=8, tile_height=8)
    tile_data = np.ones((40, 40), dtype=np.int32)  # 40x40 tiles -> 9 chunks (3x3)
    tm.load(tile_data)
    assert len(tm.chunks) == 9


def test_tilemap_with_all_zero_data_renders_without_error(gl_ctx):
    cache = ShaderCache()
    prog = cache.get(gl_ctx, "sprite")
    atlas = TextureAtlas(gl_ctx)

    tm = Tilemap(gl_ctx, prog, atlas)
    tile_data = np.zeros((10, 10), dtype=np.int32)
    tm.load(tile_data)
    assert tm.chunks == []
    cam = default_camera_matrix(800, 600)
    tm.render(cam)  # must not raise


def test_tilemap_load_replaces_previous_chunks(gl_ctx):
    cache = ShaderCache()
    prog = cache.get(gl_ctx, "sprite")
    atlas = TextureAtlas(gl_ctx)
    atlas.add_texture("dummy", gl_ctx.texture((1, 1), 4, b"\xff\xff\xff\xff"))

    tm = Tilemap(gl_ctx, prog, atlas)
    tm.load(np.ones((5, 5), dtype=np.int32))
    assert len(tm.chunks) == 1
    tm.load(np.zeros((5, 5), dtype=np.int32))
    assert tm.chunks == []
    assert tm.build_count == 2


# --- setup_renderer_2d -----------------------------------------------------

def test_setup_renderer_registers_render_phase_system(gl_ctx):
    app = _fake_app(gl_ctx)
    setup = setup_renderer_2d(app)
    assert isinstance(setup.atlas, TextureAtlas)
    assert isinstance(setup.shader_cache, ShaderCache)
    render_systems = app._scheduler.systems(Phase.RENDER)
    assert len(render_systems) == 1


def test_setup_renderer_is_idempotent(gl_ctx):
    app = _fake_app(gl_ctx)
    first = setup_renderer_2d(app)
    second = setup_renderer_2d(app)
    assert first is second
    # Render system must not be registered twice.
    render_systems = app._scheduler.systems(Phase.RENDER)
    assert len(render_systems) == 1


def test_setup_renderer_inserts_resources(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_renderer_2d(app)
    assert app.world.get_resource(moderngl.Context) is gl_ctx
    assert isinstance(app.world.get_resource(TextureAtlas), TextureAtlas)
    assert isinstance(app.world.get_resource(ShaderCache), ShaderCache)
    assert isinstance(app.world.get_resource(SpriteBatch2D), SpriteBatch2D)


def test_setup_renderer_preloads_default_white_texture(gl_ctx):
    """Phase 7 fix 1: setup_renderer_2d should ship a 1x1 white texture at id=0
    so Sprite(texture_id=0, r=g=b=...) tints render without any user setup."""
    app = _fake_app(gl_ctx)
    setup = setup_renderer_2d(app)
    atlas = setup.atlas
    assert atlas.texture_count() >= 1
    # The white texture is reachable as id=0.
    tex = atlas.get_texture(0)
    assert tex.size == (1, 1)


def test_default_camera_matrix_centers_on_framebuffer():
    """Phase 7 fix 9: with no Camera2D entity, the default camera centers on the
    framebuffer so pixel coordinates "just work"."""
    m = default_camera_matrix(800, 600)
    # World (400, 300) — the screen center in pixel coords — projects to NDC (0, 0).
    v = np.array([400.0, 300.0, 0.0, 1.0], dtype=np.float32)
    out = m @ v
    np.testing.assert_allclose(out[:2], [0.0, 0.0], atol=1e-5)
    # World (0, 0) — bottom-left in pixel coords — projects to NDC (-1, -1).
    out = m @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(out[:2], [-1.0, -1.0], atol=1e-5)
    # World (800, 600) — top-right — projects to NDC (1, 1).
    out = m @ np.array([800.0, 600.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(out[:2], [1.0, 1.0], atol=1e-5)


# --- Render integration ----------------------------------------------------

def test_render_integration_clears_and_runs_without_error(gl_ctx):
    """Spawn a Transform2D + Sprite, run one render tick, and verify ctx.clear was called."""
    app = _fake_app(gl_ctx, viewport=(64, 64))
    setup = setup_renderer_2d(app)

    tex = gl_ctx.texture((2, 2), 4, b"\xff" * 16)
    setup.atlas.add_texture("white", tex)

    app.world.spawn(
        Transform2D(x=0.0, y=0.0),
        Sprite(texture_id=0, width=16.0, height=16.0),
    )
    app.world.flush()

    with patch.object(gl_ctx, "clear", wraps=gl_ctx.clear) as mock_clear:
        app._scheduler.tick_render(app.world, 0.016)
        assert mock_clear.called, "render system must call ctx.clear() once per frame"


def test_render_uses_camera_when_present(gl_ctx):
    app = _fake_app(gl_ctx, viewport=(800, 600))
    setup_renderer_2d(app)
    app.world.spawn(Camera2D(x=10.0, y=20.0, zoom=2.0))
    app.world.spawn(
        Transform2D(),
        Sprite(texture_id=0, width=8.0, height=8.0),
    )
    app.world.flush()
    # No texture bound, but the sprite shader's discard-on-low-alpha will still
    # safely run; we just need the render system itself not to crash.
    app._scheduler.tick_render(app.world, 0.016)


def test_render_with_no_camera_uses_default(gl_ctx):
    app = _fake_app(gl_ctx, viewport=(800, 600))
    setup_renderer_2d(app)
    app.world.spawn(
        Transform2D(),
        Sprite(texture_id=0, width=8.0, height=8.0),
    )
    app.world.flush()
    app._scheduler.tick_render(app.world, 0.016)


# --- Public re-exports -----------------------------------------------------

def test_top_level_components_re_exported():
    assert pyge.Transform2D is Transform2D
    assert pyge.Sprite is Sprite
    assert pyge.Camera2D is Camera2D


def test_components_have_numpy_storage():
    assert Transform2D.__pyge_component__.is_numpy is True
    assert Sprite.__pyge_component__.is_numpy is True
    assert Camera2D.__pyge_component__.is_numpy is True


# --- setup_tilemap --------------------------------------------------------

def _basic_tile_data(rows: int = 4, cols: int = 4) -> np.ndarray:
    """A small tile array with a 1-tile border for predictable rendering."""
    td = np.zeros((rows, cols), dtype=np.int32)
    td[0, :] = 1
    td[-1, :] = 1
    td[:, 0] = 1
    td[:, -1] = 1
    return td


def test_setup_tilemap_requires_renderer_2d(gl_ctx):
    app = _fake_app(gl_ctx)
    with pytest.raises(RuntimeError):
        setup_tilemap(app, _basic_tile_data())


def test_setup_tilemap_returns_dataclass(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_renderer_2d(app)
    ts = setup_tilemap(app, _basic_tile_data())
    assert isinstance(ts, TilemapSetup)
    assert isinstance(ts.tilemap, Tilemap)


def test_setup_tilemap_inserts_world_resource(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_renderer_2d(app)
    setup_tilemap(app, _basic_tile_data())
    assert app.world.has_resource(Tilemap)


def test_setup_tilemap_registers_pre_render_system(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_renderer_2d(app)
    before = len(app._scheduler._systems[Phase.PRE_RENDER])
    setup_tilemap(app, _basic_tile_data())
    after = len(app._scheduler._systems[Phase.PRE_RENDER])
    assert after == before + 1


def test_setup_tilemap_idempotent_reload(gl_ctx):
    """A second setup_tilemap on the same app does NOT add a system; it re-bakes."""
    app = _fake_app(gl_ctx)
    setup_renderer_2d(app)
    ts1 = setup_tilemap(app, _basic_tile_data())
    initial_pre_render = len(app._scheduler._systems[Phase.PRE_RENDER])
    initial_builds = ts1.tilemap.build_count
    ts2 = setup_tilemap(app, _basic_tile_data(rows=6, cols=6))
    assert ts2 is ts1
    assert ts2.tilemap is ts1.tilemap
    assert len(app._scheduler._systems[Phase.PRE_RENDER]) == initial_pre_render
    # load() was called again — the tilemap has been re-baked.
    assert ts2.tilemap.build_count == initial_builds + 1


def test_tilemap_runs_in_pre_render_phase(gl_ctx):
    """Tilemap PRE_RENDER must execute before sprite RENDER inside tick_render."""
    app = _fake_app(gl_ctx, viewport=(64, 64))
    setup_renderer_2d(app)
    setup = setup_tilemap(app, _basic_tile_data())

    call_order: list[str] = []
    original_render = setup.tilemap.render

    def tracked_render(matrix):
        call_order.append("tilemap")
        return original_render(matrix)

    setup.tilemap.render = tracked_render

    with patch.object(
        app._pyge_renderer_2d.sprite_batch,
        "render",
        wraps=app._pyge_renderer_2d.sprite_batch.render,
    ) as sprite_render:
        def append_sprite(*args, **kwargs):
            call_order.append("sprite")
            return sprite_render._mock_wraps(*args, **kwargs)
        sprite_render.side_effect = append_sprite
        app._scheduler.tick_render(app.world, 0.016)

    assert call_order == ["tilemap", "sprite"]


def test_tilemap_and_sprite_coexist(gl_ctx):
    """Spawning a sprite while a tilemap is wired must not break rendering."""
    app = _fake_app(gl_ctx, viewport=(64, 64))
    setup_renderer_2d(app)
    setup_tilemap(app, _basic_tile_data())
    app.world.spawn(
        Transform2D(x=8.0, y=8.0),
        Sprite(texture_id=0, width=8.0, height=8.0),
    )
    app.world.flush()
    app._scheduler.tick_render(app.world, 0.016)


def test_setup_tilemap_respects_custom_tile_size(gl_ctx):
    """tile_width / tile_height must propagate to the Tilemap object."""
    app = _fake_app(gl_ctx)
    setup_renderer_2d(app)
    ts = setup_tilemap(app, _basic_tile_data(), tile_width=48, tile_height=24)
    assert ts.tilemap.tile_width == 48
    assert ts.tilemap.tile_height == 24


def test_get_active_camera_matrix_default_when_no_camera(gl_ctx):
    """The shared camera helper falls back to default_camera_matrix when no Camera2D exists."""
    from pyge.renderer import get_active_camera_matrix

    app = _fake_app(gl_ctx, viewport=(64, 64))
    setup_renderer_2d(app)
    mat = get_active_camera_matrix(app.world, 64, 64)
    expected = default_camera_matrix(64, 64)
    np.testing.assert_allclose(mat, expected)
