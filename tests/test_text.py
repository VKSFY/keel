"""Headless GL tests for Phase 8 — text rendering, fonts, glyph atlas."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import glfw
import moderngl
import pytest

import keel
from keel import Phase, Scheduler, Transform2D, World
from keel.renderer import setup_renderer_2d
from keel.text import (
    AtlasTooSmallError,
    BUILTIN_FONT,
    Font,
    FontRegistry,
    GlyphAtlas,
    GlyphInfo,
    TextBatch,
    TextLabel,
    TextSetup,
    clear_text,
    get_text,
    load_font,
    set_text,
    setup_text,
)
from keel.text.text_renderer import _reset_label_text


# --- GL fixture ----------------------------------------------------------

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
    win = glfw.create_window(1, 1, "keel-test", None, None)
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
    """A minimal app-shape compatible with setup_renderer_2d / setup_text."""
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


@pytest.fixture(autouse=True)
def _clean_text_table():
    """Each test starts with an empty _LABEL_TEXT side-table."""
    _reset_label_text()
    yield
    _reset_label_text()


# --- Font loading --------------------------------------------------------

def test_font_loads_from_builtin_font(gl_ctx):
    font = Font(gl_ctx, BUILTIN_FONT, size_px=24)
    assert font is not None
    assert font.path == os.path.abspath(BUILTIN_FONT) or font.path == BUILTIN_FONT


def test_font_measure_nonempty_string_returns_positive(gl_ctx):
    font = Font(gl_ctx, BUILTIN_FONT, size_px=24)
    w, h = font.measure("Hello")
    assert w > 0
    assert h > 0


def test_font_measure_empty_string_has_zero_width(gl_ctx):
    font = Font(gl_ctx, BUILTIN_FONT, size_px=24)
    w, h = font.measure("")
    assert w == 0
    assert h > 0  # still one line tall


def test_font_line_height_positive(gl_ctx):
    font = Font(gl_ctx, BUILTIN_FONT, size_px=24)
    assert font.line_height > 0


def test_font_registry_caches_same_path_size(gl_ctx):
    reg = FontRegistry()
    f1 = reg.load(gl_ctx, BUILTIN_FONT, size_px=24)
    f2 = reg.load(gl_ctx, BUILTIN_FONT, size_px=24)
    assert f1 is f2


def test_font_registry_different_sizes_are_different_objects(gl_ctx):
    reg = FontRegistry()
    f1 = reg.load(gl_ctx, BUILTIN_FONT, size_px=16)
    f2 = reg.load(gl_ctx, BUILTIN_FONT, size_px=32)
    assert f1 is not f2


def test_font_registry_get_by_name(gl_ctx):
    reg = FontRegistry()
    f = reg.load(gl_ctx, BUILTIN_FONT, size_px=20, name="hud")
    assert reg.get("hud") is f


def test_glyph_atlas_has_texture_after_build(gl_ctx):
    font = Font(gl_ctx, BUILTIN_FONT, size_px=24)
    assert isinstance(font.atlas, GlyphAtlas)
    assert font.atlas.texture is not None


def test_atlas_too_small_raises_for_huge_font(gl_ctx):
    # 256px font with ~224 glyphs at 1-pixel padding cannot fit in 1024x1024.
    with pytest.raises(AtlasTooSmallError):
        Font(gl_ctx, BUILTIN_FONT, size_px=256)


# --- Glyph info ----------------------------------------------------------

def test_get_glyph_A_has_positive_width(gl_ctx):
    font = Font(gl_ctx, BUILTIN_FONT, size_px=24)
    info = font.get_glyph("A")
    assert info is not None
    assert info.width > 0


def test_get_glyph_space_is_handled(gl_ctx):
    font = Font(gl_ctx, BUILTIN_FONT, size_px=24)
    info = font.get_glyph(" ")
    # Space may have zero-pixel width but must still expose advance > 0.
    assert info is not None
    assert info.advance > 0


def test_get_glyph_missing_char_returns_none_or_info(gl_ctx):
    font = Font(gl_ctx, BUILTIN_FONT, size_px=24)
    info = font.get_glyph("€")
    # Either is acceptable per spec; both must not raise.
    assert info is None or isinstance(info, GlyphInfo)


def test_glyph_uvs_in_unit_range(gl_ctx):
    font = Font(gl_ctx, BUILTIN_FONT, size_px=24)
    info = font.get_glyph("A")
    assert info is not None
    assert 0.0 <= info.uv_x <= 1.0
    assert 0.0 <= info.uv_y <= 1.0
    assert 0.0 <= info.uv_w <= 1.0
    assert 0.0 <= info.uv_h <= 1.0
    assert info.uv_x + info.uv_w <= 1.0
    assert info.uv_y + info.uv_h <= 1.0


def test_glyph_advance_positive_for_printable(gl_ctx):
    font = Font(gl_ctx, BUILTIN_FONT, size_px=24)
    for ch in "Hello":
        info = font.get_glyph(ch)
        assert info is not None
        assert info.advance > 0


# --- Text-label side-table ----------------------------------------------

def test_set_text_get_text_roundtrip():
    set_text(42, "hello")
    assert get_text(42) == "hello"


def test_get_text_unknown_entity_returns_empty():
    assert get_text(9999) == ""


def test_clear_text_removes_entry():
    set_text(7, "x")
    clear_text(7)
    assert get_text(7) == ""


def test_set_text_overwrites_previous_value():
    set_text(3, "first")
    set_text(3, "second")
    assert get_text(3) == "second"


# --- Setup ---------------------------------------------------------------

def _setup_app_with_text(gl_ctx):
    """Build a fake app with both setup_renderer_2d and setup_text wired in."""
    app = _fake_app(gl_ctx)
    setup_renderer_2d(app)
    text_setup = setup_text(app)
    return app, text_setup


def test_setup_text_returns_TextSetup_dataclass(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    assert isinstance(ts, TextSetup)
    assert isinstance(ts.font_registry, FontRegistry)
    assert isinstance(ts.text_batch, TextBatch)


def test_setup_text_inserts_world_resources(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    assert app.world.has_resource(FontRegistry)
    assert app.world.has_resource(TextBatch)


def test_setup_text_registers_post_render_system(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    # Look at scheduler entries for Phase.POST_RENDER — at least one system
    # registered against POST_RENDER after setup_text.
    sched = app.scheduler
    post = getattr(sched, "_systems", None)
    if post is not None:
        post_render_entries = [
            entry for phase, entries in post.items()
            if phase == Phase.POST_RENDER
            for entry in entries
        ]
        assert len(post_render_entries) >= 1
    else:  # pragma: no cover - scheduler internals changed
        # Fall back to running one tick and asserting no error.
        app.world.tick(0.016, scheduler=sched)


def test_setup_text_idempotent(gl_ctx):
    app, ts1 = _setup_app_with_text(gl_ctx)
    ts2 = setup_text(app)
    assert ts1 is ts2


def test_setup_text_requires_renderer_2d(gl_ctx):
    app = _fake_app(gl_ctx)
    with pytest.raises(RuntimeError):
        setup_text(app)


# --- TextBatch render ----------------------------------------------------

def test_text_batch_init_no_error(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    assert isinstance(ts.text_batch, TextBatch)


def test_text_batch_render_no_entities(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    ts.text_batch.render(app.world, ts.font_registry, 800, 600)
    assert ts.text_batch.last_glyph_count == 0
    assert ts.text_batch.last_draw_calls == 0


def test_text_batch_render_one_visible_label(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    font = load_font(app, BUILTIN_FONT, size_px=24)
    font_id = ts.font_registry.id_of(font)
    eid = app.world.spawn(
        Transform2D(x=50.0, y=100.0),
        TextLabel(font_id=font_id),
    )
    app.world.flush()
    set_text(eid, "Hello")
    ts.text_batch.render(app.world, ts.font_registry, 800, 600)
    assert ts.text_batch.last_glyph_count > 0
    assert ts.text_batch.last_draw_calls == 1


def test_text_batch_render_skips_invisible(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    font = load_font(app, BUILTIN_FONT, size_px=24)
    font_id = ts.font_registry.id_of(font)
    eid = app.world.spawn(
        Transform2D(x=10.0, y=10.0),
        TextLabel(font_id=font_id, visible=False),
    )
    app.world.flush()
    set_text(eid, "Hello")
    ts.text_batch.render(app.world, ts.font_registry, 800, 600)
    assert ts.text_batch.last_glyph_count == 0
    assert ts.text_batch.last_draw_calls == 0


def test_text_batch_render_skips_empty_string(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    font = load_font(app, BUILTIN_FONT, size_px=24)
    font_id = ts.font_registry.id_of(font)
    eid = app.world.spawn(
        Transform2D(x=10.0, y=10.0),
        TextLabel(font_id=font_id),
    )
    app.world.flush()
    # Do not set_text — the side-table has no entry, so the string is "".
    ts.text_batch.render(app.world, ts.font_registry, 800, 600)
    assert ts.text_batch.last_glyph_count == 0
    assert ts.text_batch.last_draw_calls == 0


def test_text_batch_render_entity_without_transform_warns_only(gl_ctx, caplog):
    app, ts = _setup_app_with_text(gl_ctx)
    font = load_font(app, BUILTIN_FONT, size_px=24)
    font_id = ts.font_registry.id_of(font)
    eid = app.world.spawn(TextLabel(font_id=font_id))
    app.world.flush()
    set_text(eid, "Hello")
    import logging
    with caplog.at_level(logging.WARNING):
        ts.text_batch.render(app.world, ts.font_registry, 800, 600)
    # Should not crash; should not draw.
    assert ts.text_batch.last_glyph_count == 0
    # A warning is emitted at least once. The renderer suppresses repeats.
    assert any("TextLabel" in r.getMessage() for r in caplog.records)


def test_text_batch_render_two_visible_labels_group_into_one_call(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    font = load_font(app, BUILTIN_FONT, size_px=24)
    font_id = ts.font_registry.id_of(font)
    e1 = app.world.spawn(Transform2D(x=0.0, y=0.0), TextLabel(font_id=font_id))
    e2 = app.world.spawn(Transform2D(x=0.0, y=40.0), TextLabel(font_id=font_id))
    app.world.flush()
    set_text(e1, "Hi")
    set_text(e2, "There")
    ts.text_batch.render(app.world, ts.font_registry, 800, 600)
    # Same font and same default color → one draw call.
    assert ts.text_batch.last_draw_calls == 1
    assert ts.text_batch.last_glyph_count > 0


def test_text_batch_handles_newline_and_tab(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    font = load_font(app, BUILTIN_FONT, size_px=24)
    font_id = ts.font_registry.id_of(font)
    eid = app.world.spawn(
        Transform2D(x=10.0, y=30.0),
        TextLabel(font_id=font_id),
    )
    app.world.flush()
    set_text(eid, "Line1\n\tLine2")
    ts.text_batch.render(app.world, ts.font_registry, 800, 600)
    assert ts.text_batch.last_glyph_count > 0


# --- Builtin font --------------------------------------------------------

def test_builtin_font_path_exists():
    assert os.path.exists(BUILTIN_FONT)


def test_font_loads_from_builtin_font_module(gl_ctx):
    font = Font(gl_ctx, BUILTIN_FONT, size_px=18)
    assert font.line_height > 0


def test_builtin_font_has_valid_ttf_magic_bytes():
    with open(BUILTIN_FONT, "rb") as f:
        magic = f.read(4)
    valid_magics = (b"\x00\x01\x00\x00", b"true", b"OTTO")
    assert magic in valid_magics, f"unexpected magic bytes: {magic!r}"


# --- load_font helper ----------------------------------------------------

def test_load_font_without_setup_text_raises(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_renderer_2d(app)
    with pytest.raises(RuntimeError):
        load_font(app, BUILTIN_FONT, size_px=20)


def test_load_font_returns_registered_font(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    font = load_font(app, BUILTIN_FONT, size_px=20, name="hud")
    assert ts.font_registry.get("hud") is font


def test_load_font_assigns_sequential_ids(gl_ctx):
    app, ts = _setup_app_with_text(gl_ctx)
    f1 = load_font(app, BUILTIN_FONT, size_px=12)
    f2 = load_font(app, BUILTIN_FONT, size_px=14)
    assert ts.font_registry.id_of(f1) == 0
    assert ts.font_registry.id_of(f2) == 1


def test_screen_projection_corners():
    from keel.text.text_renderer import _orthographic_screen_projection
    import numpy as np

    proj = _orthographic_screen_projection(800, 600)

    def transform(x, y):
        v = np.array([x, y, 0.0, 1.0], dtype=np.float32)
        result = proj @ v
        return result / result[3]  # perspective divide

    tl = transform(0, 0)
    tr = transform(800, 0)
    bl = transform(0, 600)
    br = transform(800, 600)

    np.testing.assert_allclose(tl[:2], [-1.0,  1.0], atol=1e-5)
    np.testing.assert_allclose(tr[:2], [ 1.0,  1.0], atol=1e-5)
    np.testing.assert_allclose(bl[:2], [-1.0, -1.0], atol=1e-5)
    np.testing.assert_allclose(br[:2], [ 1.0, -1.0], atol=1e-5)

    # w must be 1.0 everywhere — no perspective collapse
    for x, y in [(0,0), (400,300), (800,600), (100,100), (10,10)]:
        v = np.array([x, y, 0.0, 1.0], dtype=np.float32)
        result = proj @ v
        assert abs(result[3] - 1.0) < 1e-5, \
            f"w={result[3]} at ({x},{y}), expected 1.0"


def test_set_label_visible_round_trip(gl_ctx):
    from keel.text import set_label_visible

    app, ts = _setup_app_with_text(gl_ctx)
    font = load_font(app, BUILTIN_FONT, size_px=24)
    font_id = ts.font_registry.id_of(font)
    eid = app.world.spawn(
        keel.Transform2D(x=10.0, y=10.0),
        keel.TextLabel(font_id=font_id, visible=True),
    )
    app.world.flush()
    assert app.world.get(eid, keel.TextLabel)["visible"] is True
    set_label_visible(app.world, eid, False)
    assert app.world.get(eid, keel.TextLabel)["visible"] is False
    set_label_visible(app.world, eid, True)
    assert app.world.get(eid, keel.TextLabel)["visible"] is True
