"""Headless tests for Phase 7 — CLI, profiler, inspector, debug draw, DevTools."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import glfw
import moderngl
import pytest

import keel
from keel import (
    Collider2D,
    Phase,
    Physics2D,
    RigidBody2D,
    Scheduler,
    Transform2D,
    World,
    setup_physics_2d,
)
from keel.cli.commands import (
    BUILD_NOT_IMPLEMENTED_MSG,
    _build_parser,
    cmd_build,
    cmd_new,
    cmd_run,
    main,
)
from keel.cli.templates import MAIN_PY_TEMPLATE, PYPROJECT_TEMPLATE, README_TEMPLATE
from keel.physics.components2d import (
    BODY_TYPE_DYNAMIC as B2_DYNAMIC,
    BODY_TYPE_STATIC as B2_STATIC,
    SHAPE_TYPE_BOX as S2_BOX,
    SHAPE_TYPE_CIRCLE as S2_CIRCLE,
)
from keel.tools import (
    DebugDraw2D,
    FrameProfiler,
    ProfilerOverlay,
    SystemStats,
    WorldInspector,
    setup_debug_draw,
    setup_inspector,
    setup_profiler,
)


# --- GL fixture ----------------------------------------------------------

@pytest.fixture(scope="module")
def gl_ctx():
    """Hidden GLFW window + OpenGL 3.3 Core context, shared across the module."""
    if not glfw.init():
        pytest.skip("GLFW init failed (no display?)")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    if sys.platform == "darwin":
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    win = glfw.create_window(64, 64, "tooling-test", None, None)
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


def _fake_app(ctx=None) -> Any:
    """Minimal app shape: world + scheduler + system + add_shutdown_hook + ctx + window."""
    sched = Scheduler()
    world = World()
    hooks: list = []
    app = SimpleNamespace(
        world=world,
        _scheduler=sched,
        scheduler=sched,
        ctx=ctx,
        window=SimpleNamespace(get_size=lambda: (800, 600)),
        _shutdown_hooks=hooks,
    )

    def system(phase: Phase):
        def deco(fn):
            sched.register(phase, fn)
            return fn
        return deco

    app.system = system
    app.add_shutdown_hook = lambda fn: hooks.append(fn)
    return app


# --- CLI -----------------------------------------------------------------

def test_keel_new_creates_directory_layout(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = cmd_new("myproject")
    assert rc == 0
    root = tmp_path / "myproject"
    assert root.is_dir()
    assert (root / "main.py").is_file()
    assert (root / "pyproject.toml").is_file()
    assert (root / "README.md").is_file()
    assert (root / "assets" / ".gitkeep").is_file()
    assert (root / "scenes" / ".gitkeep").is_file()
    out = capsys.readouterr().out
    assert "created project" in out


def test_keel_new_main_py_contains_project_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd_new("hero_quest")
    text = (tmp_path / "hero_quest" / "main.py").read_text(encoding="utf-8")
    assert "hero_quest" in text
    assert "import keel" in text


def test_keel_new_pyproject_contains_project_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd_new("hero_quest")
    text = (tmp_path / "hero_quest" / "pyproject.toml").read_text(encoding="utf-8")
    assert "hero_quest" in text


def test_keel_new_refuses_existing_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "already_here").mkdir()
    rc = cmd_new("already_here")
    assert rc == 1
    err = capsys.readouterr().err
    assert "already_here" in err


def test_keel_build_prints_stub_and_exits_zero(capsys):
    rc = cmd_build()
    out = capsys.readouterr().out
    assert rc == 0
    assert BUILD_NOT_IMPLEMENTED_MSG in out


def _make_proc(poll_value=None):
    """Build a MagicMock that quacks like subprocess.Popen."""
    p = MagicMock(spec=subprocess.Popen)
    p.poll.return_value = poll_value
    p.wait.return_value = 0
    return p


class _ScriptedQueue:
    """Minimal queue-like with `get` and `get_nowait` driven by a list of side effects."""

    def __init__(self, gets: list, get_nowait_empty: bool = True) -> None:
        import queue as _q
        self._gets = list(gets)
        self._empty = _q.Empty
        self._get_nowait_empty = get_nowait_empty

    def get(self, timeout=None):
        if not self._gets:
            raise self._empty
        item = self._gets.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def get_nowait(self):
        if self._get_nowait_empty:
            raise self._empty
        return None


def test_keel_run_starts_subprocess_with_entry():
    """The loop spawns the entry script via the injected `spawn` callback."""
    from keel.cli.commands import _reload_loop

    spawn_calls: list[str] = []

    def spawn(entry):
        spawn_calls.append(entry)
        return _make_proc(poll_value=0)

    q = _ScriptedQueue([KeyboardInterrupt()])
    _reload_loop("main.py", q, spawn=spawn, terminate=lambda p: None, poll_timeout=0.05)

    assert spawn_calls == ["main.py"]


def test_keel_run_restarts_on_py_change(capsys):
    """A queued .py path triggers terminate(old) + spawn(new) and prints the banner."""
    from keel.cli.commands import _reload_loop

    procs = [_make_proc(poll_value=None), _make_proc(poll_value=None)]
    spawned: list[Any] = []

    def spawn(entry):
        p = procs.pop(0)
        spawned.append(p)
        return p

    terminated: list[Any] = []

    def terminate(proc):
        terminated.append(proc)

    # First get → return a path (triggers reload). Second get → KeyboardInterrupt.
    q = _ScriptedQueue(["changed.py", KeyboardInterrupt()])

    _reload_loop("main.py", q, spawn=spawn, terminate=terminate, poll_timeout=0.05)

    out = capsys.readouterr().out
    assert "reloading..." in out
    # The old proc must have been terminated before the new one was spawned.
    assert terminated[0] is spawned[0]
    assert len(spawned) == 2


def test_keel_run_terminates_subprocess_on_keyboard_interrupt():
    """On Ctrl+C the loop exits cleanly and terminate is called on the live proc."""
    from keel.cli.commands import _reload_loop

    proc = _make_proc(poll_value=None)
    terminated: list[Any] = []

    q = _ScriptedQueue([KeyboardInterrupt()])
    _reload_loop("main.py", q, spawn=lambda e: proc, terminate=terminated.append, poll_timeout=0.05)
    assert proc in terminated


def test_keel_run_terminate_uses_kill_after_timeout():
    """If a subprocess won't terminate, it must be killed — never zombied."""
    from keel.cli.commands import _terminate

    proc = MagicMock(spec=subprocess.Popen)
    proc.poll.return_value = None  # still alive
    proc.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="x", timeout=3.0),  # first wait times out
        0,  # second wait (after kill) returns
    ]
    _terminate(proc)
    assert proc.terminate.called
    assert proc.kill.called


def test_keel_run_missing_entry_exits_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = cmd_run("does_not_exist.py")
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_keel_main_unknown_command_exits_nonzero(capsys):
    # argparse exits with code 2 on unknown subcommand by raising SystemExit.
    with pytest.raises(SystemExit) as exc:
        main(["whatevenisthat"])
    assert exc.value.code != 0


def test_keel_main_no_command_returns_2(capsys):
    rc = main([])
    assert rc == 2


def test_argparse_builds_without_error():
    p = _build_parser()
    assert p.prog == "keel"


def test_templates_have_format_placeholders():
    rendered = MAIN_PY_TEMPLATE.format(project_name="zzz")
    assert "zzz" in rendered
    rendered2 = PYPROJECT_TEMPLATE.format(project_name="zzz")
    assert "zzz" in rendered2
    rendered3 = README_TEMPLATE.format(project_name="zzz")
    assert "zzz" in rendered3


# --- FrameProfiler -------------------------------------------------------

def test_profiler_begin_end_records_positive_duration():
    p = FrameProfiler()
    p.begin("sys")
    time.sleep(0.001)
    p.end("sys")
    stats = p.get_stats()
    assert "sys" in stats
    assert stats["sys"].avg_ms > 0.0


def test_profiler_get_stats_returns_system_stats():
    p = FrameProfiler()
    p.begin("a")
    p.end("a")
    stats = p.get_stats()
    s = stats["a"]
    assert isinstance(s, SystemStats)
    assert s.name == "a"


def test_profiler_avg_is_rolling_mean():
    p = FrameProfiler(history_size=5)
    # Inject deterministic fake durations by writing into the deque directly.
    from collections import deque
    p._buffers["x"] = deque([0.001, 0.002, 0.003], maxlen=5)
    s = p.get_stats()["x"]
    # avg of {1, 2, 3} ms
    assert pytest.approx(s.avg_ms, abs=1e-9) == 2.0
    assert pytest.approx(s.min_ms, abs=1e-9) == 1.0
    assert pytest.approx(s.max_ms, abs=1e-9) == 3.0
    assert pytest.approx(s.last_ms, abs=1e-9) == 3.0


def test_profiler_history_is_bounded():
    p = FrameProfiler(history_size=3)
    for _ in range(10):
        p.begin("x")
        p.end("x")
    assert len(p._buffers["x"]) == 3


def test_profiler_reset_clears_history():
    p = FrameProfiler()
    p.begin("a")
    p.end("a")
    assert "a" in p.get_stats()
    p.reset()
    assert p.get_stats() == {}


def test_scheduler_attach_profiler_records_system_calls():
    sched = Scheduler()
    world = World()
    profiler = FrameProfiler()
    sched.attach_profiler(profiler)

    def my_system(world, dt):
        time.sleep(0.001)

    sched.register(Phase.UPDATE, my_system)
    sched.tick(world, 0.016)
    stats = profiler.get_stats()
    assert "my_system" in stats
    assert stats["my_system"].avg_ms > 0.0


def test_scheduler_detach_profiler_stops_recording():
    sched = Scheduler()
    world = World()
    profiler = FrameProfiler()
    sched.attach_profiler(profiler)

    sched.register(Phase.UPDATE, lambda w, dt: None)
    sched.tick(world, 0.016)
    assert profiler.get_stats()  # something recorded

    sched.detach_profiler()
    profiler.reset()
    sched.tick(world, 0.016)
    assert profiler.get_stats() == {}  # nothing new


def test_setup_profiler_idempotent():
    app = _fake_app()
    a = setup_profiler(app)
    b = setup_profiler(app)
    assert a is b
    assert app.world.get_resource(FrameProfiler) is a


# --- WorldInspector + ProfilerOverlay (offscreen) -----------------------

def test_inspector_init(gl_ctx):
    insp = WorldInspector(gl_ctx)
    assert insp.visible is True


def test_inspector_toggle_changes_visible(gl_ctx):
    insp = WorldInspector(gl_ctx)
    initial = insp.visible
    insp.toggle()
    assert insp.visible != initial
    insp.toggle()
    assert insp.visible == initial


def test_inspector_render_with_empty_world(gl_ctx):
    from keel.tools.inspector import _ImGuiHost
    insp = WorldInspector(gl_ctx)
    host = _ImGuiHost.for_context(gl_ctx)
    world = World()
    host.begin_frame(64, 64, 0.016)
    insp.render(world)
    host.end_frame()  # must not raise


def test_inspector_render_with_populated_world(gl_ctx):
    from keel.tools.inspector import _ImGuiHost
    insp = WorldInspector(gl_ctx)
    host = _ImGuiHost.for_context(gl_ctx)
    world = World()
    world.spawn(Transform2D(x=10.0, y=20.0))
    world.spawn(Transform2D(), Collider2D(shape_type=S2_CIRCLE))
    world.flush()

    host.begin_frame(64, 64, 0.016)
    insp.render(world)
    host.end_frame()
    # The inspector recorded the total entity count it walked.
    assert insp.last_entity_count == 2


def test_inspector_render_skips_when_hidden(gl_ctx):
    from keel.tools.inspector import _ImGuiHost
    insp = WorldInspector(gl_ctx)
    insp.set_visible(False)
    host = _ImGuiHost.for_context(gl_ctx)
    host.begin_frame(64, 64, 0.016)
    insp.render(World())  # no-op
    host.end_frame()


def test_setup_inspector_registers_post_render_system(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_inspector(app)
    post = app._scheduler.systems(Phase.POST_RENDER)
    pre = app._scheduler.systems(Phase.PRE_UPDATE)
    assert len(post) == 1
    assert len(pre) == 1


def test_setup_inspector_idempotent(gl_ctx):
    app = _fake_app(gl_ctx)
    a = setup_inspector(app)
    b = setup_inspector(app)
    assert a is b
    assert len(app._scheduler.systems(Phase.POST_RENDER)) == 1
    assert len(app._scheduler.systems(Phase.PRE_UPDATE)) == 1


def test_profiler_overlay_render_with_empty_stats(gl_ctx):
    from keel.tools.inspector import _ImGuiHost
    overlay = ProfilerOverlay(gl_ctx)
    host = _ImGuiHost.for_context(gl_ctx)
    profiler = FrameProfiler()
    host.begin_frame(64, 64, 0.016)
    overlay.render(profiler)  # empty stats - must not raise
    host.end_frame()


def test_profiler_overlay_render_with_populated_stats(gl_ctx):
    from keel.tools.inspector import _ImGuiHost
    overlay = ProfilerOverlay(gl_ctx)
    host = _ImGuiHost.for_context(gl_ctx)
    profiler = FrameProfiler()
    profiler.begin("alpha")
    profiler.end("alpha")
    profiler.begin("beta")
    profiler.end("beta")
    host.begin_frame(64, 64, 0.016)
    overlay.render(profiler)
    host.end_frame()
    # Both systems should have been drawn as their own rows.
    assert overlay.last_rendered_count == 2


def test_profiler_overlay_render_with_none_profiler_is_noop(gl_ctx):
    overlay = ProfilerOverlay(gl_ctx)
    overlay.render(None)  # no host frame needed


# --- DebugDraw2D ---------------------------------------------------------

def test_debug_draw_init(gl_ctx):
    d = DebugDraw2D(gl_ctx)
    assert d.visible is False


def test_debug_draw_toggle(gl_ctx):
    d = DebugDraw2D(gl_ctx)
    d.toggle()
    assert d.visible is True
    d.toggle()
    assert d.visible is False


def test_debug_draw_render_no_entities(gl_ctx):
    import numpy as np
    from keel.renderer.camera2d import default_camera_matrix

    d = DebugDraw2D(gl_ctx)
    d.set_visible(True)
    d.render(World(), default_camera_matrix(800, 600))
    assert d.last_line_count == 0


def test_debug_draw_render_circle_collider(gl_ctx):
    from keel.renderer.camera2d import default_camera_matrix

    d = DebugDraw2D(gl_ctx)
    d.set_visible(True)
    world = World()
    world.spawn(
        Transform2D(x=0.0, y=0.0),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=4.0),
    )
    world.flush()
    d.render(world, default_camera_matrix(800, 600))
    assert d.last_line_count > 0


def test_debug_draw_render_box_collider(gl_ctx):
    from keel.renderer.camera2d import default_camera_matrix

    d = DebugDraw2D(gl_ctx)
    d.set_visible(True)
    world = World()
    world.spawn(
        Transform2D(x=10.0, y=5.0),
        RigidBody2D(body_type=B2_STATIC),
        Collider2D(shape_type=S2_BOX, width=4.0, height=2.0),
    )
    world.flush()
    d.render(world, default_camera_matrix(800, 600))
    # Box is 4 line segments.
    assert d.last_line_count == 4


def test_debug_draw_render_skipped_when_hidden(gl_ctx):
    from keel.renderer.camera2d import default_camera_matrix

    d = DebugDraw2D(gl_ctx)
    # default-hidden
    world = World()
    world.spawn(
        Transform2D(),
        RigidBody2D(mass=1.0),
        Collider2D(shape_type=S2_CIRCLE, radius=2.0),
    )
    world.flush()
    d.render(world, default_camera_matrix(800, 600))
    assert d.last_line_count == 0


def test_setup_debug_draw_idempotent(gl_ctx):
    app = _fake_app(gl_ctx)
    a = setup_debug_draw(app)
    b = setup_debug_draw(app)
    assert a is b


def test_setup_debug_draw_registers_post_render_system(gl_ctx):
    app = _fake_app(gl_ctx)
    setup_debug_draw(app)
    post = app._scheduler.systems(Phase.POST_RENDER)
    pre = app._scheduler.systems(Phase.PRE_UPDATE)
    assert len(post) == 1
    assert len(pre) == 1


# --- DevTools ------------------------------------------------------------

def test_dev_tools_returns_profiler_and_inspector(gl_ctx):
    app = _fake_app(gl_ctx)
    tools = keel.dev_tools(app)
    assert isinstance(tools.profiler, FrameProfiler)
    assert isinstance(tools.inspector, WorldInspector)


def test_dev_tools_with_physics_includes_debug_draw(gl_ctx):
    app = _fake_app(gl_ctx)
    phys = setup_physics_2d(app)
    try:
        tools = keel.dev_tools(app)
        assert tools.debug_draw is not None
        assert isinstance(tools.debug_draw, DebugDraw2D)
    finally:
        phys.cleanup()


def test_dev_tools_without_physics_has_no_debug_draw(gl_ctx):
    app = _fake_app(gl_ctx)
    tools = keel.dev_tools(app)
    assert tools.debug_draw is None


def test_dev_tools_idempotent(gl_ctx):
    app = _fake_app(gl_ctx)
    a = keel.dev_tools(app)
    b = keel.dev_tools(app)
    assert a is b
