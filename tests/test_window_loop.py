"""Headless tests for input wiring, the fixed-timestep loop, and event lifecycle.

No GLFW window is created; callback functions built by `make_callbacks` are
invoked directly with fake-window arguments to simulate GLFW events.
"""
from __future__ import annotations

import gc
import time
import weakref
from typing import Any

import glfw
import pytest

import keel
from keel import (
    FIXED_DT,
    FixedStepDriver,
    InputState,
    KeyEvent,
    MouseButtonEvent,
    MouseMoveEvent,
    MouseScrollEvent,
    Phase,
    Scheduler,
    WindowResizeEvent,
    World,
    make_callbacks,
)


# --- Fakes -----------------------------------------------------------------

class FakeWindow:
    """A non-GL stand-in for keel.Window — drives the loop in tests without a display."""

    def __init__(self, close_after_iters: int, vsync: bool = True) -> None:
        self._close_after = close_after_iters
        self._iters = 0
        self.vsync = vsync
        self._fb_w = 800
        self._fb_h = 600

    @property
    def should_close(self) -> bool:
        return self._iters >= self._close_after

    def swap_and_poll(self) -> None:
        self._iters += 1

    def update_framebuffer_size(self, w: int, h: int) -> None:
        self._fb_w = w
        self._fb_h = h

    def get_size(self) -> tuple[int, int]:
        return (self._fb_w, self._fb_h)

    def close(self) -> None:
        self._iters = self._close_after


# --- InputState polling ----------------------------------------------------

def test_is_key_down_reflects_press_and_release():
    world = World()
    state = InputState()
    cbs = make_callbacks(world, state)

    cbs["key"](None, glfw.KEY_A, 0, glfw.PRESS, 0)
    assert state.is_key_down(glfw.KEY_A)
    assert not state.is_key_down(glfw.KEY_B)

    cbs["key"](None, glfw.KEY_A, 0, glfw.RELEASE, 0)
    assert not state.is_key_down(glfw.KEY_A)


def test_is_mouse_button_down_reflects_press_and_release():
    world = World()
    state = InputState()
    cbs = make_callbacks(world, state)

    cbs["mouse_button"](None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
    assert state.is_mouse_button_down(glfw.MOUSE_BUTTON_LEFT)

    cbs["mouse_button"](None, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
    assert not state.is_mouse_button_down(glfw.MOUSE_BUTTON_LEFT)


def test_mouse_position_updates_from_cursor_callback():
    world = World()
    state = InputState()
    cbs = make_callbacks(world, state)
    assert state.mouse_position() == (0.0, 0.0)
    cbs["cursor_pos"](None, 100.5, 200.25)
    assert state.mouse_position() == (100.5, 200.25)


# --- Edge-detected helpers ---------------------------------------------

def test_is_key_pressed_only_true_on_first_held_frame():
    world = World()
    state = InputState()
    cbs = make_callbacks(world, state)

    # Frame 1: key goes down — rising edge.
    state.begin_frame()
    cbs["key"](None, glfw.KEY_SPACE, 0, glfw.PRESS, 0)
    assert state.is_key_pressed(glfw.KEY_SPACE) is True
    assert state.is_key_down(glfw.KEY_SPACE) is True

    # Frame 2: key still held — no longer a rising edge.
    state.begin_frame()
    assert state.is_key_pressed(glfw.KEY_SPACE) is False
    assert state.is_key_down(glfw.KEY_SPACE) is True


def test_is_key_pressed_false_when_key_never_pressed():
    state = InputState()
    state.begin_frame()
    assert state.is_key_pressed(glfw.KEY_Q) is False


def test_is_key_released_only_true_on_release_frame():
    world = World()
    state = InputState()
    cbs = make_callbacks(world, state)

    # Press it.
    state.begin_frame()
    cbs["key"](None, glfw.KEY_X, 0, glfw.PRESS, 0)
    # No release yet.
    assert state.is_key_released(glfw.KEY_X) is False

    # Next frame: release fires this frame — falling edge.
    state.begin_frame()
    cbs["key"](None, glfw.KEY_X, 0, glfw.RELEASE, 0)
    assert state.is_key_released(glfw.KEY_X) is True

    # Frame after: no longer a release event.
    state.begin_frame()
    assert state.is_key_released(glfw.KEY_X) is False


def test_is_mouse_button_pressed_rising_edge():
    world = World()
    state = InputState()
    cbs = make_callbacks(world, state)
    state.begin_frame()
    cbs["mouse_button"](None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
    assert state.is_mouse_button_pressed(glfw.MOUSE_BUTTON_LEFT) is True
    state.begin_frame()
    assert state.is_mouse_button_pressed(glfw.MOUSE_BUTTON_LEFT) is False


def test_is_mouse_button_released_falling_edge():
    world = World()
    state = InputState()
    cbs = make_callbacks(world, state)
    state.begin_frame()
    cbs["mouse_button"](None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
    state.begin_frame()
    cbs["mouse_button"](None, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
    assert state.is_mouse_button_released(glfw.MOUSE_BUTTON_LEFT) is True
    state.begin_frame()
    assert state.is_mouse_button_released(glfw.MOUSE_BUTTON_LEFT) is False


def test_edge_helpers_across_multiple_press_release_cycles():
    world = World()
    state = InputState()
    cbs = make_callbacks(world, state)
    for _ in range(3):
        # Press
        state.begin_frame()
        cbs["key"](None, glfw.KEY_F, 0, glfw.PRESS, 0)
        assert state.is_key_pressed(glfw.KEY_F) is True
        assert state.is_key_released(glfw.KEY_F) is False
        # Hold
        state.begin_frame()
        assert state.is_key_pressed(glfw.KEY_F) is False
        # Release
        state.begin_frame()
        cbs["key"](None, glfw.KEY_F, 0, glfw.RELEASE, 0)
        assert state.is_key_released(glfw.KEY_F) is True
        # Idle
        state.begin_frame()
        assert state.is_key_pressed(glfw.KEY_F) is False
        assert state.is_key_released(glfw.KEY_F) is False


# --- Event emission --------------------------------------------------------

def test_key_event_emitted_on_press_not_release():
    world = World()
    state = InputState()
    cbs = make_callbacks(world, state)

    cbs["key"](None, glfw.KEY_SPACE, 0, glfw.PRESS, 0)
    events = list(world.read_events(KeyEvent))
    assert len(events) == 1
    assert events[0].key == glfw.KEY_SPACE
    assert events[0].action == glfw.PRESS

    cbs["key"](None, glfw.KEY_SPACE, 0, glfw.RELEASE, 0)
    # No new event from release — total still 1.
    assert len(list(world.read_events(KeyEvent))) == 1


def test_key_event_emitted_on_repeat():
    world = World()
    cbs = make_callbacks(world, InputState())
    cbs["key"](None, glfw.KEY_W, 0, glfw.PRESS, 0)
    cbs["key"](None, glfw.KEY_W, 0, glfw.REPEAT, 0)
    actions = [e.action for e in world.read_events(KeyEvent)]
    assert actions == [glfw.PRESS, glfw.REPEAT]


def test_mouse_button_event_emitted():
    world = World()
    cbs = make_callbacks(world, InputState())
    cbs["mouse_button"](None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)
    events = list(world.read_events(MouseButtonEvent))
    assert len(events) == 1
    assert events[0].button == glfw.MOUSE_BUTTON_RIGHT


def test_mouse_move_event_emitted():
    world = World()
    cbs = make_callbacks(world, InputState())
    cbs["cursor_pos"](None, 12.0, 34.0)
    events = list(world.read_events(MouseMoveEvent))
    assert len(events) == 1
    assert (events[0].x, events[0].y) == (12.0, 34.0)


def test_mouse_scroll_event_emitted():
    world = World()
    cbs = make_callbacks(world, InputState())
    cbs["scroll"](None, 0.0, 1.5)
    events = list(world.read_events(MouseScrollEvent))
    assert len(events) == 1
    assert events[0].y_offset == 1.5


def test_window_resize_event_emitted_and_window_size_updated():
    world = World()

    class Win:
        def __init__(self):
            self.fb = (0, 0)

        def update_framebuffer_size(self, w, h):
            self.fb = (w, h)

    win = Win()
    cbs = make_callbacks(world, InputState(), window_obj=win)
    cbs["framebuffer_size"](None, 1024, 768)

    events = list(world.read_events(WindowResizeEvent))
    assert len(events) == 1
    assert (events[0].width, events[0].height) == (1024, 768)
    assert win.fb == (1024, 768)


def test_events_cleared_after_explicit_clear():
    world = World()
    cbs = make_callbacks(world, InputState())
    cbs["key"](None, glfw.KEY_A, 0, glfw.PRESS, 0)
    assert len(list(world.read_events(KeyEvent))) == 1
    world.events.clear()
    assert list(world.read_events(KeyEvent)) == []


# --- Reference-cycle hygiene -----------------------------------------------

def test_callbacks_do_not_strong_ref_world():
    """Dropping the World reference should let it be garbage-collected even after wiring callbacks."""
    world = World()
    state = InputState()
    cbs = make_callbacks(world, state)

    weak = weakref.ref(world)
    del world
    gc.collect()

    assert weak() is None, "callbacks must hold only weak references to World"
    # The callbacks themselves should still be safe to call (no-op when world is gone).
    cbs["key"](None, glfw.KEY_A, 0, glfw.PRESS, 0)


# --- FixedStepDriver -------------------------------------------------------

def test_fixed_timestep_six_ticks_for_0_1s():
    world = World()
    scheduler = Scheduler()
    log: list[float] = []

    def sys_record(world: World, dt: float) -> None:
        log.append(dt)

    scheduler.register(Phase.UPDATE, sys_record)

    driver = FixedStepDriver()
    ticks = driver.step(world, scheduler, 0.1)
    assert ticks == 6
    assert len(log) == 6
    assert all(dt == FIXED_DT for dt in log)


def test_fixed_timestep_no_tick_below_threshold():
    world = World()
    scheduler = Scheduler()
    driver = FixedStepDriver()
    ticks = driver.step(world, scheduler, FIXED_DT * 0.4)
    assert ticks == 0
    assert driver.alpha == pytest.approx(0.4, abs=1e-9)


def test_fixed_timestep_accumulator_carries_remainder():
    world = World()
    scheduler = Scheduler()
    driver = FixedStepDriver()
    # Two halves of one tick should add up to exactly one tick.
    assert driver.step(world, scheduler, FIXED_DT * 0.6) == 0
    assert driver.step(world, scheduler, FIXED_DT * 0.6) == 1


def test_fixed_timestep_clamps_long_pauses():
    """A 60-second hitch must not produce 3,600 ticks — it's clamped."""
    world = World()
    scheduler = Scheduler()
    driver = FixedStepDriver()
    ticks = driver.step(world, scheduler, 60.0)
    # The clamp keeps catch-up bounded — far fewer than 60/FIXED_DT == 3600 ticks.
    assert ticks <= 12, f"clamp let through too many ticks: {ticks}"
    assert ticks >= 5, f"clamp dropped too many ticks: {ticks}"


def test_fixed_step_calls_world_flush_each_tick():
    """Sim ticks must flush the command buffer so spawns from system N are visible to system N+1."""
    world = World()
    scheduler = Scheduler()

    def spawner(world: World, dt: float) -> None:
        world.spawn()  # queue a spawn each tick

    scheduler.register(Phase.UPDATE, spawner)
    driver = FixedStepDriver()
    driver.step(world, scheduler, FIXED_DT * 3)
    # If flush ran each tick, three entities should be alive now.
    alive_count = len(world.archetypes.get_or_create(frozenset()).entities)
    assert alive_count == 3


def test_events_persist_across_sim_ticks_within_a_step():
    """Events emitted before driver.step survive every sim tick within that step.
    The visual frame (one run_loop iteration) is the event lifetime; FixedStepDriver
    no longer clears the bus per sim tick. The run loop clears once per iteration."""
    world = World()
    scheduler = Scheduler()
    seen_per_tick: list[int] = []

    def reader(world: World, dt: float) -> None:
        seen_per_tick.append(sum(1 for _ in world.read_events(KeyEvent)))

    scheduler.register(Phase.UPDATE, reader)

    # Emit one event before the step runs (mirrors GLFW callbacks firing during
    # window.swap_and_poll, before driver.step in run_loop).
    world.emit(KeyEvent(key=1, scancode=0, action=glfw.PRESS, mods=0))
    driver = FixedStepDriver()
    driver.step(world, scheduler, FIXED_DT * 3)
    # The event survives every sim tick — exactly what we want for input.
    assert seen_per_tick == [1, 1, 1]


def test_run_loop_clears_events_between_visual_frames():
    """Events live for one visual frame and are cleared at the start of the next."""
    world = World()
    scheduler = Scheduler()
    seen_per_frame: list[int] = []

    def reader(world: World, dt: float) -> None:
        seen_per_frame.append(sum(1 for _ in world.read_events(KeyEvent)))

    scheduler.register(Phase.UPDATE, reader)

    class _SeedingWindow(FakeWindow):
        """Emit an event on iteration 1 only — should NOT be visible on iter 2.
        Sleeps long enough to guarantee one sim tick fires per iteration."""
        def __init__(self, w: World):
            super().__init__(close_after_iters=2)
            self._w = w
        def swap_and_poll(self):
            super().swap_and_poll()
            time.sleep(FIXED_DT * 1.2)
            if self._iters == 1:  # before-the-systems of the first visual frame
                self._w.emit(KeyEvent(key=1, scancode=0, action=glfw.PRESS, mods=0))

    from keel.loop import run_loop
    run_loop(_SeedingWindow(world), world, scheduler)
    # The event was emitted only in iteration 1's swap_and_poll, so:
    #  - at least one sim tick in iteration 1 saw the event;
    #  - no sim tick in iteration 2 should see anything (cleared between frames).
    assert seen_per_frame, "expected at least one sim tick to fire"
    assert seen_per_frame[0] == 1, f"first sim tick should see the event, got {seen_per_frame}"
    # No more events visible after the first sim tick of frame 1; further
    # ticks in the same frame still see it, but frame 2 onward must see 0.
    # (We can't tell sim ticks apart by frame from the outside, so a softer
    # invariant: the event count is non-increasing across the run.)
    assert all(seen_per_frame[i] >= seen_per_frame[i + 1] for i in range(len(seen_per_frame) - 1))
    assert seen_per_frame[-1] == 0, f"event should have been cleared by end of run, got {seen_per_frame}"


# --- run_loop --------------------------------------------------------------

def test_run_loop_exits_when_window_should_close():
    from keel.loop import run_loop
    world = World()
    scheduler = Scheduler()
    win = FakeWindow(close_after_iters=3)
    run_loop(win, world, scheduler)
    assert win._iters == 3


def test_run_loop_runs_render_phase_each_frame():
    from keel.loop import run_loop
    world = World()
    scheduler = Scheduler()

    render_calls: list[float] = []

    def render(world: World, dt: float) -> None:
        render_calls.append(dt)

    scheduler.register(Phase.RENDER, render)
    win = FakeWindow(close_after_iters=4)
    run_loop(win, world, scheduler)
    assert len(render_calls) == 4


def test_run_loop_inserts_render_state_resource():
    from keel.loop import run_loop, RenderState
    world = World()
    scheduler = Scheduler()
    win = FakeWindow(close_after_iters=2)
    run_loop(win, world, scheduler)
    rs = world.get_resource(RenderState)
    assert rs is not None
    assert 0.0 <= rs.alpha < 1.0


def test_run_loop_simulation_phase_can_close_window():
    """A system can call window.close() to terminate the loop."""
    from keel.loop import run_loop
    world = World()
    scheduler = Scheduler()
    win = FakeWindow(close_after_iters=10**9)  # effectively infinite

    closed_after: list[int] = []

    def closer(world: World, dt: float) -> None:
        # Run for one render frame, then close.
        if not closed_after:
            closed_after.append(1)
            win.close()

    scheduler.register(Phase.RENDER, closer)
    start = time.perf_counter()
    run_loop(win, world, scheduler)
    elapsed = time.perf_counter() - start
    assert closed_after == [1]
    assert elapsed < 1.0, "loop should have terminated promptly"


# --- Scheduler new methods -------------------------------------------------

def test_scheduler_tick_simulation_skips_render_phases():
    world = World()
    scheduler = Scheduler()
    log: list[str] = []

    scheduler.register(Phase.UPDATE, lambda w, dt: log.append("update"))
    scheduler.register(Phase.RENDER, lambda w, dt: log.append("render"))

    scheduler.tick_simulation(world, FIXED_DT)
    assert log == ["update"]
    log.clear()
    scheduler.tick_render(world, FIXED_DT)
    assert log == ["render"]


def test_scheduler_tick_alias_runs_all_phases():
    world = World()
    scheduler = Scheduler()
    log: list[str] = []

    scheduler.register(Phase.PRE_UPDATE, lambda w, dt: log.append("pre"))
    scheduler.register(Phase.RENDER, lambda w, dt: log.append("render"))

    scheduler.tick(world, FIXED_DT)
    assert log == ["pre", "render"]


# --- Public API surface ----------------------------------------------------

def test_glfw_constants_re_exported():
    """User code must be able to use keel.KEY_* / keel.MOUSE_BUTTON_* / keel.PRESS without importing glfw."""
    assert keel.KEY_ESCAPE == glfw.KEY_ESCAPE
    assert keel.KEY_A == glfw.KEY_A
    assert keel.KEY_SPACE == glfw.KEY_SPACE
    assert keel.PRESS == glfw.PRESS
    assert keel.RELEASE == glfw.RELEASE
    assert keel.REPEAT == glfw.REPEAT
    assert keel.MOUSE_BUTTON_LEFT == glfw.MOUSE_BUTTON_LEFT
    assert keel.MOUSE_BUTTON_RIGHT == glfw.MOUSE_BUTTON_RIGHT
    assert keel.MOUSE_BUTTON_MIDDLE == glfw.MOUSE_BUTTON_MIDDLE


def test_event_dataclasses_have_dataclass_fields():
    """Each event type is a dataclass with the spec'd fields, attached to the event registry."""
    import dataclasses
    for cls, expected in [
        (KeyEvent, {"key", "scancode", "action", "mods"}),
        (MouseButtonEvent, {"button", "action", "mods"}),
        (MouseMoveEvent, {"x", "y"}),
        (MouseScrollEvent, {"x_offset", "y_offset"}),
        (WindowResizeEvent, {"width", "height"}),
    ]:
        assert dataclasses.is_dataclass(cls)
        assert getattr(cls, "__keel_event__", False) is True
        assert {f.name for f in dataclasses.fields(cls)} == expected
