"""Phase 11 — gamepad polling, event emission, and setup_gamepad wiring.

GLFW is mocked: real joystick subsystem is not exercised. Each test
constructs a fake `gamepad_state` object that mirrors GLFW's struct shape
(`.buttons` + `.axes`).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import glfw
import pytest

import pyge
from pyge import (
    GAMEPAD_AXIS_LEFT_X,
    GAMEPAD_AXIS_LEFT_Y,
    GAMEPAD_AXIS_RIGHT_TRIGGER,
    GAMEPAD_BUTTON_A,
    GAMEPAD_BUTTON_B,
    GamepadAxisEvent,
    GamepadButtonEvent,
    GamepadState,
    Phase,
    Scheduler,
    World,
    setup_gamepad,
)
from pyge.gamepad import NUM_AXES, NUM_BUTTONS


# --- Helpers --------------------------------------------------------------

def _gp_state(buttons=None, axes=None):
    """Build a fake `glfw.get_gamepad_state` return value."""
    return SimpleNamespace(
        buttons=list(buttons if buttons is not None else [glfw.RELEASE] * NUM_BUTTONS),
        axes=list(axes if axes is not None else [0.0] * NUM_AXES),
    )


def _slot0_only(state):
    """Return `state` for slot 0 and `None` for every other slot.

    Without this, a naive `return_value=state` would make every slot think
    the same pad is connected, multiplying every emitted event by 4.
    """
    def _stub(slot):
        return state if slot == 0 else None
    return _stub


def _fake_app():
    """Minimal app shape compatible with setup_gamepad."""
    sched = Scheduler()
    world = World()
    app = SimpleNamespace(
        world=world,
        _scheduler=sched,
        scheduler=sched,
    )

    def system(phase: Phase):
        def deco(fn):
            sched.register(phase, fn)
            return fn
        return deco

    app.system = system
    return app


# --- Connection state ----------------------------------------------------

def test_is_connected_false_when_glfw_returns_none():
    gp = GamepadState()
    with patch("pyge.gamepad.glfw.get_gamepad_state", return_value=None):
        gp.poll(World())
    assert gp.is_connected(0) is False


def test_is_connected_true_when_glfw_returns_state():
    gp = GamepadState()
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state())):
        gp.poll(World())
    assert gp.is_connected(0) is True


def test_is_connected_out_of_range_returns_false():
    gp = GamepadState()
    assert gp.is_connected(99) is False
    assert gp.is_connected(-1) is False


# --- Axis values ---------------------------------------------------------

def test_get_axis_returns_zero_for_disconnected():
    gp = GamepadState()
    with patch("pyge.gamepad.glfw.get_gamepad_state", return_value=None):
        gp.poll(World())
    assert gp.get_axis(0, GAMEPAD_AXIS_LEFT_X) == 0.0


def test_get_axis_returns_polled_value():
    gp = GamepadState()
    axes = [0.0] * NUM_AXES
    axes[GAMEPAD_AXIS_LEFT_X] = 0.42
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(axes=axes))):
        gp.poll(World())
    assert gp.get_axis(0, GAMEPAD_AXIS_LEFT_X) == pytest.approx(0.42)


def test_get_axis_out_of_range_returns_zero():
    gp = GamepadState()
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state())):
        gp.poll(World())
    assert gp.get_axis(0, 99) == 0.0
    assert gp.get_axis(99, GAMEPAD_AXIS_LEFT_X) == 0.0


# --- Button state --------------------------------------------------------

def test_is_button_down_starts_false():
    gp = GamepadState()
    assert gp.is_button_down(0, GAMEPAD_BUTTON_A) is False


def test_is_button_down_true_after_press():
    gp = GamepadState()
    buttons = [glfw.RELEASE] * NUM_BUTTONS
    buttons[GAMEPAD_BUTTON_A] = glfw.PRESS
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(buttons=buttons))):
        gp.poll(World())
    assert gp.is_button_down(0, GAMEPAD_BUTTON_A) is True


# --- Event emission ------------------------------------------------------

def _drain(world: World, evtype):
    """Read + then clear the world's event queues so the next poll starts clean."""
    events = list(world.read_events(evtype))
    world.events.clear()
    return events


def test_poll_emits_press_event_on_button_down():
    gp = GamepadState()
    world = World()

    # Tick 1: nothing pressed; no events.
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state())):
        gp.poll(world)
    _drain(world, GamepadButtonEvent)  # discard frame 1's empty drain

    # Tick 2: A pressed; one PRESS event.
    buttons = [glfw.RELEASE] * NUM_BUTTONS
    buttons[GAMEPAD_BUTTON_A] = glfw.PRESS
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(buttons=buttons))):
        gp.poll(world)
    events = _drain(world, GamepadButtonEvent)
    assert len(events) == 1
    assert events[0].gamepad_id == 0
    assert events[0].button == GAMEPAD_BUTTON_A
    assert events[0].action == glfw.PRESS


def test_poll_emits_release_event_on_button_up():
    gp = GamepadState()
    world = World()
    buttons = [glfw.RELEASE] * NUM_BUTTONS
    buttons[GAMEPAD_BUTTON_A] = glfw.PRESS

    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(buttons=buttons))):
        gp.poll(world)
    _drain(world, GamepadButtonEvent)

    buttons[GAMEPAD_BUTTON_A] = glfw.RELEASE
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(buttons=buttons))):
        gp.poll(world)
    events = _drain(world, GamepadButtonEvent)
    assert len(events) == 1
    assert events[0].action == glfw.RELEASE


def test_poll_no_event_when_button_state_unchanged():
    gp = GamepadState()
    world = World()
    buttons = [glfw.RELEASE] * NUM_BUTTONS
    buttons[GAMEPAD_BUTTON_A] = glfw.PRESS
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(buttons=buttons))):
        gp.poll(world)
    _drain(world, GamepadButtonEvent)

    # Same state -> no events the second poll.
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(buttons=buttons))):
        gp.poll(world)
    events = _drain(world, GamepadButtonEvent)
    assert events == []


def test_poll_emits_axis_event_when_change_exceeds_deadzone():
    gp = GamepadState()
    world = World()
    axes = [0.0] * NUM_AXES
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(axes=axes))):
        gp.poll(world)
    _drain(world, GamepadAxisEvent)

    axes[GAMEPAD_AXIS_LEFT_X] = 0.5
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(axes=axes))):
        gp.poll(world)
    events = _drain(world, GamepadAxisEvent)
    assert any(
        e.axis == GAMEPAD_AXIS_LEFT_X and e.value == pytest.approx(0.5)
        for e in events
    )


def test_poll_no_axis_event_when_change_under_deadzone():
    gp = GamepadState()
    world = World()
    axes = [0.0] * NUM_AXES
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(axes=axes))):
        gp.poll(world)
    _drain(world, GamepadAxisEvent)

    axes[GAMEPAD_AXIS_LEFT_X] = 0.02  # well under 0.05 deadzone
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(axes=axes))):
        gp.poll(world)
    events = _drain(world, GamepadAxisEvent)
    assert events == []
    # get_axis still reports the raw value despite no event firing.
    assert gp.get_axis(0, GAMEPAD_AXIS_LEFT_X) == pytest.approx(0.02)


# --- Error handling ------------------------------------------------------

def test_poll_handles_disconnected_gamepad_gracefully():
    gp = GamepadState()
    with patch("pyge.gamepad.glfw.get_gamepad_state", return_value=None):
        gp.poll(World())  # must not raise
    assert gp.is_connected(0) is False


def test_poll_handles_glfw_exception_gracefully():
    gp = GamepadState()
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=Exception("backend bug")):
        gp.poll(World())  # must not raise
    assert gp.is_connected(0) is False


def test_poll_polls_all_four_slots():
    """Slots 0..3 are polled, slot 4+ is not (only 4 supported)."""
    gp = GamepadState()
    calls: list[int] = []

    def record(slot):
        calls.append(slot)
        return None

    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=record):
        gp.poll(World())
    assert calls == [0, 1, 2, 3]


def test_disconnect_releases_held_buttons():
    """If a gamepad disappears mid-game, every still-held button gets a RELEASE event."""
    gp = GamepadState()
    world = World()
    buttons = [glfw.RELEASE] * NUM_BUTTONS
    buttons[GAMEPAD_BUTTON_A] = glfw.PRESS
    buttons[GAMEPAD_BUTTON_B] = glfw.PRESS
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(buttons=buttons))):
        gp.poll(world)
    _drain(world, GamepadButtonEvent)

    # Gamepad unplugged.
    with patch("pyge.gamepad.glfw.get_gamepad_state", return_value=None):
        gp.poll(world)
    events = _drain(world, GamepadButtonEvent)
    actions = {e.button for e in events if e.action == glfw.RELEASE}
    assert GAMEPAD_BUTTON_A in actions
    assert GAMEPAD_BUTTON_B in actions
    assert gp.is_connected(0) is False
    assert gp.is_button_down(0, GAMEPAD_BUTTON_A) is False


# --- setup_gamepad -------------------------------------------------------

def test_setup_gamepad_returns_state():
    app = _fake_app()
    state = setup_gamepad(app)
    assert isinstance(state, GamepadState)


def test_setup_gamepad_inserts_resource():
    app = _fake_app()
    setup_gamepad(app)
    assert app.world.has_resource(GamepadState)


def test_setup_gamepad_registers_pre_update_system():
    app = _fake_app()
    before = len(app.scheduler._systems[Phase.PRE_UPDATE])
    setup_gamepad(app)
    after = len(app.scheduler._systems[Phase.PRE_UPDATE])
    assert after == before + 1


def test_setup_gamepad_idempotent():
    app = _fake_app()
    s1 = setup_gamepad(app)
    s2 = setup_gamepad(app)
    assert s1 is s2
    # And no second system registered.
    assert len(app.scheduler._systems[Phase.PRE_UPDATE]) == 1


# --- Constants exposed at pyge level -------------------------------------

def test_gamepad_button_constants_match_glfw():
    assert pyge.GAMEPAD_BUTTON_A == glfw.GAMEPAD_BUTTON_A
    assert pyge.GAMEPAD_BUTTON_DPAD_UP == glfw.GAMEPAD_BUTTON_DPAD_UP


def test_gamepad_axis_constants_match_glfw():
    assert pyge.GAMEPAD_AXIS_LEFT_X == glfw.GAMEPAD_AXIS_LEFT_X
    assert pyge.GAMEPAD_AXIS_RIGHT_TRIGGER == glfw.GAMEPAD_AXIS_RIGHT_TRIGGER


# --- Event-type registration --------------------------------------------

def test_gamepad_button_event_is_registered_pyge_event():
    assert hasattr(GamepadButtonEvent, "__pyge_event__")


def test_gamepad_axis_event_is_registered_pyge_event():
    assert hasattr(GamepadAxisEvent, "__pyge_event__")


def test_is_button_down_false_after_release():
    """A second poll that drops the button must flip is_button_down back to False."""
    gp = GamepadState()
    world = World()
    buttons = [glfw.RELEASE] * NUM_BUTTONS
    buttons[GAMEPAD_BUTTON_A] = glfw.PRESS
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(buttons=buttons))):
        gp.poll(world)
    assert gp.is_button_down(0, GAMEPAD_BUTTON_A) is True
    buttons[GAMEPAD_BUTTON_A] = glfw.RELEASE
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(buttons=buttons))):
        gp.poll(world)
    assert gp.is_button_down(0, GAMEPAD_BUTTON_A) is False


def test_two_simultaneous_button_presses_emit_two_events():
    """Pressing A and B in the same frame should fire two GamepadButtonEvents."""
    gp = GamepadState()
    world = World()
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state())):
        gp.poll(world)
    _drain(world, GamepadButtonEvent)

    buttons = [glfw.RELEASE] * NUM_BUTTONS
    buttons[GAMEPAD_BUTTON_A] = glfw.PRESS
    buttons[GAMEPAD_BUTTON_B] = glfw.PRESS
    with patch("pyge.gamepad.glfw.get_gamepad_state", side_effect=_slot0_only(_gp_state(buttons=buttons))):
        gp.poll(world)
    events = _drain(world, GamepadButtonEvent)
    pressed = {(e.button, e.action) for e in events}
    assert (GAMEPAD_BUTTON_A, glfw.PRESS) in pressed
    assert (GAMEPAD_BUTTON_B, glfw.PRESS) in pressed
