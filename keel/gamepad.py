"""Gamepad polling — slots 0..3, button + axis events, axis-emit deadzone."""
from __future__ import annotations

from typing import Any

import glfw

from .core import Phase
from .core.world import World
from .input import GamepadAxisEvent, GamepadButtonEvent


MAX_GAMEPADS = 4
NUM_BUTTONS = 15   # GLFW exposes 15 mapped buttons
NUM_AXES = 6       # LX, LY, RX, RY, LT, RT

# Axis must change by more than this since the last *emitted* value to fire
# a new GamepadAxisEvent. get_axis still reports the raw value.
_AXIS_EMIT_DEADZONE = 0.05


class GamepadState:
    """Polled snapshot of every connected gamepad in slots 0..3."""

    def __init__(self) -> None:
        self._connected = [False] * MAX_GAMEPADS
        self._buttons = [[False] * NUM_BUTTONS for _ in range(MAX_GAMEPADS)]
        self._axes = [[0.0] * NUM_AXES for _ in range(MAX_GAMEPADS)]
        # Last value at which each axis emitted an event (deadzone reference).
        self._last_emitted_axes = [[0.0] * NUM_AXES for _ in range(MAX_GAMEPADS)]

    def is_connected(self, gamepad_id: int) -> bool:
        """True iff slot `gamepad_id` reported a mapped gamepad last poll."""
        if not (0 <= gamepad_id < MAX_GAMEPADS):
            return False
        return self._connected[gamepad_id]

    def is_button_down(self, gamepad_id: int, button: int) -> bool:
        """True iff `button` is currently held on slot `gamepad_id`."""
        if not (0 <= gamepad_id < MAX_GAMEPADS) or not (0 <= button < NUM_BUTTONS):
            return False
        return self._buttons[gamepad_id][button]

    def get_axis(self, gamepad_id: int, axis: int) -> float:
        """Raw axis value in [-1, 1]; 0.0 for disconnected / out-of-range."""
        if not (0 <= gamepad_id < MAX_GAMEPADS) or not (0 <= axis < NUM_AXES):
            return 0.0
        if not self._connected[gamepad_id]:
            return 0.0
        return self._axes[gamepad_id][axis]

    def poll(self, world: World) -> None:
        """Refresh state from GLFW and emit transition events."""
        for slot in range(MAX_GAMEPADS):
            state = _safe_get_gamepad_state(slot)
            if state is None:
                if self._connected[slot]:
                    # On disconnect, synthesize RELEASE events for still-held
                    # buttons so game logic doesn't see stuck inputs.
                    for b in range(NUM_BUTTONS):
                        if self._buttons[slot][b]:
                            self._buttons[slot][b] = False
                            world.emit(GamepadButtonEvent(
                                gamepad_id=slot, button=b, action=glfw.RELEASE,
                            ))
                    for a in range(NUM_AXES):
                        self._axes[slot][a] = 0.0
                        self._last_emitted_axes[slot][a] = 0.0
                self._connected[slot] = False
                continue

            self._connected[slot] = True
            buttons = list(getattr(state, "buttons", ()))
            axes = list(getattr(state, "axes", ()))

            for b in range(min(NUM_BUTTONS, len(buttons))):
                pressed = _glfw_button_pressed(buttons[b])
                if pressed != self._buttons[slot][b]:
                    self._buttons[slot][b] = pressed
                    world.emit(GamepadButtonEvent(
                        gamepad_id=slot,
                        button=b,
                        action=glfw.PRESS if pressed else glfw.RELEASE,
                    ))

            for a in range(min(NUM_AXES, len(axes))):
                value = float(axes[a])
                self._axes[slot][a] = value
                if abs(value - self._last_emitted_axes[slot][a]) > _AXIS_EMIT_DEADZONE:
                    self._last_emitted_axes[slot][a] = value
                    world.emit(GamepadAxisEvent(
                        gamepad_id=slot, axis=a, value=value,
                    ))


def _safe_get_gamepad_state(slot: int) -> Any:
    """Call glfw.get_gamepad_state and swallow any backend error."""
    try:
        return glfw.get_gamepad_state(slot)
    except Exception:
        return None


def _glfw_button_pressed(value: Any) -> bool:
    """Some glfw bindings hand back bools, others PRESS/RELEASE ints."""
    if isinstance(value, bool):
        return value
    return int(value) == glfw.PRESS


def setup_gamepad(app: Any) -> GamepadState:
    """Insert GamepadState as a world resource and wire its PRE_UPDATE poll. Idempotent."""
    existing = getattr(app, "_keel_gamepad", None)
    if existing is not None:
        return existing

    state = GamepadState()
    app.world.insert_resource(state, type_=GamepadState)

    @app.system(Phase.PRE_UPDATE)
    def poll_gamepads(world: Any, dt: float, gp: GamepadState) -> None:
        gp.poll(world)

    app._keel_gamepad = state
    return state
