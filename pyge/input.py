"""Input event types, InputState polling, and GLFW callback wiring.

Two complementary input surfaces are exposed:

  * Events — KeyEvent / MouseButtonEvent / MouseMoveEvent / MouseScrollEvent
    / WindowResizeEvent — emitted into the World's EventBus, drained each
    frame by systems via world.read_events.
  * InputState — a stateful poller for "is key currently down" / "where is
    the mouse" without scanning event queues.

GLFW callbacks update both. Callbacks hold weakrefs to the World and
InputState to avoid a strong-reference cycle through the GLFW window.
"""
from __future__ import annotations

import weakref
from typing import Any, Callable

import glfw

from .core import event


@event
class KeyEvent:
    """Key transition: emitted on PRESS and REPEAT, never on RELEASE."""
    key: int
    scancode: int
    action: int
    mods: int


@event
class MouseButtonEvent:
    """Mouse button transition: emitted on every PRESS / RELEASE."""
    button: int
    action: int
    mods: int


@event
class MouseMoveEvent:
    """Cursor moved within the window's content area."""
    x: float
    y: float


@event
class MouseScrollEvent:
    """Scroll wheel motion in window coordinates."""
    x_offset: float
    y_offset: float


@event
class WindowResizeEvent:
    """Framebuffer was resized — width/height are in pixels, not screen coords."""
    width: int
    height: int


@event
class GamepadButtonEvent:
    """Gamepad button transition. Emitted on every PRESS / RELEASE state flip."""
    gamepad_id: int
    button: int
    action: int


@event
class GamepadAxisEvent:
    """Gamepad axis moved past the 0.05 emit-deadzone since the last poll."""
    gamepad_id: int
    axis: int
    value: float


class InputState:
    """Stateful, poll-friendly view of currently-pressed keys/buttons and cursor position."""

    __slots__ = (
        "_keys_down",
        "_mouse_buttons_down",
        "_prev_keys_down",
        "_prev_mouse_buttons_down",
        "_mouse_x",
        "_mouse_y",
        "_scroll_x",
        "_scroll_y",
        "__weakref__",
    )

    def __init__(self) -> None:
        self._keys_down: set[int] = set()
        self._mouse_buttons_down: set[int] = set()
        # Last frame's state; begin_frame() snapshots into these so the rising-
        # and falling-edge helpers (is_key_pressed / is_key_released etc.) can
        # diff against them.
        self._prev_keys_down: set[int] = set()
        self._prev_mouse_buttons_down: set[int] = set()
        self._mouse_x: float = 0.0
        self._mouse_y: float = 0.0
        # Scroll deltas accumulate between consume_scroll() calls — the ImGui
        # host drains them once per frame.
        self._scroll_x: float = 0.0
        self._scroll_y: float = 0.0

    def is_key_down(self, key: int) -> bool:
        """Return True if `key` is currently held."""
        return key in self._keys_down

    def is_mouse_button_down(self, button: int) -> bool:
        """Return True if the given mouse button is currently held."""
        return button in self._mouse_buttons_down

    def is_key_pressed(self, key: int) -> bool:
        """True only on the frame `key` transitions from up to down."""
        return key in self._keys_down and key not in self._prev_keys_down

    def is_key_released(self, key: int) -> bool:
        """True only on the frame `key` transitions from down to up."""
        return key in self._prev_keys_down and key not in self._keys_down

    def is_mouse_button_pressed(self, button: int) -> bool:
        """Rising edge for mouse buttons."""
        return button in self._mouse_buttons_down and button not in self._prev_mouse_buttons_down

    def is_mouse_button_released(self, button: int) -> bool:
        """Falling edge for mouse buttons."""
        return button in self._prev_mouse_buttons_down and button not in self._mouse_buttons_down

    def begin_frame(self) -> None:
        """Snapshot current input state into `_prev_*`. Called by run_loop before poll."""
        self._prev_keys_down = set(self._keys_down)
        self._prev_mouse_buttons_down = set(self._mouse_buttons_down)

    def mouse_position(self) -> tuple[float, float]:
        """Return the most recent (x, y) cursor position in window coordinates."""
        return (self._mouse_x, self._mouse_y)

    def consume_scroll(self) -> tuple[float, float]:
        """Return the accumulated (x, y) scroll deltas since the last call and reset to zero."""
        x, y = self._scroll_x, self._scroll_y
        self._scroll_x = 0.0
        self._scroll_y = 0.0
        return (x, y)

    def _on_key(self, key: int, action: int) -> None:
        if action == glfw.PRESS:
            self._keys_down.add(key)
        elif action == glfw.RELEASE:
            self._keys_down.discard(key)

    def _on_mouse_button(self, button: int, action: int) -> None:
        if action == glfw.PRESS:
            self._mouse_buttons_down.add(button)
        elif action == glfw.RELEASE:
            self._mouse_buttons_down.discard(button)

    def _on_mouse_move(self, x: float, y: float) -> None:
        self._mouse_x = x
        self._mouse_y = y

    def _on_scroll(self, x: float, y: float) -> None:
        self._scroll_x += x
        self._scroll_y += y


def make_callbacks(
    world: Any,
    input_state: InputState,
    window_obj: Any | None = None,
) -> dict[str, Callable]:
    """Build GLFW-shaped callbacks. Headless-testable — call them with fake args."""
    world_ref = weakref.ref(world)
    input_ref = weakref.ref(input_state)
    window_obj_ref = weakref.ref(window_obj) if window_obj is not None else None

    def key_cb(_glfw_window, key, scancode, action, mods):
        s = input_ref()
        if s is not None:
            s._on_key(key, action)
        if action == glfw.RELEASE:
            return
        w = world_ref()
        if w is not None:
            w.emit(KeyEvent(key=key, scancode=scancode, action=action, mods=mods))

    def mouse_button_cb(_glfw_window, button, action, mods):
        s = input_ref()
        if s is not None:
            s._on_mouse_button(button, action)
        w = world_ref()
        if w is not None:
            w.emit(MouseButtonEvent(button=button, action=action, mods=mods))

    def cursor_pos_cb(_glfw_window, x, y):
        s = input_ref()
        if s is not None:
            s._on_mouse_move(float(x), float(y))
        w = world_ref()
        if w is not None:
            w.emit(MouseMoveEvent(x=float(x), y=float(y)))

    def scroll_cb(_glfw_window, x_offset, y_offset):
        s = input_ref()
        if s is not None:
            s._on_scroll(float(x_offset), float(y_offset))
        w = world_ref()
        if w is not None:
            w.emit(MouseScrollEvent(x_offset=float(x_offset), y_offset=float(y_offset)))

    def framebuffer_size_cb(_glfw_window, width, height):
        win = window_obj_ref() if window_obj_ref is not None else None
        if win is not None:
            win.update_framebuffer_size(int(width), int(height))
        w = world_ref()
        if w is not None:
            w.emit(WindowResizeEvent(width=int(width), height=int(height)))

    return {
        "key": key_cb,
        "mouse_button": mouse_button_cb,
        "cursor_pos": cursor_pos_cb,
        "scroll": scroll_cb,
        "framebuffer_size": framebuffer_size_cb,
    }


def wire_callbacks(
    glfw_window: Any,
    world: Any,
    input_state: InputState,
    window_obj: Any | None = None,
) -> dict[str, Callable]:
    """Wire every GLFW input/resize callback for `glfw_window` and return the kept-alive callbacks."""
    cbs = make_callbacks(world, input_state, window_obj=window_obj)
    glfw.set_key_callback(glfw_window, cbs["key"])
    glfw.set_mouse_button_callback(glfw_window, cbs["mouse_button"])
    glfw.set_cursor_pos_callback(glfw_window, cbs["cursor_pos"])
    glfw.set_scroll_callback(glfw_window, cbs["scroll"])
    glfw.set_framebuffer_size_callback(glfw_window, cbs["framebuffer_size"])
    return cbs
