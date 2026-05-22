"""GLFW window + ModernGL context.

Owns one OpenGL 3.3 Core context per window. GLFW is initialized lazily on
first window creation and torn down via shutdown_glfw() (typically from
App.run when the loop exits). The framebuffer size — not the window size —
is the source of truth for GL viewport dimensions.
"""
from __future__ import annotations

import sys
from typing import Optional

import glfw
import moderngl


_GLFW_INITIALIZED: bool = False


def _glfw_error_callback(error_code: int, description) -> None:
    """GLFW error callback that raises a Python exception instead of failing silently."""
    if isinstance(description, bytes):
        description = description.decode("utf-8", errors="replace")
    raise RuntimeError(f"GLFW error 0x{error_code:x}: {description}")


def _ensure_glfw_initialized() -> None:
    """Initialize GLFW exactly once for the lifetime of the process."""
    global _GLFW_INITIALIZED
    if _GLFW_INITIALIZED:
        return
    glfw.set_error_callback(_glfw_error_callback)
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    _GLFW_INITIALIZED = True


def shutdown_glfw() -> None:
    """Terminate GLFW if it was initialized. Safe to call multiple times."""
    global _GLFW_INITIALIZED
    if _GLFW_INITIALIZED:
        glfw.terminate()
        _GLFW_INITIALIZED = False


def glfw_initialized() -> bool:
    """Return True if GLFW is currently initialized (used by tests)."""
    return _GLFW_INITIALIZED


class Window:
    """A single GLFW window owning one ModernGL OpenGL 3.3 Core context."""

    def __init__(self, title: str, width: int, height: int, vsync: bool = True) -> None:
        _ensure_glfw_initialized()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        self._title: str = title
        self._vsync: bool = vsync
        self._closed: bool = False

        self._glfw_window = glfw.create_window(width, height, title, None, None)
        if not self._glfw_window:
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._glfw_window)
        glfw.swap_interval(1 if vsync else 0)
        self._ctx: moderngl.Context = moderngl.create_context()

        fbw, fbh = glfw.get_framebuffer_size(self._glfw_window)
        self._fb_width: int = fbw
        self._fb_height: int = fbh

    @property
    def should_close(self) -> bool:
        """True if the window has been told to close (by the user or close())."""
        if self._closed or self._glfw_window is None:
            return True
        return bool(glfw.window_should_close(self._glfw_window))

    @property
    def ctx(self) -> moderngl.Context:
        """The single ModernGL context owned by this window."""
        return self._ctx

    @property
    def vsync(self) -> bool:
        """True if vertical sync was requested at construction."""
        return self._vsync

    @property
    def title(self) -> str:
        """The window's current title."""
        return self._title

    def swap_and_poll(self) -> None:
        """Swap the front/back buffers and poll GLFW events. Call once per visual frame."""
        glfw.swap_buffers(self._glfw_window)
        glfw.poll_events()

    def set_title(self, title: str) -> None:
        """Update the window's title bar text."""
        self._title = title
        glfw.set_window_title(self._glfw_window, title)

    def get_size(self) -> tuple[int, int]:
        """Return the framebuffer (drawable) size in pixels — use this for the GL viewport."""
        return (self._fb_width, self._fb_height)

    def update_framebuffer_size(self, width: int, height: int) -> None:
        """Record a new framebuffer size (called by the resize callback)."""
        self._fb_width = width
        self._fb_height = height

    def close(self) -> None:
        """Mark the window as closing so the main loop will exit on its next check."""
        self._closed = True
        if self._glfw_window is not None:
            glfw.set_window_should_close(self._glfw_window, True)

    def destroy(self) -> None:
        """Release the GLFW window. Safe to call more than once."""
        if self._glfw_window is not None:
            glfw.destroy_window(self._glfw_window)
            self._glfw_window = None
