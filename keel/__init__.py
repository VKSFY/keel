"""Keel — a Python game engine. Top-level public surface."""
from __future__ import annotations

__version__ = "0.1.0"

import glfw as _glfw
import moderngl

from .assets import (
    AssetHandle,
    AssetNotFoundError,
    AssetRegistry,
    FileWatcher,
    InvalidHandleError,
    NoLoaderError,
    Scene,
    SceneVersionError,
    setup_assets,
)
from .audio import (
    AudioEngine,
    AudioSetup,
    AudioSource,
    SoundHandle,
    play_music,
    play_sound,
    set_volume,
    setup_audio,
    stop_music,
    stop_sound,
)

_setup_assets = setup_assets
from .components import MeshRenderer, Sprite, TextLabel, Transform2D, Transform3D
from .physics import (
    Collider2D,
    Collider3D,
    CollisionEvent2D,
    CollisionEvent3D,
    Physics2D,
    Physics3D,
    RigidBody2D,
    RigidBody3D,
    setup_physics_2d,
    setup_physics_3d,
)
from .core import (
    CommandBuffer,
    NULL_ENTITY,
    Optional,
    Phase,
    QueryResult,
    Without,
    World,
    component,
    event,
)
from .core.scheduler import Scheduler
from .gamepad import GamepadState, setup_gamepad
from .input import (
    GamepadAxisEvent,
    GamepadButtonEvent,
    InputState,
    KeyEvent,
    MouseButtonEvent,
    MouseMoveEvent,
    MouseScrollEvent,
    WindowResizeEvent,
    make_callbacks,
    wire_callbacks,
)
from .loop import FIXED_DT, FixedStepDriver, RenderState, run_loop
from .renderer import (
    Renderer2DSetup,
    Tilemap,
    TilemapSetup,
    setup_renderer_2d,
    setup_tilemap,
)
from .renderer.camera2d import Camera2D
from .renderer3d import (
    Camera3D,
    DirectionalLight,
    PointLight,
    Renderer3D,
    Renderer3DSetup,
    setup_renderer_3d,
)
from .text import (
    BUILTIN_FONT,
    Font,
    TextSetup,
    clear_text,
    get_text,
    load_font,
    set_label_visible,
    set_text,
    setup_text,
)
from .tools import setup_debug_draw, setup_inspector, setup_profiler
from .window import Window, glfw_initialized, shutdown_glfw


# --- GLFW action constants -------------------------------------------------

PRESS: int = _glfw.PRESS
RELEASE: int = _glfw.RELEASE
REPEAT: int = _glfw.REPEAT

# --- Mouse buttons ---------------------------------------------------------

MOUSE_BUTTON_LEFT: int = _glfw.MOUSE_BUTTON_LEFT
MOUSE_BUTTON_RIGHT: int = _glfw.MOUSE_BUTTON_RIGHT
MOUSE_BUTTON_MIDDLE: int = _glfw.MOUSE_BUTTON_MIDDLE
MOUSE_BUTTON_4: int = _glfw.MOUSE_BUTTON_4
MOUSE_BUTTON_5: int = _glfw.MOUSE_BUTTON_5
MOUSE_BUTTON_6: int = _glfw.MOUSE_BUTTON_6
MOUSE_BUTTON_7: int = _glfw.MOUSE_BUTTON_7
MOUSE_BUTTON_8: int = _glfw.MOUSE_BUTTON_8

# --- Gamepad buttons + axes (GLFW aliases) --------------------------------

GAMEPAD_BUTTON_A: int = _glfw.GAMEPAD_BUTTON_A
GAMEPAD_BUTTON_B: int = _glfw.GAMEPAD_BUTTON_B
GAMEPAD_BUTTON_X: int = _glfw.GAMEPAD_BUTTON_X
GAMEPAD_BUTTON_Y: int = _glfw.GAMEPAD_BUTTON_Y
GAMEPAD_BUTTON_LEFT_BUMPER: int = _glfw.GAMEPAD_BUTTON_LEFT_BUMPER
GAMEPAD_BUTTON_RIGHT_BUMPER: int = _glfw.GAMEPAD_BUTTON_RIGHT_BUMPER
GAMEPAD_BUTTON_BACK: int = _glfw.GAMEPAD_BUTTON_BACK
GAMEPAD_BUTTON_START: int = _glfw.GAMEPAD_BUTTON_START
GAMEPAD_BUTTON_GUIDE: int = _glfw.GAMEPAD_BUTTON_GUIDE
GAMEPAD_BUTTON_LEFT_THUMB: int = _glfw.GAMEPAD_BUTTON_LEFT_THUMB
GAMEPAD_BUTTON_RIGHT_THUMB: int = _glfw.GAMEPAD_BUTTON_RIGHT_THUMB
GAMEPAD_BUTTON_DPAD_UP: int = _glfw.GAMEPAD_BUTTON_DPAD_UP
GAMEPAD_BUTTON_DPAD_RIGHT: int = _glfw.GAMEPAD_BUTTON_DPAD_RIGHT
GAMEPAD_BUTTON_DPAD_DOWN: int = _glfw.GAMEPAD_BUTTON_DPAD_DOWN
GAMEPAD_BUTTON_DPAD_LEFT: int = _glfw.GAMEPAD_BUTTON_DPAD_LEFT

GAMEPAD_AXIS_LEFT_X: int = _glfw.GAMEPAD_AXIS_LEFT_X
GAMEPAD_AXIS_LEFT_Y: int = _glfw.GAMEPAD_AXIS_LEFT_Y
GAMEPAD_AXIS_RIGHT_X: int = _glfw.GAMEPAD_AXIS_RIGHT_X
GAMEPAD_AXIS_RIGHT_Y: int = _glfw.GAMEPAD_AXIS_RIGHT_Y
GAMEPAD_AXIS_LEFT_TRIGGER: int = _glfw.GAMEPAD_AXIS_LEFT_TRIGGER
GAMEPAD_AXIS_RIGHT_TRIGGER: int = _glfw.GAMEPAD_AXIS_RIGHT_TRIGGER

_GAMEPAD_NAMES: list[str] = [
    "GAMEPAD_BUTTON_A", "GAMEPAD_BUTTON_B", "GAMEPAD_BUTTON_X", "GAMEPAD_BUTTON_Y",
    "GAMEPAD_BUTTON_LEFT_BUMPER", "GAMEPAD_BUTTON_RIGHT_BUMPER",
    "GAMEPAD_BUTTON_BACK", "GAMEPAD_BUTTON_START", "GAMEPAD_BUTTON_GUIDE",
    "GAMEPAD_BUTTON_LEFT_THUMB", "GAMEPAD_BUTTON_RIGHT_THUMB",
    "GAMEPAD_BUTTON_DPAD_UP", "GAMEPAD_BUTTON_DPAD_RIGHT",
    "GAMEPAD_BUTTON_DPAD_DOWN", "GAMEPAD_BUTTON_DPAD_LEFT",
    "GAMEPAD_AXIS_LEFT_X", "GAMEPAD_AXIS_LEFT_Y",
    "GAMEPAD_AXIS_RIGHT_X", "GAMEPAD_AXIS_RIGHT_Y",
    "GAMEPAD_AXIS_LEFT_TRIGGER", "GAMEPAD_AXIS_RIGHT_TRIGGER",
]

# --- Re-export every glfw.KEY_* constant under the keel.* namespace --------

_KEY_NAMES: list[str] = []
for _name in dir(_glfw):
    if _name.startswith("KEY_") and _name.isupper():
        globals()[_name] = getattr(_glfw, _name)
        _KEY_NAMES.append(_name)
del _name


class App:
    """Top-level entry point: window + world + scheduler + input wiring."""

    def __init__(
        self,
        title: str = "Keel",
        width: int = 800,
        height: int = 600,
        vsync: bool = True,
    ) -> None:
        self.world: World = World()
        self.window: Window = Window(title, width, height, vsync)
        self.input: InputState = InputState()
        # Expose the InputState as a world resource so the run_loop (and any
        # system using resource injection) can call input.begin_frame() to
        # drive edge-detected key/button helpers.
        self.world.insert_resource(self.input, type_=InputState)
        # One scheduler per app: share World.scheduler so @app.system(...) and
        # @world.system(...) target the same registry. Previously App owned a
        # separate Scheduler() that the loop drove, while @world.system fired
        # only when world.tick() was invoked manually — a silent footgun.
        self._scheduler: Scheduler = self.world.scheduler
        self._callbacks_keepalive = wire_callbacks(
            self.window._glfw_window, self.world, self.input, window_obj=self.window
        )
        self._shutdown_hooks: list = []

    @property
    def ctx(self) -> moderngl.Context:
        """The single ModernGL context owned by the window — pass to renderer systems."""
        return self.window.ctx

    @property
    def scheduler(self) -> Scheduler:
        """The scheduler driven by App.run."""
        return self._scheduler

    def system(self, phase: Phase):
        """Decorator: register a system function in the given phase."""
        def decorator(fn):
            self._scheduler.register(phase, fn)
            return fn
        return decorator

    def insert_resource(self, resource, *, type_=None) -> None:
        """Forward to world.insert_resource for ergonomic top-level access."""
        self.world.insert_resource(resource, type_=type_)

    def setup_assets(self, watch_dirs: list[str] | None = None) -> AssetRegistry:
        """Create / return the AssetRegistry, register default loaders + watcher."""
        return _setup_assets(self, watch_dirs)

    def add_shutdown_hook(self, hook) -> None:
        """Register a callable to run when App.run() exits (after the loop, before GLFW shutdown)."""
        self._shutdown_hooks.append(hook)

    def _run_shutdown_hooks(self) -> None:
        """Invoke every registered shutdown hook, swallowing per-hook exceptions."""
        for hook in self._shutdown_hooks:
            try:
                hook()
            except Exception:
                pass

    def run(self) -> None:
        """Block until the window closes, then run shutdown hooks and terminate GLFW."""
        try:
            run_loop(self.window, self.world, self._scheduler)
        finally:
            self._run_shutdown_hooks()
            self.window.destroy()
            shutdown_glfw()


class DevTools:
    """One-call developer tooling bundle: profiler, inspector, and (if physics is set up) debug draw."""

    def __init__(self, app: "App") -> None:
        from .tools.debug_draw import setup_debug_draw
        from .tools.inspector import setup_inspector
        from .tools.profiler import setup_profiler

        self.app = app
        self.profiler = setup_profiler(app)

        # Register debug_draw BEFORE the inspector so its POST_RENDER system
        # runs first; the inspector's ImGui submission then draws on top of
        # the GL line overlay rather than under it.
        try:
            from .physics import Physics2D as _Physics2D
            has_2d = app.world.has_resource(_Physics2D)
        except ImportError:  # pragma: no cover - physics is a hard dep but be safe
            has_2d = False

        if has_2d:
            self.debug_draw = setup_debug_draw(app)
        else:
            self.debug_draw = None

        self.inspector = setup_inspector(app)


def dev_tools(app: "App") -> DevTools:
    """Convenience: build (or return the cached) DevTools bundle for `app`."""
    existing = getattr(app, "_keel_dev_tools", None)
    if existing is not None:
        return existing
    tools = DevTools(app)
    app._keel_dev_tools = tools
    return tools


__all__ = [
    "App",
    "AssetHandle",
    "AssetNotFoundError",
    "AssetRegistry",
    "AudioEngine",
    "AudioSetup",
    "AudioSource",
    "BUILTIN_FONT",
    "Camera2D",
    "Camera3D",
    "Collider2D",
    "Collider3D",
    "CollisionEvent2D",
    "CollisionEvent3D",
    "CommandBuffer",
    "DevTools",
    "DirectionalLight",
    "FIXED_DT",
    "FileWatcher",
    "FixedStepDriver",
    "Font",
    "GamepadAxisEvent",
    "GamepadButtonEvent",
    "GamepadState",
    "InputState",
    "InvalidHandleError",
    "KeyEvent",
    "MouseButtonEvent",
    "MouseMoveEvent",
    "MeshRenderer",
    "MouseScrollEvent",
    "NULL_ENTITY",
    "NoLoaderError",
    "Optional",
    "Phase",
    "Physics2D",
    "Physics3D",
    "PointLight",
    "PRESS",
    "QueryResult",
    "RELEASE",
    "REPEAT",
    "Renderer2DSetup",
    "Renderer3D",
    "Renderer3DSetup",
    "RenderState",
    "RigidBody2D",
    "RigidBody3D",
    "Scene",
    "SceneVersionError",
    "Scheduler",
    "SoundHandle",
    "Sprite",
    "TextLabel",
    "TextSetup",
    "Tilemap",
    "TilemapSetup",
    "Transform2D",
    "Transform3D",
    "Window",
    "WindowResizeEvent",
    "Without",
    "World",
    "clear_text",
    "dev_tools",
    "get_text",
    "load_font",
    "play_music",
    "play_sound",
    "set_label_visible",
    "set_text",
    "set_volume",
    "setup_assets",
    "setup_audio",
    "setup_debug_draw",
    "setup_gamepad",
    "setup_inspector",
    "setup_physics_2d",
    "setup_physics_3d",
    "setup_profiler",
    "setup_renderer_2d",
    "setup_renderer_3d",
    "setup_text",
    "setup_tilemap",
    "stop_music",
    "stop_sound",
    "MOUSE_BUTTON_LEFT",
    "MOUSE_BUTTON_RIGHT",
    "MOUSE_BUTTON_MIDDLE",
    "MOUSE_BUTTON_4",
    "MOUSE_BUTTON_5",
    "MOUSE_BUTTON_6",
    "MOUSE_BUTTON_7",
    "MOUSE_BUTTON_8",
    "component",
    "event",
    "glfw_initialized",
    "make_callbacks",
    "run_loop",
    "shutdown_glfw",
    "wire_callbacks",
] + _KEY_NAMES + _GAMEPAD_NAMES
