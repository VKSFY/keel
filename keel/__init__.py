"""Keel — a Python game engine. Top-level public surface."""
from __future__ import annotations

__version__ = "0.8.1"

import glfw as _glfw
import moderngl

# Re-exports use the `from X import Y as Y` alias form so Pylance treats them
# as intentional public re-exports rather than internal side imports — this is
# what makes `from keel import setup_renderer_2d` autocomplete correctly in
# editors that use Pyright (VSCode/Cursor) for static analysis.
from .assets import AssetHandle as AssetHandle
from .assets import AssetNotFoundError as AssetNotFoundError
from .assets import AssetRegistry as AssetRegistry
from .assets import FileWatcher as FileWatcher
from .assets import InvalidHandleError as InvalidHandleError
from .assets import NoLoaderError as NoLoaderError
from .assets import Scene as Scene
from .assets import SceneVersionError as SceneVersionError
from .assets import setup_assets as setup_assets

from .audio import AudioEngine as AudioEngine
from .audio import AudioSetup as AudioSetup
from .audio import AudioSource as AudioSource
from .audio import SoundHandle as SoundHandle
from .audio import play_music as play_music
from .audio import play_sound as play_sound
from .audio import set_volume as set_volume
from .audio import setup_audio as setup_audio
from .audio import stop_music as stop_music
from .audio import stop_sound as stop_sound

_setup_assets = setup_assets

from .components import MeshRenderer as MeshRenderer
from .components import Sprite as Sprite
from .components import TextLabel as TextLabel
from .components import Transform2D as Transform2D
from .components import Transform3D as Transform3D

from .physics import BodyType as BodyType
from .physics import Collider2D as Collider2D
from .physics import Collider3D as Collider3D
from .physics import CollisionEvent2D as CollisionEvent2D
from .physics import CollisionEvent3D as CollisionEvent3D
from .physics import Physics2D as Physics2D
from .physics import Physics3D as Physics3D
from .physics import RigidBody2D as RigidBody2D
from .physics import RigidBody3D as RigidBody3D
from .physics import ShapeType2D as ShapeType2D
from .physics import ShapeType3D as ShapeType3D
from .physics import setup_physics_2d as setup_physics_2d
from .physics import setup_physics_3d as setup_physics_3d

from .core import CommandBuffer as CommandBuffer
from .core import NULL_ENTITY as NULL_ENTITY
from .core import Optional as Optional
from .core import Phase as Phase
from .core import QueryResult as QueryResult
from .core import Without as Without
from .core import World as World
from .core import component as component
from .core import event as event
from .core.scheduler import Scheduler as Scheduler

from .gamepad import GamepadState as GamepadState
from .gamepad import setup_gamepad as setup_gamepad

from .input import GamepadAxisEvent as GamepadAxisEvent
from .input import GamepadButtonEvent as GamepadButtonEvent
from .input import InputState as InputState
from .input import KeyEvent as KeyEvent
from .input import MouseButtonEvent as MouseButtonEvent
from .input import MouseMoveEvent as MouseMoveEvent
from .input import MouseScrollEvent as MouseScrollEvent
from .input import WindowResizeEvent as WindowResizeEvent
from .input import make_callbacks as make_callbacks
from .input import wire_callbacks as wire_callbacks

from .loop import FIXED_DT as FIXED_DT
from .loop import FixedStepDriver as FixedStepDriver
from .loop import RenderState as RenderState
from .loop import run_loop as run_loop

from .renderer import Renderer2DSetup as Renderer2DSetup
from .renderer import Tilemap as Tilemap
from .renderer import TilemapSetup as TilemapSetup
from .renderer import setup_renderer_2d as setup_renderer_2d
from .renderer import setup_tilemap as setup_tilemap
from .renderer.camera2d import Camera2D as Camera2D

from .renderer3d import Camera3D as Camera3D
from .renderer3d import DirectionalLight as DirectionalLight
from .renderer3d import PointLight as PointLight
from .renderer3d import Renderer3D as Renderer3D
from .renderer3d import Renderer3DSetup as Renderer3DSetup
from .renderer3d import setup_renderer_3d as setup_renderer_3d

from .text import BUILTIN_FONT as BUILTIN_FONT
from .text import Font as Font
from .text import TextSetup as TextSetup
from .text import clear_text as clear_text
from .text import get_text as get_text
from .text import load_font as load_font
from .text import set_label_visible as set_label_visible
from .text import set_text as set_text
from .text import setup_text as setup_text

from .tools import setup_debug_draw as setup_debug_draw
from .tools import setup_inspector as setup_inspector
from .tools import setup_profiler as setup_profiler

from .window import Window as Window
from .window import glfw_initialized as glfw_initialized
from .window import shutdown_glfw as shutdown_glfw


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
        self._has_run: bool = False

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
        """Block until the window closes, then run shutdown hooks and terminate GLFW.

        Single-shot: calling run() again after the loop has exited (the window
        was closed, an exception propagated, etc.) raises RuntimeError. Build
        a new App instead.
        """
        if self._has_run:
            raise RuntimeError(
                "App.run() has already been called — the window and GLFW "
                "context were torn down on exit. Build a new App() instance."
            )
        self._has_run = True
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
    setattr(app, "_keel_dev_tools", tools)
    return tools


__all__ = [
    "App",
    "AssetHandle",
    "AssetNotFoundError",
    "AssetRegistry",
    "AudioEngine",
    "AudioSetup",
    "AudioSource",
    "BodyType",
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
    "ShapeType2D",
    "ShapeType3D",
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
