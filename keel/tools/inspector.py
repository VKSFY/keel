"""WorldInspector + ProfilerOverlay — ImGui-driven dev tooling.

Both tools share one ImGui context per app, owned by `_ImGuiHost`. The host
takes care of creating the context, wiring a ProgrammablePipelineRenderer
against the existing ModernGL context, and bracketing draw calls with
`new_frame()` / `render()` each visual frame. WorldInspector and
ProfilerOverlay are independent windows drawn between those brackets.

`setup_inspector` registers a single PRE_UPDATE input system (F1 toggles
the inspector, F2 the profiler overlay) and a single POST_RENDER system
that runs both windows. F1 and F2 are toggled via edge-detected
`app.input.is_key_down` polling. KeyEvents are not used because input
events can be dropped on visual frames where no sim tick fires.
"""
from __future__ import annotations

from typing import Any

from ..core import Phase
from .profiler import FrameProfiler


def _require_imgui():
    """Import imgui-bundle and find the OpenGL backend renderer across versions."""
    try:
        from imgui_bundle import imgui
    except ImportError as e:
        raise ImportError(
            "install imgui-bundle: pip install 'imgui-bundle==1.5.2'"
        ) from e

    # imgui-bundle reorganized its python backend layout between 1.5.x and the
    # 1.6+/1.9x releases. Try every known location for the programmable GL
    # renderer; bail with a helpful message if none match.
    import importlib
    # imgui-bundle reorganized paths between releases. Known layouts:
    #   1.5.x : python_backends.opengl_backend
    #   1.6+  : python_backends.opengl_backend_programmable
    candidates = (
        "imgui_bundle.python_backends.opengl_backend",
        "imgui_bundle.python_backends.opengl_backend_programmable",
        "imgui_bundle.python_backends.opengl3_backend",
        "imgui_bundle.python_backends_disabled.opengl_backend",
    )
    failures: list[str] = []
    for module_path in candidates:
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError:
            # Path simply doesn't exist in this version — try the next one.
            continue
        except ImportError as e:
            # Path exists but a dep it imports is missing (e.g. PyOpenGL).
            # That's the user-actionable failure; record and report it below.
            failures.append(f"{module_path}: {e}")
            continue
        renderer = getattr(mod, "ProgrammablePipelineRenderer", None)
        if renderer is not None:
            return imgui, renderer

    if failures:
        raise ImportError(
            "imgui-bundle's OpenGL backend is present but failed to import:\n"
            + "\n".join(f"  - {f}" for f in failures)
            + "\nA missing 'OpenGL' module usually means PyOpenGL isn't installed:\n"
            "  pip install PyOpenGL"
        )
    raise ImportError(
        "imgui-bundle is installed but its OpenGL backend layout is unfamiliar.\n"
        "Keel supports the layouts in 1.5.x and 1.6+/1.9x releases.\n"
        "Try: pip install --only-binary=:all: imgui-bundle"
    )


class _ImGuiHost:
    """Owns the ImGui context, font texture, and ModernGL backend renderer."""

    _shared: dict[int, "_ImGuiHost"] = {}

    def __init__(self, ctx: Any) -> None:
        imgui, ProgrammablePipelineRenderer = _require_imgui()
        self.imgui = imgui
        self.ctx = ctx
        self._context = imgui.create_context()
        io = imgui.get_io()
        io.display_size = imgui.ImVec2(800.0, 600.0)
        # Build the font atlas before constructing the renderer (which uploads it).
        try:
            io.fonts.get_tex_data_as_rgba32()
        except Exception:
            pass
        self._renderer = ProgrammablePipelineRenderer()
        self._frame_active = False
        # Track previous mouse-button state so we only forward transitions to
        # imgui's event-based input API. Keys: 0=LEFT, 1=RIGHT, 2=MIDDLE.
        self._last_mouse_buttons: dict[int, bool] = {}

    @classmethod
    def for_context(cls, ctx: Any) -> "_ImGuiHost":
        """Return (or build) the singleton ImGui host for `ctx`."""
        key = id(ctx)
        host = cls._shared.get(key)
        if host is None:
            host = cls(ctx)
            cls._shared[key] = host
        return host

    def begin_frame(
        self,
        viewport_w: int,
        viewport_h: int,
        dt: float,
        *,
        input_state: Any = None,
    ) -> None:
        """Start an ImGui frame, optionally forwarding input from a Keel InputState.

        Without `input_state`, ImGui has no idea where the mouse is, whether
        any buttons are held, or that the wheel moved — so all of its widgets
        (scrollbars, buttons, tree nodes) are inert. `input_state` should be
        the same object as `app.input`.
        """
        if self._frame_active:
            return
        io = self.imgui.get_io()
        w = max(1, int(viewport_w))
        h = max(1, int(viewport_h))
        io.display_size = self.imgui.ImVec2(float(w), float(h))
        io.delta_time = max(float(dt), 1.0 / 60.0)

        if input_state is not None:
            self._forward_input(io, input_state)

        self.imgui.new_frame()
        self._frame_active = True

    def _forward_input(self, io: Any, input_state: Any) -> None:
        """Push mouse position, button transitions, and wheel deltas into ImGui IO."""
        # Mouse position. GLFW gives top-left origin; ImGui expects the same.
        try:
            mx, my = input_state.mouse_position()
            io.add_mouse_pos_event(float(mx), float(my))
        except Exception:
            pass

        # Mouse buttons — fire on transitions only, otherwise imgui spuriously
        # detects new clicks every frame the button stays down.
        try:
            import glfw
            for imgui_button, glfw_button in (
                (0, glfw.MOUSE_BUTTON_LEFT),
                (1, glfw.MOUSE_BUTTON_RIGHT),
                (2, glfw.MOUSE_BUTTON_MIDDLE),
            ):
                cur = bool(input_state.is_mouse_button_down(glfw_button))
                last = self._last_mouse_buttons.get(imgui_button, False)
                if cur != last:
                    io.add_mouse_button_event(imgui_button, cur)
                    self._last_mouse_buttons[imgui_button] = cur
        except Exception:
            pass

        # Scroll wheel. consume_scroll() drains and resets, so each frame
        # only reports the deltas accumulated since the last begin_frame.
        try:
            sx, sy = input_state.consume_scroll()
            if sx != 0.0 or sy != 0.0:
                io.add_mouse_wheel_event(float(sx), float(sy))
        except Exception:
            pass

    def end_frame(self) -> None:
        """Finalize and submit the ImGui frame."""
        if not self._frame_active:
            return
        self.imgui.render()
        self._renderer.render(self.imgui.get_draw_data())
        self._frame_active = False

    def shutdown(self) -> None:
        """Tear down the renderer + ImGui context."""
        try:
            self._renderer.shutdown()
        except Exception:
            pass
        try:
            self.imgui.destroy_context(self._context)
        except Exception:
            pass


class WorldInspector:
    """Live ImGui inspector showing entity / archetype / component data."""

    def __init__(self, ctx: Any) -> None:
        self._host: _ImGuiHost = _ImGuiHost.for_context(ctx)
        self.imgui = self._host.imgui
        self._visible: bool = True
        self._filter: str = ""
        self._selected_entity: int | None = None
        # Updated by render() — last entity total the inspector saw. Useful for
        # tests that want to confirm the inspector actually walked the world.
        self.last_entity_count: int = 0

    @property
    def visible(self) -> bool:
        """Whether the inspector window draws each frame."""
        return self._visible

    def toggle(self) -> None:
        """Flip the visibility flag."""
        self._visible = not self._visible

    def set_visible(self, visible: bool) -> None:
        """Set the visibility flag explicitly."""
        self._visible = bool(visible)

    def render(self, world: Any) -> None:
        """Draw the inspector window. Caller is responsible for new_frame/render."""
        if not self._visible:
            return
        imgui = self.imgui
        try:
            try:
                imgui.set_next_window_size(
                    imgui.ImVec2(380.0, 480.0),
                    int(imgui.Cond_.first_use_ever),
                )
            except Exception:
                pass  # size hint is optional
            # Don't pass p_open — its return semantics drift between
            # imgui-bundle versions and we don't need the close-button X
            # since F1 already toggles visibility.
            result = imgui.begin("World Inspector")
            shown = bool(result[0]) if isinstance(result, tuple) else bool(result)
            if shown:
                try:
                    self._draw_contents(world)
                except Exception:
                    pass  # don't let widget errors break the begin/end pair
            imgui.end()
        except Exception:
            try:
                imgui.end()
            except Exception:
                pass

    def _draw_contents(self, world: Any) -> None:
        imgui = self.imgui
        archetypes = world.archetypes.all_archetypes()
        total_entities = sum(arch.length for arch in archetypes)
        self.last_entity_count = total_entities
        non_empty = sum(1 for arch in archetypes if arch.length > 0)
        imgui.text(f"Entities: {total_entities}")
        imgui.text(f"Archetypes: {non_empty} (registered: {len(archetypes)})")
        imgui.separator()

        try:
            changed, new_filter = imgui.input_text("Filter", self._filter)
            if changed:
                self._filter = new_filter
        except Exception:
            pass
        imgui.separator()

        # Scrollable child region so the entity list can grow past the window
        # height without clipping. begin_child returns a bool; end_child must
        # be called unconditionally after.
        child_open = False
        try:
            child_open = imgui.begin_child("entity_list", imgui.ImVec2(0.0, 0.0))
        except Exception:
            pass
        try:
            if child_open:
                for arch in archetypes:
                    n = arch.length
                    if n == 0:
                        continue
                    comp_names = sorted(c.__name__ for c in arch.component_types)
                    if self._filter:
                        needle = self._filter.lower()
                        if not any(needle in name.lower() for name in comp_names):
                            continue
                    for i in range(n):
                        eid = int(arch.entities[i])
                        label = f"#{eid}: {', '.join(comp_names) or '(no components)'}"
                        if imgui.tree_node(label):
                            self._draw_entity_components(arch, i)
                            imgui.tree_pop()
        finally:
            try:
                imgui.end_child()
            except Exception:
                pass

    def _draw_entity_components(self, arch: Any, row: int) -> None:
        imgui = self.imgui
        for ct in sorted(arch.component_types, key=lambda c: c.__name__):
            inst = arch.get_component(row, ct)
            try:
                import dataclasses as _dc
                fields = _dc.fields(inst)
            except Exception:
                imgui.text(f"{ct.__name__}: <opaque>")
                continue
            for f in fields:
                value = getattr(inst, f.name, None)
                imgui.text(f"  {ct.__name__}.{f.name} = {value!r}")


class ProfilerOverlay:
    """Top-right ImGui overlay listing per-system avg_ms with a unit-scaled bar."""

    def __init__(self, ctx: Any) -> None:
        self._host: _ImGuiHost = _ImGuiHost.for_context(ctx)
        self.imgui = self._host.imgui
        self._visible: bool = True
        # Updated by render() — count of system rows actually drawn last frame.
        self.last_rendered_count: int = 0

    @property
    def visible(self) -> bool:
        """Whether the overlay draws each frame."""
        return self._visible

    def toggle(self) -> None:
        """Flip the visibility flag."""
        self._visible = not self._visible

    def set_visible(self, visible: bool) -> None:
        """Set the visibility flag explicitly."""
        self._visible = bool(visible)

    def render(self, profiler: FrameProfiler | None) -> None:
        """Draw the overlay. No-op if hidden, no profiler given, or no samples yet."""
        if not self._visible or profiler is None:
            return
        stats = profiler.get_stats()
        if not stats:
            return
        imgui = self.imgui
        try:
            try:
                io = imgui.get_io()
                display = io.display_size
                pos = imgui.ImVec2(max(display.x - 320.0, 0.0), 10.0)
                imgui.set_next_window_pos(pos, int(imgui.Cond_.always))
                imgui.set_next_window_size(imgui.ImVec2(310.0, 0.0), int(imgui.Cond_.always))
            except Exception:
                pass  # positioning hints are optional

            flags = 0
            try:
                flags = (
                    int(imgui.WindowFlags_.no_decoration)
                    | int(imgui.WindowFlags_.no_move)
                    | int(imgui.WindowFlags_.always_auto_resize)
                    | int(imgui.WindowFlags_.no_focus_on_appearing)
                )
            except Exception:
                pass
            # Don't pass p_open; F2 controls visibility.
            result = imgui.begin("Profiler", None, flags) if flags else imgui.begin("Profiler")
            shown = bool(result[0]) if isinstance(result, tuple) else bool(result)
            if shown:
                try:
                    ordered = sorted(stats.values(), key=lambda s: -s.avg_ms)
                    max_avg = max(s.avg_ms for s in ordered) if ordered else 1.0
                    if max_avg <= 0.0:
                        max_avg = 1.0
                    drawn = 0
                    for s in ordered:
                        fraction = max(0.0, min(1.0, s.avg_ms / max_avg))
                        imgui.text(f"{s.name:<20} {s.avg_ms:6.2f} ms")
                        imgui.same_line()
                        imgui.progress_bar(fraction, imgui.ImVec2(120.0, 0.0), "")
                        drawn += 1
                    self.last_rendered_count = drawn
                except Exception:
                    pass
            imgui.end()
        except Exception:
            try:
                imgui.end()
            except Exception:
                pass


def setup_inspector(app: Any) -> WorldInspector:
    """Create the inspector + overlay + key bindings + render system. Idempotent."""
    existing = getattr(app, "_keel_inspector_setup", None)
    if existing is not None:
        return existing["inspector"]

    inspector = WorldInspector(app.ctx)
    overlay = ProfilerOverlay(app.ctx)
    host = _ImGuiHost.for_context(app.ctx)

    import glfw

    # Edge-detected polling instead of KeyEvent. KeyEvents only reach
    # PRE_UPDATE systems through a sim tick, and a high-refresh visual
    # frame can have zero sim ticks → missed presses. is_key_down via
    # InputState is updated synchronously in the GLFW callback and reading
    # it from any phase always reflects the current key state.
    _was: dict[int, bool] = {}

    def _edge_pressed(key: int) -> bool:
        is_down = app.input.is_key_down(key)
        was = _was.get(key, False)
        _was[key] = is_down
        return is_down and not was

    @app.system(Phase.PRE_UPDATE)
    def inspector_input_system(world: Any, dt: float) -> None:
        if _edge_pressed(glfw.KEY_F1):
            inspector.toggle()
        if _edge_pressed(glfw.KEY_F2):
            overlay.toggle()

    @app.system(Phase.POST_RENDER)
    def inspector_render_system(world: Any, dt: float) -> None:
        viewport_w, viewport_h = (800, 600)
        try:
            viewport_w, viewport_h = app.window.get_size()
        except Exception:
            pass
        host.begin_frame(
            viewport_w,
            viewport_h,
            dt,
            input_state=getattr(app, "input", None),
        )
        inspector.render(world)
        profiler = world.get_resource(FrameProfiler)
        overlay.render(profiler)
        host.end_frame()

    setup = {
        "inspector": inspector,
        "overlay": overlay,
        "host": host,
        "input_system": inspector_input_system,
        "render_system": inspector_render_system,
    }
    app._keel_inspector_setup = setup

    add_hook = getattr(app, "add_shutdown_hook", None)
    if callable(add_hook):
        add_hook(host.shutdown)

    return inspector
