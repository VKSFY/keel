"""Text rendering: setup_text, load_font, set_text, BUILTIN_FONT, TextLabel.

Text is drawn in screen-space pixels with y growing downward and is not
camera-transformed. World-space labels are a v0.2 feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import moderngl

from ..components.text_label import TextLabel
from ..core import Phase
from .builtin_font import (
    get_builtin_font_bytes,
    get_builtin_font_path,
)
from .font import (
    AtlasTooSmallError,
    Font,
    FontRegistry,
    GlyphAtlas,
    GlyphInfo,
)
from .shader_text import TEXT_FRAG_SRC, TEXT_VERT_SRC
from .text_renderer import (
    TextBatch,
    clear_text,
    get_text,
    set_text,
)


__all__ = [
    "AtlasTooSmallError",
    "BUILTIN_FONT",
    "Font",
    "FontRegistry",
    "GlyphAtlas",
    "GlyphInfo",
    "TEXT_FRAG_SRC",
    "TEXT_VERT_SRC",
    "TextBatch",
    "TextLabel",
    "TextSetup",
    "clear_text",
    "get_builtin_font_bytes",
    "get_builtin_font_path",
    "get_text",
    "load_font",
    "set_label_visible",
    "set_text",
    "setup_text",
]


def set_label_visible(world: Any, entity_id: int, visible: bool) -> None:
    """Show or hide a TextLabel entity."""
    world.set(entity_id, TextLabel, visible=visible)


# Bundled DejaVu Sans Mono — materialized on import so `load_font(app,
# BUILTIN_FONT)` works without an external asset.
BUILTIN_FONT: str = get_builtin_font_path()


@dataclass
class TextSetup:
    """The wired text resources: FontRegistry + TextBatch."""
    font_registry: FontRegistry
    text_batch: TextBatch


def setup_text(app: Any) -> TextSetup:
    """Create FontRegistry + TextBatch and register the POST_RENDER draw system.

    Must follow `setup_renderer_2d(app)` — the text shader runs over the
    framebuffer the 2D pipeline cleared. Idempotent.
    """
    existing = getattr(app, "_pyge_text", None)
    if existing is not None:
        return existing

    try:
        from ..renderer import SpriteBatch2D
    except ImportError:  # pragma: no cover - 2D renderer is in-tree
        raise RuntimeError("setup_text requires pyge.renderer.")
    if not app.world.has_resource(SpriteBatch2D):
        raise RuntimeError(
            "setup_text: call setup_renderer_2d(app) first — text needs the "
            "2D framebuffer to be cleared this frame."
        )

    ctx = app.ctx
    text_program = ctx.program(
        vertex_shader=TEXT_VERT_SRC,
        fragment_shader=TEXT_FRAG_SRC,
    )
    font_registry = FontRegistry()
    text_batch = TextBatch(ctx, text_program)
    app.world.insert_resource(font_registry, type_=FontRegistry)
    app.world.insert_resource(text_batch, type_=TextBatch)

    @app.system(Phase.POST_RENDER)
    def render_text(
        world: Any,
        dt: float,
        registry: FontRegistry,
        batch: TextBatch,
    ) -> None:
        viewport_w, viewport_h = app.window.get_size()
        if viewport_w <= 0 or viewport_h <= 0:
            return
        batch.render(world, registry, viewport_w, viewport_h)

    setup = TextSetup(font_registry=font_registry, text_batch=text_batch)
    app._pyge_text = setup
    return setup


def load_font(
    app: Any,
    path: str,
    size_px: int = 24,
    name: Optional[str] = None,
) -> Font:
    """Load a font through the world's FontRegistry. Requires setup_text first."""
    registry = app.world.get_resource(FontRegistry)
    if registry is None:
        raise RuntimeError("load_font: call setup_text(app) first.")
    return registry.load(app.ctx, path, size_px=size_px, name=name)
