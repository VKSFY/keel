"""TextLabel — screen-space text component. Text string lives in a side dict."""
from __future__ import annotations

from ..core import component


@component
class TextLabel:
    """A screen-space text label.

    The text label's Transform2D is interpreted in **screen pixels** with
    y = 0 at the **top** of the window and y growing downward — this is UI
    space, not world space, and the camera does NOT affect it. The baseline
    of the text sits at Transform2D.y, so place labels at y ≈ font ascender
    (~20-30 px for a 28 px font) to keep the top of the glyphs on screen.

    The actual text string is NOT stored on this component (numpy structured
    arrays cannot hold variable-length strings). Use
    ``keel.text.set_text(entity_id, "...")`` to set it,
    ``get_text(entity_id)`` / ``clear_text(entity_id)`` to read or remove.
    A typed string column is planned for v0.2.
    """
    font_id: int = 0
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0
    scale: float = 1.0
    visible: bool = True
