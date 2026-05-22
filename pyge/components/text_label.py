"""TextLabel — screen-space text component. Text string lives in a side dict."""
from __future__ import annotations

from ..core import component


@component
class TextLabel:
    """A screen-space text label.

    The text string is NOT stored here (numpy structured arrays cannot hold
    variable-length strings). Use `pyge.text.set_text(entity_id, "...")` to
    set the text, and `get_text(entity_id)` / `clear_text(entity_id)` to read
    or remove it. This is a known v0.1 limitation; a typed string column is
    planned for v0.2.
    """
    font_id: int = 0
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0
    scale: float = 1.0
    visible: bool = True
