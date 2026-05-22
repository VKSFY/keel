"""Freetype-driven font loader + glyph atlas + Font / FontRegistry.

A single Font owns one R8 ModernGL texture (the glyph atlas) baked once
at __init__ time. The atlas stores coverage only — color comes from
the shader's u_color uniform, so one atlas works for any tint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import moderngl
import numpy as np

try:
    import freetype
except ImportError as e:  # pragma: no cover - import-time fallback
    raise ImportError("install freetype-py: pip install freetype-py") from e


# --- Default character set ------------------------------------------------

# ASCII printable + full Latin-1 Supplement. Missing glyphs in the source
# font are silently skipped at atlas build time.
_DEFAULT_CHARS: str = "".join(chr(c) for c in list(range(32, 127)) + list(range(160, 256)))


# --- Errors ---------------------------------------------------------------

class AtlasTooSmallError(RuntimeError):
    """Raised when the requested character set does not fit in a 1024x1024 atlas.

    The remedy is to use a smaller font size_px when constructing the Font.
    """


# --- GlyphInfo ------------------------------------------------------------

@dataclass
class GlyphInfo:
    """Atlas placement + metrics for a single glyph."""
    char: str
    uv_x: float
    uv_y: float
    uv_w: float
    uv_h: float
    width: int
    height: int
    bearing_x: int
    bearing_y: int
    advance: int


# --- GlyphAtlas -----------------------------------------------------------

_MAX_ATLAS_SIZE: int = 1024


class GlyphAtlas:
    """Shelf-packed R8 atlas. One texture, one build pass, never repacked."""

    def __init__(self, ctx: moderngl.Context, size: int = 512) -> None:
        self._ctx: moderngl.Context = ctx
        self._size: int = int(size)
        self._glyphs: dict[str, GlyphInfo] = {}
        self._texture: Optional[moderngl.Texture] = None

    @property
    def size(self) -> int:
        """Side length of the square atlas in pixels."""
        return self._size

    @property
    def texture(self) -> moderngl.Texture:
        """The R8 atlas texture. Build() must have been called first."""
        if self._texture is None:
            raise RuntimeError("GlyphAtlas.build() has not been called")
        return self._texture

    def get_glyph(self, char: str) -> Optional[GlyphInfo]:
        """Return GlyphInfo for char, or None if the glyph is not in the atlas."""
        return self._glyphs.get(char)

    def build(self, face: "freetype.Face", char_set: str) -> None:
        """Render every char in `char_set` into the atlas. Doubles atlas to 1024 if 512 is too small."""
        for attempt_size in (self._size, _MAX_ATLAS_SIZE):
            self._size = attempt_size
            try:
                self._pack_and_upload(face, char_set)
                return
            except _ShelfOverflow:
                if attempt_size >= _MAX_ATLAS_SIZE:
                    raise AtlasTooSmallError(
                        f"Glyph atlas overflowed at {attempt_size}x{attempt_size}; "
                        "use a smaller font size_px."
                    )
                # Otherwise loop and retry at the larger size.
                continue

    # --- Internal -------------------------------------------------------

    def _pack_and_upload(self, face: "freetype.Face", char_set: str) -> None:
        """Render each requested char into an in-memory R8 buffer and upload once."""
        # First: render all glyphs to bitmaps + metrics in memory.
        rendered: list[tuple[str, int, int, int, int, int, bytes]] = []
        for ch in char_set:
            if face.get_char_index(ch) == 0:
                # Glyph not present in the source font. Skip silently.
                continue
            try:
                face.load_char(ch, freetype.FT_LOAD_RENDER)
            except Exception:
                continue
            g = face.glyph
            bmp = g.bitmap
            w = int(bmp.width)
            h = int(bmp.rows)
            bx = int(g.bitmap_left)
            by = int(g.bitmap_top)
            adv = int(g.advance.x) >> 6
            # The pitch may be larger than width for some bitmap formats;
            # copy row-by-row into a tight w*h buffer.
            if w == 0 or h == 0:
                # Whitespace glyph (e.g. space). Still record metrics so we
                # know its advance, but it has no pixels to pack.
                rendered.append((ch, w, h, bx, by, adv, b""))
                continue
            pitch = bmp.pitch
            src = bytes(bmp.buffer)
            if pitch == w:
                pixels = src[: w * h]
            else:
                rows = []
                for r in range(h):
                    start = r * pitch
                    rows.append(src[start : start + w])
                pixels = b"".join(rows)
            rendered.append((ch, w, h, bx, by, adv, pixels))

        # Sort by height descending — classic shelf packer. Items with
        # h==0 (whitespace) sort to the end and consume no shelf space.
        rendered.sort(key=lambda r: r[2], reverse=True)

        atlas_buf = np.zeros((self._size, self._size), dtype=np.uint8)
        shelf_x = 0
        shelf_y = 0
        shelf_h = 0
        padding = 1  # 1px gutter between glyphs to avoid sample bleed.

        self._glyphs = {}

        for ch, w, h, bx, by, adv, pixels in rendered:
            if w == 0 or h == 0:
                # No pixel data; store metrics with a zero-size UV rect.
                self._glyphs[ch] = GlyphInfo(
                    char=ch,
                    uv_x=0.0,
                    uv_y=0.0,
                    uv_w=0.0,
                    uv_h=0.0,
                    width=w,
                    height=h,
                    bearing_x=bx,
                    bearing_y=by,
                    advance=adv,
                )
                continue

            # Advance to next shelf if this glyph would overflow horizontally.
            if shelf_x + w + padding > self._size:
                shelf_y += shelf_h + padding
                shelf_x = 0
                shelf_h = 0

            # Vertical overflow: atlas is too small. Caller handles retry/raise.
            if shelf_y + h > self._size:
                raise _ShelfOverflow()

            # Blit into the atlas buffer.
            glyph_arr = np.frombuffer(pixels, dtype=np.uint8).reshape((h, w))
            atlas_buf[shelf_y : shelf_y + h, shelf_x : shelf_x + w] = glyph_arr

            self._glyphs[ch] = GlyphInfo(
                char=ch,
                uv_x=shelf_x / self._size,
                uv_y=shelf_y / self._size,
                uv_w=w / self._size,
                uv_h=h / self._size,
                width=w,
                height=h,
                bearing_x=bx,
                bearing_y=by,
                advance=adv,
            )

            shelf_x += w + padding
            if h > shelf_h:
                shelf_h = h

        # Upload as a single-channel R8 texture.
        if self._texture is not None:
            try:
                self._texture.release()
            except Exception:
                pass
        tex = self._ctx.texture(
            (self._size, self._size),
            components=1,
            data=atlas_buf.tobytes(),
            dtype="f1",
        )
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        self._texture = tex


class _ShelfOverflow(Exception):
    """Internal: the current atlas size is too small for the requested char set."""


# --- Font -----------------------------------------------------------------

class Font:
    """A loaded font face at a fixed pixel size, with its glyph atlas already built."""

    def __init__(
        self,
        ctx: moderngl.Context,
        path: str,
        size_px: int = 24,
        char_set: Optional[str] = None,
    ) -> None:
        self._path: str = str(path)
        self._size_px: int = int(size_px)
        self._face: "freetype.Face" = freetype.Face(self._path)
        # Width=0 means "scale to match height". 26.6 fixed-point not needed
        # for set_pixel_sizes — it takes integer pixels directly.
        self._face.set_pixel_sizes(0, self._size_px)
        # FreeType reports metrics in 26.6 fixed-point, so >>6 to get pixels.
        self._line_height: int = self._face.size.height >> 6
        self._ascender: int = self._face.size.ascender >> 6
        self._descender: int = self._face.size.descender >> 6
        self._space_advance: int = self._compute_space_advance()
        chars = char_set if char_set is not None else _DEFAULT_CHARS
        self._atlas: GlyphAtlas = GlyphAtlas(ctx, size=512)
        self._atlas.build(self._face, chars)

    @property
    def path(self) -> str:
        """The file path the font was loaded from."""
        return self._path

    @property
    def size_px(self) -> int:
        """Pixel size used when loading the face (NOT a recomputed font height)."""
        return self._size_px

    @property
    def line_height(self) -> int:
        """Pixels to advance Y by for a newline."""
        return self._line_height

    @property
    def ascender(self) -> int:
        """Pixels above the baseline for the tallest glyph."""
        return self._ascender

    @property
    def descender(self) -> int:
        """Pixels below the baseline (negative or zero)."""
        return self._descender

    @property
    def space_advance(self) -> int:
        """Advance width of the space glyph in pixels (also used as missing-glyph fallback)."""
        return self._space_advance

    @property
    def atlas(self) -> GlyphAtlas:
        """The packed glyph atlas for this font."""
        return self._atlas

    def get_glyph(self, char: str) -> Optional[GlyphInfo]:
        """Return GlyphInfo for char, or None if the glyph is missing."""
        return self._atlas.get_glyph(char)

    def measure(self, text: str) -> tuple[float, float]:
        """Return (width, height) in pixels for `text` rendered at this font's size.

        Width is the sum of advances along the current line; for multi-line
        strings it is the max line width. Height is line_height * line_count.
        An empty string measures (0, line_height).
        """
        if not text:
            return (0.0, float(self._line_height))
        width = 0.0
        line_widths: list[float] = []
        lines = 1
        for ch in text:
            if ch == "\n":
                line_widths.append(width)
                width = 0.0
                lines += 1
                continue
            if ch == "\t":
                width += self._space_advance * 4
                continue
            info = self._atlas.get_glyph(ch)
            if info is None:
                width += self._space_advance
                continue
            width += info.advance
        line_widths.append(width)
        max_w = max(line_widths) if line_widths else 0.0
        return (float(max_w), float(self._line_height * lines))

    # --- Internal -------------------------------------------------------

    def _compute_space_advance(self) -> int:
        """Get the advance width of the space glyph in pixels."""
        if self._face.get_char_index(" ") == 0:
            # No space glyph in this font; fall back to a quarter of the size.
            return max(1, self._size_px // 4)
        try:
            self._face.load_char(" ", freetype.FT_LOAD_DEFAULT)
        except Exception:
            return max(1, self._size_px // 4)
        adv = int(self._face.glyph.advance.x) >> 6
        return adv if adv > 0 else max(1, self._size_px // 4)


# --- FontRegistry ---------------------------------------------------------

class FontRegistry:
    """Cache of loaded fonts keyed by (absolute_path, size_px) plus optional name alias."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, int], Font] = {}
        self._by_name: dict[str, Font] = {}
        self._by_id: list[Font] = []
        self._id_of: dict[Font, int] = {}

    def load(
        self,
        ctx: moderngl.Context,
        path: str,
        size_px: int = 24,
        name: Optional[str] = None,
    ) -> Font:
        """Load a font once. Subsequent calls with the same (path, size_px) return the cached Font."""
        import os
        key = (os.path.abspath(str(path)), int(size_px))
        font = self._by_key.get(key)
        if font is None:
            font = Font(ctx, key[0], size_px=key[1])
            self._by_key[key] = font
            self._id_of[font] = len(self._by_id)
            self._by_id.append(font)
        if name is not None:
            self._by_name[str(name)] = font
        return font

    def get(self, name: str) -> Font:
        """Look up a font by alias name. Raises KeyError if unknown."""
        return self._by_name[name]

    def get_by_path(self, path: str, size_px: int) -> Font:
        """Look up a font by (path, size_px). Raises KeyError if not loaded."""
        import os
        return self._by_key[(os.path.abspath(str(path)), int(size_px))]

    def get_by_id(self, font_id: int) -> Optional[Font]:
        """Look up a font by index (the order it was loaded). Returns None if out of range."""
        if 0 <= font_id < len(self._by_id):
            return self._by_id[font_id]
        return None

    def id_of(self, font: Font) -> int:
        """Return the integer id for `font`. Raises KeyError if the font is not registered."""
        return self._id_of[font]

    def __len__(self) -> int:
        return len(self._by_id)
