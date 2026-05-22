"""TextureAtlas: integer-keyed registry of GPU textures.

Each loaded image becomes one moderngl.Texture bound to one of the 16
guaranteed texture units. Texture IDs are dense integers (0, 1, 2, ...) in
load order. Loading the same path returns the cached ID without re-uploading.
"""
from __future__ import annotations

from typing import Any

import moderngl


MAX_TEXTURE_UNITS: int = 16


class TextureAtlas:
    """Owns up to 16 moderngl.Textures, addressable by stable integer IDs."""

    __slots__ = ("ctx", "_textures", "_path_to_id")

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx: moderngl.Context = ctx
        self._textures: list[moderngl.Texture] = []
        self._path_to_id: dict[str, int] = {}

    def load(self, path: str) -> int:
        """Load `path`, upload to the GPU, and return a stable integer ID. Cached by path."""
        cached = self._path_to_id.get(path)
        if cached is not None:
            return cached
        if len(self._textures) >= MAX_TEXTURE_UNITS:
            raise RuntimeError(
                f"TextureAtlas exceeded {MAX_TEXTURE_UNITS} texture limit "
                "(GL guarantees at most 16 sampler units in fragment shaders)"
            )
        from PIL import Image
        with Image.open(path) as raw:
            img = raw.convert("RGBA")
            data = img.tobytes()
            size = img.size
        tex = self.ctx.texture(size, 4, data)
        tex.repeat_x = False
        tex.repeat_y = False
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tid = len(self._textures)
        self._textures.append(tex)
        self._path_to_id[path] = tid
        return tid

    def add_texture(self, key: str, texture: moderngl.Texture) -> int:
        """Register an already-built moderngl.Texture under `key`. For tests / procedural assets."""
        cached = self._path_to_id.get(key)
        if cached is not None:
            return cached
        if len(self._textures) >= MAX_TEXTURE_UNITS:
            raise RuntimeError(
                f"TextureAtlas exceeded {MAX_TEXTURE_UNITS} texture limit"
            )
        tid = len(self._textures)
        self._textures.append(texture)
        self._path_to_id[key] = tid
        return tid

    def bind_all(self, ctx: moderngl.Context) -> None:
        """Bind every loaded texture to its corresponding texture unit (0..N-1)."""
        for i, tex in enumerate(self._textures):
            tex.use(location=i)

    def reload(self, texture_id: int) -> None:
        """Re-read the source file for `texture_id` and re-upload to the GPU. ID is preserved."""
        if texture_id < 0 or texture_id >= len(self._textures):
            raise IndexError(f"texture_id {texture_id} out of range")
        path = next(
            (p for p, tid in self._path_to_id.items() if tid == texture_id),
            None,
        )
        if path is None:
            raise KeyError(f"texture_id {texture_id} has no associated source path")
        from PIL import Image
        with Image.open(path) as raw:
            img = raw.convert("RGBA")
            data = img.tobytes()
            size = img.size
        old_tex = self._textures[texture_id]
        if old_tex.size == size:
            old_tex.write(data)
            return
        try:
            old_tex.release()
        except Exception:
            pass
        new_tex = self.ctx.texture(size, 4, data)
        new_tex.repeat_x = False
        new_tex.repeat_y = False
        new_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._textures[texture_id] = new_tex

    def get_texture(self, texture_id: int) -> moderngl.Texture:
        """Return the moderngl.Texture for the given ID."""
        return self._textures[texture_id]

    def __len__(self) -> int:
        return len(self._textures)

    def texture_count(self) -> int:
        """Number of textures currently registered."""
        return len(self._textures)

    def release(self) -> None:
        """Release every owned GL texture."""
        for tex in self._textures:
            try:
                tex.release()
            except Exception:
                pass
        self._textures.clear()
        self._path_to_id.clear()
