"""Texture loader factory.

The returned closure is registered with AssetRegistry for image extensions.
On first load it delegates to TextureAtlas.load and returns the new ID; on
re-load (registry.reload(handle)) it detects the cached path and calls
TextureAtlas.reload(id) so the GPU texture is re-uploaded in-place while the
ID itself stays stable.
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ...renderer.texture import TextureAtlas


def make_texture_loader(atlas: "TextureAtlas") -> Callable[[str], int]:
    """Return a path -> int loader bound to `atlas` and reload-aware."""

    def loader(path: str) -> int:
        existing = atlas._path_to_id.get(path)
        if existing is not None:
            atlas.reload(existing)
            return existing
        return atlas.load(path)

    return loader
