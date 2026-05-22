"""AssetRegistry: handle-based asset cache with extension-dispatched loaders.

The registry is the single user-facing entry point for loading assets. All
paths are normalized via os.path.normpath(os.path.abspath(...)) so that
"./hero.png" and "/abs/.../hero.png" hit the same cache slot. Asset IDs are
strictly monotonically increasing — never reused after unload.
"""
from __future__ import annotations

import os
from typing import Any, Callable


class AssetNotFoundError(FileNotFoundError):
    """Raised when AssetRegistry.load is given a path that doesn't exist."""


class NoLoaderError(KeyError):
    """Raised when no loader is registered for a path's extension."""


class InvalidHandleError(KeyError):
    """Raised when AssetRegistry is asked to act on a handle whose asset has been unloaded."""


class AssetHandle:
    """Immutable, hashable reference to a loaded asset. Equality is by normalized path."""

    __slots__ = ("id", "path", "asset_type")

    def __init__(self, id: int, path: str, asset_type: type) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "asset_type", asset_type)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("AssetHandle is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("AssetHandle is immutable")

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AssetHandle):
            return NotImplemented
        return self.path == other.path

    def __repr__(self) -> str:
        type_name = self.asset_type.__name__ if isinstance(self.asset_type, type) else str(self.asset_type)
        return f"AssetHandle(id={self.id}, path={self.path!r}, type={type_name})"


def _normalize(path: str) -> str:
    """Canonical path used as the cache key — absolute + normalized separators."""
    return os.path.normpath(os.path.abspath(path))


class AssetRegistry:
    """Central asset cache: path -> handle, id -> asset, ext -> loader."""

    def __init__(self) -> None:
        self._loaders: dict[str, Callable[[str], Any]] = {}
        self._handles: dict[str, AssetHandle] = {}
        self._assets: dict[int, Any] = {}
        self._next_id: int = 1

    def register_loader(
        self,
        extensions: list[str],
        loader_fn: Callable[[str], Any],
    ) -> None:
        """Register `loader_fn` as the loader for every extension in `extensions`."""
        for ext in extensions:
            ext_norm = ext.lower()
            if not ext_norm.startswith("."):
                ext_norm = "." + ext_norm
            self._loaders[ext_norm] = loader_fn

    def loader_for(self, path: str) -> Callable[[str], Any] | None:
        """Return the registered loader for `path`'s extension, or None."""
        ext = os.path.splitext(path)[1].lower()
        return self._loaders.get(ext)

    def load(self, path: str) -> AssetHandle:
        """Load (or return the cached handle for) the asset at `path`."""
        normalized = _normalize(path)
        cached = self._handles.get(normalized)
        if cached is not None:
            return cached
        if not os.path.exists(normalized):
            raise AssetNotFoundError(f"Asset not found: {path!r} (resolved to {normalized!r})")
        loader = self.loader_for(normalized)
        if loader is None:
            ext = os.path.splitext(normalized)[1].lower()
            raise NoLoaderError(
                f"No loader registered for extension {ext!r} (path: {path!r})"
            )
        asset = loader(normalized)
        handle = AssetHandle(self._next_id, normalized, type(asset))
        self._next_id += 1
        self._handles[normalized] = handle
        self._assets[handle.id] = asset
        return handle

    def get(self, handle: AssetHandle) -> Any:
        """Return the loaded asset for `handle`. Raises InvalidHandleError if unloaded."""
        if not isinstance(handle, AssetHandle):
            raise InvalidHandleError(f"Not an AssetHandle: {handle!r}")
        if handle.id not in self._assets:
            raise InvalidHandleError(
                f"Handle {handle!r} has no live asset — was it unloaded?"
            )
        return self._assets[handle.id]

    def reload(self, handle: AssetHandle) -> None:
        """Re-invoke the loader for `handle` and replace the stored asset."""
        if not isinstance(handle, AssetHandle):
            raise InvalidHandleError(f"Not an AssetHandle: {handle!r}")
        if handle.id not in self._assets:
            raise InvalidHandleError(
                f"Handle {handle!r} has no live asset — cannot reload"
            )
        loader = self.loader_for(handle.path)
        if loader is None:
            ext = os.path.splitext(handle.path)[1].lower()
            raise NoLoaderError(f"No loader registered for extension {ext!r}")
        if not os.path.exists(handle.path):
            raise AssetNotFoundError(f"Asset disappeared: {handle.path!r}")
        new_asset = loader(handle.path)
        self._assets[handle.id] = new_asset

    def unload(self, handle: AssetHandle) -> None:
        """Drop the asset for `handle`. Subsequent get/reload on this handle raise."""
        if not isinstance(handle, AssetHandle):
            raise InvalidHandleError(f"Not an AssetHandle: {handle!r}")
        if handle.id not in self._assets:
            raise InvalidHandleError(f"Handle {handle!r} already unloaded")
        del self._assets[handle.id]
        cached = self._handles.get(handle.path)
        if cached is not None and cached.id == handle.id:
            del self._handles[handle.path]

    def loaded_count(self) -> int:
        """Number of assets currently loaded."""
        return len(self._assets)

    def handles(self) -> list[AssetHandle]:
        """Snapshot of every live handle."""
        return list(self._handles.values())

    def handle_for_path(self, path: str) -> AssetHandle | None:
        """Return the handle whose normalized path matches `path`, or None."""
        return self._handles.get(_normalize(path))

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, AssetHandle):
            return item.id in self._assets
        if isinstance(item, str):
            return _normalize(item) in self._handles
        return False
