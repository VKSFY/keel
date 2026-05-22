"""Asset pipeline for Keel: registry, loaders, hot reload, scene save/load."""
from __future__ import annotations

from typing import Any

from ..core import Phase
from .hot_reload import FileWatcher
from .loaders import json_loader, make_texture_loader
from .registry import (
    AssetHandle,
    AssetNotFoundError,
    AssetRegistry,
    InvalidHandleError,
    NoLoaderError,
)
from .scene import Scene, SceneVersionError

__all__ = [
    "AssetHandle",
    "AssetNotFoundError",
    "AssetRegistry",
    "FileWatcher",
    "InvalidHandleError",
    "NoLoaderError",
    "Scene",
    "SceneVersionError",
    "json_loader",
    "make_texture_loader",
    "setup_assets",
]


def setup_assets(app: Any, watch_dirs: list[str] | None = None) -> AssetRegistry:
    """Create or return the AssetRegistry on `app`. Idempotent; registers default loaders + watcher."""
    existing = getattr(app, "_keel_asset_setup", None)
    if existing is not None:
        return existing["registry"]

    registry = AssetRegistry()
    registry.register_loader([".json"], json_loader)

    # If the renderer has already been initialised, hook the texture loader to
    # its atlas right away. Otherwise setup_renderer_2d() will wire it later.
    renderer_setup = getattr(app, "_keel_renderer_2d", None)
    if renderer_setup is not None:
        atlas = renderer_setup.atlas
        registry.register_loader(
            [".png", ".jpg", ".jpeg", ".bmp", ".tga"],
            make_texture_loader(atlas),
        )

    app.world.insert_resource(registry, type_=AssetRegistry)

    file_watcher: FileWatcher | None = None
    if watch_dirs:
        file_watcher = FileWatcher(registry)
        for d in watch_dirs:
            file_watcher.watch(d)

        @app.system(Phase.PRE_UPDATE)
        def _keel_file_watcher_poll(world: Any, dt: float) -> None:
            file_watcher.poll()

        app.world.insert_resource(file_watcher, type_=FileWatcher)

    setup = {"registry": registry, "file_watcher": file_watcher}
    app._keel_asset_setup = setup
    return registry
