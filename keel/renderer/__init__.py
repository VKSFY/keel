"""2D renderer for Keel: textures, shaders, sprite batching, tilemaps, and a setup helper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import moderngl

from ..components import Sprite, Transform2D
from ..core import Phase
from .batch2d import SpriteBatch2D
from .camera2d import Camera2D, build_camera_matrix, default_camera_matrix
from .shader import ShaderCache
from .texture import MAX_TEXTURE_UNITS, TextureAtlas
from .tilemap import Tilemap

__all__ = [
    "Camera2D",
    "MAX_TEXTURE_UNITS",
    "Renderer2DSetup",
    "ShaderCache",
    "Sprite",
    "SpriteBatch2D",
    "TextureAtlas",
    "Tilemap",
    "TilemapSetup",
    "Transform2D",
    "build_camera_matrix",
    "default_camera_matrix",
    "get_active_camera_matrix",
    "setup_renderer_2d",
    "setup_tilemap",
]


@dataclass
class Renderer2DSetup:
    """What `setup_renderer_2d` returns: registries already wired into the app."""
    atlas: TextureAtlas
    shader_cache: ShaderCache
    sprite_batch: SpriteBatch2D
    render_system: Callable


@dataclass
class TilemapSetup:
    """What `setup_tilemap` returns — the wired Tilemap world resource."""
    tilemap: "Tilemap"


def get_active_camera_matrix(world: Any, viewport_w: int, viewport_h: int):
    """Return the first Camera2D's matrix, or the default centered ortho if none exists."""
    for (cameras,) in world.query(Camera2D):
        if len(cameras) > 0:
            return build_camera_matrix(cameras[0], viewport_w, viewport_h)
    return default_camera_matrix(viewport_w, viewport_h)


def setup_renderer_2d(app: Any) -> "Renderer2DSetup":
    """Create + register the 2D renderer on `app`. Idempotent — second call is a no-op."""
    existing = getattr(app, "_keel_renderer_2d", None)
    if existing is not None:
        return existing

    ctx = app.ctx
    atlas = TextureAtlas(ctx)
    # Preload a 1x1 white texture at id=0 so Sprite(texture_id=0, r=g=b=...)
    # tints render correctly without the user wiring an image asset first.
    atlas.add_texture("__keel_default_white__", ctx.texture((1, 1), 4, b"\xff\xff\xff\xff"))
    shader_cache = ShaderCache()
    sprite_shader = shader_cache.get(ctx, "sprite")
    sprite_batch = SpriteBatch2D(ctx, sprite_shader)

    app.world.insert_resource(ctx, type_=moderngl.Context)
    app.world.insert_resource(atlas, type_=TextureAtlas)
    app.world.insert_resource(shader_cache, type_=ShaderCache)
    app.world.insert_resource(sprite_batch, type_=SpriteBatch2D)

    # If the asset registry has already been set up, register the image loader
    # against this atlas so registry.load("foo.png") routes through it.
    try:
        from ..assets import AssetRegistry
        from ..assets.loaders.texture_loader import make_texture_loader
    except ImportError:  # pragma: no cover - defensive
        AssetRegistry = None
        make_texture_loader = None
    if AssetRegistry is not None:
        registry = app.world.get_resource(AssetRegistry)
        if registry is not None:
            registry.register_loader(
                [".png", ".jpg", ".jpeg", ".bmp", ".tga"],
                make_texture_loader(atlas),
            )

    @app.system(Phase.RENDER)
    def render_2d(
        world: Any,
        dt: float,
        ctx: moderngl.Context,
        atlas: TextureAtlas,
        batch: SpriteBatch2D,
    ) -> None:
        viewport_w, viewport_h = app.window.get_size()
        if viewport_w > 0 and viewport_h > 0:
            ctx.viewport = (0, 0, viewport_w, viewport_h)

        cam_matrix = get_active_camera_matrix(world, viewport_w, viewport_h)

        # When a tilemap is active its PRE_RENDER system already cleared
        # the framebuffer. Clearing again here would wipe the tile layer.
        if getattr(app, "_keel_tilemap", None) is None:
            ctx.clear(0.05, 0.05, 0.08, 1.0)
        atlas.bind_all(ctx)
        batch.render(world.query(Transform2D, Sprite), cam_matrix)

    setup = Renderer2DSetup(
        atlas=atlas,
        shader_cache=shader_cache,
        sprite_batch=sprite_batch,
        render_system=render_2d,
    )
    app._keel_renderer_2d = setup
    return setup


def setup_tilemap(
    app: Any,
    tile_data: "np.ndarray",
    tile_width: int = 32,
    tile_height: int = 32,
) -> TilemapSetup:
    """Create / reload a Tilemap on `app`. Requires setup_renderer_2d to be live.

    On the first call we build the Tilemap, bake chunks from `tile_data`,
    insert it as a world resource, and register a PRE_RENDER system that
    draws every chunk before SpriteBatch2D runs. Subsequent calls re-bake
    the chunks against the new tile data without registering a second system.
    """
    import numpy as np

    if not app.world.has_resource(SpriteBatch2D):
        raise RuntimeError(
            "setup_tilemap: call setup_renderer_2d(app) first — the tilemap "
            "shares its shader and atlas with the 2D sprite pipeline."
        )

    existing: TilemapSetup | None = getattr(app, "_keel_tilemap", None)
    if existing is not None:
        existing.tilemap.tile_width = int(tile_width)
        existing.tilemap.tile_height = int(tile_height)
        existing.tilemap.load(np.asarray(tile_data, dtype=np.int32))
        return existing

    renderer_setup: Renderer2DSetup = app._keel_renderer_2d
    atlas = renderer_setup.atlas
    shader_cache = renderer_setup.shader_cache
    sprite_shader = shader_cache.get(app.ctx, "sprite")

    tilemap = Tilemap(
        app.ctx,
        sprite_shader,
        atlas,
        tile_width=int(tile_width),
        tile_height=int(tile_height),
    )
    tilemap.load(np.asarray(tile_data, dtype=np.int32))

    app.world.insert_resource(tilemap, type_=Tilemap)

    @app.system(Phase.PRE_RENDER)
    def render_tilemap(world: Any, dt: float, tm: Tilemap) -> None:
        viewport_w, viewport_h = app.window.get_size()
        if viewport_w <= 0 or viewport_h <= 0:
            return
        ctx = app.ctx
        ctx.viewport = (0, 0, viewport_w, viewport_h)
        # Tilemap owns the framebuffer clear when active — the sprite
        # RENDER system detects this and skips its own clear so the
        # tilemap pixels survive into the sprite draw.
        ctx.clear(0.05, 0.05, 0.08, 1.0)
        cam_matrix = get_active_camera_matrix(world, viewport_w, viewport_h)
        tm.render(cam_matrix)

    setup = TilemapSetup(tilemap=tilemap)
    app._keel_tilemap = setup
    return setup
