"""tilemap_demo.py — a tilemap with a player sprite and a follow camera.

Demonstrates: setup_tilemap, Tilemap chunk rendering under live sprites,
Camera2D following the player. WASD to move, Escape to quit.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np

import keel
from keel.renderer import setup_renderer_2d, setup_tilemap


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIDTH = 800
HEIGHT = 600

TILE = 32                          # tile size in pixels (square)
ROWS, COLS = 15, 20                # 20×32 = 640 wide, 15×32 = 480 tall
SPEED = 220.0                      # player speed in px/s
PLAYER_SIZE = 24.0                 # sprite side length

# Tile-id mapping. Atlas slot 0 is the default white texture, so tile id 1
# would draw white if used. We paint two extra atlas tiles (floor + wall)
# at slots 1 and 2 — tile ids 2 and 3 reference them.
TILE_ID_FLOOR = 2
TILE_ID_WALL = 3

# Pixel colours used to hand-paint the floor and wall textures.
FLOOR_RGBA = (90, 70, 45, 255)
WALL_RGBA = (45, 45, 60, 255)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

@keel.component
class Player:
    pass


# ---------------------------------------------------------------------------
# App + renderer + tilemap setup
# ---------------------------------------------------------------------------

app = keel.App(title="Tilemap Demo", width=WIDTH, height=HEIGHT)
renderer = setup_renderer_2d(app)

# Hand-paint two extra atlas tiles. add_texture appends to the atlas; the
# next available slot is index 1 (floor), then 2 (wall). The tilemap
# references them as tile ids 2 and 3 below (id - 1 maps to atlas slot).
floor_px = bytes(FLOOR_RGBA) * (TILE * TILE)
wall_px = bytes(WALL_RGBA) * (TILE * TILE)
renderer.atlas.add_texture("floor", app.ctx.texture((TILE, TILE), 4, floor_px))
renderer.atlas.add_texture("wall", app.ctx.texture((TILE, TILE), 4, wall_px))

# Build a 15×20 tile array: floor everywhere, walls around the border.
tile_data = np.zeros((ROWS, COLS), dtype=np.int32)
tile_data[1:ROWS - 1, 1:COLS - 1] = TILE_ID_FLOOR
tile_data[0, :] = TILE_ID_WALL
tile_data[ROWS - 1, :] = TILE_ID_WALL
tile_data[:, 0] = TILE_ID_WALL
tile_data[:, COLS - 1] = TILE_ID_WALL

setup_tilemap(app, tile_data, tile_width=TILE, tile_height=TILE)


# ---------------------------------------------------------------------------
# Initial entities
# ---------------------------------------------------------------------------

# Player, placed in the centre of the playable area.
app.world.spawn(
    keel.Transform2D(x=COLS * TILE / 2.0, y=ROWS * TILE / 2.0),
    keel.Sprite(
        texture_id=0,
        width=PLAYER_SIZE,
        height=PLAYER_SIZE,
        r=1.0,
        g=0.7,
        b=0.3,
    ),
    Player(),
)

# Camera2D — moved every frame to follow the player.
camera = app.world.spawn(
    keel.Camera2D(x=COLS * TILE / 2.0, y=ROWS * TILE / 2.0, zoom=1.0),
)

app.world.flush()


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

# Read WASD, move the player, then move the camera to match.
@app.system(keel.Phase.UPDATE)
def move_player(world, dt):
    if app.input.is_key_down(keel.KEY_ESCAPE):
        app.window.close()

    right = 1.0 if app.input.is_key_down(keel.KEY_D) else 0.0
    left  = 1.0 if app.input.is_key_down(keel.KEY_A) else 0.0
    up    = 1.0 if app.input.is_key_down(keel.KEY_W) else 0.0
    down  = 1.0 if app.input.is_key_down(keel.KEY_S) else 0.0

    dx = right - left
    dy = up - down

    px = 0.0
    py = 0.0
    for transforms, _ in world.query(keel.Transform2D, Player):
        for i in range(len(transforms)):
            transforms["x"][i] += dx * SPEED * dt
            transforms["y"][i] += dy * SPEED * dt
            px = float(transforms["x"][i])
            py = float(transforms["y"][i])

    # Snap the camera to the player. world.query returns archetype column
    # tuples; the [0] indexing is the column view of Camera2D itself.
    for cameras in world.query(keel.Camera2D):
        if len(cameras[0]) > 0:
            cameras[0]["x"][0] = px
            cameras[0]["y"][0] = py


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app.run()
