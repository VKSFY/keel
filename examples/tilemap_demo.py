"""tilemap_demo.py — a tilemap with a player sprite and a follow camera.

Demonstrates: setup_tilemap, Tilemap chunk rendering under live sprites,
Camera2D following the player. WASD to move, Escape to quit.
"""
import numpy as np

import keel
from keel.renderer import setup_renderer_2d, setup_tilemap


TILE = 32
ROWS, COLS = 15, 20  # 20*32=640 wide, 15*32=480 tall — fits 800x600
SPEED = 220.0


@keel.component
class Player:
    pass


app = keel.App(title="Tilemap Demo", width=800, height=600)
renderer = setup_renderer_2d(app)

# Hand-paint two extra atlas tiles (the default white is at id=0):
# id 1 → atlas slot 0 (default white, tinted by the chunk's instance color)
# We add a brown floor at slot 1 and a slate wall at slot 2, so tile ids
# 1, 2, 3 in the data map to white(unused), floor, wall.
floor_px = bytes([90, 70, 45, 255]) * (TILE * TILE)
wall_px = bytes([45, 45, 60, 255]) * (TILE * TILE)
renderer.atlas.add_texture("floor", app.ctx.texture((TILE, TILE), 4, floor_px))
renderer.atlas.add_texture("wall", app.ctx.texture((TILE, TILE), 4, wall_px))

tile_data = np.zeros((ROWS, COLS), dtype=np.int32)
tile_data[1:ROWS - 1, 1:COLS - 1] = 2  # floor everywhere inside
tile_data[0, :] = 3                    # top wall row
tile_data[ROWS - 1, :] = 3             # bottom wall row
tile_data[:, 0] = 3                    # left wall col
tile_data[:, COLS - 1] = 3             # right wall col

setup_tilemap(app, tile_data, tile_width=TILE, tile_height=TILE)

# Center the player in the playable area.
app.world.spawn(
    keel.Transform2D(x=COLS * TILE / 2, y=ROWS * TILE / 2),
    keel.Sprite(texture_id=0, width=24.0, height=24.0, r=1.0, g=0.7, b=0.3),
    Player(),
)
# Camera2D centered on (0, 0) by default — we move it to track the player.
camera = app.world.spawn(keel.Camera2D(x=COLS * TILE / 2, y=ROWS * TILE / 2, zoom=1.0))
app.world.flush()


@app.system(keel.Phase.UPDATE)
def move_player(world, dt):
    if app.input.is_key_down(keel.KEY_ESCAPE):
        app.window.close()
    dx = (1.0 if app.input.is_key_down(keel.KEY_D) else 0.0) - \
         (1.0 if app.input.is_key_down(keel.KEY_A) else 0.0)
    dy = (1.0 if app.input.is_key_down(keel.KEY_W) else 0.0) - \
         (1.0 if app.input.is_key_down(keel.KEY_S) else 0.0)
    px = py = 0.0
    for transforms, _ in world.query(keel.Transform2D, Player):
        for i in range(len(transforms)):
            transforms["x"][i] += dx * SPEED * dt
            transforms["y"][i] += dy * SPEED * dt
            px = float(transforms["x"][i])
            py = float(transforms["y"][i])
    # Follow camera — write through to the structured array.
    for cameras in world.query(keel.Camera2D):
        if len(cameras[0]) > 0:
            cameras[0]["x"][0] = px
            cameras[0]["y"][0] = py


app.run()
