"""Platformer demo: tilemap + physics + camera follow + text + sensor polling.

Controls: A/D or arrows = move, Space = jump (on ground only), R = restart,
Esc = quit. Collect every yellow coin to win; touching a red enemy or red
spike ends the run.
"""
# === Imports ===
import numpy as np
import keel
from keel.physics import setup_physics_2d
from keel.renderer import setup_renderer_2d, setup_tilemap
from keel.renderer.texture import TextureAtlas
from keel.text import BUILTIN_FONT, load_font, set_text, setup_text


# === Constants ===
SCREEN_W, SCREEN_H = 800, 600
TILE_SIZE = 32
LEVEL_COLS, LEVEL_ROWS = 50, 20
LEVEL_W = LEVEL_COLS * TILE_SIZE      # 1600 px; camera scrolls within this
GRAVITY = -800.0
PLAYER_W, PLAYER_H, PLAYER_SPEED, JUMP_FORCE = 24.0, 32.0, 220.0, 520.0
ENEMY_W, ENEMY_H, ENEMY_SPEED = 24.0, 24.0, 80.0

# Tile IDs into tile_data. 0 = empty; tilemap renders id → atlas slot (id-1).
# Slot 0 is the default white from setup_renderer_2d (= id 1); _add_atlas_colors
# registers slots 1..3 for ids 2..4 (ground/platform/spikes).
TILE_EMPTY, TILE_GROUND, TILE_PLATFORM, TILE_SPIKES = 0, 2, 3, 4

FONT_ID = 0
SCORE_LABEL_XY = (10.0, 30.0)
MESSAGE_LABEL_XY = (200.0, 280.0)


# === Components ===

@keel.component
class Player:
    on_ground: bool = False
    dead: bool = False


@keel.component
class Enemy:
    speed: float = ENEMY_SPEED


@keel.component
class Coin:
    collected: bool = False


@keel.component
class Tile:
    """Marker on every per-tile static collider — distinguishes solid tiles
    from enemies / coins / player inside the collision system."""
    kind: int = 0


@keel.component
class GameState:
    coins_total: int = 0
    coins_collected: int = 0
    dead: bool = False
    won: bool = False


# === App + subsystem setup ===
app = keel.App(title="Platformer", width=SCREEN_W, height=SCREEN_H)
phys = setup_physics_2d(app, gravity_y=GRAVITY)
setup_renderer_2d(app)


# === Tile data + atlas colors ===
# Build the level in "row 0 = top" convention, then flipud so Keel's
# y-up world places the ground row at world y=0 (the bottom).
_spec = np.zeros((LEVEL_ROWS, LEVEL_COLS), dtype=np.int32)
_spec[18:20, :] = TILE_GROUND                   # ground (two rows tall)
_spec[:, 0] = TILE_GROUND                       # left wall
_spec[:, LEVEL_COLS - 1] = TILE_GROUND          # right wall
for _row, _c0, _c1 in [(14, 5, 10), (11, 12, 17), (14, 20, 25),
                       (10, 28, 33), (14, 36, 41), (8, 43, 48)]:
    _spec[_row, _c0:_c1] = TILE_PLATFORM
_spec[17, 15:18] = TILE_SPIKES
_spec[17, 30:33] = TILE_SPIKES
tile_data = np.flipud(_spec).astype(np.int32)


def _add_atlas_colors() -> None:
    """Register a colored 32×32 RGBA texture for tile ids 2..4."""
    atlas: TextureAtlas = app.world.get_resource(TextureAtlas)
    for key, (r, g, b) in [("platformer.ground", (0.50, 0.35, 0.20)),
                           ("platformer.platform", (0.20, 0.70, 0.30)),
                           ("platformer.spikes", (0.90, 0.20, 0.20))]:
        img = np.empty((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
        img[:, :, 0] = int(r * 255); img[:, :, 1] = int(g * 255)
        img[:, :, 2] = int(b * 255); img[:, :, 3] = 255
        atlas.add_texture(key, app.ctx.texture((TILE_SIZE, TILE_SIZE), 4, img.tobytes()))


_add_atlas_colors()
setup_tilemap(app, tile_data, TILE_SIZE, TILE_SIZE)
setup_text(app)
load_font(app, BUILTIN_FONT, size_px=22)
world = app.world


# === Module-level state (populated by spawn helpers below) ===
PLAYER_ENTITY = GS_ENTITY = CAMERA_ENTITY = SCORE_LABEL = MESSAGE_LABEL = 0
ENEMY_ENTITIES: set[int] = set()
COIN_ENTITIES: set[int] = set()
SPIKE_ENTITIES: set[int] = set()
DESPAWN_QUEUE: list[int] = []


# === Spawn helpers ===

def spawn_tile_colliders() -> None:
    """One static box per solid tile; spikes are sensors. Aligns 1:1 with the rendered tilemap."""
    for row in range(LEVEL_ROWS):
        for col in range(LEVEL_COLS):
            tid = int(tile_data[row, col])
            if tid == TILE_EMPTY:
                continue
            sensor = (tid == TILE_SPIKES)
            eid = world.spawn(
                keel.Transform2D(x=col * TILE_SIZE + TILE_SIZE * 0.5,
                                 y=row * TILE_SIZE + TILE_SIZE * 0.5),
                keel.RigidBody2D(body_type=keel.BodyType.STATIC),
                keel.Collider2D(shape_type=keel.ShapeType2D.BOX,
                                width=float(TILE_SIZE), height=float(TILE_SIZE),
                                friction=0.5, elasticity=0.0, sensor=sensor),
                Tile(kind=tid),
            )
            world.flush()
            if sensor:
                SPIKE_ENTITIES.add(eid)


def spawn_player() -> None:
    """DYNAMIC body. Pymunk gravity drives vy so CollisionEvent2D fires against STATIC ground."""
    global PLAYER_ENTITY
    PLAYER_ENTITY = world.spawn(
        keel.Transform2D(x=96.0, y=200.0),
        keel.RigidBody2D(mass=1.0, body_type=keel.BodyType.DYNAMIC),
        keel.Collider2D(shape_type=keel.ShapeType2D.BOX,
                        width=PLAYER_W, height=PLAYER_H, friction=0.0, elasticity=0.0),
        keel.Sprite(texture_id=0, width=PLAYER_W, height=PLAYER_H,
                    r=0.3, g=0.55, b=1.0),
        Player(),
    )
    world.flush()


def spawn_enemies() -> None:
    """Patrolling red boxes, DYNAMIC so collision events vs the player fire."""
    for ex, ey in [(400.0, 200.0), (720.0, 200.0), (960.0, 360.0),
                   (1200.0, 200.0), (1400.0, 130.0)]:
        eid = world.spawn(
            keel.Transform2D(x=ex, y=ey),
            keel.RigidBody2D(mass=1.0, body_type=keel.BodyType.DYNAMIC, vel_x=ENEMY_SPEED),
            keel.Collider2D(shape_type=keel.ShapeType2D.BOX,
                            width=ENEMY_W, height=ENEMY_H, friction=0.0, elasticity=0.0),
            keel.Sprite(texture_id=0, width=ENEMY_W, height=ENEMY_H,
                        r=1.0, g=0.25, b=0.25),
            Enemy(),
        )
        world.flush()
        ENEMY_ENTITIES.add(eid)


def spawn_coins() -> None:
    """STATIC sensor circles. coins_total drives the win condition."""
    positions = [(224.0, 230.0), (464.0, 320.0), (704.0, 230.0),
                 (960.0, 380.0), (1216.0, 230.0), (1440.0, 180.0),
                 (300.0, 100.0), (600.0, 100.0), (1100.0, 100.0)]
    world.set(GS_ENTITY, GameState, coins_total=len(positions))
    for cx, cy in positions:
        eid = world.spawn(
            keel.Transform2D(x=cx, y=cy),
            keel.RigidBody2D(body_type=keel.BodyType.STATIC),
            keel.Collider2D(shape_type=keel.ShapeType2D.CIRCLE,
                            radius=10.0, sensor=True),
            keel.Sprite(texture_id=0, width=20.0, height=20.0,
                        r=1.0, g=0.9, b=0.15),
            Coin(),
        )
        world.flush()
        COIN_ENTITIES.add(eid)


def spawn_text() -> None:
    """HUD: coin counter (always visible) + center message (hidden by default)."""
    global SCORE_LABEL, MESSAGE_LABEL
    SCORE_LABEL = world.spawn(keel.Transform2D(x=SCORE_LABEL_XY[0], y=SCORE_LABEL_XY[1]),
                              keel.TextLabel(font_id=FONT_ID, r=1.0, g=1.0, b=1.0))
    MESSAGE_LABEL = world.spawn(keel.Transform2D(x=MESSAGE_LABEL_XY[0], y=MESSAGE_LABEL_XY[1]),
                                keel.TextLabel(font_id=FONT_ID, r=1.0, g=1.0, b=0.2, visible=False))
    set_text(SCORE_LABEL, "Coins: 0 / 0")
    set_text(MESSAGE_LABEL, "")
    world.flush()


# === Initial entities ===
CAMERA_ENTITY = world.spawn(keel.Camera2D(x=SCREEN_W * 0.5, y=SCREEN_H * 0.5))
GS_ENTITY = world.spawn(GameState())
world.flush()

spawn_tile_colliders()
spawn_player()
spawn_enemies()
spawn_coins()
spawn_text()


# === Systems ===

@app.system(keel.Phase.PRE_UPDATE)
def camera_follow(world, dt):
    pos = world.get(PLAYER_ENTITY, keel.Transform2D)
    if pos is None:
        return
    half_w = SCREEN_W * 0.5
    cam_x = max(half_w, min(LEVEL_W - half_w, float(pos["x"])))
    world.set(CAMERA_ENTITY, keel.Camera2D, x=cam_x, y=SCREEN_H * 0.5)


# Clear on_ground each tick; collision_system sets it True on ground contact.
@app.system(keel.Phase.PRE_UPDATE)
def reset_ground(world, dt):
    world.set(PLAYER_ENTITY, Player, on_ground=False)


# Override horizontal velocity from input; jump on SPACE if on_ground.
# Vy is preserved from pymunk so gravity keeps integrating against the body.
@app.system(keel.Phase.PRE_UPDATE)
def player_input(world, dt):
    gs = world.query_one(GameState)
    pl = world.query_one(Player)
    if gs is None or pl is None or gs["dead"] or gs["won"]:
        return
    left = app.input.is_key_down(keel.KEY_LEFT) or app.input.is_key_down(keel.KEY_A)
    right = app.input.is_key_down(keel.KEY_RIGHT) or app.input.is_key_down(keel.KEY_D)
    vx = PLAYER_SPEED * (int(right) - int(left))
    rb = world.get(PLAYER_ENTITY, keel.RigidBody2D)
    vy = float(rb["vel_y"]) if rb else 0.0
    if pl["on_ground"] and app.input.is_key_pressed(keel.KEY_SPACE):
        vy = JUMP_FORCE
    phys.set_velocity(PLAYER_ENTITY, vx, vy)


@app.system(keel.Phase.PRE_UPDATE)
def enemy_movement(world, dt):
    """Patrol horizontally; reverse at level walls; kick out of zero-velocity stalls."""
    for eid in list(ENEMY_ENTITIES):
        if not world.is_alive(eid):
            ENEMY_ENTITIES.discard(eid)
            continue
        pos = world.get(eid, keel.Transform2D)
        rb = world.get(eid, keel.RigidBody2D)
        if pos is None or rb is None:
            continue
        vx, vy = float(rb["vel_x"]), float(rb["vel_y"])
        x = float(pos["x"])
        if x < TILE_SIZE + ENEMY_W * 0.5 and vx < 0.0:
            vx = ENEMY_SPEED
        elif x > LEVEL_W - TILE_SIZE - ENEMY_W * 0.5 and vx > 0.0:
            vx = -ENEMY_SPEED
        elif abs(vx) < 1.0:
            vx = ENEMY_SPEED
        phys.set_velocity(eid, vx, vy)


# Drain CollisionEvent2D once per frame: coin pickup, spike / enemy death,
# and ground contact for the player.
@app.system(keel.Phase.POST_UPDATE)
def collision_system(world, dt):
    gs = world.query_one(GameState)
    if gs is None:
        return
    player_pos = world.get(PLAYER_ENTITY, keel.Transform2D)
    if player_pos is None:
        return

    coins_collected, coins_total, dead = gs["coins_collected"], gs["coins_total"], gs["dead"]
    on_ground = False
    px, py = float(player_pos["x"]), float(player_pos["y"])

    # Event-driven: enemy hits + ground contact (DYNAMIC pairs fire post_solve).
    for event in world.read_events(keel.CollisionEvent2D):
        a, b = int(event.entity_a), int(event.entity_b)
        if PLAYER_ENTITY not in (a, b):
            continue
        other = b if a == PLAYER_ENTITY else a
        if other in ENEMY_ENTITIES:
            dead = True
        else:
            other_pos = world.get(other, keel.Transform2D)
            if other_pos is not None and py > float(other_pos["y"]):
                on_ground = True

    # Sensors skip pymunk's solver, so post_solve never fires — poll AABB.
    if not dead:
        for eid in list(COIN_ENTITIES):
            cp = world.get(eid, keel.Transform2D)
            if cp is not None and _aabb(px, py, PLAYER_W, PLAYER_H,
                                         float(cp["x"]), float(cp["y"]), 20.0, 20.0):
                COIN_ENTITIES.discard(eid)
                if eid not in DESPAWN_QUEUE:
                    DESPAWN_QUEUE.append(eid)
                coins_collected += 1
        for eid in SPIKE_ENTITIES:
            sp = world.get(eid, keel.Transform2D)
            if sp is not None and _aabb(px, py, PLAYER_W, PLAYER_H,
                                         float(sp["x"]), float(sp["y"]),
                                         float(TILE_SIZE), float(TILE_SIZE)):
                dead = True
                break

    won = coins_total > 0 and coins_collected >= coins_total
    world.set(GS_ENTITY, GameState,
              coins_collected=coins_collected, dead=dead, won=won)
    if on_ground:
        world.set(PLAYER_ENTITY, Player, on_ground=True)
    if dead:
        world.set(PLAYER_ENTITY, Player, dead=True)


# Drain the despawn queue once per frame so we never despawn inside a query.
@app.system(keel.Phase.POST_UPDATE)
def despawn_system(world, dt):
    if not DESPAWN_QUEUE:
        return
    for eid in DESPAWN_QUEUE:
        if world.is_alive(eid):
            world.despawn(eid)
    DESPAWN_QUEUE.clear()
    world.flush()


# HUD strings + win / loss message toggle.
@app.system(keel.Phase.POST_UPDATE)
def text_update(world, dt):
    gs = world.query_one(GameState)
    if gs is None:
        return
    set_text(SCORE_LABEL, f"Coins: {gs['coins_collected']} / {gs['coins_total']}")
    if gs["dead"]:
        world.set(MESSAGE_LABEL, keel.TextLabel, visible=True)
        set_text(MESSAGE_LABEL, "YOU DIED  -  press R to restart")
    elif gs["won"]:
        world.set(MESSAGE_LABEL, keel.TextLabel, visible=True)
        set_text(MESSAGE_LABEL, "YOU WIN!  -  press R to restart")
    else:
        world.set(MESSAGE_LABEL, keel.TextLabel, visible=False)


@app.system(keel.Phase.UPDATE)
def restart_system(world, dt):
    """R restarts after death or win."""
    gs = world.query_one(GameState)
    if gs is None or not (gs["dead"] or gs["won"]):
        return
    if app.input.is_key_pressed(keel.KEY_R):
        _restart()


@app.system(keel.Phase.UPDATE)
def quit_system(world, dt):
    """Esc quits cleanly."""
    if app.input.is_key_pressed(keel.KEY_ESCAPE):
        app.window.close()


# === Restart + despawn helpers ===

def _aabb(ax: float, ay: float, aw: float, ah: float,
          bx: float, by: float, bw: float, bh: float) -> bool:
    """Center-based AABB overlap test."""
    return abs(ax - bx) < (aw + bw) * 0.5 and abs(ay - by) < (ah + bh) * 0.5


def _restart() -> None:
    """Tear down player + enemies + coins; respawn the lot; reset GameState."""
    for eid in list(ENEMY_ENTITIES | COIN_ENTITIES | {PLAYER_ENTITY}):
        if world.is_alive(eid):
            world.despawn(eid)
    world.flush()
    ENEMY_ENTITIES.clear()
    COIN_ENTITIES.clear()
    world.set(GS_ENTITY, GameState,
              coins_collected=0, coins_total=0, dead=False, won=False)
    spawn_player()
    spawn_enemies()
    spawn_coins()


# === Entry point ===
app.run()
