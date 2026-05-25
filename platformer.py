"""Platformer demo for Keel: 2400-px scrolling level with platforms,
patrolling enemies, sensor coins, jump, camera follow, and HUD text.

Controls: A/D or arrows = move, Space = jump (on ground), R = restart,
Escape = quit.
"""

# === Imports ===
import keel
from keel.physics import setup_physics_2d
from keel.renderer import setup_renderer_2d
from keel.text import BUILTIN_FONT, load_font, set_text, setup_text


# === Constants ===
SCREEN_W, SCREEN_H = 800, 600
LEVEL_W, LEVEL_H = 2400, 600
GRAVITY = -800.0                          # px/s², downward
GROUND_THICKNESS = 32.0
WALL_THICKNESS = 32.0
WALL_TOP_Y = LEVEL_H * 0.5                # walls span full height, centered
PLAYER_W, PLAYER_H, PLAYER_SPEED, JUMP_FORCE = 24.0, 32.0, 220.0, 520.0
ENEMY_W, ENEMY_H, ENEMY_SPEED = 24.0, 24.0, 80.0
COIN_RADIUS, COIN_VISUAL = 12.0, 24.0
FONT_ID = 0

# Spawn positions. (x, y) is the center of the entity in world space.
PLATFORM_DEFS = [
    (300.0,  200.0, 200.0, 24.0),
    (600.0,  280.0, 200.0, 24.0),
    (900.0,  200.0, 200.0, 24.0),
    (1200.0, 300.0, 200.0, 24.0),
    (1500.0, 200.0, 200.0, 24.0),
    (1800.0, 280.0, 200.0, 24.0),
    (2100.0, 200.0, 200.0, 24.0),
]
ENEMY_SPAWNS = [(500.0, 80.0), (1000.0, 80.0), (1500.0, 80.0), (2000.0, 80.0)]
COIN_POSITIONS = [
    (300.0, 240.0), (600.0, 320.0), (900.0, 240.0),
    (1200.0, 340.0), (1500.0, 240.0), (1800.0, 320.0),
    (2100.0, 240.0), (400.0, 90.0), (1000.0, 90.0),
]


# === Components ===

@keel.component
class Player:
    pass


@keel.component
class Enemy:
    pass


@keel.component
class Coin:
    pass


@keel.component
class GameState:
    """Singleton on GS_ENTITY: HUD totals + end-of-run flags."""
    coins_collected: int = 0
    dead: bool = False
    won: bool = False


# === App + subsystem setup ===
app = keel.App(title="Platformer", width=SCREEN_W, height=SCREEN_H)
phys = setup_physics_2d(app, gravity_y=GRAVITY)
setup_renderer_2d(app)
setup_text(app)
load_font(app, BUILTIN_FONT, size_px=22)
world = app.world


# === Module-level state ===
PLAYER_ENTITY = GS_ENTITY = CAM_ENTITY = SCORE_LABEL = MESSAGE_LABEL = 0
ENEMY_ENTITIES: set[int] = set()
COIN_ENTITIES: set[int] = set()
GROUND_ENTITIES: set[int] = set()        # ground + walls + platforms (permanent)
ENEMY_VEL: dict[int, float] = {}         # entity_id -> signed horizontal vx
DESPAWN_QUEUE: list[int] = []
ON_GROUND: bool = False                   # set by collision_system, cleared by reset_on_ground


# === Spawn helpers ===

def _spawn_static_box(x: float, y: float, w: float, h: float,
                      color: tuple[float, float, float] | None) -> int:
    """STATIC box collider with optional colored Sprite. Returns entity id."""
    extra = ((keel.Sprite(texture_id=0, width=w, height=h,
                          r=color[0], g=color[1], b=color[2]),) if color else ())
    eid = world.spawn(
        keel.Transform2D(x=x, y=y),
        keel.RigidBody2D(body_type=keel.BodyType.STATIC),
        keel.Collider2D(shape_type=keel.ShapeType2D.BOX,
                        width=w, height=h, elasticity=0.0, friction=0.8),
        *extra,
    )
    world.flush()
    return eid


def spawn_world_geometry() -> None:
    """Ground (visible brown), two invisible side walls, and seven green platforms."""
    GROUND_ENTITIES.add(_spawn_static_box(
        LEVEL_W * 0.5, GROUND_THICKNESS * 0.5,
        float(LEVEL_W), GROUND_THICKNESS, (0.40, 0.28, 0.18)))
    for wx in (-WALL_THICKNESS * 0.5, LEVEL_W + WALL_THICKNESS * 0.5):
        GROUND_ENTITIES.add(_spawn_static_box(
            wx, WALL_TOP_Y, WALL_THICKNESS, float(LEVEL_H), None))
    for px, py, pw, ph in PLATFORM_DEFS:
        GROUND_ENTITIES.add(_spawn_static_box(px, py, pw, ph, (0.20, 0.60, 0.30)))


def spawn_player() -> None:
    """DYNAMIC body. Pymunk applies gravity; player_input overrides vx each frame."""
    global PLAYER_ENTITY
    PLAYER_ENTITY = world.spawn(
        keel.Transform2D(x=80.0, y=200.0),
        keel.RigidBody2D(mass=1.0, body_type=keel.BodyType.DYNAMIC),
        keel.Collider2D(shape_type=keel.ShapeType2D.BOX,
                        width=PLAYER_W, height=PLAYER_H, friction=0.0, elasticity=0.0),
        keel.Sprite(texture_id=0, width=PLAYER_W, height=PLAYER_H,
                    r=0.30, g=0.55, b=1.0),
        Player(),
    )
    world.flush()


def spawn_enemies() -> None:
    """Patrolling DYNAMIC reds; ENEMY_VEL tracks each one's signed vx."""
    for ex, ey in ENEMY_SPAWNS:
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
        ENEMY_VEL[eid] = ENEMY_SPEED


def spawn_coins() -> None:
    """STATIC sensor circles; CollisionEvent2D fires on contact (v0.8.3+)."""
    COIN_ENTITIES.clear()
    for cx, cy in COIN_POSITIONS:
        eid = world.spawn(
            keel.Transform2D(x=cx, y=cy),
            keel.RigidBody2D(body_type=keel.BodyType.STATIC),
            keel.Collider2D(shape_type=keel.ShapeType2D.CIRCLE,
                            radius=COIN_RADIUS, sensor=True),
            keel.Sprite(texture_id=0, width=COIN_VISUAL, height=COIN_VISUAL,
                        r=1.0, g=0.9, b=0.15),
            Coin(),
        )
        world.flush()
        COIN_ENTITIES.add(eid)


def spawn_text() -> None:
    """HUD: coin counter (always visible) + center message (hidden until end)."""
    global SCORE_LABEL, MESSAGE_LABEL
    SCORE_LABEL = world.spawn(keel.Transform2D(x=10.0, y=30.0),
                              keel.TextLabel(font_id=FONT_ID, r=1.0, g=1.0, b=1.0))
    MESSAGE_LABEL = world.spawn(keel.Transform2D(x=200.0, y=280.0),
                                keel.TextLabel(font_id=FONT_ID, r=1.0, g=1.0, b=0.2, visible=False))
    set_text(SCORE_LABEL, f"Coins: 0 / {len(COIN_POSITIONS)}")
    set_text(MESSAGE_LABEL, "")
    world.flush()


# === Initial entities ===
CAM_ENTITY = world.spawn(keel.Camera2D(x=SCREEN_W * 0.5, y=SCREEN_H * 0.5))
GS_ENTITY = world.spawn(GameState())
world.flush()

spawn_world_geometry()
spawn_player()
spawn_enemies()
spawn_coins()
spawn_text()


# === Systems ===
#
# Ordering invariant: player_input MUST be registered (and therefore run)
# BEFORE reset_on_ground in PRE_UPDATE. ON_GROUND is set by collision_system
# in POST_UPDATE and consumed by the NEXT tick's player_input; clearing it
# before the jump check reads it makes Space silently never jump.

# Camera tracks the player horizontally, clamped to the level edges.
@app.system(keel.Phase.PRE_UPDATE)
def camera_follow(world, dt):
    pos = world.get(PLAYER_ENTITY, keel.Transform2D)
    if pos is None:
        return
    half_w = SCREEN_W * 0.5
    cam_x = max(half_w, min(LEVEL_W - half_w, float(pos["x"])))
    world.set(CAM_ENTITY, keel.Camera2D, x=cam_x, y=SCREEN_H * 0.5)


# Player input: Esc (always), horizontal movement, jump (on ground only).
# Vy is preserved from pymunk so gravity keeps integrating against the body.
# Must run BEFORE reset_on_ground.
@app.system(keel.Phase.PRE_UPDATE)
def player_input(world, dt):
    # Esc is intentionally handled here too (not only in quit_system) so the
    # close request happens immediately, not after one extra tick of input.
    if app.input.is_key_pressed(keel.KEY_ESCAPE):
        app.window.close()
        return

    gs = world.query_one(GameState)
    if gs is None or gs["dead"] or gs["won"]:
        return

    rb = world.get(PLAYER_ENTITY, keel.RigidBody2D)
    if rb is None:
        return

    vx = 0.0
    if app.input.is_key_down(keel.KEY_LEFT) or app.input.is_key_down(keel.KEY_A):
        vx = -PLAYER_SPEED
    elif app.input.is_key_down(keel.KEY_RIGHT) or app.input.is_key_down(keel.KEY_D):
        vx = PLAYER_SPEED

    vy = float(rb["vel_y"])
    if ON_GROUND and app.input.is_key_pressed(keel.KEY_SPACE):
        vy = JUMP_FORCE

    phys.set_velocity(PLAYER_ENTITY, vx, vy)


# Clear ON_GROUND so collision_system has a clean slate this tick. Must run
# AFTER player_input (which consumes the previous tick's ON_GROUND).
@app.system(keel.Phase.PRE_UPDATE)
def reset_on_ground(world, dt):
    global ON_GROUND
    ON_GROUND = False


# Enemies patrol horizontally; flip ENEMY_VEL[eid] on wall contact.
@app.system(keel.Phase.PRE_UPDATE)
def enemy_movement(world, dt):
    for eid in list(ENEMY_ENTITIES):
        if not world.is_alive(eid):
            ENEMY_ENTITIES.discard(eid)
            ENEMY_VEL.pop(eid, None)
            continue
        pos = world.get(eid, keel.Transform2D)
        rb = world.get(eid, keel.RigidBody2D)
        if pos is None or rb is None:
            continue
        vx = ENEMY_VEL.get(eid, ENEMY_SPEED)
        x = float(pos["x"])
        if x < 100.0 and vx < 0.0:
            vx = ENEMY_SPEED
        elif x > LEVEL_W - 100.0 and vx > 0.0:
            vx = -ENEMY_SPEED
        ENEMY_VEL[eid] = vx
        phys.set_velocity(eid, vx, float(rb["vel_y"]))


# Drains CollisionEvent2D once per tick. Sets ON_GROUND, collects coins,
# kills the player on enemy contact. Runs in POST_UPDATE after physics step.
@app.system(keel.Phase.POST_UPDATE)
def collision_system(world, dt):
    global ON_GROUND
    gs = world.query_one(GameState)
    if gs is None:
        return
    coins_collected = gs["coins_collected"]
    dead = gs["dead"]

    for event in world.read_events(keel.CollisionEvent2D):
        a, b = int(event.entity_a), int(event.entity_b)
        if PLAYER_ENTITY not in (a, b):
            continue
        other = b if a == PLAYER_ENTITY else a

        if other in COIN_ENTITIES:
            if not dead:
                COIN_ENTITIES.discard(other)
                if other not in DESPAWN_QUEUE:
                    DESPAWN_QUEUE.append(other)
                coins_collected += 1
        elif other in ENEMY_ENTITIES:
            dead = True
        elif other in GROUND_ENTITIES:
            ON_GROUND = True

    won = coins_collected >= len(COIN_POSITIONS)
    world.set(GS_ENTITY, GameState,
              coins_collected=coins_collected, dead=dead, won=won)


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


# HUD strings + win/loss message toggle.
@app.system(keel.Phase.POST_UPDATE)
def text_update(world, dt):
    gs = world.query_one(GameState)
    if gs is None:
        return
    set_text(SCORE_LABEL, f"Coins: {gs['coins_collected']} / {len(COIN_POSITIONS)}")
    if gs["dead"]:
        world.set(MESSAGE_LABEL, keel.TextLabel, visible=True)
        set_text(MESSAGE_LABEL, "YOU DIED  -  press R to restart")
    elif gs["won"]:
        world.set(MESSAGE_LABEL, keel.TextLabel, visible=True)
        set_text(MESSAGE_LABEL, "YOU WIN!  -  press R to restart")
    else:
        world.set(MESSAGE_LABEL, keel.TextLabel, visible=False)


# Always-on Esc handler — runs regardless of dead / won state.
@app.system(keel.Phase.UPDATE)
def quit_system(world, dt):
    if app.input.is_key_pressed(keel.KEY_ESCAPE):
        app.window.close()


# R restarts after death or win.
@app.system(keel.Phase.UPDATE)
def restart_system(world, dt):
    gs = world.query_one(GameState)
    if gs is None or not (gs["dead"] or gs["won"]):
        return
    if app.input.is_key_pressed(keel.KEY_R):
        do_restart()


# === Restart helper ===

def do_restart() -> None:
    """Despawn player + enemies + coins, respawn them, reset GameState.
    Ground / walls / platforms stay in GROUND_ENTITIES for the whole session.
    """
    global ON_GROUND
    ON_GROUND = False
    for eid in list(ENEMY_ENTITIES | COIN_ENTITIES | {PLAYER_ENTITY}):
        if world.is_alive(eid):
            world.despawn(eid)
    world.flush()
    ENEMY_ENTITIES.clear()
    ENEMY_VEL.clear()
    COIN_ENTITIES.clear()
    DESPAWN_QUEUE.clear()
    world.set(GS_ENTITY, GameState, coins_collected=0, dead=False, won=False)
    spawn_player()
    spawn_enemies()
    spawn_coins()


# === Entry point ===
app.run()
