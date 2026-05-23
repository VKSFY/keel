"""asteroids.py — a small Asteroids clone built on Keel.

Run with:  python asteroids.py

Controls
--------
Left / Right arrows : rotate the ship
Up arrow            : thrust
Space               : fire
R                   : restart (only on the game-over screen)
Escape              : quit
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import math
import random

import keel
from keel.physics import setup_physics_2d
from keel.renderer import setup_renderer_2d
from keel.text import (
    BUILTIN_FONT,
    load_font,
    set_label_visible,
    set_text,
    setup_text,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Window
WIDTH = 800
HEIGHT = 600
SCREEN_CX = WIDTH / 2.0
SCREEN_CY = HEIGHT / 2.0

# Ship motion
SHIP_ROT_DEG_PER_S = 180.0   # rotation speed (degrees per second)
SHIP_THRUST = 300.0          # pixels / second² added per frame UP is held
SHIP_MAX_SPEED = 400.0       # max linear speed in px/s
NOSE_OFFSET = 18.0           # distance from ship center to bullet spawn point

# Bullets
BULLET_SPEED = 500.0         # bullet launch speed in px/s
BULLET_LIFETIME = 3.0        # seconds before a bullet self-despawns
FIRE_COOLDOWN = 0.18         # seconds between consecutive shots

# Lives / wave timing
RESPAWN_DELAY = 2.0          # seconds between ship death and respawn
INVINCIBLE_TIME = 2.0        # seconds of grace after respawn
NEW_WAVE_EXCLUSION_RADIUS = 150.0  # asteroids spawn at least this far from center

# Asteroid splitting
SPLIT_SPEED_BOOST = 1.2      # children fly 20 % faster than the parent
SPLIT_ANGLE_OFFSET = 0.4     # radians each child deflects from parent direction
NEW_WAVE_MIN_SPEED = 40.0    # randomized asteroid speed lower bound (px/s)
NEW_WAVE_MAX_SPEED = 80.0    # randomized asteroid speed upper bound (px/s)

# Body and shape type enums — IntEnum, so DYNAMIC == 0 / KINEMATIC == 2 still hold.
# CollisionEvent2D does NOT fire for KINEMATIC-vs-KINEMATIC pairs (a pymunk
# quirk), so bullets and asteroids run as DYNAMIC. The ship stays kinematic
# so the player isn't pushed around by physics responses.
DYNAMIC = keel.BodyType.DYNAMIC
KINEMATIC = keel.BodyType.KINEMATIC
CIRCLE = keel.ShapeType2D.CIRCLE

# Collision filter bits — every collider can be in one or more categories,
# and only collides with categories listed in its mask.
#   ship      collides with everything
#   asteroid  collides with everything EXCEPT other asteroids
#   bullet    collides ONLY with asteroids (no self-firing, no bullet-bullet)
CAT_SHIP = 1
CAT_ASTEROID = 2
CAT_BULLET = 4
MASK_SHIP = 0xFFFF
MASK_ASTEROID = 0xFFFF ^ CAT_ASTEROID
MASK_BULLET = CAT_ASTEROID

# Asteroid sizes — keyed by an integer "size class" (3 = large, 1 = small).
ASTEROID_RADIUS = {3: 40.0, 2: 22.0, 1: 12.0}
ASTEROID_PX     = {3: 80.0, 2: 44.0, 1: 24.0}
ASTEROID_SCORE  = {3: 20,   2: 50,   1: 100}

# Text label placement (screen-space pixels — Keel text renders with y=0 at
# the TOP of the window, and Transform2D.y is the BASELINE of the glyphs).
SCORE_LABEL_X, SCORE_LABEL_Y     = 10.0, 35.0
LIVES_LABEL_X, LIVES_LABEL_Y     = 590.0, 35.0
GAMEOVER_LABEL_X, GAMEOVER_LABEL_Y = 280.0, 270.0
RESTART_LABEL_X, RESTART_LABEL_Y = 210.0, 320.0
FONT_SIZE_PX = 28


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

@keel.component
class Ship:
    rotation_speed: float = SHIP_ROT_DEG_PER_S
    thrust: float = SHIP_THRUST
    invincible_timer: float = 0.0
    fire_cooldown: float = 0.0


@keel.component
class Bullet:
    lifetime: float = BULLET_LIFETIME


@keel.component
class Asteroid:
    size: int = 3
    vel_x: float = 0.0
    vel_y: float = 0.0


@keel.component
class GameState:
    score: int = 0
    lives: int = 3
    wave: int = 1
    game_over: bool = False
    respawn_timer: float = 0.0
    ship_alive: bool = True
    restart_pending: bool = False


# ---------------------------------------------------------------------------
# App + subsystem setup
# ---------------------------------------------------------------------------

app = keel.App(title="Asteroids", width=WIDTH, height=HEIGHT)
setup_renderer_2d(app)                          # must precede setup_text
phys = setup_physics_2d(app, gravity_y=0.0)
text_setup = setup_text(app)

_font = load_font(app, BUILTIN_FONT, size_px=FONT_SIZE_PX)
FONT_ID = text_setup.font_registry.id_of(_font)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
#
# A few pieces of state can't live on components because Keel components are
# backed by numpy structured arrays and can only hold numeric / bool scalars.
# Anything variable-sized (a list, a tuple, a dict) goes here instead.

# Current ship velocity. Read+written by input_system and pushed into pymunk
# by apply_ship_vel each tick.
SHIP_VEL = {"x": 0.0, "y": 0.0}

# Per-bullet velocity. A bullet keeps a constant (vx, vy) for its whole
# lifetime; we can't put a tuple on the Bullet component.
BULLET_VEL: dict[int, tuple[float, float]] = {}

# Entities scheduled for despawn at the end of the frame. A list (rather
# than a set) because order can matter for debug output; queue_despawn()
# below deduplicates.
DESPAWN_QUEUE: list[int] = []

# We track entity IDs by category here so the collision_system can quickly
# classify a CollisionEvent2D pair without a per-entity component lookup.
SHIP_ENTITIES: set[int] = set()
BULLET_ENTITIES: set[int] = set()
ASTEROID_ENTITIES: set[int] = set()

# Filled in below once the text labels are spawned.
SCORE_LABEL = 0
LIVES_LABEL = 0
GAMEOVER_LABEL = 0
RESTART_LABEL = 0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def queue_despawn(eid):
    """Schedule `eid` for destruction at end of frame (deduplicated)."""
    if eid not in DESPAWN_QUEUE:
        DESPAWN_QUEUE.append(eid)


def wrap(x, y):
    """Zero-margin screen wrap — entity reappears at the exact opposite edge."""
    if x < 0:
        x += WIDTH
    if x > WIDTH:
        x -= WIDTH
    if y < 0:
        y += HEIGHT
    if y > HEIGHT:
        y -= HEIGHT
    return x, y


def _gs(world):
    """Return the singleton GameState column-view, or None on the bootstrap frame."""
    for (g,) in world.query(GameState):
        if len(g) > 0:
            return g
    return None


# ---------------------------------------------------------------------------
# Spawn functions
# ---------------------------------------------------------------------------

def spawn_ship(world, _phys):
    """Spawn a fresh ship at screen center with INVINCIBLE_TIME of grace."""
    SHIP_VEL["x"] = 0.0
    SHIP_VEL["y"] = 0.0

    eid = world.spawn(
        keel.Transform2D(x=SCREEN_CX, y=SCREEN_CY, rotation=0.0),
        keel.RigidBody2D(mass=1.0, body_type=KINEMATIC),
        keel.Collider2D(
            shape_type=CIRCLE,
            radius=14.0,
            elasticity=0.0,
            friction=0.0,
            category_bits=CAT_SHIP,
            mask_bits=MASK_SHIP,
        ),
        keel.Sprite(texture_id=0, width=28.0, height=28.0, r=1.0, g=1.0, b=1.0),
        Ship(invincible_timer=INVINCIBLE_TIME),
    )

    world.flush()
    SHIP_ENTITIES.add(eid)
    return eid


def spawn_bullet(world, x, y, vx, vy):
    """Spawn a bullet at (x, y) heading at (vx, vy)."""
    eid = world.spawn(
        keel.Transform2D(x=x, y=y),
        # DYNAMIC so the bullet-vs-asteroid collision callback fires.
        keel.RigidBody2D(mass=0.1, body_type=DYNAMIC),
        keel.Collider2D(
            shape_type=CIRCLE,
            radius=3.0,
            elasticity=0.0,
            friction=0.0,
            category_bits=CAT_BULLET,
            mask_bits=MASK_BULLET,
        ),
        keel.Sprite(texture_id=0, width=6.0, height=6.0, r=1.0, g=0.85, b=0.4),
        Bullet(lifetime=BULLET_LIFETIME),
    )

    world.flush()
    BULLET_VEL[eid] = (vx, vy)
    BULLET_ENTITIES.add(eid)
    return eid


def _spawn_asteroid(world, size, x, y, vx, vy):
    """Spawn one asteroid with an explicit velocity (used by splits + random spawns)."""
    radius = ASTEROID_RADIUS[size]
    px = ASTEROID_PX[size]

    eid = world.spawn(
        keel.Transform2D(x=x, y=y, rotation=random.uniform(0.0, math.tau)),
        # DYNAMIC because kinematic-vs-kinematic pairs don't emit collision
        # events in pymunk. apply_asteroid_vel re-pushes the stored velocity
        # each frame so the asteroid keeps its constant drift speed.
        keel.RigidBody2D(mass=float(size), body_type=DYNAMIC),
        keel.Collider2D(
            shape_type=CIRCLE,
            radius=radius,
            elasticity=0.0,
            friction=0.0,
            category_bits=CAT_ASTEROID,
            mask_bits=MASK_ASTEROID,
        ),
        keel.Sprite(texture_id=0, width=px, height=px, r=0.78, g=0.78, b=0.82),
        Asteroid(size=size, vel_x=vx, vel_y=vy),
    )

    world.flush()
    ASTEROID_ENTITIES.add(eid)
    return eid


def spawn_asteroid(world, _phys, size, x, y):
    """Spawn an asteroid with a random velocity (used by spawn_wave)."""
    speed = random.uniform(NEW_WAVE_MIN_SPEED, NEW_WAVE_MAX_SPEED) * (4 - size)
    angle = random.uniform(0.0, math.tau)
    vx = math.cos(angle) * speed
    vy = math.sin(angle) * speed
    return _spawn_asteroid(world, size, x, y, vx, vy)


def spawn_wave(world, phys, wave):
    """Spawn `3 + wave` large asteroids, all at least 150 px from screen center."""
    for _ in range(3 + wave):
        # Reject candidate positions that land too close to the ship.
        for _attempt in range(50):
            x = random.uniform(0.0, WIDTH)
            y = random.uniform(0.0, HEIGHT)
            if math.hypot(x - SCREEN_CX, y - SCREEN_CY) > NEW_WAVE_EXCLUSION_RADIUS:
                break

        spawn_asteroid(world, phys, 3, x, y)


# ---------------------------------------------------------------------------
# Initial entities (run at module import time, before app.run())
# ---------------------------------------------------------------------------

app.world.spawn(GameState())

# Transform2D.y for a TextLabel is the BASELINE of the glyph row, so labels
# need to sit at least ~ascender pixels (about 25 px at size 28) below the
# top of the window to remain on-screen.
SCORE_LABEL = app.world.spawn(
    keel.Transform2D(x=SCORE_LABEL_X, y=SCORE_LABEL_Y),
    keel.TextLabel(font_id=FONT_ID),
)
LIVES_LABEL = app.world.spawn(
    keel.Transform2D(x=LIVES_LABEL_X, y=LIVES_LABEL_Y),
    keel.TextLabel(font_id=FONT_ID),
)
GAMEOVER_LABEL = app.world.spawn(
    keel.Transform2D(x=GAMEOVER_LABEL_X, y=GAMEOVER_LABEL_Y),
    keel.TextLabel(font_id=FONT_ID, r=1.0, g=1.0, b=0.2, visible=False),
)
RESTART_LABEL = app.world.spawn(
    keel.Transform2D(x=RESTART_LABEL_X, y=RESTART_LABEL_Y),
    keel.TextLabel(font_id=FONT_ID, r=0.8, g=0.8, b=0.8, visible=False),
)
app.world.flush()

set_text(SCORE_LABEL, "Score: 0")
set_text(LIVES_LABEL, "Lives: 3")
set_text(GAMEOVER_LABEL, "GAME OVER")
set_text(RESTART_LABEL, "Press R to restart")

spawn_ship(app.world, phys)
spawn_wave(app.world, phys, 1)


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

# Read the keyboard. Rotates / thrusts / fires the ship; pumps the restart
# flag while on the game-over screen.
@app.system(keel.Phase.PRE_UPDATE)
def input_system(world, dt):
    if app.input.is_key_down(keel.KEY_ESCAPE):
        app.window.close()

    g = _gs(world)
    if g is None:
        return

    if g["game_over"][0]:
        if app.input.is_key_down(keel.KEY_R):
            g["restart_pending"][0] = True
        return

    if not g["ship_alive"][0]:
        return

    for ts, ships in world.query(keel.Transform2D, Ship):
        for i in range(len(ts)):
            ships["fire_cooldown"][i] = max(0.0, ships["fire_cooldown"][i] - dt)

            rotation_rate_rad = math.radians(float(ships["rotation_speed"][i]))
            if app.input.is_key_down(keel.KEY_LEFT):
                ts["rotation"][i] += rotation_rate_rad * dt
            if app.input.is_key_down(keel.KEY_RIGHT):
                ts["rotation"][i] -= rotation_rate_rad * dt

            angle = float(ts["rotation"][i])   # rotation 0 points at +x
            forward_x = math.cos(angle)
            forward_y = math.sin(angle)

            if app.input.is_key_down(keel.KEY_UP):
                SHIP_VEL["x"] += forward_x * float(ships["thrust"][i]) * dt
                SHIP_VEL["y"] += forward_y * float(ships["thrust"][i]) * dt

            if (
                app.input.is_key_down(keel.KEY_SPACE)
                and ships["fire_cooldown"][i] <= 0.0
            ):
                nose_x = float(ts["x"][i]) + forward_x * NOSE_OFFSET
                nose_y = float(ts["y"][i]) + forward_y * NOSE_OFFSET
                spawn_bullet(
                    world,
                    nose_x,
                    nose_y,
                    forward_x * BULLET_SPEED + SHIP_VEL["x"],
                    forward_y * BULLET_SPEED + SHIP_VEL["y"],
                )
                ships["fire_cooldown"][i] = FIRE_COOLDOWN


# Push the accumulated ship velocity into pymunk (clamped to SHIP_MAX_SPEED).
@app.system(keel.Phase.PRE_UPDATE)
def apply_ship_vel(world, dt):
    speed_sq = SHIP_VEL["x"] ** 2 + SHIP_VEL["y"] ** 2
    max_sq = SHIP_MAX_SPEED * SHIP_MAX_SPEED
    if speed_sq > max_sq:
        scale = SHIP_MAX_SPEED / math.sqrt(speed_sq)
        SHIP_VEL["x"] *= scale
        SHIP_VEL["y"] *= scale

    for sid in SHIP_ENTITIES:
        if world.is_alive(sid):
            phys.set_velocity(sid, SHIP_VEL["x"], SHIP_VEL["y"])


# Re-push each asteroid's stored velocity into pymunk every frame, so they
# keep their constant drift speed even after a kinematic-vs-dynamic bump.
@app.system(keel.Phase.PRE_UPDATE)
def apply_asteroid_vel(world, dt):
    for arch in world.query(Asteroid).archetypes():
        n = arch.length
        a = arch.columns[Asteroid][:n]
        for i in range(n):
            phys.set_velocity(
                int(arch.entities[i]),
                float(a["vel_x"][i]),
                float(a["vel_y"][i]),
            )


# Same idea for bullets — their velocity lives in BULLET_VEL, not on the
# component, so we re-apply it each frame.
@app.system(keel.Phase.PRE_UPDATE)
def apply_bullet_vel(world, dt):
    for bid, v in list(BULLET_VEL.items()):
        if world.is_alive(bid):
            phys.set_velocity(bid, v[0], v[1])


# Wrap the ship + asteroids around the screen; bullets get destroyed instead.
@app.system(keel.Phase.UPDATE)
def wrap_system(world, dt):
    for marker in (Ship, Asteroid):
        for arch in world.query(keel.Transform2D, marker).archetypes():
            n = arch.length
            t = arch.columns[keel.Transform2D][:n]
            for i in range(n):
                x = float(t["x"][i])
                y = float(t["y"][i])
                nx, ny = wrap(x, y)
                if nx != x or ny != y:
                    phys.set_position(int(arch.entities[i]), nx, ny)

    # Bullets that leave the screen die on the spot.
    for arch in world.query(keel.Transform2D, Bullet).archetypes():
        n = arch.length
        t = arch.columns[keel.Transform2D][:n]
        for i in range(n):
            x = float(t["x"][i])
            y = float(t["y"][i])
            if x < 0.0 or x > WIDTH or y < 0.0 or y > HEIGHT:
                queue_despawn(int(arch.entities[i]))


# Tick down each bullet's lifetime; queue expired bullets for despawn.
@app.system(keel.Phase.UPDATE)
def bullet_lifetime(world, dt):
    for arch in world.query(Bullet).archetypes():
        n = arch.length
        bs = arch.columns[Bullet][:n]
        for i in range(n):
            bs["lifetime"][i] -= dt
            if bs["lifetime"][i] <= 0.0:
                queue_despawn(int(arch.entities[i]))


# Count down respawn_timer; when it hits zero either spawn a new ship or
# flip game_over and show the end-screen labels.
@app.system(keel.Phase.UPDATE)
def respawn_system(world, dt):
    g = _gs(world)
    if g is None or g["game_over"][0] or g["restart_pending"][0]:
        return
    if g["ship_alive"][0]:
        return

    remaining = g["respawn_timer"][0] - dt
    if remaining > 0.0:
        g["respawn_timer"][0] = remaining
        return

    if g["lives"][0] <= 0:
        g["game_over"][0] = True
        set_label_visible(world, GAMEOVER_LABEL, True)
        set_label_visible(world, RESTART_LABEL, True)
        return

    spawn_ship(world, phys)
    g["ship_alive"][0] = True
    g["respawn_timer"][0] = 0.0


# When every asteroid is gone, bump the wave counter and spawn the next set.
@app.system(keel.Phase.UPDATE)
def wave_system(world, dt):
    g = _gs(world)
    if g is None or g["game_over"][0] or g["restart_pending"][0]:
        return
    if len(ASTEROID_ENTITIES) > 0:
        return

    g["wave"][0] += 1
    spawn_wave(world, phys, int(g["wave"][0]))


# Decay each ship's invincibility timer + blink the sprite alpha while > 0.
@app.system(keel.Phase.UPDATE)
def invincibility_system(world, dt):
    for ships, sprites in world.query(Ship, keel.Sprite):
        for i in range(len(ships)):
            t = ships["invincible_timer"][i]
            if t > 0.0:
                t = max(0.0, t - dt)
                ships["invincible_timer"][i] = t
                # Blink at ~6 Hz: 0.45 alpha every other half-cycle.
                sprites["a"][i] = 0.45 if (int(t * 12) % 2) else 1.0
            else:
                sprites["a"][i] = 1.0


# Drain CollisionEvent2D and route bullet-vs-asteroid + ship-vs-asteroid hits.
@app.system(keel.Phase.POST_UPDATE)
def collision_system(world, dt):
    g = _gs(world)
    if g is None:
        return

    for event in world.read_events(keel.CollisionEvent2D):
        a = int(event.entity_a)
        b = int(event.entity_b)
        if not world.is_alive(a) or not world.is_alive(b):
            continue

        if a in BULLET_ENTITIES and b in ASTEROID_ENTITIES:
            _hit_asteroid(world, g, a, b)
        elif b in BULLET_ENTITIES and a in ASTEROID_ENTITIES:
            _hit_asteroid(world, g, b, a)
        elif a in SHIP_ENTITIES and b in ASTEROID_ENTITIES:
            _ship_hit(world, g, a)
        elif b in SHIP_ENTITIES and a in ASTEROID_ENTITIES:
            _ship_hit(world, g, b)


# Apply every queued despawn and clean up the tracking sets / dicts.
@app.system(keel.Phase.POST_UPDATE)
def despawn_system(world, dt):
    for eid in DESPAWN_QUEUE:
        if world.is_alive(eid):
            world.despawn(eid)
            world.flush()
        BULLET_VEL.pop(eid, None)
        SHIP_ENTITIES.discard(eid)
        BULLET_ENTITIES.discard(eid)
        ASTEROID_ENTITIES.discard(eid)

    DESPAWN_QUEUE.clear()


# Update the score / lives label strings; trigger restart if R was pressed.
@app.system(keel.Phase.POST_UPDATE)
def text_update(world, dt):
    g = _gs(world)
    if g is None:
        return

    set_text(SCORE_LABEL, f"Score: {int(g['score'][0])}")
    set_text(LIVES_LABEL, f"Lives: {int(g['lives'][0])}")

    if g["restart_pending"][0]:
        restart_game(world, phys)


# ---------------------------------------------------------------------------
# Collision + restart helpers
# ---------------------------------------------------------------------------

def _hit_asteroid(world, g, bullet_eid, ast_eid):
    """A bullet hit an asteroid: award points, queue despawns, and split if large."""
    if bullet_eid in DESPAWN_QUEUE or ast_eid in DESPAWN_QUEUE:
        return

    info = world.get_component(ast_eid, Asteroid)
    pos = world.get_component(ast_eid, keel.Transform2D)
    if info is None or pos is None:
        return

    size = int(info.size)
    g["score"][0] += ASTEROID_SCORE[size]
    print(f"score {int(g['score'][0])}  (asteroid size={size})")

    queue_despawn(bullet_eid)
    queue_despawn(ast_eid)

    # Large + medium asteroids split into two smaller children, deflected
    # symmetrically off the parent's heading.
    if size > 1:
        parent_speed = math.hypot(info.vel_x, info.vel_y) or 60.0
        child_speed = parent_speed * SPLIT_SPEED_BOOST
        base_angle = math.atan2(info.vel_y, info.vel_x)
        for sign in (-1, 1):
            angle = base_angle + sign * SPLIT_ANGLE_OFFSET
            _spawn_asteroid(
                world,
                size - 1,
                float(pos.x),
                float(pos.y),
                math.cos(angle) * child_speed,
                math.sin(angle) * child_speed,
            )


def _ship_hit(world, g, ship_eid):
    """An asteroid hit the ship: cost a life unless invincibility is active."""
    ship = world.get_component(ship_eid, Ship)
    if ship is None or ship.invincible_timer > 0.0:
        return

    queue_despawn(ship_eid)
    g["lives"][0] -= 1
    g["ship_alive"][0] = False
    g["respawn_timer"][0] = RESPAWN_DELAY
    SHIP_VEL["x"] = 0.0
    SHIP_VEL["y"] = 0.0


def restart_game(world, phys):
    """Wipe the field and start a fresh game (called from text_update on R)."""
    # Despawn every gameplay entity in one pass.
    for eid in list(SHIP_ENTITIES | BULLET_ENTITIES | ASTEROID_ENTITIES):
        if world.is_alive(eid):
            world.despawn(eid)
    world.flush()

    SHIP_ENTITIES.clear()
    BULLET_ENTITIES.clear()
    ASTEROID_ENTITIES.clear()
    BULLET_VEL.clear()
    DESPAWN_QUEUE.clear()
    SHIP_VEL["x"] = 0.0
    SHIP_VEL["y"] = 0.0

    # Reset the singleton GameState.
    g = _gs(world)
    if g is not None:
        g["score"][0] = 0
        g["lives"][0] = 3
        g["wave"][0] = 1
        g["game_over"][0] = False
        g["ship_alive"][0] = True
        g["respawn_timer"][0] = 0.0
        g["restart_pending"][0] = False

    set_label_visible(world, GAMEOVER_LABEL, False)
    set_label_visible(world, RESTART_LABEL, False)

    spawn_ship(world, phys)
    spawn_wave(world, phys, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run()
