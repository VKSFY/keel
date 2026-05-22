"""asteroids.py — Asteroids in Keel (v2: wrap + text fixes).
Run with:  python asteroids.py
Arrows rotate / thrust, Space fires, R restarts on game over, Esc quits.
"""
import math, random
import keel
from keel.physics import setup_physics_2d
from keel.renderer import setup_renderer_2d
from keel.text import BUILTIN_FONT, load_font, set_label_visible, set_text, setup_text

WIDTH, HEIGHT = 800, 600
SHIP_ROT_DEG_PER_S = 180.0
SHIP_THRUST = 300.0
SHIP_MAX_SPEED = 400.0
BULLET_SPEED = 500.0
BULLET_LIFETIME = 3.0
FIRE_COOLDOWN = 0.18
RESPAWN_DELAY = 2.0
INVINCIBLE_TIME = 2.0
NOSE_OFFSET = 18.0
# Body types: Keel has DYNAMIC=0, STATIC=1, KINEMATIC=2. pymunk's collision
# callbacks DO NOT fire for KINEMATIC-vs-KINEMATIC pairs, so bullets and
# asteroids must be DYNAMIC for CollisionEvent2D to reach our handler.
DYNAMIC = 0
KINEMATIC = 2
# Collision filter bits — ship<->everything, asteroids ignore each other,
# bullets only collide with asteroids (no self-firing, no bullet-bullet).
CAT_SHIP, CAT_ASTEROID, CAT_BULLET = 1, 2, 4
MASK_SHIP = 0xFFFF
MASK_ASTEROID = 0xFFFF ^ CAT_ASTEROID
MASK_BULLET = CAT_ASTEROID                    # bullets see only asteroids
ASTEROID_RADIUS = {3: 40.0, 2: 22.0, 1: 12.0}
ASTEROID_PX    = {3: 80.0, 2: 44.0, 1: 24.0}
ASTEROID_SCORE = {3: 20, 2: 50, 1: 100}
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
app = keel.App(title="Asteroids", width=WIDTH, height=HEIGHT)
setup_renderer_2d(app)                                    # MUST precede setup_text
phys = setup_physics_2d(app, gravity_y=0.0)
text_setup = setup_text(app)
_font = load_font(app, BUILTIN_FONT, size_px=28)
FONT_ID = text_setup.font_registry.id_of(_font)           # NOT 0 by assumption
SHIP_VEL = {"x": 0.0, "y": 0.0}
BULLET_VEL: dict[int, tuple[float, float]] = {}
DESPAWN_QUEUE: list[int] = []
SHIP_ENTITIES: set[int] = set()
BULLET_ENTITIES: set[int] = set()
ASTEROID_ENTITIES: set[int] = set()
SCORE_LABEL = LIVES_LABEL = GAMEOVER_LABEL = RESTART_LABEL = 0

def queue_despawn(eid):
    if eid not in DESPAWN_QUEUE: DESPAWN_QUEUE.append(eid)
def wrap(x, y):
    """Zero-margin screen wrap. Entity reappears at the exact opposite edge."""
    if x < 0: x += WIDTH
    if x > WIDTH: x -= WIDTH
    if y < 0: y += HEIGHT
    if y > HEIGHT: y -= HEIGHT
    return x, y
def _gs(world):
    for (g,) in world.query(GameState):
        if len(g) > 0: return g
    return None
# TextLabel visibility now goes through keel.text.set_label_visible (wraps
# world.set under the hood). The bespoke archetype-surgery helper this file
# used to carry is gone.

def spawn_ship(world, _phys):
    SHIP_VEL["x"] = SHIP_VEL["y"] = 0.0
    eid = world.spawn(
        keel.Transform2D(x=WIDTH/2, y=HEIGHT/2, rotation=0.0),
        keel.RigidBody2D(mass=1.0, body_type=KINEMATIC),
        keel.Collider2D(shape_type=0, radius=14.0, elasticity=0.0, friction=0.0,
                        category_bits=CAT_SHIP, mask_bits=MASK_SHIP),
        keel.Sprite(texture_id=0, width=28.0, height=28.0, r=1.0, g=1.0, b=1.0),
        Ship(invincible_timer=INVINCIBLE_TIME))
    world.flush()
    SHIP_ENTITIES.add(eid)
    return eid
def spawn_bullet(world, x, y, vx, vy):
    eid = world.spawn(
        keel.Transform2D(x=x, y=y),
        # DYNAMIC so the bullet-vs-dynamic-asteroid collision callback fires.
        keel.RigidBody2D(mass=0.1, body_type=DYNAMIC),
        keel.Collider2D(shape_type=0, radius=3.0, elasticity=0.0, friction=0.0,
                        category_bits=CAT_BULLET, mask_bits=MASK_BULLET),
        keel.Sprite(texture_id=0, width=6.0, height=6.0, r=1.0, g=0.85, b=0.4),
        Bullet(lifetime=BULLET_LIFETIME))
    world.flush()
    BULLET_VEL[eid] = (vx, vy)
    BULLET_ENTITIES.add(eid)
    return eid
def _spawn_asteroid(world, size, x, y, vx, vy):
    r, s = ASTEROID_RADIUS[size], ASTEROID_PX[size]
    eid = world.spawn(
        keel.Transform2D(x=x, y=y, rotation=random.uniform(0.0, math.tau)),
        # DYNAMIC because kinematic-vs-kinematic pairs don't emit collision
        # events in pymunk. apply_asteroid_vel re-pushes the stored velocity
        # each frame so the asteroid keeps its constant drift speed.
        keel.RigidBody2D(mass=float(size), body_type=DYNAMIC),
        keel.Collider2D(shape_type=0, radius=r, elasticity=0.0, friction=0.0,
                        category_bits=CAT_ASTEROID, mask_bits=MASK_ASTEROID),
        keel.Sprite(texture_id=0, width=s, height=s, r=0.78, g=0.78, b=0.82),
        Asteroid(size=size, vel_x=vx, vel_y=vy))
    world.flush()
    ASTEROID_ENTITIES.add(eid)
    return eid
def spawn_asteroid(world, _phys, size, x, y):
    speed = random.uniform(40.0, 80.0) * (4 - size)
    a = random.uniform(0.0, math.tau)
    return _spawn_asteroid(world, size, x, y, math.cos(a)*speed, math.sin(a)*speed)
def spawn_wave(world, phys, wave):
    cx, cy = WIDTH/2.0, HEIGHT/2.0
    for _ in range(3 + wave):
        for _try in range(50):
            x = random.uniform(0.0, WIDTH); y = random.uniform(0.0, HEIGHT)
            if math.hypot(x - cx, y - cy) > 150.0: break
        spawn_asteroid(world, phys, 3, x, y)
# Initial entities (text positions are in SCREEN space: y=0 is top).
app.world.spawn(GameState())
# Transform2D.y for a TextLabel is the BASELINE, so we need to push it
# down by ~ascender (~25 px at size 28) to keep the top of the glyphs on screen.
SCORE_LABEL = app.world.spawn(
    keel.Transform2D(x=10.0, y=35.0), keel.TextLabel(font_id=FONT_ID))
LIVES_LABEL = app.world.spawn(
    keel.Transform2D(x=590.0, y=35.0), keel.TextLabel(font_id=FONT_ID))
GAMEOVER_LABEL = app.world.spawn(
    keel.Transform2D(x=280.0, y=270.0),
    keel.TextLabel(font_id=FONT_ID, r=1.0, g=1.0, b=0.2, visible=False))
RESTART_LABEL = app.world.spawn(
    keel.Transform2D(x=210.0, y=320.0),
    keel.TextLabel(font_id=FONT_ID, r=0.8, g=0.8, b=0.8, visible=False))
app.world.flush()
set_text(SCORE_LABEL, "Score: 0")
set_text(LIVES_LABEL, "Lives: 3")
set_text(GAMEOVER_LABEL, "GAME OVER")
set_text(RESTART_LABEL, "Press R to restart")
spawn_ship(app.world, phys)
spawn_wave(app.world, phys, 1)
@app.system(keel.Phase.PRE_UPDATE)
def input_system(world, dt):
    if app.input.is_key_down(keel.KEY_ESCAPE): app.window.close()
    g = _gs(world)
    if g is None: return
    if g["game_over"][0]:
        if app.input.is_key_down(keel.KEY_R): g["restart_pending"][0] = True
        return
    if not g["ship_alive"][0]: return
    for ts, ships in world.query(keel.Transform2D, Ship):
        for i in range(len(ts)):
            ships["fire_cooldown"][i] = max(0.0, ships["fire_cooldown"][i] - dt)
            rps = math.radians(float(ships["rotation_speed"][i]))
            if app.input.is_key_down(keel.KEY_LEFT):  ts["rotation"][i] += rps * dt
            if app.input.is_key_down(keel.KEY_RIGHT): ts["rotation"][i] -= rps * dt
            angle = float(ts["rotation"][i])       # rotation 0 = +x
            cx, cy = math.cos(angle), math.sin(angle)
            if app.input.is_key_down(keel.KEY_UP):
                SHIP_VEL["x"] += cx * float(ships["thrust"][i]) * dt
                SHIP_VEL["y"] += cy * float(ships["thrust"][i]) * dt
            if app.input.is_key_down(keel.KEY_SPACE) and ships["fire_cooldown"][i] <= 0.0:
                bx = float(ts["x"][i]) + cx * NOSE_OFFSET
                by = float(ts["y"][i]) + cy * NOSE_OFFSET
                spawn_bullet(world, bx, by,
                             cx * BULLET_SPEED + SHIP_VEL["x"],
                             cy * BULLET_SPEED + SHIP_VEL["y"])
                ships["fire_cooldown"][i] = FIRE_COOLDOWN
@app.system(keel.Phase.PRE_UPDATE)
def apply_ship_vel(world, dt):
    s2 = SHIP_VEL["x"]**2 + SHIP_VEL["y"]**2
    if s2 > SHIP_MAX_SPEED * SHIP_MAX_SPEED:
        k = SHIP_MAX_SPEED / math.sqrt(s2)
        SHIP_VEL["x"] *= k; SHIP_VEL["y"] *= k
    for sid in SHIP_ENTITIES:
        if world.is_alive(sid):
            phys.set_velocity(sid, SHIP_VEL["x"], SHIP_VEL["y"])
@app.system(keel.Phase.PRE_UPDATE)
def apply_asteroid_vel(world, dt):
    for arch in world.query(Asteroid).archetypes():
        n = arch.length
        a = arch.columns[Asteroid][:n]
        for i in range(n):
            phys.set_velocity(int(arch.entities[i]),
                              float(a["vel_x"][i]), float(a["vel_y"][i]))
@app.system(keel.Phase.PRE_UPDATE)
def apply_bullet_vel(world, dt):
    for bid, v in list(BULLET_VEL.items()):
        if world.is_alive(bid):
            phys.set_velocity(bid, v[0], v[1])
@app.system(keel.Phase.UPDATE)
def wrap_system(world, dt):
    """Wrap ship + asteroids at ZERO margin; bullets are DESPAWNED on exit."""
    # Wrap ship + asteroids; sync both Transform2D and pymunk body.
    for marker in (Ship, Asteroid):
        for arch in world.query(keel.Transform2D, marker).archetypes():
            n = arch.length
            t = arch.columns[keel.Transform2D][:n]
            for i in range(n):
                x, y = float(t["x"][i]), float(t["y"][i])
                nx, ny = wrap(x, y)
                if nx != x or ny != y:
                    phys.set_position(int(arch.entities[i]), nx, ny)
    # Bullets that leave the screen die on the spot.
    for arch in world.query(keel.Transform2D, Bullet).archetypes():
        n = arch.length
        t = arch.columns[keel.Transform2D][:n]
        for i in range(n):
            x, y = float(t["x"][i]), float(t["y"][i])
            if x < 0.0 or x > WIDTH or y < 0.0 or y > HEIGHT:
                queue_despawn(int(arch.entities[i]))
@app.system(keel.Phase.UPDATE)
def bullet_lifetime(world, dt):
    for arch in world.query(Bullet).archetypes():
        n = arch.length
        bs = arch.columns[Bullet][:n]
        for i in range(n):
            bs["lifetime"][i] -= dt
            if bs["lifetime"][i] <= 0.0:
                queue_despawn(int(arch.entities[i]))
@app.system(keel.Phase.UPDATE)
def respawn_system(world, dt):
    g = _gs(world)
    if g is None or g["game_over"][0] or g["restart_pending"][0]: return
    if g["ship_alive"][0]: return
    t = g["respawn_timer"][0] - dt
    if t > 0.0:
        g["respawn_timer"][0] = t
        return
    if g["lives"][0] <= 0:
        g["game_over"][0] = True
        set_label_visible(world, GAMEOVER_LABEL, True)
        set_label_visible(world, RESTART_LABEL, True)
        return
    spawn_ship(world, phys)
    g["ship_alive"][0] = True
    g["respawn_timer"][0] = 0.0
@app.system(keel.Phase.UPDATE)
def wave_system(world, dt):
    g = _gs(world)
    if g is None or g["game_over"][0] or g["restart_pending"][0]: return
    if len(ASTEROID_ENTITIES) > 0: return
    g["wave"][0] += 1
    spawn_wave(world, phys, int(g["wave"][0]))
@app.system(keel.Phase.UPDATE)
def invincibility_system(world, dt):
    for ships, sprites in world.query(Ship, keel.Sprite):
        for i in range(len(ships)):
            t = ships["invincible_timer"][i]
            if t > 0.0:
                t = max(0.0, t - dt)
                ships["invincible_timer"][i] = t
                sprites["a"][i] = 0.45 if (int(t * 12) % 2) else 1.0
            else:
                sprites["a"][i] = 1.0
@app.system(keel.Phase.POST_UPDATE)
def collision_system(world, dt):
    g = _gs(world)
    if g is None: return
    for event in world.read_events(keel.CollisionEvent2D):
        a, b = int(event.entity_a), int(event.entity_b)
        if not world.is_alive(a) or not world.is_alive(b): continue
        if a in BULLET_ENTITIES and b in ASTEROID_ENTITIES:
            _hit_asteroid(world, g, a, b)
        elif b in BULLET_ENTITIES and a in ASTEROID_ENTITIES:
            _hit_asteroid(world, g, b, a)
        elif a in SHIP_ENTITIES and b in ASTEROID_ENTITIES:
            _ship_hit(world, g, a)
        elif b in SHIP_ENTITIES and a in ASTEROID_ENTITIES:
            _ship_hit(world, g, b)
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
@app.system(keel.Phase.POST_UPDATE)
def text_update(world, dt):
    g = _gs(world)
    if g is None: return
    set_text(SCORE_LABEL, f"Score: {int(g['score'][0])}")
    set_text(LIVES_LABEL, f"Lives: {int(g['lives'][0])}")
    if g["restart_pending"][0]: restart_game(world, phys)
def _hit_asteroid(world, g, bullet_eid, ast_eid):
    if bullet_eid in DESPAWN_QUEUE or ast_eid in DESPAWN_QUEUE: return
    info = world.get_component(ast_eid, Asteroid)
    pos = world.get_component(ast_eid, keel.Transform2D)
    if info is None or pos is None: return
    size = int(info.size)
    g["score"][0] += ASTEROID_SCORE[size]
    print(f"score {int(g['score'][0])}  (asteroid size={size})")
    queue_despawn(bullet_eid); queue_despawn(ast_eid)
    if size > 1:
        speed = (math.hypot(info.vel_x, info.vel_y) or 60.0) * 1.2
        base = math.atan2(info.vel_y, info.vel_x)
        for offset in (-0.4, 0.4):
            ang = base + offset
            _spawn_asteroid(world, size - 1, float(pos.x), float(pos.y),
                            math.cos(ang) * speed, math.sin(ang) * speed)
def _ship_hit(world, g, ship_eid):
    ship = world.get_component(ship_eid, Ship)
    if ship is None or ship.invincible_timer > 0.0: return
    queue_despawn(ship_eid)
    g["lives"][0] -= 1
    g["ship_alive"][0] = False
    g["respawn_timer"][0] = RESPAWN_DELAY
    SHIP_VEL["x"] = SHIP_VEL["y"] = 0.0
def restart_game(world, phys):
    for eid in list(SHIP_ENTITIES | BULLET_ENTITIES | ASTEROID_ENTITIES):
        if world.is_alive(eid): world.despawn(eid)
    world.flush()
    SHIP_ENTITIES.clear(); BULLET_ENTITIES.clear(); ASTEROID_ENTITIES.clear()
    BULLET_VEL.clear(); DESPAWN_QUEUE.clear()
    SHIP_VEL["x"] = SHIP_VEL["y"] = 0.0
    g = _gs(world)
    if g is not None:
        g["score"][0] = 0; g["lives"][0] = 3; g["wave"][0] = 1
        g["game_over"][0] = False; g["ship_alive"][0] = True
        g["respawn_timer"][0] = 0.0; g["restart_pending"][0] = False
    set_label_visible(world, GAMEOVER_LABEL, False)
    set_label_visible(world, RESTART_LABEL, False)
    spawn_ship(world, phys)
    spawn_wave(world, phys, 1)

if __name__ == "__main__":
    app.run()
