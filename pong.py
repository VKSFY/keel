"""Pong, built on Keel. Run with: python pong.py

Audit notes from initial development have been resolved.
See README.md for known v0.1 limitations.

Controls:
    W / S            — left paddle up / down
    Up / Down arrow  — right paddle up / down
    Escape           — quit
    First to 7 wins; scores print to terminal.
"""
import math
import random

import keel
from keel.physics import setup_physics_2d
from keel.renderer import setup_renderer_2d


# --- Components ----------------------------------------------------------

@keel.component
class LeftPaddle: pass

@keel.component
class RightPaddle: pass

@keel.component
class Ball: pass

@keel.component
class LeftScore:
    value: int = 0

@keel.component
class RightScore:
    value: int = 0

@keel.component
class GameState:
    running: bool = True
    winner: int = 0       # 0=none, 1=left, 2=right
    reset_timer: float = 0.0


# --- Constants -----------------------------------------------------------

PADDLE_SPEED = 300.0
PADDLE_W, PADDLE_H = 15.0, 80.0
PADDLE_MIN_Y = PADDLE_H * 0.5
PADDLE_MAX_Y = 600.0 - PADDLE_H * 0.5
BALL_RADIUS = 8.0
BALL_INITIAL_SPEED = 250.0
BALL_MAX_MULT = 3.0
WIN_SCORE = 7
RESET_DELAY = 1.5


# --- Setup ---------------------------------------------------------------

app = keel.App(title="Pong", width=800, height=600)
setup_renderer_2d(app)
phys = setup_physics_2d(app, gravity_y=0.0)


# --- State ---------------------------------------------------------------

ENTITIES: dict[str, int] = {}

# multiplier: cumulative speed-up across paddle hits, capped at BALL_MAX_MULT.
# frozen: post-score pause flag. last_hit: dedupes pymunk's continued-contact
# CollisionEvent2D burst so we only deflect once per "new contact".
BALL_STATE = {"multiplier": 1.0, "frozen": False, "last_hit": 0}


def make_random_launch():
    angle = random.uniform(-30.0, 30.0) * math.pi / 180.0
    direction = random.choice([-1.0, 1.0])
    return (direction * BALL_INITIAL_SPEED * math.cos(angle),
            BALL_INITIAL_SPEED * math.sin(angle))


# --- Spawning -----------------------------------------------------------

def spawn_paddle(x, marker_cls):
    # body_type=2 = kinematic. Pymunk drives the bounce direction (elasticity=1
    # for both paddle and ball); our collision_system layers Pong's gameplay
    # rules (y-deflection by hit position + speed-up) on top.
    return app.world.spawn(
        keel.Transform2D(x=x, y=300.0),
        keel.RigidBody2D(body_type=2),
        keel.Collider2D(shape_type=1, width=PADDLE_W, height=PADDLE_H,
                        friction=0.0, elasticity=1.0),
        keel.Sprite(texture_id=0, width=PADDLE_W, height=PADDLE_H),
        marker_cls(),
    )


def spawn_ball(vx, vy):
    return app.world.spawn(
        keel.Transform2D(x=400.0, y=300.0),
        keel.RigidBody2D(mass=1.0, body_type=0, vel_x=vx, vel_y=vy),
        keel.Collider2D(shape_type=0, radius=BALL_RADIUS,
                        friction=0.0, elasticity=1.0),
        keel.Sprite(texture_id=0, width=BALL_RADIUS * 2.0, height=BALL_RADIUS * 2.0,
                    r=1.0, g=0.85, b=0.25),
        Ball(),
    )


def spawn_wall(y):
    return app.world.spawn(
        keel.Transform2D(x=400.0, y=y),
        keel.RigidBody2D(body_type=1),
        keel.Collider2D(shape_type=1, width=800.0, height=10.0,
                        friction=0.0, elasticity=1.0),
    )


_init_vx, _init_vy = make_random_launch()
ENTITIES["left_paddle"]  = spawn_paddle(40.0, LeftPaddle)
ENTITIES["right_paddle"] = spawn_paddle(760.0, RightPaddle)
ENTITIES["ball"]         = spawn_ball(_init_vx, _init_vy)
ENTITIES["top_wall"]     = spawn_wall(595.0)
ENTITIES["bottom_wall"]  = spawn_wall(5.0)
ENTITIES["game"]         = app.world.spawn(GameState(), LeftScore(), RightScore())
app.world.flush()


# --- Systems ------------------------------------------------------------

def _paddle_input_velocity(world, eid, up_key, down_key):
    """Read the up/down keys for one paddle and clamp the resulting velocity
    to zero when the paddle is already at a wall."""
    v = ((PADDLE_SPEED if app.input.is_key_down(up_key)   else 0.0) -
         (PADDLE_SPEED if app.input.is_key_down(down_key) else 0.0))
    t = world.get_component(eid, keel.Transform2D)
    if t is not None:
        if t.y >= PADDLE_MAX_Y and v > 0.0: v = 0.0
        elif t.y <= PADDLE_MIN_Y and v < 0.0: v = 0.0
    return v


@app.system(keel.Phase.PRE_UPDATE)
def input_system(world, dt):
    if app.input.is_key_down(keel.KEY_ESCAPE):
        app.window.close()
    phys.set_velocity(
        ENTITIES["left_paddle"],
        0.0,
        _paddle_input_velocity(world, ENTITIES["left_paddle"], keel.KEY_W, keel.KEY_S),
    )
    phys.set_velocity(
        ENTITIES["right_paddle"],
        0.0,
        _paddle_input_velocity(world, ENTITIES["right_paddle"], keel.KEY_UP, keel.KEY_DOWN),
    )


@app.system(keel.Phase.PRE_UPDATE)
def paddle_clamp_system(world, dt):
    """Snap any paddle that drifted past its boundary back into bounds."""
    for marker in (LeftPaddle, RightPaddle):
        for transforms, _ in world.query(keel.Transform2D, marker):
            ys = transforms["y"]
            for i in range(len(transforms)):
                if ys[i] > PADDLE_MAX_Y: ys[i] = PADDLE_MAX_Y
                elif ys[i] < PADDLE_MIN_Y: ys[i] = PADDLE_MIN_Y


@app.system(keel.Phase.UPDATE)
def ball_out_of_bounds_system(world, dt):
    if BALL_STATE["frozen"]:
        return
    ball_t = world.get_component(ENTITIES["ball"], keel.Transform2D)
    if ball_t is None:
        return
    for arch in world.query(GameState, LeftScore, RightScore).archetypes():
        n = arch.length
        gs = arch.columns[GameState][:n]
        ls = arch.columns[LeftScore][:n]
        rs = arch.columns[RightScore][:n]
        for i in range(n):
            if float(gs["reset_timer"][i]) > 0.0:
                continue
            scored = False
            if ball_t.x < 0.0:
                rs["value"][i] = int(rs["value"][i]) + 1
                scored = True
            elif ball_t.x > 800.0:
                ls["value"][i] = int(ls["value"][i]) + 1
                scored = True
            if scored:
                gs["reset_timer"][i] = RESET_DELAY
                BALL_STATE["frozen"] = True
                phys.set_velocity(ENTITIES["ball"], 0.0, 0.0)
                print(f"[Pong] Left: {int(ls['value'][i])}  Right: {int(rs['value'][i])}")


@app.system(keel.Phase.UPDATE)
def reset_timer_system(world, dt):
    for arch in world.query(GameState).archetypes():
        n = arch.length
        gs = arch.columns[GameState][:n]
        for i in range(n):
            t = float(gs["reset_timer"][i])
            if t <= 0.0:
                continue
            new_t = max(0.0, t - dt)
            gs["reset_timer"][i] = new_t
            if new_t == 0.0:
                vx, vy = make_random_launch()
                BALL_STATE["multiplier"] = 1.0
                BALL_STATE["frozen"] = False
                BALL_STATE["last_hit"] = 0
                phys.set_position(ENTITIES["ball"], 400.0, 300.0)
                phys.set_velocity(ENTITIES["ball"], vx, vy)


@app.system(keel.Phase.POST_UPDATE)
def collision_system(world, dt):
    """Layer Pong gameplay on top of pymunk's natural paddle bounce: y-deflection
    by hit position + cumulative speed multiplier. Pymunk has already flipped vx
    via paddle elasticity=1.0 by the time we read the ball's velocity."""
    if BALL_STATE["frozen"]:
        for _ in world.read_events(keel.CollisionEvent2D):
            pass
        return

    ball_eid = ENTITIES["ball"]
    left_eid = ENTITIES["left_paddle"]
    right_eid = ENTITIES["right_paddle"]
    ball_t = world.get_component(ball_eid, keel.Transform2D)
    if ball_t is None:
        return

    # Once the ball has moved well past the paddle that hit it, allow re-deflect.
    if BALL_STATE["last_hit"] == left_eid and ball_t.x > 100.0:
        BALL_STATE["last_hit"] = 0
    elif BALL_STATE["last_hit"] == right_eid and ball_t.x < 700.0:
        BALL_STATE["last_hit"] = 0

    for event in world.read_events(keel.CollisionEvent2D):
        a, b = int(event.entity_a), int(event.entity_b)
        if ball_eid not in (a, b):
            continue
        other = b if a == ball_eid else a
        if other not in (left_eid, right_eid):
            continue
        if BALL_STATE["last_hit"] == other:
            continue  # still in contact — pymunk's post_solve fires every tick

        paddle_t = world.get_component(other, keel.Transform2D)
        ball_rb = world.get_component(ball_eid, keel.RigidBody2D)
        if paddle_t is None or ball_rb is None:
            continue

        vx, vy = ball_rb.vel_x, ball_rb.vel_y
        rel_y = ball_t.y - paddle_t.y
        if rel_y > PADDLE_H * 0.166:
            new_vy = abs(vx) * 0.5
        elif rel_y < -PADDLE_H * 0.166:
            new_vy = -abs(vx) * 0.5
        else:
            new_vy = vy

        BALL_STATE["multiplier"] = min(BALL_STATE["multiplier"] * 1.1, BALL_MAX_MULT)
        target = BALL_INITIAL_SPEED * BALL_STATE["multiplier"]
        mag = math.sqrt(vx * vx + new_vy * new_vy)
        if mag > 1e-6:
            scale = target / mag
            phys.set_velocity(ball_eid, vx * scale, new_vy * scale)

        BALL_STATE["last_hit"] = other


@app.system(keel.Phase.UPDATE)
def win_check_system(world, dt):
    for arch in world.query(GameState, LeftScore, RightScore).archetypes():
        n = arch.length
        gs = arch.columns[GameState][:n]
        ls = arch.columns[LeftScore][:n]
        rs = arch.columns[RightScore][:n]
        for i in range(n):
            left, right = int(ls["value"][i]), int(rs["value"][i])
            if left < WIN_SCORE and right < WIN_SCORE:
                continue
            print("[Pong] Left wins!" if left >= WIN_SCORE else "[Pong] Right wins!")
            ls["value"][i] = 0
            rs["value"][i] = 0
            gs["winner"][i] = 1 if left >= WIN_SCORE else 2
            gs["reset_timer"][i] = RESET_DELAY
            BALL_STATE["frozen"] = True
            BALL_STATE["last_hit"] = 0
            phys.set_velocity(ENTITIES["ball"], 0.0, 0.0)


app.run()
