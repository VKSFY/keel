"""Pong, built on Keel. Run with: python pong.py

Controls
--------
W / S            : left paddle up / down
Up / Down arrow  : right paddle up / down
Escape           : quit

First to 7 wins; scores print to terminal.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import math
import random

import keel
from keel.physics import setup_physics_2d
from keel.renderer import setup_renderer_2d


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Window
WIDTH = 800
HEIGHT = 600
SCREEN_CX = WIDTH / 2.0
SCREEN_CY = HEIGHT / 2.0

# Paddle geometry
PADDLE_W = 15.0
PADDLE_H = 80.0
LEFT_PADDLE_X = 40.0
RIGHT_PADDLE_X = 760.0
PADDLE_SPEED = 300.0
PADDLE_MIN_Y = PADDLE_H * 0.5
PADDLE_MAX_Y = HEIGHT - PADDLE_H * 0.5

# Ball
BALL_RADIUS = 8.0
BALL_INITIAL_SPEED = 250.0
BALL_MAX_MULT = 3.0           # cumulative speed cap across paddle hits
BALL_SPEED_BOOST = 1.1        # multiplier added on each paddle bounce
BALL_DEFLECT_ZONE = 1.0 / 6.0 # |rel_y| > PADDLE_H * this gives a y-deflection
LAUNCH_ANGLE_DEG = 30.0       # ball launches within ±this many degrees of horizontal

# Walls
TOP_WALL_Y = HEIGHT - 5.0
BOTTOM_WALL_Y = 5.0
WALL_THICKNESS = 10.0

# Game flow
WIN_SCORE = 7
RESET_DELAY = 1.5

# Body and shape type enums from Keel — IntEnum so they're drop-in replacements
# for the raw ints we used in older Pong revisions.
DYNAMIC = keel.BodyType.DYNAMIC
STATIC = keel.BodyType.STATIC
KINEMATIC = keel.BodyType.KINEMATIC
BOX = keel.ShapeType2D.BOX
CIRCLE = keel.ShapeType2D.CIRCLE


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

@keel.component
class LeftPaddle:
    pass


@keel.component
class RightPaddle:
    pass


@keel.component
class Ball:
    pass


@keel.component
class LeftScore:
    value: int = 0


@keel.component
class RightScore:
    value: int = 0


@keel.component
class GameState:
    running: bool = True
    winner: int = 0           # 0 = none, 1 = left, 2 = right
    reset_timer: float = 0.0


# ---------------------------------------------------------------------------
# App + subsystem setup
# ---------------------------------------------------------------------------

app = keel.App(title="Pong", width=WIDTH, height=HEIGHT)
setup_renderer_2d(app)
phys = setup_physics_2d(app, gravity_y=0.0)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Entity ids by role — populated once below, used by every system.
ENTITIES: dict[str, int] = {}

# Per-rally ball state that won't fit inside a numpy-backed component:
#   multiplier — speed multiplier accumulated across paddle hits, capped at
#                BALL_MAX_MULT.
#   frozen     — True during the post-score / post-win pause.
#   last_hit   — entity id of the paddle that most recently deflected the
#                ball, to dedupe pymunk's continued-contact CollisionEvent2D
#                burst (we only deflect once per "new contact").
BALL_STATE = {"multiplier": 1.0, "frozen": False, "last_hit": 0}


# ---------------------------------------------------------------------------
# Spawn helpers
# ---------------------------------------------------------------------------

def make_random_launch():
    """Pick a launch velocity within ±LAUNCH_ANGLE_DEG of pure horizontal."""
    angle = math.radians(random.uniform(-LAUNCH_ANGLE_DEG, LAUNCH_ANGLE_DEG))
    direction = random.choice([-1.0, 1.0])
    vx = direction * BALL_INITIAL_SPEED * math.cos(angle)
    vy = BALL_INITIAL_SPEED * math.sin(angle)
    return vx, vy


def spawn_paddle(x, marker_cls):
    """Spawn a kinematic paddle at horizontal `x`, vertically centred."""
    # KINEMATIC bodies are driven by phys.set_velocity each frame. Pymunk
    # still handles the ball bounce (paddle elasticity=1.0); our
    # collision_system layers Pong's y-deflection + speed-up on top.
    return app.world.spawn(
        keel.Transform2D(x=x, y=SCREEN_CY),
        keel.RigidBody2D(body_type=KINEMATIC),
        keel.Collider2D(
            shape_type=BOX,
            width=PADDLE_W,
            height=PADDLE_H,
            friction=0.0,
            elasticity=1.0,
        ),
        keel.Sprite(texture_id=0, width=PADDLE_W, height=PADDLE_H),
        marker_cls(),
    )


def spawn_ball(vx, vy):
    """Spawn the ball at screen center with the given launch velocity."""
    return app.world.spawn(
        keel.Transform2D(x=SCREEN_CX, y=SCREEN_CY),
        keel.RigidBody2D(mass=1.0, body_type=DYNAMIC, vel_x=vx, vel_y=vy),
        keel.Collider2D(
            shape_type=CIRCLE,
            radius=BALL_RADIUS,
            friction=0.0,
            elasticity=1.0,
        ),
        keel.Sprite(
            texture_id=0,
            width=BALL_RADIUS * 2.0,
            height=BALL_RADIUS * 2.0,
            r=1.0,
            g=0.85,
            b=0.25,
        ),
        Ball(),
    )


def spawn_wall(y):
    """Spawn a horizontal static wall spanning the full window width."""
    return app.world.spawn(
        keel.Transform2D(x=SCREEN_CX, y=y),
        keel.RigidBody2D(body_type=STATIC),
        keel.Collider2D(
            shape_type=BOX,
            width=float(WIDTH),
            height=WALL_THICKNESS,
            friction=0.0,
            elasticity=1.0,
        ),
    )


# ---------------------------------------------------------------------------
# Initial entities
# ---------------------------------------------------------------------------

_init_vx, _init_vy = make_random_launch()
ENTITIES["left_paddle"]  = spawn_paddle(LEFT_PADDLE_X, LeftPaddle)
ENTITIES["right_paddle"] = spawn_paddle(RIGHT_PADDLE_X, RightPaddle)
ENTITIES["ball"]         = spawn_ball(_init_vx, _init_vy)
ENTITIES["top_wall"]     = spawn_wall(TOP_WALL_Y)
ENTITIES["bottom_wall"]  = spawn_wall(BOTTOM_WALL_Y)
ENTITIES["game"]         = app.world.spawn(GameState(), LeftScore(), RightScore())
app.world.flush()


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def _paddle_input_velocity(world, eid, up_key, down_key):
    """Read up/down keys and clamp velocity to zero when the paddle is at a wall."""
    up = PADDLE_SPEED if app.input.is_key_down(up_key) else 0.0
    down = PADDLE_SPEED if app.input.is_key_down(down_key) else 0.0
    v = up - down

    t = world.get_component(eid, keel.Transform2D)
    if t is not None:
        if t.y >= PADDLE_MAX_Y and v > 0.0:
            v = 0.0
        elif t.y <= PADDLE_MIN_Y and v < 0.0:
            v = 0.0
    return v


# Read the keyboard and write paddle velocities into pymunk.
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


# Snap any paddle that drifted past its top/bottom limit back into bounds.
@app.system(keel.Phase.PRE_UPDATE)
def paddle_clamp_system(world, dt):
    for marker in (LeftPaddle, RightPaddle):
        for transforms, _ in world.query(keel.Transform2D, marker):
            ys = transforms["y"]
            for i in range(len(transforms)):
                if ys[i] > PADDLE_MAX_Y:
                    ys[i] = PADDLE_MAX_Y
                elif ys[i] < PADDLE_MIN_Y:
                    ys[i] = PADDLE_MIN_Y


# Award a point when the ball leaves either side of the window.
@app.system(keel.Phase.UPDATE)
def ball_out_of_bounds_system(world, dt):
    if BALL_STATE["frozen"]:
        return

    ball_t = world.get_component(ENTITIES["ball"], keel.Transform2D)
    if ball_t is None:
        return

    gs = world.query_one(GameState)
    if gs is None or gs["reset_timer"] > 0.0:
        return

    scored_right = ball_t.x < 0.0
    scored_left = ball_t.x > WIDTH
    if not (scored_right or scored_left):
        return

    ls = world.query_one(LeftScore) or {"value": 0}
    rs = world.query_one(RightScore) or {"value": 0}
    left = ls["value"]
    right = rs["value"]
    if scored_right:
        right += 1
        world.set(ENTITIES["game"], RightScore, value=right)
    else:
        left += 1
        world.set(ENTITIES["game"], LeftScore, value=left)

    world.set(ENTITIES["game"], GameState, reset_timer=RESET_DELAY)
    BALL_STATE["frozen"] = True
    phys.set_velocity(ENTITIES["ball"], 0.0, 0.0)
    print(f"[Pong] Left: {left}  Right: {right}")


# Count down the post-score pause; when it hits 0, relaunch the ball.
@app.system(keel.Phase.UPDATE)
def reset_timer_system(world, dt):
    gs = world.query_one(GameState)
    if gs is None or gs["reset_timer"] <= 0.0:
        return

    new_t = max(0.0, gs["reset_timer"] - dt)
    world.set(ENTITIES["game"], GameState, reset_timer=new_t)

    if new_t == 0.0:
        vx, vy = make_random_launch()
        BALL_STATE["multiplier"] = 1.0
        BALL_STATE["frozen"] = False
        BALL_STATE["last_hit"] = 0
        phys.set_position(ENTITIES["ball"], SCREEN_CX, SCREEN_CY)
        phys.set_velocity(ENTITIES["ball"], vx, vy)


# Layer Pong gameplay on top of pymunk's natural bounce: deflect by hit y
# offset, multiply ball speed slightly, dedupe continued-contact events.
@app.system(keel.Phase.POST_UPDATE)
def collision_system(world, dt):
    if BALL_STATE["frozen"]:
        # Still drain the event queue so it doesn't grow unbounded.
        for _ in world.read_events(keel.CollisionEvent2D):
            pass
        return

    ball_eid = ENTITIES["ball"]
    left_eid = ENTITIES["left_paddle"]
    right_eid = ENTITIES["right_paddle"]

    ball_t = world.get_component(ball_eid, keel.Transform2D)
    if ball_t is None:
        return

    # Clear the "last hit" lock once the ball has moved well past the paddle
    # that hit it — that way a paddle CAN re-hit a ball that came back around.
    if BALL_STATE["last_hit"] == left_eid and ball_t.x > 100.0:
        BALL_STATE["last_hit"] = 0
    elif BALL_STATE["last_hit"] == right_eid and ball_t.x < 700.0:
        BALL_STATE["last_hit"] = 0

    for event in world.read_events(keel.CollisionEvent2D):
        a = int(event.entity_a)
        b = int(event.entity_b)
        if ball_eid not in (a, b):
            continue

        other = b if a == ball_eid else a
        if other not in (left_eid, right_eid):
            continue

        if BALL_STATE["last_hit"] == other:
            # Pymunk's post_solve fires every tick while contact persists;
            # we already deflected on the first frame of this contact.
            continue

        paddle_t = world.get_component(other, keel.Transform2D)
        ball_rb = world.get_component(ball_eid, keel.RigidBody2D)
        if paddle_t is None or ball_rb is None:
            continue

        # Pymunk has already flipped vx via paddle elasticity=1.0 by the
        # time we read the velocity here. We tweak vy based on where on the
        # paddle the ball struck, then re-normalize to the target speed.
        vx, vy = ball_rb.vel_x, ball_rb.vel_y
        rel_y = ball_t.y - paddle_t.y

        if rel_y > PADDLE_H * BALL_DEFLECT_ZONE:
            new_vy = abs(vx) * 0.5
        elif rel_y < -PADDLE_H * BALL_DEFLECT_ZONE:
            new_vy = -abs(vx) * 0.5
        else:
            new_vy = vy

        BALL_STATE["multiplier"] = min(
            BALL_STATE["multiplier"] * BALL_SPEED_BOOST,
            BALL_MAX_MULT,
        )
        target_speed = BALL_INITIAL_SPEED * BALL_STATE["multiplier"]

        mag = math.sqrt(vx * vx + new_vy * new_vy)
        if mag > 1e-6:
            scale = target_speed / mag
            phys.set_velocity(ball_eid, vx * scale, new_vy * scale)

        BALL_STATE["last_hit"] = other


# Detect WIN_SCORE — announce the winner, reset scores, freeze for one round.
@app.system(keel.Phase.UPDATE)
def win_check_system(world, dt):
    ls = world.query_one(LeftScore)
    rs = world.query_one(RightScore)
    if ls is None or rs is None:
        return
    left, right = ls["value"], rs["value"]
    if left < WIN_SCORE and right < WIN_SCORE:
        return

    print("[Pong] Left wins!" if left >= WIN_SCORE else "[Pong] Right wins!")
    world.set(ENTITIES["game"], LeftScore, value=0)
    world.set(ENTITIES["game"], RightScore, value=0)
    world.set(
        ENTITIES["game"],
        GameState,
        winner=1 if left >= WIN_SCORE else 2,
        reset_timer=RESET_DELAY,
    )
    BALL_STATE["frozen"] = True
    BALL_STATE["last_hit"] = 0
    phys.set_velocity(ENTITIES["ball"], 0.0, 0.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app.run()
