"""devtools_overlay.py — F1 inspector, F2 profiler, F3 physics debug draw.

Shows the dev tooling running on top of a non-trivial ECS workload. Drops
50 coloured balls into an 800×600 box, lets pymunk simulate, and turns on
every overlay through one call to keel.dev_tools(app).

Run with:
    python examples/devtools_overlay.py

Controls
--------
F1     : toggle World Inspector window (entity / component browser)
F2     : toggle Profiler overlay (per-system avg ms with bars)
F3     : toggle physics debug draw (collider outlines, by body type)
Escape : quit
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import random

import keel
from keel.physics import setup_physics_2d
from keel.renderer import setup_renderer_2d


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIDTH = 800
HEIGHT = 600

BALL_COUNT = 50
BALL_RADIUS = 12.0
BALL_SPEED_RANGE = (-300.0, 300.0)
BALL_SPAWN_PADDING = 40.0          # keep ball spawn inside the wall border

WALL_THICKNESS = 10.0

# Body type ids used by Keel's RigidBody2D.
DYNAMIC = 0
STATIC = 1


# ---------------------------------------------------------------------------
# App + subsystem setup
# ---------------------------------------------------------------------------

app = keel.App(title="DevTools Overlay", width=WIDTH, height=HEIGHT)
setup_renderer_2d(app)

# Zero gravity + zero friction + perfect elasticity = chaotic perpetual
# motion. Exactly what we want for a tooling demo: the profiler always has
# something to measure and the debug draw always shows shifting outlines.
setup_physics_2d(app, gravity_y=0.0)

# keel.dev_tools enables the profiler (F2), the inspector (F1), and —
# because Physics2D is set up — the 2D physics debug draw (F3). The call
# is idempotent and returns the same DevTools bundle for this app.
keel.dev_tools(app)


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------

def spawn_wall(x: float, y: float, w: float, h: float):
    """Spawn a static rectangle collider at (x, y) with size (w, h)."""
    return app.world.spawn(
        keel.Transform2D(x=x, y=y),
        keel.RigidBody2D(body_type=STATIC),
        keel.Collider2D(
            shape_type=1,
            width=w,
            height=h,
            friction=0.0,
            elasticity=1.0,
        ),
    )


def spawn_ball():
    """Spawn one dynamic ball at a random position + velocity + tint."""
    return app.world.spawn(
        keel.Transform2D(
            x=random.uniform(BALL_SPAWN_PADDING, WIDTH - BALL_SPAWN_PADDING),
            y=random.uniform(BALL_SPAWN_PADDING, HEIGHT - BALL_SPAWN_PADDING),
        ),
        keel.RigidBody2D(
            mass=1.0,
            body_type=DYNAMIC,
            vel_x=random.uniform(*BALL_SPEED_RANGE),
            vel_y=random.uniform(*BALL_SPEED_RANGE),
        ),
        keel.Collider2D(
            shape_type=0,
            radius=BALL_RADIUS,
            friction=0.0,
            elasticity=1.0,
        ),
        keel.Sprite(
            # texture_id=0 is the default 1×1 white texture installed by
            # setup_renderer_2d; the r/g/b tint below colours each ball.
            texture_id=0,
            width=BALL_RADIUS * 2.0,
            height=BALL_RADIUS * 2.0,
            r=random.uniform(0.4, 1.0),
            g=random.uniform(0.4, 1.0),
            b=random.uniform(0.4, 1.0),
        ),
    )


# ---------------------------------------------------------------------------
# Initial entities
# ---------------------------------------------------------------------------

# A box of walls around the visible area.
half = WALL_THICKNESS * 0.5
spawn_wall(WIDTH / 2,           half,                 WIDTH,           WALL_THICKNESS)   # floor
spawn_wall(WIDTH / 2,           HEIGHT - half,        WIDTH,           WALL_THICKNESS)   # ceiling
spawn_wall(half,                HEIGHT / 2,           WALL_THICKNESS,  HEIGHT)           # left
spawn_wall(WIDTH - half,        HEIGHT / 2,           WALL_THICKNESS,  HEIGHT)           # right

for _ in range(BALL_COUNT):
    spawn_ball()

app.world.flush()


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

# Quit when Escape is pressed.
@app.system(keel.Phase.UPDATE)
def quit_on_escape(world, dt):
    if app.input.is_key_down(keel.KEY_ESCAPE):
        app.window.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

print(f"[devtools] {BALL_COUNT} balls + 4 walls, zero gravity, perfect elasticity")
print("[devtools] F1 = inspector  F2 = profiler  F3 = debug draw  Escape = quit")
app.run()
