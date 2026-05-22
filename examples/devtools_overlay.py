"""devtools_overlay.py — F1 inspector, F2 profiler, F3 physics debug draw.

Shows the dev tooling running on top of a non-trivial ECS workload. Drops
50 colored balls into an 800x600 box, lets pymunk simulate, and turns on
every overlay through one call to keel.dev_tools(app).

Run: python examples/devtools_overlay.py

Controls:
    F1     — toggle World Inspector window (entity / component browser)
    F2     — toggle Profiler overlay (per-system avg ms with bars)
    F3     — toggle physics debug draw (collider outlines, color-coded by body type)
    Escape — quit
"""
import random

import keel
from keel.physics import setup_physics_2d
from keel.renderer import setup_renderer_2d


BALL_COUNT = 50
BALL_RADIUS = 12.0
WALL_THICKNESS = 10.0


# --- Setup --------------------------------------------------------------

app = keel.App(title="DevTools Overlay", width=800, height=600)
setup_renderer_2d(app)
# Zero gravity + zero friction + perfect elasticity = chaotic perpetual motion,
# which is what we want for a tooling demo (the profiler should always have
# something to measure, the debug draw should always show shifting outlines).
setup_physics_2d(app, gravity_y=0.0)

# keel.dev_tools enables: profiler (F2), inspector (F1), and — because
# Physics2D is set up — the 2D physics debug draw (F3). Idempotent and
# always returns the same DevTools bundle for this app.
keel.dev_tools(app)


# --- Entities ----------------------------------------------------------

def spawn_wall(x: float, y: float, w: float, h: float):
    return app.world.spawn(
        keel.Transform2D(x=x, y=y),
        keel.RigidBody2D(body_type=1),  # static
        keel.Collider2D(shape_type=1, width=w, height=h,
                        friction=0.0, elasticity=1.0),
    )


# Box of walls around the visible area.
spawn_wall(400.0, WALL_THICKNESS * 0.5, 800.0, WALL_THICKNESS)            # floor
spawn_wall(400.0, 600.0 - WALL_THICKNESS * 0.5, 800.0, WALL_THICKNESS)    # ceiling
spawn_wall(WALL_THICKNESS * 0.5, 300.0, WALL_THICKNESS, 600.0)            # left
spawn_wall(800.0 - WALL_THICKNESS * 0.5, 300.0, WALL_THICKNESS, 600.0)    # right


def spawn_ball():
    return app.world.spawn(
        keel.Transform2D(
            x=random.uniform(40.0, 760.0),
            y=random.uniform(40.0, 560.0),
        ),
        keel.RigidBody2D(
            mass=1.0,
            body_type=0,  # dynamic
            vel_x=random.uniform(-300.0, 300.0),
            vel_y=random.uniform(-300.0, 300.0),
        ),
        keel.Collider2D(
            shape_type=0,
            radius=BALL_RADIUS,
            friction=0.0,
            elasticity=1.0,
        ),
        keel.Sprite(
            texture_id=0,  # default white from setup_renderer_2d, tinted below
            width=BALL_RADIUS * 2.0,
            height=BALL_RADIUS * 2.0,
            r=random.uniform(0.4, 1.0),
            g=random.uniform(0.4, 1.0),
            b=random.uniform(0.4, 1.0),
        ),
    )


for _ in range(BALL_COUNT):
    spawn_ball()
app.world.flush()


# --- Quit hotkey -------------------------------------------------------

@app.system(keel.Phase.UPDATE)
def quit_on_escape(world, dt):
    if app.input.is_key_down(keel.KEY_ESCAPE):
        app.window.close()


print(f"[devtools] {BALL_COUNT} balls + 4 walls, zero gravity, perfect elasticity")
print("[devtools] F1 = inspector  F2 = profiler  F3 = debug draw  Escape = quit")
app.run()
