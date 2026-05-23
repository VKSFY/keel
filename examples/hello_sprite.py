"""hello_sprite.py — the smallest interesting Keel program.

A blue square you can drive around the window with WASD or the arrow keys.
Demonstrates: App setup, custom @keel.component, world.spawn, world.query
in-place mutation, the input poller, and Phase.UPDATE.

Run with:
    python examples/hello_sprite.py
"""

import keel
from keel.renderer import setup_renderer_2d


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIDTH = 800
HEIGHT = 600

# Hero motion (pixels per second). Stored on the Hero component below so a
# system can read it per-entity through a numpy column view.
SPEED = 250.0

# Default camera is centered on the framebuffer, so this is the middle of
# the window — handy as a spawn point.
SCREEN_CX = WIDTH / 2.0
SCREEN_CY = HEIGHT / 2.0


# ---------------------------------------------------------------------------
# A custom component
# ---------------------------------------------------------------------------
#
# Every @keel.component is a Python dataclass under the hood. Keel inspects
# the field types (float / int / bool) and stores them as a numpy structured
# array column per archetype, so queries iterate components in tight loops.

@keel.component
class Hero:
    speed: float = SPEED


# ---------------------------------------------------------------------------
# App + renderer setup
# ---------------------------------------------------------------------------

app = keel.App(title="Hello Sprite", width=WIDTH, height=HEIGHT)
setup_renderer_2d(app)


# ---------------------------------------------------------------------------
# Initial entities
# ---------------------------------------------------------------------------

# texture_id=0 is the engine's default 1x1 white texture; the renderer
# multiplies the sprite's RGB tint into it, so this is a solid blue 64x64
# quad. world.spawn returns the entity id (we ignore it here).
app.world.spawn(
    keel.Transform2D(x=SCREEN_CX, y=SCREEN_CY),
    keel.Sprite(texture_id=0, width=64.0, height=64.0, r=0.40, g=0.65, b=1.0),
    Hero(),
)


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

# UPDATE-phase system: read the keyboard and slide every Hero entity.
@app.system(keel.Phase.UPDATE)
def move_hero(world, dt):
    if app.input.is_key_down(keel.KEY_ESCAPE):
        app.window.close()

    # Compose a unit-ish direction vector from the WASD + arrow keys.
    right = app.input.is_key_down(keel.KEY_D) or app.input.is_key_down(keel.KEY_RIGHT)
    left  = app.input.is_key_down(keel.KEY_A) or app.input.is_key_down(keel.KEY_LEFT)
    up    = app.input.is_key_down(keel.KEY_W) or app.input.is_key_down(keel.KEY_UP)
    down  = app.input.is_key_down(keel.KEY_S) or app.input.is_key_down(keel.KEY_DOWN)

    dx = (1.0 if right else 0.0) - (1.0 if left else 0.0)
    dy = (1.0 if up    else 0.0) - (1.0 if down else 0.0)

    # world.query yields per-archetype column VIEWS, not copies, so the
    # in-place assignments below write straight through to the ECS storage.
    for transforms, heroes in world.query(keel.Transform2D, Hero):
        for i in range(len(transforms)):
            transforms["x"][i] += dx * heroes["speed"][i] * dt
            transforms["y"][i] += dy * heroes["speed"][i] * dt


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app.run()
