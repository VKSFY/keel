"""hello_sprite.py — the smallest interesting Keel program.

A blue square you can drive around the window with WASD or the arrow keys.
Demonstrates: App setup, custom @keel.component, world.spawn, world.query
in-place mutation, the input poller, and Phase.UPDATE.

Run with:
    python examples/hello_sprite.py
"""
import keel
from keel.renderer import setup_renderer_2d


SPEED = 250.0  # pixels per second


@keel.component
class Hero:
    speed: float = SPEED


app = keel.App(title="Hello Sprite", width=800, height=600)
setup_renderer_2d(app)

# texture_id=0 is the engine's default 1x1 white texture, multiplied by the
# r/g/b tint — so this is a solid blue 64x64 quad. The default camera is
# centered on the framebuffer, so (400, 300) is the middle of the window.
app.world.spawn(
    keel.Transform2D(x=400.0, y=300.0),
    keel.Sprite(texture_id=0, width=64.0, height=64.0, r=0.40, g=0.65, b=1.0),
    Hero(),
)


@app.system(keel.Phase.UPDATE)
def move_hero(world, dt):
    if app.input.is_key_down(keel.KEY_ESCAPE):
        app.window.close()

    dx = (1.0 if app.input.is_key_down(keel.KEY_D) or app.input.is_key_down(keel.KEY_RIGHT) else 0.0) - \
         (1.0 if app.input.is_key_down(keel.KEY_A) or app.input.is_key_down(keel.KEY_LEFT)  else 0.0)
    dy = (1.0 if app.input.is_key_down(keel.KEY_W) or app.input.is_key_down(keel.KEY_UP)    else 0.0) - \
         (1.0 if app.input.is_key_down(keel.KEY_S) or app.input.is_key_down(keel.KEY_DOWN)  else 0.0)

    # The query yields per-archetype views into the structured-array storage,
    # so the assignments below write through to the ECS in place.
    for transforms, heroes in world.query(keel.Transform2D, Hero):
        for i in range(len(transforms)):
            transforms["x"][i] += dx * heroes["speed"][i] * dt
            transforms["y"][i] += dy * heroes["speed"][i] * dt


app.run()
