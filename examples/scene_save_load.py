"""scene_save_load.py — round-trip world state through atomic JSON.

Demonstrates Scene.save / Scene.load: every @pyge.component (built-in or
custom) on every live entity is serialized to disk and read back. The example
spawns 30 colored "particles" that bounce off the window edges, with the
per-particle velocity stored on a custom Bouncer component so it survives
the round-trip.

Press F5 to snapshot, mess things up or wait, then F9 to wipe the world and
reload exactly what you saved.

Run: python examples/scene_save_load.py

Controls:
    F5     — save world to saves/scene.json
    F9     — clear world, then load from saves/scene.json
    F10    — clear world
    Escape — quit
"""
import random
from pathlib import Path

import pyge
from pyge.renderer import setup_renderer_2d


SAVE_PATH = Path(__file__).resolve().parent / "saves" / "scene.json"
SPAWN_COUNT = 30
PARTICLE_SIZE = 24.0


@pyge.component
class Bouncer:
    """Per-particle velocity. Numpy-backed floats round-trip as JSON numbers."""
    vx: float = 0.0
    vy: float = 0.0


# --- App ----------------------------------------------------------------

app = pyge.App(title="Scene Save / Load", width=800, height=600)
setup_renderer_2d(app)


def spawn_particle():
    return app.world.spawn(
        pyge.Transform2D(
            x=random.uniform(40.0, 760.0),
            y=random.uniform(40.0, 560.0),
        ),
        pyge.Sprite(
            texture_id=0,  # default white from setup_renderer_2d, tinted by r/g/b
            width=PARTICLE_SIZE,
            height=PARTICLE_SIZE,
            r=random.uniform(0.3, 1.0),
            g=random.uniform(0.3, 1.0),
            b=random.uniform(0.3, 1.0),
        ),
        Bouncer(
            vx=random.uniform(-160.0, 160.0),
            vy=random.uniform(-160.0, 160.0),
        ),
    )


for _ in range(SPAWN_COUNT):
    spawn_particle()
app.world.flush()


def entity_count() -> int:
    return sum(arch.length for arch in app.world.archetypes.all_archetypes())


def clear_world() -> None:
    """Despawn every live entity, then flush."""
    for arch in app.world.archetypes.all_archetypes():
        for eid in list(arch.entities[: arch.length]):
            app.world.despawn(eid)
    app.world.flush()


# --- Edge-detected hotkeys ---------------------------------------------

_was_down: set[int] = set()


def _edge_pressed(key: int) -> bool:
    is_down = app.input.is_key_down(key)
    was = key in _was_down
    if is_down:
        _was_down.add(key)
    else:
        _was_down.discard(key)
    return is_down and not was


# --- Systems ------------------------------------------------------------

@app.system(pyge.Phase.UPDATE)
def hotkey_input(world, dt):
    if app.input.is_key_down(pyge.KEY_ESCAPE):
        app.window.close()
        return

    if _edge_pressed(pyge.KEY_F5):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pyge.Scene.save(world, str(SAVE_PATH))
        print(f"[scene] saved {entity_count()} entities → {SAVE_PATH}")

    if _edge_pressed(pyge.KEY_F9):
        if not SAVE_PATH.exists():
            print(f"[scene] nothing to load: {SAVE_PATH} doesn't exist")
        else:
            clear_world()
            ids = pyge.Scene.load(world, str(SAVE_PATH))
            print(f"[scene] loaded {len(ids)} entities from {SAVE_PATH}")

    if _edge_pressed(pyge.KEY_F10):
        clear_world()
        print(f"[scene] cleared (entities now: {entity_count()})")


@app.system(pyge.Phase.UPDATE)
def bounce(world, dt):
    """Move every Bouncer by its velocity; reflect off the 800x600 window edges."""
    half = PARTICLE_SIZE * 0.5
    for transforms, bouncers in world.query(pyge.Transform2D, Bouncer):
        for i in range(len(transforms)):
            x = transforms["x"][i] + bouncers["vx"][i] * dt
            y = transforms["y"][i] + bouncers["vy"][i] * dt
            if x < half:
                x = half
                bouncers["vx"][i] = abs(bouncers["vx"][i])
            elif x > 800.0 - half:
                x = 800.0 - half
                bouncers["vx"][i] = -abs(bouncers["vx"][i])
            if y < half:
                y = half
                bouncers["vy"][i] = abs(bouncers["vy"][i])
            elif y > 600.0 - half:
                y = 600.0 - half
                bouncers["vy"][i] = -abs(bouncers["vy"][i])
            transforms["x"][i] = x
            transforms["y"][i] = y


print("[scene] F5=save  F9=clear+load  F10=clear  Escape=quit")
app.run()
