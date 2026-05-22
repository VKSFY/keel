# Keel

*The backbone of your game.*

A Python game engine with archetype ECS, built on ModernGL and GLFW.

## Why Keel

Pygame is old, single-threaded, and bound to CPU blits. Panda3D is a Python wrapper around a C++ engine, with the cognitive load that implies. Other Python options stop at hobby scope or no longer maintain a release. None of them provide a modern, data-oriented ECS as the core data model. Keel is for developers who want to stay in Python and write structured game code on top of a real archetype-based ECS. The tradeoff is honest: Python has interpreter overhead, so Keel pushes hot paths into numpy and C extensions (ModernGL, pymunk, pybullet) and exposes the rest as plain Python.

## Features

### ECS

- Archetype storage with one numpy structured array per component type per archetype.
- Struct-of-arrays layout, so iteration is a numpy slice rather than a Python loop.
- Query DSL with `world.query(A, B, Without[C], Optional[D])`.
- Command buffer for deferred structural changes (spawn, despawn, add, remove).
- Typed event bus (`world.emit(...)`, `world.read_events(EventType)`), cleared at the start of each frame.
- Resource injection into systems via parameter type annotations.

### 2D rendering

- Instanced sprite batcher: one draw call per texture group.
- Texture atlas with up to 16 texture units.
- Orthographic camera with translation, rotation, zoom.
<!-- Tilemap class exists in keel.renderer.tilemap but has no setup helper
     and is not exported from the top-level package, so it's roadmap, not
     a feature. -->


### 3D rendering

- OBJ loader (positions, normals, UVs, n-gon triangulation, missing-normal fallback).
- PBR-lite material (albedo, roughness, metallic, emissive scalars).
- Directional light plus up to 8 point lights, sorted nearest-first.
- Sphere-based frustum culling using Gribb/Hartmann plane extraction.
- Transform3D hierarchy with parent chains and cycle detection.
- Cube, plane, and UV sphere primitive generators.

### Physics

- 2D bridge to pymunk: rigid bodies, shapes (circle, box, segment), collision events, segment-query raycast.
- 3D bridge to pybullet (DIRECT mode only, never GUI): rigid bodies, sphere/box/capsule shapes, contact events, ray tests.
- Both bridges run at `Phase.POST_UPDATE`. ECS data is the source of truth on the way in; physics owns the result on the way out.

> **Collision events and body types:** `CollisionEvent2D` and `CollisionEvent3D` only fire when at least one body is **dynamic** (`body_type=0`). Two kinematic or two static bodies that overlap will not emit events — this is pymunk/Bullet behavior, not a Keel bug. Make at least one side dynamic if you need a collision to be detected. Keel prints a one-time `UserWarning` the first time a second kinematic body joins `Physics2D` to flag the trap early.

### Assets

- Handle-based `AssetRegistry` with extension-dispatched loaders.
- Built-in loaders for JSON and image formats (PNG, JPG, BMP, TGA).
- Hot reload via watchdog: file change is queued on the watchdog thread and drained on the main thread inside a PRE_UPDATE system, so GL re-uploads stay on the right thread.
- Scene save/load to JSON, atomic write (`.tmp` + `os.replace`), versioned schema.

### Tooling

- ImGui world inspector (F1).
- Per-system frame profiler overlay (F2).
- 2D physics debug draw (F3).
- CLI: `keel new`, `keel run`, `keel build`.

## Requirements

- Python 3.11 or newer.
- A GPU and driver supporting OpenGL 3.3 Core.
- Pip dependencies: `moderngl`, `pyglfw`, `PyGLM`, `numpy`, `Pillow`, `pymunk`, `pybullet`, `watchdog`, `imgui-bundle`.

## Installation

Keel is not yet on PyPI. Install from source:

```bash
git clone https://github.com/yourusername/keel
cd keel
pip install -e .
```

Once it is published, the supported install will be:

```bash
pip install keel
```

## Quickstart

The fastest way to start a project is the CLI scaffold:

```bash
keel new mygame
cd mygame
keel run
```

`keel new` creates the project tree, and `keel run` watches every `.py` file in the directory and restarts the process on save.

A minimal working example, a ball that falls under gravity and bounces on a floor, looks like this:

```python
import keel
from keel.renderer import setup_renderer_2d
from keel.physics import setup_physics_2d

app = keel.App(title="Bouncing Ball", width=800, height=600)
setup_renderer_2d(app)
setup_physics_2d(app, gravity_y=-980.0)

tools = keel.dev_tools(app)
tools.debug_draw.set_visible(True)

# Static floor.
app.world.spawn(
    keel.Transform2D(x=400.0, y=50.0),
    keel.RigidBody2D(body_type=1),  # 1 = static
    keel.Collider2D(shape_type=1, width=600.0, height=20.0, elasticity=0.6),
)

# Dynamic ball.
app.world.spawn(
    keel.Transform2D(x=400.0, y=500.0),
    keel.RigidBody2D(mass=1.0),
    keel.Collider2D(shape_type=0, radius=20.0, elasticity=0.75),
)

@app.system(keel.Phase.UPDATE)
def log_bounces(world, dt):
    for event in world.read_events(keel.CollisionEvent2D):
        if event.impulse > 100.0:
            print(f"bounce: impulse={event.impulse:.0f}")

app.run()
```

That is a complete program. Save it as `main.py` and run it with `python main.py` or `keel run`. Press F1 to open the world inspector, F2 for the profiler overlay, and F3 to toggle the debug draw of the physics shapes.

## ECS concepts

Components are plain dataclasses, decorated with `@keel.component`. Field types map to numpy dtypes when possible (`float` to `float64`, `int` to `int64`, `bool` to `bool_`). Components with non-numpy fields fall back to a Python list column. Systems are plain functions registered with `@app.system(phase)`. The first two parameters are always `(world, dt)`. Any further parameters annotated with a registered resource type are injected by the scheduler.

Queries return per-archetype numpy array views. Mutations write through to the underlying storage in place:

```python
for pos, vel in world.query(Position, Velocity):
    pos['x'] += vel['x'] * dt
    pos['y'] += vel['y'] * dt
```

That loop runs once per matching archetype, not once per entity. Each iteration is a vectorized numpy operation over the entire archetype's rows.

Structural changes are deferred. Calling `world.spawn`, `world.despawn`, `world.add_component`, or `world.remove_component` queues a command in the buffer. Nothing moves between archetypes until `world.flush()` runs, which the main loop calls at the end of every frame. This keeps query iteration stable for the entire frame: you can spawn entities from inside a system without invalidating the views you are iterating.

## Project structure

`keel new mygame` produces:

```
mygame/
├── main.py
├── pyproject.toml
├── README.md
├── assets/
│   └── .gitkeep
└── scenes/
    └── .gitkeep
```

`assets/` is monitored by the asset hot reload (textures, JSON data, anything `setup_assets` knows how to load). `scenes/` is the conventional home for `Scene.save` JSON output.

## Developer tools

### World inspector (F1)

`WorldInspector` opens an ImGui window listing every archetype and its entities. Each row expands to show component field values, sourced live from the structured arrays. There is a filter box: typing `Sprite` narrows the list to archetypes that contain a `Sprite` component. Useful for verifying that a system actually wrote what you think it wrote.

### Profiler overlay (F2)

`FrameProfiler` wraps every scheduler-invoked system in `time.perf_counter` markers. The overlay (top right) lists each system with its rolling 60-frame average in milliseconds and a unit-scaled bar. min, max, and last-sample stats are also tracked and available via `profiler.get_stats()` for programmatic use.

### Debug draw (F3)

`DebugDraw2D` walks every `Transform2D + Collider2D + RigidBody2D` entity and draws the collider outline as GL line segments: 32-segment circles, 4-segment rectangles, single segments. Lines are grouped by color (green for dynamic, gray for static, blue for kinematic, yellow for sensor) so the whole overlay is one draw call per color. The shader is a 2-uniform line program (`u_camera`, `u_color`).

### Enable everything

```python
tools = keel.dev_tools(app)
```

That call sets up the profiler, the inspector, and (if `setup_physics_2d` has already been called on this app) the debug draw. F1, F2, and F3 use edge-detected polling via `app.input.is_key_down`, checked once per sim tick. `KeyEvent`s are not used here because input events can be dropped on visual frames where no sim tick fires.

## Architecture overview

`App` owns one `World` (the ECS), one `Scheduler` (phase-ordered system runner), one `Window` (GLFW + ModernGL context), and one `InputState`. The fixed-timestep loop drives the scheduler at 60 Hz simulation, render-once-per-visual-frame. Renderers are plain systems registered at `Phase.RENDER`: they read components through `world.query` and issue draw calls. Physics bridges run at `Phase.POST_UPDATE`, sync ECS state into the engine, step, and write results back into Transform components. Assets and scenes go through their own resources but never own simulation state. Every layer talks to the next through public ECS APIs only, so adding or replacing a layer does not require changes to the others.

## Roadmap

- [ ] Tilemap (`keel.renderer.tilemap.Tilemap` exists but has no `setup_tilemap` helper or top-level export yet)
- [ ] Text rendering
- [ ] Skeletal animation
- [ ] Parallel system execution
- [ ] WASM export via Pyodide
- [ ] Audio
- [ ] Visual scene editor

## License

MIT.

## Contributing

Pull requests are welcome. Run `pytest` before submitting; the suite is fast and covers every phase. There is no formal contribution guide yet, so open an issue if you want to discuss a larger change before writing it.
