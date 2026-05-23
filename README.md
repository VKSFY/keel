# Keel

*The backbone of your game.*

A Python game engine with archetype ECS, built on ModernGL and GLFW.

## Why Keel

Pygame is old, single-threaded, and bound to CPU blits. Panda3D is a Python wrapper around a C++ engine, with the cognitive load that implies. Other Python options stop at hobby scope or no longer maintain a release. None of them provide a modern, data-oriented ECS as the core data model. Keel is for developers who want to stay in Python and write structured game code on top of a real archetype-based ECS. The tradeoff is honest: Python has interpreter overhead, so Keel pushes hot paths into numpy and C extensions (ModernGL, pymunk, pybullet) and exposes the rest as plain Python.

## Installation

Keel is on PyPI. The distribution name is `keelpy`, the import name is `keel`:

```bash
pip install keelpy
```

For 3D physics (requires pybullet, which only has wheels on Windows and a few older Python combinations until cross-platform wheels exist on PyPI):

```bash
pip install keelpy[physics3d]
```

For developer tooling (ImGui inspector, profiler, debug draw):

```bash
pip install keelpy[tools]
```

For everything:

```bash
pip install "keelpy[physics3d,tools]"
```

To install from source:

```bash
git clone https://github.com/VKSFY/keel
cd keel
pip install -e .
```

### Requirements

Python 3.11 or newer. A GPU and driver supporting OpenGL 3.3 Core. The base install pulls in `numpy`, `moderngl`, `glfw`, `Pillow`, `watchdog`, `pymunk`, `freetype-py`, and `miniaudio` transitively from PyPI. `pybullet` is optional (Windows-only until cross-platform wheels exist); install it with the `[physics3d]` extra. `imgui-bundle` is also optional and ships with the `[tools]` extra.

## Quickstart

### Minimal example

A window with a sprite you can move around the screen with WASD. `texture_id=0` is always a white 1x1 pixel, so no asset file is needed to get started.

```python
import keel
from keel.renderer import setup_renderer_2d

app = keel.App(title="Hello Keel", width=800, height=600)
setup_renderer_2d(app)

@keel.component
class Player:
    speed: float = 200.0

player = app.world.spawn(
    keel.Transform2D(x=400.0, y=300.0),
    keel.Sprite(texture_id=0, width=32.0, height=32.0),
    Player(),
)

@app.system(keel.Phase.UPDATE)
def move(world, dt):
    for transform, player in world.query(keel.Transform2D, Player):
        if app.input.is_key_down(keel.KEY_D):
            transform['x'] += player['speed'] * dt
        if app.input.is_key_down(keel.KEY_A):
            transform['x'] -= player['speed'] * dt
        if app.input.is_key_down(keel.KEY_W):
            transform['y'] += player['speed'] * dt
        if app.input.is_key_down(keel.KEY_S):
            transform['y'] -= player['speed'] * dt

app.run()
```

### Physics example

A ball that falls under gravity and bounces on a static floor.

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
    keel.RigidBody2D(body_type=keel.BodyType.STATIC),
    keel.Collider2D(
        shape_type=keel.ShapeType2D.BOX,
        width=600.0,
        height=20.0,
        elasticity=0.6,
    ),
)

# Dynamic ball.
app.world.spawn(
    keel.Transform2D(x=400.0, y=500.0),
    keel.RigidBody2D(mass=1.0, body_type=keel.BodyType.DYNAMIC),
    keel.Collider2D(
        shape_type=keel.ShapeType2D.CIRCLE,
        radius=20.0,
        elasticity=0.75,
    ),
)

@app.system(keel.Phase.UPDATE)
def log_bounces(world, dt):
    for event in world.read_events(keel.CollisionEvent2D):
        if event.impulse > 100.0:
            print(f"bounce: impulse={event.impulse:.0f}")

app.run()
```

That is a complete program. Save it as `main.py` and run it with `python main.py`. Press F1 to open the world inspector, F2 for the profiler overlay, and F3 to toggle the debug draw of the physics shapes.

### Project scaffold

The CLI sets up the file tree and a hot-reloading dev loop:

```bash
keel new mygame
cd mygame
keel run
```

`keel run` watches every `.py` file in the directory and restarts the process on save.

## ECS concepts

Components are plain dataclasses, decorated with `@keel.component`. Field types map to numpy dtypes when possible (`float` to `float64`, `int` to `int64`, `bool` to `bool_`). Components with non-numpy fields fall back to a Python list column. Systems are plain functions registered with `@app.system(phase)`. The first two parameters are always `(world, dt)`. Any further parameters annotated with a registered resource type are injected by the scheduler.

Queries return per-archetype numpy array views. Mutations write through to the underlying storage in place:

```python
for pos, vel in world.query(Position, Velocity):
    pos['x'] += vel['x'] * dt
    pos['y'] += vel['y'] * dt
```

That loop runs once per matching archetype, not once per entity. Each iteration is a vectorized numpy operation over the entire archetype's rows.

### Per-entity reads and writes

For one-off entity access (UI labels, paddles, a specific bullet), use `world.get` and `world.set`:

```python
# Read a specific entity's component fields as a plain dict.
pos = world.get(player, keel.Transform2D)
if pos:
    print(pos['x'], pos['y'])

# Write fields in place. Returns False if the entity lacks the component.
world.set(player, keel.Transform2D, x=100.0, y=200.0)
world.set(label, keel.TextLabel, visible=False)
```

### Deferred structural changes

Structural changes are deferred. Calling `world.spawn`, `world.despawn`, `world.add_component`, or `world.remove_component` queues a command in the buffer. Nothing moves between archetypes until `world.flush()` runs, which the main loop calls at the end of every frame. This keeps query iteration stable for the entire frame: you can spawn entities from inside a system without invalidating the views you are iterating.

### Events

The event bus is typed. Emit one type, read the same type:

```python
@keel.event
class ScoredEvent:
    team: int
    points: int

world.emit(ScoredEvent(team=1, points=3))

for ev in world.read_events(ScoredEvent):
    handle(ev)
```

Each frame starts with all event queues cleared.

## Input

`app.input` exposes both held-state and edge-detected helpers. Edge detection is updated once per simulation tick, so `is_key_pressed` is True for exactly one frame after the key transitions.

```python
# Held down every frame.
app.input.is_key_down(keel.KEY_W)

# Only True on the first frame the key is pressed.
app.input.is_key_pressed(keel.KEY_SPACE)

# Only True on the frame the key is released.
app.input.is_key_released(keel.KEY_SPACE)

# Mouse.
app.input.is_mouse_button_pressed(keel.MOUSE_BUTTON_LEFT)
app.input.mouse_position()  # returns (x, y) tuple
```

For one-shot reactions, read events from the world bus:

```python
for ev in world.read_events(keel.KeyEvent):
    if ev.key == keel.KEY_ESCAPE and ev.action == keel.PRESS:
        app.window.close()
```

## Physics enums

Body type and shape type are `IntEnum`, so the named form and the raw integer form are interchangeable. Use the enums for readability.

```python
# Body types.
keel.BodyType.DYNAMIC    # affected by physics, generates collision events
keel.BodyType.STATIC     # immovable, generates events with dynamic bodies
keel.BodyType.KINEMATIC  # moved manually, no events with other kinematics

# 2D shape types.
keel.ShapeType2D.CIRCLE
keel.ShapeType2D.BOX
keel.ShapeType2D.SEGMENT

# 3D shape types.
keel.ShapeType3D.SPHERE
keel.ShapeType3D.BOX
keel.ShapeType3D.CAPSULE
keel.ShapeType3D.MESH
```

Raw integers still work for backwards compatibility (`keel.BodyType.STATIC == 1` evaluates to `True`).

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
- Tilemap with chunked baking, accessible through `setup_tilemap(app, tile_data)`.

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

`CollisionEvent2D` and `CollisionEvent3D` only fire when at least one body is dynamic. Two kinematic or two static bodies that overlap will not emit events. This is pymunk/Bullet behavior, not a Keel bug. Make at least one side dynamic if you need the collision to be detected. Keel prints a one-time `UserWarning` the first time a second kinematic body joins `Physics2D` to flag the trap early.

### Text rendering

- `TextLabel` component with `font_id`, color (`r`, `g`, `b`, `a`), `scale`, `visible`.
- Font loading through freetype-py, with a bundled DejaVu Sans Mono fallback at `keel.BUILTIN_FONT`.
- Glyph atlas baked once at font load time (R8 texture, shelf packer).
- Screen-space rendering. `y=0` is the top of the screen and the y axis grows downward.
- Newline and tab support during layout.
- Side-table API for text content: `set_text(entity_id, str)`, `get_text(entity_id)`, `clear_text(entity_id)`.

```python
import keel
from keel.text import setup_text, load_font, set_text, BUILTIN_FONT

text = setup_text(app)
font = load_font(app, BUILTIN_FONT, size_px=28)

score_label = app.world.spawn(
    keel.Transform2D(x=10.0, y=35.0),
    keel.TextLabel(font_id=0, r=1.0, g=1.0, b=1.0),
)
set_text(score_label, "Score: 0")

# Later, inside a system:
set_text(score_label, f"Score: {score}")
```

Text renders in screen space, not world space. The position is the text baseline.

### Audio

- `AudioEngine` backed by miniaudio for playback.
- `play_sound(app, path)` for one-shot effects, returns a `SoundHandle`.
- `play_music(app, path, loop=True, fade_in=0.0)` for streaming music tracks.
- `set_volume(app, master=..., sfx=..., music=...)` for per-channel volume control.
- Fade in and fade out support on music transitions.
- `AudioSource` component for entity-bound sounds.

```python
import keel
from keel.audio import setup_audio, play_sound, play_music

audio = setup_audio(app)
play_music(app, "assets/music/theme.ogg", loop=True, fade_in=2.0)

@app.system(keel.Phase.UPDATE)
def handle_jump(world, dt):
    for event in world.read_events(keel.KeyEvent):
        if event.key == keel.KEY_SPACE and event.action == keel.PRESS:
            play_sound(app, "assets/sounds/jump.wav", volume=0.8)
```

### Gamepad

- `GamepadState` with polling and event-based input for up to 4 gamepads.
- `is_connected(gamepad_id)`, `is_button_down(gamepad_id, button)`, `get_axis(gamepad_id, axis)`.
- `GamepadButtonEvent` and `GamepadAxisEvent`, with a 0.05 deadzone applied before axis events are emitted.
- Named constants under `keel.GAMEPAD_BUTTON_A`, `keel.GAMEPAD_AXIS_LEFT_X`, and the rest of the GLFW gamepad mapping.
- Disconnect handling: held buttons emit a synthesized `RELEASE` event when the pad drops.

```python
import keel
from keel.gamepad import setup_gamepad

gamepad = setup_gamepad(app)

@app.system(keel.Phase.UPDATE)
def move(world, dt):
    if gamepad.is_connected(0):
        x = gamepad.get_axis(0, keel.GAMEPAD_AXIS_LEFT_X)
        y = gamepad.get_axis(0, keel.GAMEPAD_AXIS_LEFT_Y)
```

### Assets

- Handle-based `AssetRegistry` with extension-dispatched loaders.
- Built-in loaders for JSON and image formats (PNG, JPG, BMP, TGA).
- Hot reload via watchdog: file change is queued on the watchdog thread and drained on the main thread inside a `PRE_UPDATE` system, so GL re-uploads stay on the right thread.
- Scene save/load to JSON, atomic write (`.tmp` + `os.replace`), versioned schema.

### Tooling

- ImGui world inspector (F1).
- Per-system frame profiler overlay (F2).
- 2D physics debug draw (F3).
- CLI: `keel new`, `keel run`, `keel build`.

## Developer tools

### World inspector (F1)

`WorldInspector` opens an ImGui window listing every archetype and its entities. Each row expands to show component field values, sourced live from the structured arrays. The filter box accepts a component name (typing `Sprite` narrows the list to archetypes that contain a `Sprite` component). Useful for verifying that a system actually wrote what you think it wrote.

### Profiler overlay (F2)

`FrameProfiler` wraps every scheduler-invoked system in `time.perf_counter` markers. The overlay (top right) lists each system with its rolling 60-frame average in milliseconds and a unit-scaled bar. Min, max, and last-sample stats are also tracked and available via `profiler.get_stats()` for programmatic use.

### Debug draw (F3)

`DebugDraw2D` walks every `Transform2D + Collider2D + RigidBody2D` entity and draws the collider outline as GL line segments: 32-segment circles, 4-segment rectangles, single segments. Lines are grouped by color (green for dynamic, gray for static, blue for kinematic, yellow for sensor) so the whole overlay is one draw call per color. The shader is a 2-uniform line program (`u_camera`, `u_color`).

### Enable everything

```python
tools = keel.dev_tools(app)
```

That call sets up the profiler, the inspector, and (if `setup_physics_2d` has already been called on this app) the debug draw. F1, F2, and F3 use edge-detected polling via `app.input.is_key_down`, checked once per sim tick. `KeyEvent`s are not used here because input events can be dropped on visual frames where no sim tick fires.

## Architecture overview

`App` owns one `World` (the ECS), one `Scheduler` (phase-ordered system runner), one `Window` (GLFW + ModernGL context), and one `InputState`. The fixed-timestep loop drives the scheduler at 60 Hz simulation, render-once-per-visual-frame. Renderers are plain systems registered at `Phase.RENDER`: they read components through `world.query` and issue draw calls. Physics bridges run at `Phase.POST_UPDATE`, sync ECS state into the engine, step, and write results back into Transform components. Assets and scenes go through their own resources but never own simulation state. Every layer talks to the next through public ECS APIs only, so adding or replacing a layer does not require changes to the others.

## Project structure

`keel new mygame` produces:

```text
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

## Roadmap

- [ ] Skeletal animation
- [ ] Parallel system execution
- [ ] WASM export via Pyodide
- [ ] Visual scene editor
- [ ] World-space text (text that moves with the camera)
- [ ] 3D audio / positional audio
- [ ] PhysicsMaterial system

## License

MIT.

## Contributing

Pull requests are welcome. Run `pytest` before submitting; the suite is fast and covers every phase. There is no formal contribution guide yet, so open an issue if you want to discuss a larger change before writing it.
