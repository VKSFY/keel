"""cube_demo.py — a lit, spinning cube under a sun and an orbiting point light.

Demonstrates the 3D pipeline end-to-end: setup_renderer_3d, mesh primitives,
MaterialRegistry, Transform3D + MeshRenderer, Camera3D, DirectionalLight,
and a PointLight whose position is driven by its co-located Transform3D.

The cube spins on two axes; an emissive sphere ("lamp") orbits it at a fixed
radius and height, and the point light pinned to that sphere's transform
illuminates the cube as it moves. A weak directional light keeps the dark
side visible.

Run with:
    python examples/cube_demo.py

Controls:
    Escape — quit
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import math

import keel
from keel.renderer3d import Material, make_cube, make_sphere, setup_renderer_3d


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIDTH = 800
HEIGHT = 600

# Camera placement (world units). The camera points at the origin.
CAMERA_X = 3.0
CAMERA_Y = 3.0
CAMERA_Z = 5.0
CAMERA_FOV_DEG = 60.0
CAMERA_NEAR = 0.1
CAMERA_FAR = 100.0

# Spin rates for the cube (radians / second).
CUBE_SPIN_YAW = 0.6
CUBE_SPIN_PITCH = 0.3

# Lamp orbit — horizontal radius, fixed height, scale of the lamp sphere.
LAMP_ORBIT_RADIUS = 2.5
LAMP_HEIGHT = 1.5
LAMP_SCALE = 0.15
LAMP_LIGHT_INTENSITY = 3.0
LAMP_LIGHT_RADIUS = 8.0

# A weak sun to keep the cube's dark side visible.
SUN_DIR = (-0.4, -0.7, -0.5)
SUN_INTENSITY = 0.6


# ---------------------------------------------------------------------------
# Custom components (just markers used by the animation systems)
# ---------------------------------------------------------------------------

@keel.component
class SpinningCube:
    pass


@keel.component
class OrbitingLamp:
    pass


# ---------------------------------------------------------------------------
# App + renderer setup
# ---------------------------------------------------------------------------

app = keel.App(title="Cube Demo", width=WIDTH, height=HEIGHT)
renderer = setup_renderer_3d(app)

mesh_registry = renderer.mesh_registry
material_registry = renderer.material_registry

# One Mesh per primitive shape we'll draw (the registry uploads vertex /
# index buffers to the GPU and hands back an integer id).
cube_mesh = mesh_registry.add(make_cube())
sphere_mesh = mesh_registry.add(make_sphere(subdivisions=2))

# One Material per visual identity. The cube is matte red-orange; the lamp
# is a strongly emissive yellow so it visibly glows.
cube_material = material_registry.add(Material(
    albedo_r=0.85, albedo_g=0.40, albedo_b=0.30,
    roughness=0.4,
    metallic=0.0,
))
lamp_material = material_registry.add(Material(
    albedo_r=1.0, albedo_g=0.9, albedo_b=0.5,
    roughness=0.2,
    emissive_r=1.0, emissive_g=0.85, emissive_b=0.4,
))


def _look_at_origin(x: float, y: float, z: float) -> tuple[float, float]:
    """Return (yaw, pitch) for a Camera3D at (x, y, z) that aims at the origin."""
    yaw = math.atan2(x, z)
    pitch = -math.atan2(y, math.sqrt(x * x + z * z))
    return yaw, pitch


# ---------------------------------------------------------------------------
# Initial entities
# ---------------------------------------------------------------------------

# Camera, aimed at the origin from CAMERA_X/Y/Z.
cam_yaw, cam_pitch = _look_at_origin(CAMERA_X, CAMERA_Y, CAMERA_Z)
app.world.spawn(
    keel.Camera3D(
        x=CAMERA_X,
        y=CAMERA_Y,
        z=CAMERA_Z,
        yaw=cam_yaw,
        pitch=cam_pitch,
        fov=math.radians(CAMERA_FOV_DEG),
        near=CAMERA_NEAR,
        far=CAMERA_FAR,
    ),
)

# The cube at the origin. Transform3D has identity defaults so we don't
# need to set x/y/z explicitly.
app.world.spawn(
    keel.Transform3D(),
    keel.MeshRenderer(mesh_id=cube_mesh, material_id=cube_material),
    SpinningCube(),
)

# Emissive sphere acting as the visible "lamp". The PointLight has no
# position of its own — it reads the entity's Transform3D, so moving the
# transform also moves the light.
app.world.spawn(
    keel.Transform3D(scale_x=LAMP_SCALE, scale_y=LAMP_SCALE, scale_z=LAMP_SCALE),
    keel.MeshRenderer(mesh_id=sphere_mesh, material_id=lamp_material),
    keel.PointLight(
        r=1.0, g=0.85, b=0.4,
        intensity=LAMP_LIGHT_INTENSITY,
        radius=LAMP_LIGHT_RADIUS,
    ),
    OrbitingLamp(),
)

# Sun-like directional light. No Transform3D — direction is in the component.
app.world.spawn(keel.DirectionalLight(
    dir_x=SUN_DIR[0], dir_y=SUN_DIR[1], dir_z=SUN_DIR[2],
    r=0.95, g=0.95, b=0.85,
    intensity=SUN_INTENSITY,
))

app.world.flush()


# ---------------------------------------------------------------------------
# Animation state + systems
# ---------------------------------------------------------------------------

# Module-level clock used by orbit_lamp; the cube's spin only needs `dt`.
_t = 0.0


# Quit when Escape is pressed.
@app.system(keel.Phase.UPDATE)
def quit_on_escape(world, dt):
    if app.input.is_key_down(keel.KEY_ESCAPE):
        app.window.close()


# Rotate every SpinningCube on its X and Y axes.
@app.system(keel.Phase.UPDATE)
def spin_cubes(world, dt):
    for transforms, _ in world.query(keel.Transform3D, SpinningCube):
        for i in range(len(transforms)):
            transforms["rot_y"][i] += dt * CUBE_SPIN_YAW
            transforms["rot_x"][i] += dt * CUBE_SPIN_PITCH


# Move every OrbitingLamp around the Y axis at LAMP_ORBIT_RADIUS / LAMP_HEIGHT.
# Writing into Transform3D in place also moves the co-located PointLight.
@app.system(keel.Phase.UPDATE)
def orbit_lamp(world, dt):
    global _t
    _t += dt
    cx = math.cos(_t) * LAMP_ORBIT_RADIUS
    cz = math.sin(_t) * LAMP_ORBIT_RADIUS

    for transforms, _ in world.query(keel.Transform3D, OrbitingLamp):
        for i in range(len(transforms)):
            transforms["x"][i] = cx
            transforms["y"][i] = LAMP_HEIGHT
            transforms["z"][i] = cz


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

print("[cube] spinning cube + orbiting lamp; Escape to quit")
app.run()
