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
import math

import pyge
from pyge.renderer3d import Material, make_cube, make_sphere, setup_renderer_3d


@pyge.component
class SpinningCube: pass

@pyge.component
class OrbitingLamp: pass


# --- App ----------------------------------------------------------------

app = pyge.App(title="Cube Demo", width=800, height=600)
renderer = setup_renderer_3d(app)

mesh_registry = renderer.mesh_registry
material_registry = renderer.material_registry

# One mesh per primitive shape we use; one material per visual identity.
cube_mesh = mesh_registry.add(make_cube())
sphere_mesh = mesh_registry.add(make_sphere(subdivisions=2))

cube_material = material_registry.add(Material(
    albedo_r=0.85, albedo_g=0.40, albedo_b=0.30,
    roughness=0.4, metallic=0.0,
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


# Camera at (3, 3, 5) facing the origin.
cam_yaw, cam_pitch = _look_at_origin(3.0, 3.0, 5.0)
app.world.spawn(
    pyge.Camera3D(x=3.0, y=3.0, z=5.0, yaw=cam_yaw, pitch=cam_pitch,
                  fov=math.radians(60.0), near=0.1, far=100.0),
)

# The cube at the origin.
app.world.spawn(
    pyge.Transform3D(),
    pyge.MeshRenderer(mesh_id=cube_mesh, material_id=cube_material),
    SpinningCube(),
)

# Emissive sphere acting as the visible "lamp" — Transform3D drives the
# PointLight, since PointLight has no position field of its own.
app.world.spawn(
    pyge.Transform3D(scale_x=0.15, scale_y=0.15, scale_z=0.15),
    pyge.MeshRenderer(mesh_id=sphere_mesh, material_id=lamp_material),
    pyge.PointLight(r=1.0, g=0.85, b=0.4, intensity=3.0, radius=8.0),
    OrbitingLamp(),
)

# Sun-like directional light. No Transform3D — the direction is in the component.
app.world.spawn(pyge.DirectionalLight(
    dir_x=-0.4, dir_y=-0.7, dir_z=-0.5,
    r=0.95, g=0.95, b=0.85, intensity=0.6,
))

app.world.flush()


# --- Animation ----------------------------------------------------------

# Module-level clock used by the orbit system; cube spin is per-frame dt only.
_t = 0.0


@app.system(pyge.Phase.UPDATE)
def quit_on_escape(world, dt):
    if app.input.is_key_down(pyge.KEY_ESCAPE):
        app.window.close()


@app.system(pyge.Phase.UPDATE)
def spin_cubes(world, dt):
    """Rotate every SpinningCube on its X and Y axes."""
    for transforms, _ in world.query(pyge.Transform3D, SpinningCube):
        for i in range(len(transforms)):
            transforms["rot_y"][i] += dt * 0.6
            transforms["rot_x"][i] += dt * 0.3


@app.system(pyge.Phase.UPDATE)
def orbit_lamp(world, dt):
    """Move every OrbitingLamp around the Y axis at a fixed radius and height.
    Writing into Transform3D in place also moves the co-located PointLight."""
    global _t
    _t += dt
    cx = math.cos(_t) * 2.5
    cz = math.sin(_t) * 2.5
    for transforms, _ in world.query(pyge.Transform3D, OrbitingLamp):
        for i in range(len(transforms)):
            transforms["x"][i] = cx
            transforms["y"][i] = 1.5
            transforms["z"][i] = cz


print("[cube] spinning cube + orbiting lamp; Escape to quit")
app.run()
