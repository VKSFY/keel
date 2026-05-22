"""3D renderer for PyGE: Renderer3D + setup_renderer_3d.

Coexistence with the 2D renderer
--------------------------------
If both renderers are active, the 2D system's clear is the only clear per
frame — Renderer3D detects the presence of SpriteBatch2D as a world resource
and skips its own clear. Depth test is enabled while meshes draw and disabled
on exit, so the 2D overlay (or a future 2D system) sees no depth state from us.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import moderngl
import numpy as np

from ..components.mesh_renderer import MeshRenderer
from ..components.transform3d import Transform3D
from ..core import Phase
from .camera3d import (
    Camera3D,
    build_projection_matrix,
    build_view_matrix,
    _read_camera,
)
from .frustum import BoundingSphere, FrustumCuller
from .lighting import (
    MAX_POINT_LIGHTS,
    AmbientLight,
    DirectionalLight,
    PointLight,
)
from .material import Material, MaterialRegistry
from .mesh import (
    Mesh,
    MeshBuffer,
    MeshRegistry,
    OBJLoader,
    make_cube,
    make_plane,
    make_sphere,
)
from .shader3d import PBR_LITE_FRAG_SRC, PBR_LITE_VERT_SRC, ShaderCache3D
from .transform3d import build_model_matrix, resolve_world_matrix


__all__ = [
    "AmbientLight",
    "BoundingSphere",
    "Camera3D",
    "DirectionalLight",
    "FrustumCuller",
    "MAX_POINT_LIGHTS",
    "Material",
    "MaterialRegistry",
    "Mesh",
    "MeshBuffer",
    "MeshRegistry",
    "MeshRenderer",
    "OBJLoader",
    "PBR_LITE_FRAG_SRC",
    "PBR_LITE_VERT_SRC",
    "PointLight",
    "Renderer3D",
    "Renderer3DSetup",
    "ShaderCache3D",
    "Transform3D",
    "build_model_matrix",
    "build_projection_matrix",
    "build_view_matrix",
    "make_cube",
    "make_plane",
    "make_sphere",
    "resolve_world_matrix",
    "setup_renderer_3d",
]


@dataclass
class Renderer3DSetup:
    """What `setup_renderer_3d` returns: registries already wired into the app."""
    mesh_registry: MeshRegistry
    material_registry: MaterialRegistry
    renderer3d: "Renderer3D"
    shader_cache: ShaderCache3D
    render_system: Callable


_DEFAULT_AMBIENT: tuple[float, float, float] = (0.1, 0.1, 0.1)
_BACKGROUND_COLOR: tuple[float, float, float, float] = (0.05, 0.05, 0.08, 1.0)
_DEFAULT_BOUND_RADIUS: float = 2.0


class Renderer3D:
    """Per-frame 3D pass: build VP, cull, set lights, draw every visible MeshRenderer."""

    def __init__(
        self,
        ctx: moderngl.Context,
        mesh_registry: MeshRegistry,
        material_registry: MaterialRegistry,
        shader: moderngl.Program,
    ) -> None:
        self.ctx: moderngl.Context = ctx
        self.mesh_registry: MeshRegistry = mesh_registry
        self.material_registry: MaterialRegistry = material_registry
        self.shader: moderngl.Program = shader
        self.frustum_culler: FrustumCuller = FrustumCuller()
        self.last_draw_calls: int = 0
        self.last_culled: int = 0
        self.last_point_lights: int = 0

        # Pre-allocated upload buffers for the per-frame light arrays.
        self._point_pos = np.zeros((MAX_POINT_LIGHTS, 3), dtype=np.float32)
        self._point_color = np.zeros((MAX_POINT_LIGHTS, 3), dtype=np.float32)
        self._point_intensity = np.zeros(MAX_POINT_LIGHTS, dtype=np.float32)
        self._point_radius = np.zeros(MAX_POINT_LIGHTS, dtype=np.float32)

    # --- helpers ---------------------------------------------------------

    def _try_set(self, name: str, value: Any) -> None:
        """Set a uniform that may have been optimized out of the program."""
        try:
            self.shader[name].value = value
        except KeyError:
            pass

    def _try_write(self, name: str, data: bytes) -> None:
        """Write to a uniform array / matrix that may have been optimized out."""
        try:
            self.shader[name].write(data)
        except KeyError:
            pass

    def _has_2d_renderer(self, world: Any) -> bool:
        """True if SpriteBatch2D is registered as a world resource."""
        try:
            from ..renderer.batch2d import SpriteBatch2D
        except ImportError:
            return False
        return world.has_resource(SpriteBatch2D)

    def _find_camera(self, world: Any) -> Camera3D:
        for (cams,) in world.query(Camera3D):
            if len(cams) > 0:
                rec = cams[0]
                x, y, z, pitch, yaw, roll, fov, near, far = _read_camera(rec)
                return Camera3D(
                    x=x, y=y, z=z,
                    pitch=pitch, yaw=yaw, roll=roll,
                    fov=fov, near=near, far=far,
                )
        return Camera3D()

    def _find_directional_light(self, world: Any) -> DirectionalLight:
        for (lights,) in world.query(DirectionalLight):
            if len(lights) > 0:
                rec = lights[0]
                return DirectionalLight(
                    dir_x=float(rec["dir_x"]),
                    dir_y=float(rec["dir_y"]),
                    dir_z=float(rec["dir_z"]),
                    r=float(rec["r"]),
                    g=float(rec["g"]),
                    b=float(rec["b"]),
                    intensity=float(rec["intensity"]),
                )
        # Default: a dim sun pointing diagonally down so unlit scenes still show shape.
        return DirectionalLight()

    def _collect_point_lights(self, world: Any, camera: Camera3D) -> int:
        """Pack up to MAX_POINT_LIGHTS lights (sorted nearest-to-camera) into the upload buffers."""
        records: list[tuple[float, tuple[float, float, float], tuple[float, float, float], float, float]] = []
        for transforms, lights in world.query(Transform3D, PointLight):
            n = len(transforms)
            for i in range(n):
                pos = (
                    float(transforms["x"][i]),
                    float(transforms["y"][i]),
                    float(transforms["z"][i]),
                )
                dx = pos[0] - camera.x
                dy = pos[1] - camera.y
                dz = pos[2] - camera.z
                dist_sq = dx * dx + dy * dy + dz * dz
                color = (
                    float(lights["r"][i]),
                    float(lights["g"][i]),
                    float(lights["b"][i]),
                )
                records.append(
                    (
                        dist_sq,
                        pos,
                        color,
                        float(lights["intensity"][i]),
                        float(lights["radius"][i]),
                    )
                )
        records.sort(key=lambda rec: rec[0])
        used = min(len(records), MAX_POINT_LIGHTS)

        self._point_pos.fill(0.0)
        self._point_color.fill(0.0)
        self._point_intensity.fill(0.0)
        self._point_radius.fill(0.0)
        for i in range(used):
            _, pos, color, intensity, radius = records[i]
            self._point_pos[i] = pos
            self._point_color[i] = color
            self._point_intensity[i] = intensity
            self._point_radius[i] = radius
        return used

    # --- main entry point ------------------------------------------------

    def render(self, world: Any, viewport_width: int, viewport_height: int) -> None:
        """One pass: VP, lights, depth-tested mesh draw."""
        if viewport_width <= 0 or viewport_height <= 0:
            return

        camera = self._find_camera(world)
        proj = build_projection_matrix(camera, viewport_width, viewport_height)
        view = build_view_matrix(camera)
        vp = proj @ view
        self.frustum_culler.update(vp)

        if not self._has_2d_renderer(world):
            self.ctx.clear(*_BACKGROUND_COLOR)

        self.ctx.enable(moderngl.DEPTH_TEST)
        try:
            self._upload_frame_uniforms(world, camera, view, proj)
            self._draw_meshes(world)
        finally:
            self.ctx.disable(moderngl.DEPTH_TEST)

    def _upload_frame_uniforms(
        self,
        world: Any,
        camera: Camera3D,
        view: np.ndarray,
        proj: np.ndarray,
    ) -> None:
        view_bytes = np.ascontiguousarray(view.T).tobytes()
        proj_bytes = np.ascontiguousarray(proj.T).tobytes()
        self._try_write("u_view", view_bytes)
        self._try_write("u_projection", proj_bytes)
        self._try_set("u_camera_pos", (camera.x, camera.y, camera.z))

        ambient = world.get_resource(AmbientLight)
        if ambient is None:
            ar, ag, ab = _DEFAULT_AMBIENT
        else:
            ar, ag, ab = float(ambient.r), float(ambient.g), float(ambient.b)
        self._try_set("u_ambient", (ar, ag, ab))

        dir_light = self._find_directional_light(world)
        self._try_set(
            "u_dir_light_dir",
            (float(dir_light.dir_x), float(dir_light.dir_y), float(dir_light.dir_z)),
        )
        self._try_set(
            "u_dir_light_color",
            (float(dir_light.r), float(dir_light.g), float(dir_light.b)),
        )
        self._try_set("u_dir_light_intensity", float(dir_light.intensity))

        n_points = self._collect_point_lights(world, camera)
        self.last_point_lights = n_points
        self._try_write("u_point_light_pos", self._point_pos.tobytes())
        self._try_write("u_point_light_color", self._point_color.tobytes())
        self._try_write("u_point_light_intensity", self._point_intensity.tobytes())
        self._try_write("u_point_light_radius", self._point_radius.tobytes())
        self._try_set("u_num_point_lights", n_points)

    def _draw_meshes(self, world: Any) -> None:
        self.last_draw_calls = 0
        self.last_culled = 0
        mesh_count = self.mesh_registry.count()
        if mesh_count == 0:
            return

        for arch in world.query(Transform3D, MeshRenderer).archetypes():
            n = arch.length
            transforms = arch.columns[Transform3D][:n]
            renderers = arch.columns[MeshRenderer][:n]
            entities = arch.entities[:n]

            t_xs = transforms["x"]
            t_ys = transforms["y"]
            t_zs = transforms["z"]
            visibles = renderers["visible"]
            mesh_ids = renderers["mesh_id"]
            material_ids = renderers["material_id"]

            for i in range(n):
                if not bool(visibles[i]):
                    continue
                mid = int(mesh_ids[i])
                if mid < 0 or mid >= mesh_count:
                    continue

                sphere = BoundingSphere(
                    float(t_xs[i]),
                    float(t_ys[i]),
                    float(t_zs[i]),
                    _DEFAULT_BOUND_RADIUS,
                )
                if not self.frustum_culler.is_visible(sphere):
                    self.last_culled += 1
                    continue

                model = resolve_world_matrix(int(entities[i]), world)
                self._try_write(
                    "u_model",
                    np.ascontiguousarray(model.T).tobytes(),
                )

                material = self.material_registry.get(int(material_ids[i]))
                self._try_set(
                    "u_albedo",
                    (float(material.albedo_r), float(material.albedo_g), float(material.albedo_b)),
                )
                self._try_set("u_roughness", float(material.roughness))
                self._try_set("u_metallic", float(material.metallic))
                self._try_set(
                    "u_emissive",
                    (
                        float(material.emissive_r),
                        float(material.emissive_g),
                        float(material.emissive_b),
                    ),
                )

                buf = self.mesh_registry.get_buffer(mid)
                buf.bind(self.shader)
                buf.render()
                self.last_draw_calls += 1


def setup_renderer_3d(app: Any) -> "Renderer3DSetup":
    """Create + register the 3D renderer on `app`. Idempotent — second call is a no-op."""
    existing = getattr(app, "_pyge_renderer_3d", None)
    if existing is not None:
        return existing

    ctx = app.ctx
    shader_cache = ShaderCache3D()
    shader = shader_cache.get(ctx, "pbr_lite")

    mesh_registry = MeshRegistry(ctx, shader)
    material_registry = MaterialRegistry()

    renderer = Renderer3D(ctx, mesh_registry, material_registry, shader)

    app.world.insert_resource(mesh_registry, type_=MeshRegistry)
    app.world.insert_resource(material_registry, type_=MaterialRegistry)
    app.world.insert_resource(renderer, type_=Renderer3D)
    app.world.insert_resource(shader_cache, type_=ShaderCache3D)

    @app.system(Phase.RENDER)
    def render_3d(world: Any, dt: float, renderer: Renderer3D) -> None:
        viewport_w, viewport_h = app.window.get_size()
        renderer.render(world, viewport_w, viewport_h)

    setup = Renderer3DSetup(
        mesh_registry=mesh_registry,
        material_registry=material_registry,
        renderer3d=renderer,
        shader_cache=shader_cache,
        render_system=render_3d,
    )
    app._pyge_renderer_3d = setup
    return setup
