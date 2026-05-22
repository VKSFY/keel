"""Physics3D — pybullet bridge for 3D simulation.

Always uses ``p.connect(p.DIRECT)`` — never GUI. Every pybullet call passes
``physicsClientId=self._client`` explicitly so multiple Physics3D instances
(in tests, in editors, etc.) don't fight over the default client. ECS data
flows pymunk-style: sync_to_physics → step → sync_from_physics each tick.
"""
from __future__ import annotations

import warnings
from typing import Any

import pybullet as _pb
import pybullet_data as _pb_data

from ..components.transform3d import Transform3D
from ..core.query import Without
from .components3d import (
    BODY_TYPE_DYNAMIC,
    BODY_TYPE_KINEMATIC,
    BODY_TYPE_STATIC,
    SHAPE_TYPE_BOX,
    SHAPE_TYPE_CAPSULE,
    SHAPE_TYPE_MESH,
    SHAPE_TYPE_SPHERE,
    Collider3D,
    CollisionEvent3D,
    RigidBody3D,
)


_FIXED_DT: float = 1.0 / 240.0
_MAX_SUBSTEPS: int = 10


class Physics3D:
    """ECS ↔ pybullet bridge owning a single DIRECT-mode pybullet client."""

    def __init__(
        self,
        gravity_x: float = 0.0,
        gravity_y: float = -9.81,
        gravity_z: float = 0.0,
        world: Any = None,
    ) -> None:
        self._p = _pb
        # Headless-only: GUI mode is forbidden by Phase 6's quality rules.
        self._client: int = _pb.connect(_pb.DIRECT)
        assert self._client >= 0, "pybullet DIRECT connect returned a negative client id"
        _pb.setGravity(
            float(gravity_x),
            float(gravity_y),
            float(gravity_z),
            physicsClientId=self._client,
        )
        _pb.setAdditionalSearchPath(_pb_data.getDataPath(), physicsClientId=self._client)

        self._bodies: dict[int, int] = {}            # entity_id -> bullet body id
        self._collision_shapes: dict[int, int] = {}  # entity_id -> bullet shape id
        self._body_types: dict[int, int] = {}        # entity_id -> last seen body_type
        self._mismatch_warned: set[int] = set()
        self._collision_buffer: list[
            tuple[int, int, float, float, float, float, float, float]
        ] = []
        self._connected: bool = True
        # Optional world reference so set_velocity / set_position can mirror
        # the change into the ECS RigidBody3D / Transform3D columns. Captured
        # by setup_physics_3d.
        self.world: Any = world

    # --- Per-tick API -------------------------------------------------------

    def sync_to_physics(self, world: Any) -> None:
        """Build, update, or remove bullet bodies so they mirror current ECS state."""
        if not self._connected:
            return
        self._warn_mismatched_components(world)
        seen: set[int] = set()

        for arch in world.query(Transform3D, RigidBody3D, Collider3D).archetypes():
            n = arch.length
            transforms = arch.columns[Transform3D][:n]
            rbs = arch.columns[RigidBody3D][:n]
            colliders = arch.columns[Collider3D][:n]
            entities = arch.entities[:n]

            for i in range(n):
                eid = int(entities[i])
                seen.add(eid)
                bt = int(rbs["body_type"][i])

                existing_bt = self._body_types.get(eid)
                if eid not in self._bodies or existing_bt != bt:
                    if eid in self._bodies:
                        self._remove_entity(eid)
                    self._create(eid, transforms, rbs, colliders, i)
                    continue

                body_id = self._bodies[eid]
                if bt == BODY_TYPE_KINEMATIC:
                    pos = (
                        float(transforms["x"][i]),
                        float(transforms["y"][i]),
                        float(transforms["z"][i]),
                    )
                    orn = self._p.getQuaternionFromEuler(
                        [
                            float(transforms["rot_x"][i]),
                            float(transforms["rot_y"][i]),
                            float(transforms["rot_z"][i]),
                        ]
                    )
                    self._p.resetBasePositionAndOrientation(
                        body_id, pos, orn, physicsClientId=self._client
                    )
                if bt == BODY_TYPE_DYNAMIC:
                    self._p.resetBaseVelocity(
                        body_id,
                        linearVelocity=(
                            float(rbs["vel_x"][i]),
                            float(rbs["vel_y"][i]),
                            float(rbs["vel_z"][i]),
                        ),
                        angularVelocity=(
                            float(rbs["ang_vel_x"][i]),
                            float(rbs["ang_vel_y"][i]),
                            float(rbs["ang_vel_z"][i]),
                        ),
                        physicsClientId=self._client,
                    )

        for eid in [eid for eid in self._bodies if eid not in seen]:
            self._remove_entity(eid)

    def step(self, dt: float) -> None:
        """Advance the bullet simulation. Substeps are clamped per the spec to 1..10."""
        if not self._connected:
            return
        substeps = max(1, min(_MAX_SUBSTEPS, int(float(dt) / _FIXED_DT)))
        for _ in range(substeps):
            self._p.stepSimulation(physicsClientId=self._client)
        self._collect_contacts()

    def sync_from_physics(self, world: Any) -> None:
        """Write bullet body state back into Transform3D / RigidBody3D arrays in place."""
        if not self._connected:
            return
        for arch in world.query(Transform3D, RigidBody3D).archetypes():
            n = arch.length
            transforms = arch.columns[Transform3D][:n]
            rbs = arch.columns[RigidBody3D][:n]
            entities = arch.entities[:n]

            t_x = transforms["x"]
            t_y = transforms["y"]
            t_z = transforms["z"]
            t_rx = transforms["rot_x"]
            t_ry = transforms["rot_y"]
            t_rz = transforms["rot_z"]

            r_vx = rbs["vel_x"]
            r_vy = rbs["vel_y"]
            r_vz = rbs["vel_z"]
            r_ax = rbs["ang_vel_x"]
            r_ay = rbs["ang_vel_y"]
            r_az = rbs["ang_vel_z"]
            body_types = rbs["body_type"]

            for i in range(n):
                eid = int(entities[i])
                body_id = self._bodies.get(eid)
                if body_id is None:
                    continue
                if int(body_types[i]) == BODY_TYPE_STATIC:
                    continue

                pos, orn = self._p.getBasePositionAndOrientation(
                    body_id, physicsClientId=self._client
                )
                t_x[i] = pos[0]
                t_y[i] = pos[1]
                t_z[i] = pos[2]

                euler = self._p.getEulerFromQuaternion(orn)
                t_rx[i] = euler[0]
                t_ry[i] = euler[1]
                t_rz[i] = euler[2]

                lin_vel, ang_vel = self._p.getBaseVelocity(
                    body_id, physicsClientId=self._client
                )
                r_vx[i] = lin_vel[0]
                r_vy[i] = lin_vel[1]
                r_vz[i] = lin_vel[2]
                r_ax[i] = ang_vel[0]
                r_ay[i] = ang_vel[1]
                r_az[i] = ang_vel[2]

    # --- Convenience controls ---------------------------------------------

    def apply_impulse(self, entity_id: int, ix: float, iy: float, iz: float) -> None:
        """Add a velocity-change impulse (impulse = mass × Δv) at the body's center."""
        body_id = self._bodies.get(int(entity_id))
        if body_id is None or not self._connected:
            return
        info = self._p.getDynamicsInfo(body_id, -1, physicsClientId=self._client)
        mass = float(info[0])
        if mass <= 0.0:
            return
        lin_vel, ang_vel = self._p.getBaseVelocity(body_id, physicsClientId=self._client)
        new_lin = (
            lin_vel[0] + float(ix) / mass,
            lin_vel[1] + float(iy) / mass,
            lin_vel[2] + float(iz) / mass,
        )
        self._p.resetBaseVelocity(
            body_id,
            linearVelocity=new_lin,
            angularVelocity=ang_vel,
            physicsClientId=self._client,
        )

    def apply_force(self, entity_id: int, fx: float, fy: float, fz: float) -> None:
        """Apply a continuous world-space force at the body's center of mass."""
        body_id = self._bodies.get(int(entity_id))
        if body_id is None or not self._connected:
            return
        self._p.applyExternalForce(
            body_id,
            -1,
            forceObj=[float(fx), float(fy), float(fz)],
            posObj=[0.0, 0.0, 0.0],
            flags=self._p.WORLD_FRAME,
            physicsClientId=self._client,
        )

    def set_velocity(
        self,
        entity_id: int,
        vx: float,
        vy: float,
        vz: float,
    ) -> None:
        """Overwrite a body's linear velocity. Also mirrors the change into
        RigidBody3D.vel_x/y/z when a world is attached, so the value survives
        the next sync_to_physics. No-op if entity has no body."""
        eid = int(entity_id)
        body_id = self._bodies.get(eid)
        if body_id is None or not self._connected:
            return
        vx_f, vy_f, vz_f = float(vx), float(vy), float(vz)
        # Preserve angular velocity (resetBaseVelocity overwrites both).
        try:
            _, ang_vel = self._p.getBaseVelocity(body_id, physicsClientId=self._client)
        except Exception:
            ang_vel = (0.0, 0.0, 0.0)
        self._p.resetBaseVelocity(
            body_id,
            linearVelocity=(vx_f, vy_f, vz_f),
            angularVelocity=ang_vel,
            physicsClientId=self._client,
        )
        self._mirror_velocity_to_ecs(eid, vx_f, vy_f, vz_f)

    def set_position(
        self,
        entity_id: int,
        x: float,
        y: float,
        z: float,
    ) -> None:
        """Teleport a body to (x, y, z). Also writes Transform3D in the ECS
        when a world is attached, so subsequent sync_to_physics doesn't fight
        the move. No-op if entity has no body."""
        eid = int(entity_id)
        body_id = self._bodies.get(eid)
        if body_id is None or not self._connected:
            return
        px, py, pz = float(x), float(y), float(z)
        # Preserve current orientation (resetBasePositionAndOrientation needs both).
        try:
            _, orn = self._p.getBasePositionAndOrientation(
                body_id, physicsClientId=self._client
            )
        except Exception:
            orn = (0.0, 0.0, 0.0, 1.0)
        self._p.resetBasePositionAndOrientation(
            body_id, (px, py, pz), orn, physicsClientId=self._client
        )
        self._mirror_position_to_ecs(eid, px, py, pz)

    def _mirror_velocity_to_ecs(
        self,
        entity_id: int,
        vx: float,
        vy: float,
        vz: float,
    ) -> None:
        if self.world is None:
            return
        loc = self.world.location_of(entity_id)
        if loc is None:
            return
        arch, row = loc
        if RigidBody3D not in arch.component_types:
            return
        col = arch.columns[RigidBody3D]
        col["vel_x"][row] = vx
        col["vel_y"][row] = vy
        col["vel_z"][row] = vz

    def _mirror_position_to_ecs(
        self,
        entity_id: int,
        x: float,
        y: float,
        z: float,
    ) -> None:
        if self.world is None:
            return
        loc = self.world.location_of(entity_id)
        if loc is None:
            return
        arch, row = loc
        if Transform3D not in arch.component_types:
            return
        col = arch.columns[Transform3D]
        col["x"][row] = x
        col["y"][row] = y
        col["z"][row] = z

    def raycast_3d(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> list[dict]:
        """Single-ray query against the world. Returns hits sorted by `fraction`."""
        if not self._connected:
            return []
        results = self._p.rayTest(
            list(start),
            list(end),
            physicsClientId=self._client,
        )
        out: list[dict] = []
        for hit in results:
            obj_uid = hit[0]
            if obj_uid < 0:
                continue
            eid = next(
                (k for k, v in self._bodies.items() if v == obj_uid),
                0,
            )
            out.append(
                {
                    "entity_id": eid,
                    "point": (float(hit[3][0]), float(hit[3][1]), float(hit[3][2])),
                    "normal": (float(hit[4][0]), float(hit[4][1]), float(hit[4][2])),
                    "fraction": float(hit[2]),
                }
            )
        out.sort(key=lambda h: h["fraction"])
        return out

    def _emit_collisions(self, world: Any) -> None:
        """Drain the contact buffer into world.emit(CollisionEvent3D(...))."""
        if not self._collision_buffer:
            return
        for entry in self._collision_buffer:
            (
                eid_a, eid_b,
                contact_x, contact_y, contact_z,
                normal_x, normal_y, normal_z,
            ) = entry
            world.emit(
                CollisionEvent3D(
                    entity_a=eid_a,
                    entity_b=eid_b,
                    contact_x=contact_x,
                    contact_y=contact_y,
                    contact_z=contact_z,
                    normal_x=normal_x,
                    normal_y=normal_y,
                    normal_z=normal_z,
                )
            )
        self._collision_buffer.clear()

    # --- Cleanup ----------------------------------------------------------

    def disconnect(self) -> None:
        """Close the pybullet client. Idempotent — second call is a no-op."""
        if not self._connected:
            return
        try:
            self._p.disconnect(physicsClientId=self._client)
        except Exception:
            pass
        self._connected = False
        self._bodies.clear()
        self._collision_shapes.clear()
        self._body_types.clear()

    def cleanup(self) -> None:
        """Alias for disconnect — used by App shutdown hooks."""
        self.disconnect()

    @property
    def connected(self) -> bool:
        """True if the underlying pybullet client is still open."""
        return self._connected

    @property
    def client_id(self) -> int:
        """Underlying pybullet physics client ID — exposed for tests / debug."""
        return self._client

    # --- Internals --------------------------------------------------------

    def _create(
        self,
        eid: int,
        transforms: Any,
        rbs: Any,
        colliders: Any,
        i: int,
    ) -> None:
        shape_id = self._create_shape(colliders, i)
        bt = int(rbs["body_type"][i])
        mass = float(rbs["mass"][i]) if bt == BODY_TYPE_DYNAMIC else 0.0

        pos = (
            float(transforms["x"][i]),
            float(transforms["y"][i]),
            float(transforms["z"][i]),
        )
        orn = self._p.getQuaternionFromEuler(
            [
                float(transforms["rot_x"][i]),
                float(transforms["rot_y"][i]),
                float(transforms["rot_z"][i]),
            ]
        )
        body_id = self._p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=shape_id,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=self._client,
        )

        # Material + damping properties.
        self._p.changeDynamics(
            body_id,
            -1,
            lateralFriction=float(colliders["friction"][i]),
            restitution=float(colliders["restitution"][i]),
            linearDamping=float(rbs["damping"][i]),
            angularDamping=float(rbs["ang_damping"][i]),
            physicsClientId=self._client,
        )

        if bt == BODY_TYPE_KINEMATIC:
            # Pybullet flag bits — KINEMATIC body never integrates, only resets.
            try:
                self._p.changeDynamics(
                    body_id,
                    -1,
                    activationState=self._p.ACTIVATION_STATE_DISABLE_DEACTIVATION,
                    physicsClientId=self._client,
                )
            except Exception:
                pass

        if bt == BODY_TYPE_DYNAMIC:
            self._p.resetBaseVelocity(
                body_id,
                linearVelocity=(
                    float(rbs["vel_x"][i]),
                    float(rbs["vel_y"][i]),
                    float(rbs["vel_z"][i]),
                ),
                angularVelocity=(
                    float(rbs["ang_vel_x"][i]),
                    float(rbs["ang_vel_y"][i]),
                    float(rbs["ang_vel_z"][i]),
                ),
                physicsClientId=self._client,
            )

        self._bodies[eid] = body_id
        self._collision_shapes[eid] = shape_id
        self._body_types[eid] = bt

    def _create_shape(self, colliders: Any, i: int) -> int:
        st = int(colliders["shape_type"][i])
        if st == SHAPE_TYPE_BOX:
            return self._p.createCollisionShape(
                self._p.GEOM_BOX,
                halfExtents=[
                    float(colliders["size_x"][i]),
                    float(colliders["size_y"][i]),
                    float(colliders["size_z"][i]),
                ],
                physicsClientId=self._client,
            )
        if st == SHAPE_TYPE_CAPSULE:
            return self._p.createCollisionShape(
                self._p.GEOM_CAPSULE,
                radius=float(colliders["size_x"][i]),
                height=float(colliders["size_y"][i]),
                physicsClientId=self._client,
            )
        if st == SHAPE_TYPE_MESH:
            warnings.warn(
                "Physics3D: shape_type=MESH is not implemented; falling back to a sphere",
                RuntimeWarning,
                stacklevel=3,
            )
            return self._p.createCollisionShape(
                self._p.GEOM_SPHERE,
                radius=float(colliders["radius"][i]),
                physicsClientId=self._client,
            )
        # Default / SHAPE_TYPE_SPHERE.
        return self._p.createCollisionShape(
            self._p.GEOM_SPHERE,
            radius=float(colliders["radius"][i]),
            physicsClientId=self._client,
        )

    def _remove_entity(self, eid: int) -> None:
        body_id = self._bodies.pop(eid, None)
        self._collision_shapes.pop(eid, None)
        self._body_types.pop(eid, None)
        if body_id is None or not self._connected:
            return
        try:
            self._p.removeBody(body_id, physicsClientId=self._client)
        except Exception:
            pass

    def _collect_contacts(self) -> None:
        """Drain pybullet's per-step contact list into self._collision_buffer."""
        try:
            contacts = self._p.getContactPoints(physicsClientId=self._client)
        except Exception:
            return
        for c in contacts:
            uid_a = c[1]
            uid_b = c[2]
            eid_a = next((k for k, v in self._bodies.items() if v == uid_a), 0)
            eid_b = next((k for k, v in self._bodies.items() if v == uid_b), 0)
            if eid_a == 0 or eid_b == 0:
                continue
            point_b = c[6]
            normal_on_b = c[7]
            self._collision_buffer.append(
                (
                    eid_a, eid_b,
                    float(point_b[0]), float(point_b[1]), float(point_b[2]),
                    float(normal_on_b[0]), float(normal_on_b[1]), float(normal_on_b[2]),
                )
            )

    def _warn_mismatched_components(self, world: Any) -> None:
        for arch in world.query(Collider3D, Without[RigidBody3D]).archetypes():
            for eid in arch.entities[: arch.length]:
                eid_int = int(eid)
                if eid_int in self._mismatch_warned:
                    continue
                self._mismatch_warned.add(eid_int)
                warnings.warn(
                    f"Physics3D: entity {eid_int} has Collider3D without RigidBody3D — skipping",
                    RuntimeWarning,
                    stacklevel=2,
                )
        for arch in world.query(RigidBody3D, Without[Collider3D]).archetypes():
            for eid in arch.entities[: arch.length]:
                eid_int = int(eid)
                if eid_int in self._mismatch_warned:
                    continue
                self._mismatch_warned.add(eid_int)
                warnings.warn(
                    f"Physics3D: entity {eid_int} has RigidBody3D without Collider3D — skipping",
                    RuntimeWarning,
                    stacklevel=2,
                )
