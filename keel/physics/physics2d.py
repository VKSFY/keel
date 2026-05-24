"""Physics2D — pymunk (Chipmunk2D) bridge for the ECS.

Bridge invariants:
  * pymunk owns the simulation truth. ECS reads body state back into
    Transform2D / RigidBody2D each tick.
  * sync_to_physics first creates / updates pymunk bodies from ECS state,
    then step() advances the simulation, then sync_from_physics writes
    pymunk's results back into the archetype's structured arrays in place.
  * The collision handler is registered exactly once at construction. The
    pymunk callback only appends to a small buffer; the buffer is drained
    into world.emit() during _emit_collisions on the main thread.
"""
from __future__ import annotations

import warnings
from typing import Any

import pymunk

from ..components.transform2d import Transform2D
from ..core.query import Without
from .components2d import (
    BODY_TYPE_DYNAMIC,
    BODY_TYPE_KINEMATIC,
    BODY_TYPE_STATIC,
    SHAPE_TYPE_BOX,
    SHAPE_TYPE_CIRCLE,
    SHAPE_TYPE_SEGMENT,
    Collider2D,
    CollisionEvent2D,
    RigidBody2D,
)


_BODY_TYPE_MAP: dict[int, int] = {
    BODY_TYPE_DYNAMIC: pymunk.Body.DYNAMIC,
    BODY_TYPE_STATIC: pymunk.Body.STATIC,
    BODY_TYPE_KINEMATIC: pymunk.Body.KINEMATIC,
}


def _shape_entity_id(shape: pymunk.Shape) -> int:
    """Read the entity_id we attached to a pymunk shape, or 0 if absent."""
    return int(getattr(shape, "keel_entity_id", 0))


class Physics2D:
    """ECS ↔ pymunk bridge owning a single pymunk Space and all bodies/shapes."""

    def __init__(
        self,
        gravity_x: float = 0.0,
        gravity_y: float = -980.0,
        world: Any = None,
    ) -> None:
        self._space: pymunk.Space = pymunk.Space()
        self._space.gravity = (float(gravity_x), float(gravity_y))
        self._bodies: dict[int, pymunk.Body] = {}
        self._shapes: dict[int, pymunk.Shape] = {}
        self._body_types: dict[int, int] = {}
        self._collision_buffer: list[tuple[int, int, float, float, float]] = []
        self._mismatch_warned: set[int] = set()
        # One-shot latch — pymunk silently drops KINEMATIC-vs-KINEMATIC
        # callbacks, so we surface a UserWarning the first time a second
        # kinematic body joins.
        self._warned_kinematic_pair = False
        # Optional world reference so set_velocity / set_position can keep
        # the ECS components in sync. Captured by setup_physics_2d.
        self.world: Any = world
        self._setup_collision_handler()

    # ----------------------------------------------------------------------
    # Collision handler — pymunk fires two callbacks we care about:
    #   begin(arbiter, space, data) -> bool : called ONCE when two shapes
    #       first start touching. Fires for sensors too. Returning True
    #       lets the contact proceed; returning False would cancel it.
    #   post_solve(arbiter, ...)            : called every step the contact
    #       persists, ONLY for non-sensor pairs (sensors skip the solver,
    #       so post_solve never fires for them). Carries impulse + normal.
    #
    # Both callbacks append to _collision_buffer; the main thread drains it
    # in _emit_collisions(). The split lets non-sensor contacts ship real
    # impulse / normal data while sensor pickups still emit a CollisionEvent2D
    # (the bug platformers tripped over before — sensors were silent).
    # ----------------------------------------------------------------------

    def _setup_collision_handler(self) -> None:
        """Register pymunk 7 begin + post_solve handlers. See class banner above."""
        buffer = self._collision_buffer

        def _on_begin(arbiter: pymunk.Arbiter, space: pymunk.Space, data: Any) -> bool:
            shapes = arbiter.shapes
            if len(shapes) < 2:
                return True
            # Non-sensor pairs get their event from post_solve, where we have
            # impulse + normal data. Begin only handles sensors here.
            if not (shapes[0].sensor or shapes[1].sensor):
                return True
            eid_a = _shape_entity_id(shapes[0])
            eid_b = _shape_entity_id(shapes[1])
            if eid_a == 0 or eid_b == 0:
                return True
            # Sensor contacts give no useful impulse / normal; emit zeros so
            # _emit_collisions can unpack a uniform 5-tuple either way.
            buffer.append((eid_a, eid_b, 0.0, 0.0, 0.0))
            # Returning True lets pymunk run the (empty) sensor solve so that
            # separate() still fires; returning False would suppress it.
            return True

        def _post_solve(arbiter: pymunk.Arbiter, space: pymunk.Space, data: Any) -> None:
            shapes = arbiter.shapes
            if len(shapes) < 2:
                return
            eid_a = _shape_entity_id(shapes[0])
            eid_b = _shape_entity_id(shapes[1])
            if eid_a == 0 or eid_b == 0:
                return
            # Defensive: pymunk never calls post_solve for sensor pairs in
            # practice, but bail anyway so a future pymunk doesn't surprise us.
            if shapes[0].sensor or shapes[1].sensor:
                return
            try:
                normal = arbiter.normal
                nx, ny = float(normal.x), float(normal.y)
            except Exception:
                nx, ny = 0.0, 0.0
            try:
                impulse_vec = arbiter.total_impulse
                impulse_mag = float((impulse_vec.x ** 2 + impulse_vec.y ** 2) ** 0.5)
            except Exception:
                impulse_mag = 0.0
            buffer.append((eid_a, eid_b, nx, ny, impulse_mag))

        # collision_type_a/b = None => default handler that fires for every pair.
        self._space.on_collision(begin=_on_begin, post_solve=_post_solve)

    # ----------------------------------------------------------------------
    # Per-tick API — called by the POST_UPDATE physics_2d_system in this
    # order: sync_to_physics → step → sync_from_physics → _emit_collisions.
    # ----------------------------------------------------------------------

    def sync_to_physics(self, world: Any) -> None:
        """Build, update, or remove pymunk objects so they mirror current ECS state."""
        self._warn_mismatched_components(world)
        seen: set[int] = set()

        for arch in world.query(Transform2D, RigidBody2D, Collider2D).archetypes():
            n = arch.length
            transforms = arch.columns[Transform2D][:n]
            rbs = arch.columns[RigidBody2D][:n]
            colliders = arch.columns[Collider2D][:n]
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

                body = self._bodies[eid]
                if bt == BODY_TYPE_KINEMATIC:
                    body.position = (float(transforms["x"][i]), float(transforms["y"][i]))
                    body.angle = float(transforms["rotation"][i])
                if bt != BODY_TYPE_STATIC:
                    body.velocity = (
                        float(rbs["vel_x"][i]),
                        float(rbs["vel_y"][i]),
                    )
                    body.angular_velocity = float(rbs["ang_vel"][i])

        for eid in [eid for eid in self._bodies if eid not in seen]:
            self._remove_entity(eid)

    def step(self, dt: float) -> None:
        """Advance the pymunk simulation by `dt` seconds."""
        self._space.step(float(dt))

    def sync_from_physics(self, world: Any) -> None:
        """Write pymunk body state back into Transform2D / RigidBody2D arrays in place."""
        for arch in world.query(Transform2D, RigidBody2D).archetypes():
            n = arch.length
            transforms = arch.columns[Transform2D][:n]
            rbs = arch.columns[RigidBody2D][:n]
            entities = arch.entities[:n]

            t_x = transforms["x"]
            t_y = transforms["y"]
            t_rot = transforms["rotation"]
            r_vx = rbs["vel_x"]
            r_vy = rbs["vel_y"]
            r_avel = rbs["ang_vel"]
            body_types = rbs["body_type"]

            for i in range(n):
                eid = int(entities[i])
                body = self._bodies.get(eid)
                if body is None:
                    continue
                if int(body_types[i]) == BODY_TYPE_STATIC:
                    continue
                pos = body.position
                vel = body.velocity
                t_x[i] = pos.x
                t_y[i] = pos.y
                t_rot[i] = body.angle
                r_vx[i] = vel.x
                r_vy[i] = vel.y
                r_avel[i] = body.angular_velocity

    # ----------------------------------------------------------------------
    # Convenience controls — write into a body from gameplay code. Each one
    # also mirrors the value back into the ECS so the next sync_to_physics
    # doesn't undo the write.
    # ----------------------------------------------------------------------

    def apply_impulse(self, entity_id: int, impulse_x: float, impulse_y: float) -> None:
        """Apply a world-space impulse at the body's center of mass. No-op if absent."""
        body = self._bodies.get(int(entity_id))
        if body is None:
            return
        body.apply_impulse_at_local_point((float(impulse_x), float(impulse_y)))

    def apply_force(self, entity_id: int, force_x: float, force_y: float) -> None:
        """Apply a continuous world-space force at the body's center of mass."""
        body = self._bodies.get(int(entity_id))
        if body is None:
            return
        body.apply_force_at_local_point((float(force_x), float(force_y)))

    def set_velocity(self, entity_id: int, vel_x: float, vel_y: float) -> None:
        """Overwrite a body's linear velocity. Also mirrors the change into the ECS
        RigidBody2D fields when a world is attached, so the value survives the next
        sync_to_physics. No-op if entity has no body."""
        eid = int(entity_id)
        body = self._bodies.get(eid)
        if body is None:
            return
        vx, vy = float(vel_x), float(vel_y)
        body.velocity = (vx, vy)
        self._mirror_velocity_to_ecs(eid, vx, vy)

    def set_position(self, entity_id: int, x: float, y: float) -> None:
        """Teleport a body to (x, y). Also writes Transform2D in the ECS when a world
        is attached, so subsequent sync_to_physics doesn't fight the move (matters for
        kinematic bodies; cheap for dynamic). No-op if entity has no body."""
        eid = int(entity_id)
        body = self._bodies.get(eid)
        if body is None:
            return
        px, py = float(x), float(y)
        body.position = (px, py)
        self._mirror_position_to_ecs(eid, px, py)

    def _mirror_velocity_to_ecs(self, entity_id: int, vx: float, vy: float) -> None:
        if self.world is None:
            return
        loc = self.world.location_of(entity_id)
        if loc is None:
            return
        arch, row = loc
        if RigidBody2D not in arch.component_types:
            return
        col = arch.columns[RigidBody2D]
        col["vel_x"][row] = vx
        col["vel_y"][row] = vy

    def _mirror_position_to_ecs(self, entity_id: int, x: float, y: float) -> None:
        if self.world is None:
            return
        loc = self.world.location_of(entity_id)
        if loc is None:
            return
        arch, row = loc
        if Transform2D not in arch.component_types:
            return
        col = arch.columns[Transform2D]
        col["x"][row] = x
        col["y"][row] = y

    def raycast_2d(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        radius: float = 0.0,
    ) -> list[dict]:
        """Segment query in pymunk; results sorted nearest-first by `alpha`."""
        hits = self._space.segment_query(
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])),
            float(radius),
            pymunk.ShapeFilter(),
        )
        out: list[dict] = []
        for h in hits:
            shape = h.shape
            if shape is None:
                continue
            out.append(
                {
                    "entity_id": _shape_entity_id(shape),
                    "point": (float(h.point.x), float(h.point.y)),
                    "normal": (float(h.normal.x), float(h.normal.y)),
                    "alpha": float(h.alpha),
                }
            )
        out.sort(key=lambda hit: hit["alpha"])
        return out

    def _emit_collisions(self, world: Any) -> None:
        """Drain the collision buffer into world.emit(CollisionEvent2D(...))."""
        if not self._collision_buffer:
            return
        for eid_a, eid_b, nx, ny, impulse in self._collision_buffer:
            world.emit(
                CollisionEvent2D(
                    entity_a=eid_a,
                    entity_b=eid_b,
                    normal_x=nx,
                    normal_y=ny,
                    impulse=impulse,
                )
            )
        self._collision_buffer.clear()

    # ----------------------------------------------------------------------
    # Cleanup — called when the owning App shuts down.
    # ----------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove every body and shape from the pymunk space. Idempotent."""
        for eid in list(self._bodies.keys()):
            self._remove_entity(eid)

    # ----------------------------------------------------------------------
    # Internals — body / shape construction, archetype migration, helpers.
    # ----------------------------------------------------------------------

    def _create(
        self,
        eid: int,
        transforms: Any,
        rbs: Any,
        colliders: Any,
        i: int,
    ) -> None:
        """Build a pymunk body + shape from one row of the archetype views."""
        bt = int(rbs["body_type"][i])
        body_type_pm = _BODY_TYPE_MAP.get(bt, pymunk.Body.DYNAMIC)

        if body_type_pm == pymunk.Body.DYNAMIC:
            mass = float(rbs["mass"][i])
            moment_input = float(rbs["moment"][i])
            moment = moment_input if moment_input > 0.0 else self._auto_moment(mass, colliders, i)
            body = pymunk.Body(mass, moment, body_type=body_type_pm)
        else:
            body = pymunk.Body(body_type=body_type_pm)

        body.position = (float(transforms["x"][i]), float(transforms["y"][i]))
        body.angle = float(transforms["rotation"][i])
        if body_type_pm != pymunk.Body.STATIC:
            body.velocity = (
                float(rbs["vel_x"][i]),
                float(rbs["vel_y"][i]),
            )
            body.angular_velocity = float(rbs["ang_vel"][i])

        shape = self._create_shape(body, colliders, i)
        shape.friction = float(colliders["friction"][i])
        shape.elasticity = float(colliders["elasticity"][i])
        shape.sensor = bool(colliders["sensor"][i])
        shape.filter = pymunk.ShapeFilter(
            categories=int(colliders["category_bits"][i]),
            mask=int(colliders["mask_bits"][i]),
        )
        # Tag the shape with the entity ID so the collision callback can recover it.
        shape.keel_entity_id = eid  # type: ignore[attr-defined]

        self._space.add(body, shape)
        self._bodies[eid] = body
        self._shapes[eid] = shape
        self._body_types[eid] = bt
        # pymunk doesn't emit collision callbacks for KINEMATIC-vs-KINEMATIC
        # OR KINEMATIC-vs-STATIC pairs. Surface a one-time UserWarning when
        # this entity creates such a pairing with any existing body, so the
        # silent-failure trap is visible before it bites in gameplay code.
        if not self._warned_kinematic_pair:
            others = [t for e, t in self._body_types.items() if e != eid]
            offender = (
                (bt == BODY_TYPE_KINEMATIC and any(
                    t in (BODY_TYPE_KINEMATIC, BODY_TYPE_STATIC) for t in others))
                or
                (bt == BODY_TYPE_STATIC and any(
                    t == BODY_TYPE_KINEMATIC for t in others))
            )
            if offender:
                self._warned_kinematic_pair = True
                warnings.warn(
                    f"Physics2D: entity {eid} created a KINEMATIC/STATIC or "
                    "KINEMATIC/KINEMATIC body pair. pymunk does NOT emit "
                    "CollisionEvent2D for these pairs (it's a pymunk "
                    "limitation, not a Keel bug). Fix: change one body to "
                    "keel.BodyType.DYNAMIC so the collision callback fires. "
                    "See the 'Choosing body types for games' section in the "
                    "Keel README for the full guide.",
                    UserWarning,
                    stacklevel=2,
                )

    def _create_shape(self, body: pymunk.Body, colliders: Any, i: int) -> pymunk.Shape:
        st = int(colliders["shape_type"][i])
        if st == SHAPE_TYPE_CIRCLE:
            return pymunk.Circle(body, float(colliders["radius"][i]))
        if st == SHAPE_TYPE_BOX:
            return pymunk.Poly.create_box(
                body,
                (float(colliders["width"][i]), float(colliders["height"][i])),
            )
        if st == SHAPE_TYPE_SEGMENT:
            half_w = float(colliders["width"][i]) * 0.5
            return pymunk.Segment(body, (-half_w, 0.0), (half_w, 0.0), 1.0)
        # Unknown shape type — fall back to a circle so the entity at least exists.
        return pymunk.Circle(body, float(colliders["radius"][i]))

    @staticmethod
    def _auto_moment(mass: float, colliders: Any, i: int) -> float:
        st = int(colliders["shape_type"][i])
        if st == SHAPE_TYPE_CIRCLE:
            return float(pymunk.moment_for_circle(mass, 0.0, float(colliders["radius"][i])))
        if st == SHAPE_TYPE_BOX:
            return float(
                pymunk.moment_for_box(
                    mass,
                    (float(colliders["width"][i]), float(colliders["height"][i])),
                )
            )
        return 1.0

    def _remove_entity(self, eid: int) -> None:
        body = self._bodies.pop(eid, None)
        shape = self._shapes.pop(eid, None)
        self._body_types.pop(eid, None)
        if shape is not None and shape in self._space.shapes:
            self._space.remove(shape)
        if body is not None and body in self._space.bodies:
            self._space.remove(body)

    def _warn_mismatched_components(self, world: Any) -> None:
        """Emit one-time warnings for entities missing Transform2D, RigidBody2D, or Collider2D."""
        for arch in world.query(Collider2D, Without[RigidBody2D]).archetypes():
            for eid in arch.entities[: arch.length]:
                eid_int = int(eid)
                if eid_int in self._mismatch_warned:
                    continue
                self._mismatch_warned.add(eid_int)
                warnings.warn(
                    f"Physics2D: entity {eid_int} has Collider2D without RigidBody2D — skipping",
                    RuntimeWarning,
                    stacklevel=2,
                )
        for arch in world.query(RigidBody2D, Without[Collider2D]).archetypes():
            for eid in arch.entities[: arch.length]:
                eid_int = int(eid)
                if eid_int in self._mismatch_warned:
                    continue
                self._mismatch_warned.add(eid_int)
                warnings.warn(
                    f"Physics2D: entity {eid_int} has RigidBody2D without Collider2D — skipping",
                    RuntimeWarning,
                    stacklevel=2,
                )
        # A body without a Transform2D has no position to mirror — the main
        # sync loop requires all three components and silently skips this
        # case. Emit a one-time warning so users notice the misconfiguration.
        for arch in world.query(RigidBody2D, Collider2D, Without[Transform2D]).archetypes():
            for eid in arch.entities[: arch.length]:
                eid_int = int(eid)
                if eid_int in self._mismatch_warned:
                    continue
                self._mismatch_warned.add(eid_int)
                warnings.warn(
                    f"Physics2D: entity {eid_int} has RigidBody2D + Collider2D "
                    "without Transform2D — skipping (a body has no position "
                    "to sync without Transform2D)",
                    RuntimeWarning,
                    stacklevel=2,
                )
