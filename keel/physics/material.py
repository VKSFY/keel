"""Reusable surface presets for 2D physics shapes.

A `PhysicsMaterial2D` carries a friction + elasticity pair. Apply it to an
entity with `keel.apply_material(world, entity_id, material)` immediately
after the entity is spawned and flushed; Physics2D will use the material's
values instead of the Collider2D component's friction / elasticity when it
builds the pymunk shape on the next sync_to_physics.

Six built-in presets are exposed as class attributes:

    PhysicsMaterial2D.DEFAULT  friction=0.50  elasticity=0.30
    PhysicsMaterial2D.BOUNCY   friction=0.30  elasticity=0.90
    PhysicsMaterial2D.ICE      friction=0.05  elasticity=0.10
    PhysicsMaterial2D.RUBBER   friction=0.90  elasticity=0.80
    PhysicsMaterial2D.WOOD     friction=0.60  elasticity=0.20
    PhysicsMaterial2D.METAL    friction=0.30  elasticity=0.10
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhysicsMaterial2D:
    """Surface properties for a 2D physics shape."""
    friction: float = 0.5
    elasticity: float = 0.3


PhysicsMaterial2D.DEFAULT = PhysicsMaterial2D(friction=0.50, elasticity=0.30)
PhysicsMaterial2D.BOUNCY = PhysicsMaterial2D(friction=0.30, elasticity=0.90)
PhysicsMaterial2D.ICE = PhysicsMaterial2D(friction=0.05, elasticity=0.10)
PhysicsMaterial2D.RUBBER = PhysicsMaterial2D(friction=0.90, elasticity=0.80)
PhysicsMaterial2D.WOOD = PhysicsMaterial2D(friction=0.60, elasticity=0.20)
PhysicsMaterial2D.METAL = PhysicsMaterial2D(friction=0.30, elasticity=0.10)
