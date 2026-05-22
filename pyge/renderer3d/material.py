"""Material data + MaterialRegistry.

PBR-lite — albedo + roughness + metallic + emissive scalars only. Texture
maps (albedo / normal / metallic-roughness textures) are deliberately out
of scope for v1.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Material:
    """Per-mesh shading parameters. Floats only — no texture maps in v1."""
    albedo_r: float = 0.8
    albedo_g: float = 0.8
    albedo_b: float = 0.8
    roughness: float = 0.5
    metallic: float = 0.0
    emissive_r: float = 0.0
    emissive_g: float = 0.0
    emissive_b: float = 0.0


class MaterialRegistry:
    """Indexed store of Materials. ID 0 is always a default mid-gray surface."""

    __slots__ = ("_materials", "_default_id")

    def __init__(self) -> None:
        self._materials: list[Material] = []
        # Always allocate a default at id=0 so MeshRenderer.material_id=0 is safe.
        self._default_id: int = self._add(Material())

    def _add(self, mat: Material) -> int:
        idx = len(self._materials)
        self._materials.append(mat)
        return idx

    def add(self, material: Material) -> int:
        """Append `material` and return its integer ID."""
        return self._add(material)

    def get(self, material_id: int) -> Material:
        """Return the Material for `material_id`. Falls back to the default for out-of-range IDs."""
        if material_id < 0 or material_id >= len(self._materials):
            return self._materials[self._default_id]
        return self._materials[material_id]

    def default_id(self) -> int:
        """ID of the default mid-gray material — always 0, always valid."""
        return self._default_id

    def __len__(self) -> int:
        return len(self._materials)
