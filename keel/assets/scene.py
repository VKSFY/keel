"""Scene save/load: JSON serialization of every entity + component in a World.

Component classes are resolved at load time by walking sys.modules for
classes carrying the `__keel_component__` metadata attached by Phase 1's
@component decorator. This avoids modifying core/ to add a global registry.

The on-disk schema:

    {
      "version": 1,
      "entities": [
        {"id": 42, "components": {"Transform2D": {"x": 1.0, ...}, ...}},
        ...
      ]
    }

Save is atomic: data is written to <path>.tmp and then os.replace'd onto the
target, so a crash mid-write can never corrupt a previous save file.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import warnings
from typing import Any

import numpy as np

from .registry import AssetRegistry  # noqa: F401  (kept for the optional load() arg)


SCENE_VERSION: int = 1


class SceneVersionError(Exception):
    """Raised when Scene.load is given a file with a version field it can't handle."""


def _is_jsonable_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _to_python_scalar(v: Any) -> Any:
    """Convert numpy scalars / arrays to native Python so json.dumps doesn't choke."""
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def _serialize_component(inst: Any) -> dict[str, Any]:
    """Convert a component dataclass instance into a JSON-safe field dict."""
    fields: dict[str, Any] = {}
    for f in dataclasses.fields(inst):
        v = _to_python_scalar(getattr(inst, f.name))
        if _is_jsonable_scalar(v):
            fields[f.name] = v
            continue
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            warnings.warn(
                f"Scene.save: skipping unserializable field "
                f"{type(inst).__name__}.{f.name} = {v!r}",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        fields[f.name] = v
    return fields


def _serialize_world(world: Any) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for arch in world.archetypes.all_archetypes():
        for row in range(arch.length):
            entity_id = int(arch.entities[row])
            comps: dict[str, dict[str, Any]] = {}
            for ct in arch.component_types:
                inst = arch.get_component(row, ct)
                comps[ct.__name__] = _serialize_component(inst)
            entities.append({"id": entity_id, "components": comps})
    return entities


def _discover_components() -> dict[str, type]:
    """Walk every loaded module and collect classes carrying __keel_component__."""
    found: dict[str, type] = {}
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            members = vars(mod)
        except Exception:
            continue
        for obj in members.values():
            if not isinstance(obj, type):
                continue
            if getattr(obj, "__keel_component__", None) is None:
                continue
            name = obj.__name__
            if name not in found:
                found[name] = obj
    return found


def _deserialize_component(cls: type, fields: dict[str, Any]) -> Any:
    """Construct a component instance from a field dict, dropping unknown keys."""
    valid = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in fields.items() if k in valid}
    return cls(**kwargs)


class Scene:
    """Static facade for save/load of a World to a JSON file."""

    VERSION: int = SCENE_VERSION

    @staticmethod
    def save(world: Any, path: str) -> None:
        """Atomically serialize every component of every live entity in `world` to `path`."""
        data = {
            "version": Scene.VERSION,
            "entities": _serialize_world(world),
        }
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)

    @staticmethod
    def load(
        world: Any,
        path: str,
        registry: AssetRegistry | None = None,
    ) -> list[int]:
        """Deserialize entities from `path`, spawn them additively into `world`. Returns the new IDs."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        version = data.get("version")
        if version != Scene.VERSION:
            raise SceneVersionError(
                f"Scene at {path!r} has version={version!r}, expected {Scene.VERSION}"
            )
        components = _discover_components()
        spawned: list[int] = []
        for entry in data.get("entities", []):
            comps_dict = entry.get("components", {})
            instances: list[Any] = []
            for name, fields in comps_dict.items():
                cls = components.get(name)
                if cls is None:
                    warnings.warn(
                        f"Scene.load: unknown component {name!r} — entry skipped. "
                        "Make sure the component module is imported before loading.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                try:
                    instances.append(_deserialize_component(cls, fields))
                except Exception as e:
                    warnings.warn(
                        f"Scene.load: failed to construct {name}({fields!r}): {e!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if not instances:
                continue
            new_id = world.spawn(*instances)
            spawned.append(new_id)
        world.flush()
        return spawned

    @staticmethod
    def load_additive(
        world: Any,
        path: str,
        registry: AssetRegistry | None = None,
    ) -> list[int]:
        """Identical to load() — name documents that the world is not cleared first."""
        return Scene.load(world, path, registry)
