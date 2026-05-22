"""Headless tests for the Phase 4 asset pipeline.

Most tests use mocked loaders so no GL context or watchdog observer is
required. The texture-loader integration test boots a hidden GLFW context
on demand and skips if one can't be created.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from watchdog.events import FileCreatedEvent, FileModifiedEvent

import keel
from keel import (
    AssetHandle,
    AssetNotFoundError,
    AssetRegistry,
    FileWatcher,
    InvalidHandleError,
    NoLoaderError,
    Phase,
    Scene,
    SceneVersionError,
    Scheduler,
    Sprite,
    Transform2D,
    World,
    component,
    setup_assets as setup_assets_fn,
)
from keel.assets import setup_assets
from keel.assets.loaders import json_loader, make_texture_loader
from keel.assets.registry import _normalize


# --- Fakes ----------------------------------------------------------------

def _fake_app() -> Any:
    """A minimal app-shape for setup_assets that doesn't need a real window/context."""
    sched = Scheduler()
    world = World()
    app = SimpleNamespace(
        world=world,
        _scheduler=sched,
        scheduler=sched,
    )

    def system(phase: Phase):
        def deco(fn):
            sched.register(phase, fn)
            return fn
        return deco

    app.system = system
    return app


def _write_text(tmp_path, name: str, content: str = "{}") -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# --- AssetHandle ----------------------------------------------------------

def test_handle_is_immutable():
    h = AssetHandle(7, "/abs/p", str)
    with pytest.raises(AttributeError):
        h.id = 99
    with pytest.raises(AttributeError):
        h.path = "/other"


def test_handle_is_hashable_and_dict_keyable():
    h1 = AssetHandle(1, "/abs/a", str)
    h2 = AssetHandle(2, "/abs/b", int)
    d = {h1: "alpha", h2: "beta"}
    assert d[h1] == "alpha"
    assert d[h2] == "beta"


def test_handle_equality_by_path():
    h1 = AssetHandle(1, "/abs/same", str)
    h2 = AssetHandle(2, "/abs/same", int)  # different id and type, same path
    h3 = AssetHandle(3, "/abs/other", str)
    assert h1 == h2
    assert hash(h1) == hash(h2)
    assert h1 != h3


# --- AssetRegistry --------------------------------------------------------

def test_registry_load_returns_same_handle_for_same_path(tmp_path):
    reg = AssetRegistry()
    reg.register_loader([".json"], json_loader)
    p = _write_text(tmp_path, "data.json", "{}")
    h1 = reg.load(p)
    h2 = reg.load(p)
    assert h1 is h2  # identity, not just equality


def test_registry_load_normalizes_paths(tmp_path):
    reg = AssetRegistry()
    reg.register_loader([".json"], json_loader)
    p = _write_text(tmp_path, "data.json", "{}")
    h1 = reg.load(p)
    h2 = reg.load(os.path.join(str(tmp_path), ".", "data.json"))
    assert h1 is h2


def test_registry_missing_file_raises(tmp_path):
    reg = AssetRegistry()
    reg.register_loader([".json"], json_loader)
    with pytest.raises(AssetNotFoundError):
        reg.load(str(tmp_path / "ghost.json"))


def test_registry_unregistered_extension_raises(tmp_path):
    p = _write_text(tmp_path, "weird.xyz", "data")
    reg = AssetRegistry()
    with pytest.raises(NoLoaderError):
        reg.load(p)


def test_registry_get_returns_loader_value(tmp_path):
    reg = AssetRegistry()
    reg.register_loader([".foo"], lambda path: {"loaded_from": path})
    p = _write_text(tmp_path, "x.foo", "hi")
    h = reg.load(p)
    asset = reg.get(h)
    assert asset["loaded_from"] == _normalize(p)


def test_registry_reload_calls_loader_again(tmp_path):
    counter = {"n": 0}

    def loader(path: str) -> int:
        counter["n"] += 1
        return counter["n"]

    reg = AssetRegistry()
    reg.register_loader([".foo"], loader)
    p = _write_text(tmp_path, "x.foo", "hi")
    h = reg.load(p)
    assert reg.get(h) == 1
    reg.reload(h)
    assert reg.get(h) == 2
    assert counter["n"] == 2


def test_registry_unload_makes_handle_invalid(tmp_path):
    reg = AssetRegistry()
    reg.register_loader([".foo"], lambda p: "asset")
    p = _write_text(tmp_path, "x.foo", "hi")
    h = reg.load(p)
    assert reg.loaded_count() == 1
    reg.unload(h)
    assert reg.loaded_count() == 0
    with pytest.raises(InvalidHandleError):
        reg.get(h)
    with pytest.raises(InvalidHandleError):
        reg.reload(h)


def test_registry_ids_are_monotonic_and_never_reused(tmp_path):
    reg = AssetRegistry()
    reg.register_loader([".foo"], lambda p: "asset")
    p1 = _write_text(tmp_path, "a.foo", "hi")
    p2 = _write_text(tmp_path, "b.foo", "hi")

    h1 = reg.load(p1)
    h2 = reg.load(p2)
    assert h2.id > h1.id

    reg.unload(h1)
    h1_again = reg.load(p1)
    # New handle, brand-new ID even though the path is reused.
    assert h1_again.id > h2.id


def test_registry_get_rejects_non_handle():
    reg = AssetRegistry()
    with pytest.raises(InvalidHandleError):
        reg.get("not-a-handle")


# --- json_loader ----------------------------------------------------------

def test_json_loader_parses_dict(tmp_path):
    p = _write_text(tmp_path, "x.json", '{"a": 1, "b": [2, 3]}')
    assert json_loader(p) == {"a": 1, "b": [2, 3]}


# --- texture_loader -------------------------------------------------------

class _FakeAtlas:
    """Minimal TextureAtlas substitute used to verify make_texture_loader semantics."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._path_to_id: dict[str, int] = {}
        self._next_id: int = 0

    def load(self, path: str) -> int:
        self.calls.append(("load", path))
        tid = self._next_id
        self._next_id += 1
        self._path_to_id[path] = tid
        return tid

    def reload(self, texture_id: int) -> None:
        self.calls.append(("reload", str(texture_id)))


def test_make_texture_loader_returns_int(tmp_path):
    atlas = _FakeAtlas()
    loader = make_texture_loader(atlas)
    p = _write_text(tmp_path, "img.png", "fake")
    tid = loader(p)
    assert isinstance(tid, int)
    assert atlas.calls == [("load", p)]


def test_texture_loader_reload_path_calls_reload_not_load(tmp_path):
    atlas = _FakeAtlas()
    loader = make_texture_loader(atlas)
    p = _write_text(tmp_path, "img.png", "fake")
    first = loader(p)
    second = loader(p)
    assert first == second
    assert atlas.calls == [("load", p), ("reload", str(first))]


# --- FileWatcher ----------------------------------------------------------

class _CountingRegistry:
    """Stand-in for AssetRegistry that just counts reloads on registered paths."""

    def __init__(self) -> None:
        self._handles: dict[str, AssetHandle] = {}
        self.reload_count: int = 0

    def register(self, path: str) -> AssetHandle:
        normalized = _normalize(path)
        h = AssetHandle(len(self._handles) + 1, normalized, str)
        self._handles[normalized] = h
        return h

    def reload(self, handle: AssetHandle) -> None:
        self.reload_count += 1


def test_filewatcher_poll_drains_queue_and_reloads(tmp_path):
    reg = _CountingRegistry()
    p = str(tmp_path / "watched.png")
    reg.register(p)

    fw = FileWatcher(reg)
    fw._queue.put(p)
    fw._queue.put(p)  # duplicate within same poll — coalesced
    reloads = fw.poll()
    assert reloads == 1
    assert reg.reload_count == 1


def test_filewatcher_callback_does_not_reload_directly(tmp_path):
    """Watchdog handler must only enqueue — reloads must wait until poll()."""
    reg = _CountingRegistry()
    p = str(tmp_path / "watched.png")
    reg.register(p)

    fw = FileWatcher(reg)
    # Simulate watchdog firing on the background thread.
    fw._handler.on_modified(FileModifiedEvent(p))
    fw._handler.on_created(FileCreatedEvent(p))
    assert reg.reload_count == 0, "callback must not touch the registry"

    # Now drain on the main thread.
    fw.poll()
    assert reg.reload_count == 1  # coalesced to one reload for that path


def test_filewatcher_ignores_directory_events(tmp_path):
    reg = _CountingRegistry()
    fw = FileWatcher(reg)

    dir_event = FileModifiedEvent(str(tmp_path))
    dir_event.is_directory = True
    fw._handler.on_modified(dir_event)

    fw.poll()
    assert reg.reload_count == 0


def test_filewatcher_ignores_unwatched_paths(tmp_path):
    reg = _CountingRegistry()
    fw = FileWatcher(reg)
    fw._queue.put(str(tmp_path / "not_registered.png"))
    fw.poll()
    assert reg.reload_count == 0


def test_filewatcher_watch_starts_observer_lazily(tmp_path):
    reg = _CountingRegistry()
    fw = FileWatcher(reg)
    assert fw.started is False
    fw.watch(str(tmp_path))
    try:
        assert fw.started is True
        assert str(tmp_path) in [os.path.abspath(d) for d in fw.watched_directories()]
    finally:
        fw.stop()
    assert fw.started is False


# --- Hot reload integration ----------------------------------------------

def test_hot_reload_updates_asset_after_poll(tmp_path):
    """End-to-end: enqueue path -> poll -> registry.get returns refreshed value."""
    versions = {"n": 0}

    def loader(path: str):
        versions["n"] += 1
        return f"version_{versions['n']}"

    reg = AssetRegistry()
    reg.register_loader([".foo"], loader)
    p = _write_text(tmp_path, "thing.foo", "hi")
    handle = reg.load(p)
    assert reg.get(handle) == "version_1"

    fw = FileWatcher(reg)
    fw._queue.put(p)
    # Before poll: stale value. After poll: refreshed.
    assert reg.get(handle) == "version_1"
    fw.poll()
    assert reg.get(handle) == "version_2"


# --- Scene save/load ------------------------------------------------------

@component
class _SceneTestMarker:
    """Dataclass component used only by scene tests so we have a non-Phase-3 type to round-trip."""
    flag: bool = False
    count: int = 0


def test_scene_save_writes_versioned_json(tmp_path):
    world = World()
    world.spawn(Transform2D(x=1.0, y=2.0))
    world.flush()

    out = str(tmp_path / "save.json")
    Scene.save(world, out)

    raw = json.loads((tmp_path / "save.json").read_text())
    assert raw["version"] == Scene.VERSION
    assert isinstance(raw["entities"], list)
    assert len(raw["entities"]) == 1
    entry = raw["entities"][0]
    assert "Transform2D" in entry["components"]
    assert entry["components"]["Transform2D"]["x"] == 1.0


def test_scene_save_is_atomic_no_tmp_left_over(tmp_path):
    world = World()
    world.spawn(Transform2D())
    world.flush()
    out = str(tmp_path / "save.json")
    Scene.save(world, out)
    # The .tmp scratch file must not survive a successful save.
    assert not os.path.exists(out + ".tmp")
    assert os.path.exists(out)


def test_scene_save_creates_missing_parent_directory(tmp_path):
    world = World()
    world.spawn(Transform2D())
    world.flush()
    out = str(tmp_path / "saves" / "deep" / "level.json")
    Scene.save(world, out)
    assert os.path.exists(out)


def test_scene_round_trip_preserves_component_values(tmp_path):
    world = World()
    e = world.spawn(
        Transform2D(x=10.0, y=20.0, rotation=0.5, scale_x=2.0, scale_y=3.0),
        Sprite(texture_id=4, r=0.1, g=0.2, b=0.3, a=0.5, width=32.0, height=64.0, flip_x=True),
    )
    world.flush()

    out = str(tmp_path / "round.json")
    Scene.save(world, out)

    fresh = World()
    new_ids = Scene.load(fresh, out)
    assert len(new_ids) == 1

    new_eid = new_ids[0]
    t = fresh.get_component(new_eid, Transform2D)
    s = fresh.get_component(new_eid, Sprite)
    assert (t.x, t.y, t.rotation, t.scale_x, t.scale_y) == (10.0, 20.0, 0.5, 2.0, 3.0)
    assert s.texture_id == 4
    assert s.flip_x is True
    assert s.width == 32.0
    _ = e  # original entity ID isn't preserved across worlds — that's intentional


def test_scene_load_additive_doubles_entities(tmp_path):
    src = World()
    src.spawn(Transform2D(x=1.0))
    src.flush()
    out = str(tmp_path / "s.json")
    Scene.save(src, out)

    dst = World()
    Scene.load_additive(dst, out)
    Scene.load_additive(dst, out)
    total = sum(arch.length for arch in dst.archetypes.all_archetypes())
    assert total == 2


def test_scene_load_unknown_component_warns_and_skips(tmp_path):
    bad = {
        "version": Scene.VERSION,
        "entities": [
            {
                "id": 1,
                "components": {
                    "Transform2D": {"x": 5.0, "y": 6.0},
                    "ThisDoesNotExistAnywhere": {"foo": 1},
                },
            }
        ],
    }
    out = str(tmp_path / "mixed.json")
    (tmp_path / "mixed.json").write_text(json.dumps(bad))

    fresh = World()
    with pytest.warns(RuntimeWarning, match="ThisDoesNotExistAnywhere"):
        ids = Scene.load(fresh, out)
    # The known component still spawned; the unknown one was dropped.
    assert len(ids) == 1
    t = fresh.get_component(ids[0], Transform2D)
    assert (t.x, t.y) == (5.0, 6.0)


def test_scene_load_version_mismatch_raises(tmp_path):
    out = str(tmp_path / "wrong.json")
    (tmp_path / "wrong.json").write_text(json.dumps({"version": 999, "entities": []}))
    with pytest.raises(SceneVersionError):
        Scene.load(World(), out)


def test_scene_round_trip_for_custom_component(tmp_path):
    """Scene.load must resolve any registered @component, not just Phase 3 types."""
    world = World()
    world.spawn(_SceneTestMarker(flag=True, count=42))
    world.flush()
    out = str(tmp_path / "marker.json")
    Scene.save(world, out)

    fresh = World()
    ids = Scene.load(fresh, out)
    inst = fresh.get_component(ids[0], _SceneTestMarker)
    assert inst.flag is True
    assert inst.count == 42


# --- setup_assets ---------------------------------------------------------

def test_setup_assets_registers_default_loaders():
    app = _fake_app()
    reg = setup_assets(app)
    assert reg.loader_for("a.json") is not None
    # No texture loader unless renderer is set up first.
    assert reg.loader_for("a.png") is None


def test_setup_assets_inserts_registry_as_resource():
    app = _fake_app()
    reg = setup_assets(app)
    assert app.world.get_resource(AssetRegistry) is reg


def test_setup_assets_is_idempotent():
    app = _fake_app()
    reg1 = setup_assets(app)
    reg2 = setup_assets(app)
    assert reg1 is reg2
    # Only one PRE_UPDATE poll system, max — none in this case (no watch_dirs).
    assert len(app._scheduler.systems(Phase.PRE_UPDATE)) == 0


def test_setup_assets_with_watch_dirs_registers_pre_update_system(tmp_path):
    app = _fake_app()
    reg = setup_assets(app, watch_dirs=[str(tmp_path)])
    try:
        # Exactly one PRE_UPDATE system added.
        pre = app._scheduler.systems(Phase.PRE_UPDATE)
        assert len(pre) == 1
        # FileWatcher available as a resource.
        fw = app.world.get_resource(FileWatcher)
        assert fw is not None
        assert str(tmp_path) in [os.path.abspath(d) for d in fw.watched_directories()]
    finally:
        fw = app.world.get_resource(FileWatcher)
        if fw is not None:
            fw.stop()
    assert reg is app.world.get_resource(AssetRegistry)


def test_setup_assets_idempotent_with_watch_dirs(tmp_path):
    app = _fake_app()
    setup_assets(app, watch_dirs=[str(tmp_path)])
    try:
        setup_assets(app, watch_dirs=[str(tmp_path)])
        # Should still be exactly one poll system, not two.
        assert len(app._scheduler.systems(Phase.PRE_UPDATE)) == 1
    finally:
        fw = app.world.get_resource(FileWatcher)
        if fw is not None:
            fw.stop()


def test_setup_assets_module_function_matches_app_method():
    """The keel.setup_assets re-export and the App method are the same code path."""
    assert setup_assets is setup_assets_fn


# --- keel re-exports ------------------------------------------------------

def test_top_level_re_exports():
    assert keel.AssetRegistry is AssetRegistry
    assert keel.AssetHandle is AssetHandle
    assert keel.Scene is Scene
    assert keel.SceneVersionError is SceneVersionError
    assert keel.AssetNotFoundError is AssetNotFoundError
    assert keel.NoLoaderError is NoLoaderError
    assert keel.InvalidHandleError is InvalidHandleError
    assert keel.FileWatcher is FileWatcher


# --- v0.1.1: hot reload failures now log a warning -------------------------

def test_filewatcher_reload_failure_logs_a_warning(tmp_path, caplog):
    import logging

    class _ExplodingRegistry(_CountingRegistry):
        def reload(self, handle):
            raise RuntimeError("decode bombed")

    reg = _ExplodingRegistry()
    p = str(tmp_path / "boom.png")
    reg.register(p)

    fw = FileWatcher(reg)
    fw._queue.put(p)

    with caplog.at_level(logging.WARNING, logger="keel.assets.hot_reload"):
        fw.poll()

    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("hot reload failed" in m and "decode bombed" in m for m in msgs), \
        f"expected a hot-reload warning naming the exception; got {msgs!r}"
