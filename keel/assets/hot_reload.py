"""FileWatcher: watchdog-driven hot reload with main-thread polling.

Watchdog runs its filesystem observer on a background thread. We do NOT call
the registry from that thread — GL operations (texture re-upload) must run
on the main thread that owns the GL context. Instead, the watchdog event
handler does nothing except `queue.put(path)`. The main thread drains the
queue inside `poll()`, which is registered as a PRE_UPDATE system by
setup_assets so it runs once per frame.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .registry import AssetRegistry, _normalize

_log = logging.getLogger(__name__)


class _ReloadHandler(FileSystemEventHandler):
    """Watchdog handler — only enqueues paths, never touches the registry directly."""

    __slots__ = ("_queue",)

    def __init__(self, q: "queue.Queue[str]") -> None:
        self._queue = q

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._queue.put(event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._queue.put(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        dest = getattr(event, "dest_path", None)
        if dest:
            self._queue.put(dest)


class FileWatcher:
    """Polls a thread-safe queue of file change events and reloads matching assets."""

    def __init__(self, registry: AssetRegistry) -> None:
        self.registry: AssetRegistry = registry
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._handler: _ReloadHandler = _ReloadHandler(self._queue)
        self._observer: Observer = Observer()
        self._watched: dict[str, Any] = {}
        self._started: bool = False
        self._stopped: bool = False
        self._lock: threading.Lock = threading.Lock()
        self.last_reload_count: int = 0

    def watch(self, directory: str) -> None:
        """Start watching `directory` recursively (idempotent per directory)."""
        if self._stopped:
            raise RuntimeError("FileWatcher has been stopped — create a new one")
        directory = os.path.abspath(directory)
        with self._lock:
            if directory in self._watched:
                return
            watch = self._observer.schedule(self._handler, directory, recursive=True)
            self._watched[directory] = watch
            if not self._started:
                self._observer.start()
                self._started = True

    def unwatch(self, directory: str) -> None:
        """Stop watching `directory`. No-op if it wasn't being watched."""
        directory = os.path.abspath(directory)
        with self._lock:
            watch = self._watched.pop(directory, None)
            if watch is not None:
                try:
                    self._observer.unschedule(watch)
                except Exception:
                    pass

    def poll(self) -> int:
        """Drain the queue on the calling thread and reload any matching assets. Returns the reload count."""
        seen: set[str] = set()
        reloads = 0
        while True:
            try:
                raw = self._queue.get_nowait()
            except queue.Empty:
                break
            normalized = _normalize(raw)
            if normalized in seen:
                continue
            seen.add(normalized)
            handle = self.registry._handles.get(normalized)
            if handle is None:
                continue
            try:
                self.registry.reload(handle)
                reloads += 1
            except Exception as exc:
                # Don't let a failed reload kill the loop, but tell the user
                # which path failed — silent reload failure is a notorious
                # "why isn't my edited PNG updating?" trap.
                _log.warning(
                    "hot reload failed for %s: %s: %s",
                    normalized, type(exc).__name__, exc,
                )
        self.last_reload_count = reloads
        return reloads

    def stop(self) -> None:
        """Stop the observer and join its thread. Idempotent."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            if self._started:
                try:
                    self._observer.stop()
                    self._observer.join(timeout=2.0)
                except Exception:
                    pass

    @property
    def started(self) -> bool:
        """True if the watchdog observer has been started."""
        return self._started and not self._stopped

    def watched_directories(self) -> list[str]:
        """Snapshot of the currently watched directories."""
        return list(self._watched.keys())
