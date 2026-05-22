"""Per-system frame-time profiler.

Each system call recorded as a single entry in a fixed-size deque (one per
system name). `get_stats` summarises the rolling window into mean / min /
max / last in milliseconds. Hooked into the scheduler via
`Scheduler.attach_profiler`.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

from ..core import Phase
from ..core.scheduler import Scheduler


@dataclass
class SystemStats:
    """Per-system rolling-window timing summary, all values in milliseconds."""
    name: str
    avg_ms: float
    min_ms: float
    max_ms: float
    last_ms: float


class FrameProfiler:
    """Records per-system elapsed time over the last `history_size` calls."""

    __slots__ = ("history_size", "_buffers", "_starts")

    def __init__(self, history_size: int = 60) -> None:
        self.history_size: int = int(history_size)
        self._buffers: dict[str, Deque[float]] = {}
        self._starts: dict[str, float] = {}

    def begin(self, system_name: str) -> None:
        """Mark the start of a system run. Stores `time.perf_counter()` for that name."""
        self._starts[system_name] = time.perf_counter()

    def end(self, system_name: str) -> None:
        """Mark the end of a system run; appends elapsed seconds to its rolling buffer."""
        start = self._starts.pop(system_name, None)
        if start is None:
            return
        elapsed = time.perf_counter() - start
        buf = self._buffers.get(system_name)
        if buf is None:
            buf = deque(maxlen=self.history_size)
            self._buffers[system_name] = buf
        buf.append(elapsed)

    def get_stats(self) -> dict[str, SystemStats]:
        """Snapshot of every active system as `SystemStats` (ms-scaled)."""
        out: dict[str, SystemStats] = {}
        for name, buf in self._buffers.items():
            if not buf:
                continue
            samples = list(buf)
            avg_s = sum(samples) / len(samples)
            out[name] = SystemStats(
                name=name,
                avg_ms=avg_s * 1000.0,
                min_ms=min(samples) * 1000.0,
                max_ms=max(samples) * 1000.0,
                last_ms=samples[-1] * 1000.0,
            )
        return out

    def reset(self) -> None:
        """Forget every recorded sample and any in-flight begin/end pair."""
        self._buffers.clear()
        self._starts.clear()


def setup_profiler(app: Any) -> FrameProfiler:
    """Attach a FrameProfiler to `app`'s scheduler. Idempotent — caches on the app."""
    existing = getattr(app, "_keel_profiler", None)
    if existing is not None:
        return existing

    profiler = FrameProfiler()
    scheduler: Scheduler = app._scheduler
    scheduler.attach_profiler(profiler)
    app.world.insert_resource(profiler, type_=FrameProfiler)
    app._keel_profiler = profiler
    return profiler
