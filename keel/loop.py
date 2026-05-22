"""Fixed-timestep game loop.

Simulation phases (PRE_UPDATE / UPDATE / POST_UPDATE) advance at a fixed
60 Hz tick — multiple sim ticks may run per visual frame to catch up.
Render phases (PRE_RENDER / RENDER / POST_RENDER) run exactly once per
visual frame at the real elapsed delta. The accumulator is clamped to
prevent the spiral of death when the host is overloaded.
"""
from __future__ import annotations

import time
from typing import Any

from .core.scheduler import Phase, Scheduler


FIXED_DT: float = 1.0 / 60.0
# Cap how far the simulation can fall behind in a single frame. Below this we
# always run every queued tick so 0.1s elapsed -> 6 ticks; above it we drop
# excess time to prevent the spiral of death on a hitch (e.g. a 60s pause).
MAX_ACCUMULATED: float = 10.0 * FIXED_DT


class RenderState:
    """Per-frame interpolation info exposed to render systems as a resource."""

    __slots__ = ("alpha", "frame_dt")

    def __init__(self) -> None:
        self.alpha: float = 0.0
        self.frame_dt: float = 0.0


class FixedStepDriver:
    """Drives fixed-timestep simulation ticks given variable real-time elapsed deltas."""

    __slots__ = ("fixed_dt", "max_accumulated", "accumulator")

    def __init__(
        self,
        fixed_dt: float = FIXED_DT,
        max_accumulated: float = MAX_ACCUMULATED,
    ) -> None:
        self.fixed_dt: float = fixed_dt
        self.max_accumulated: float = max_accumulated
        self.accumulator: float = 0.0

    def step(self, world: Any, scheduler: Scheduler, elapsed: float) -> int:
        """Add `elapsed` to the accumulator, run sim ticks, return how many fired.

        Note: this method does NOT clear `world.events`. The event lifetime is
        the visual frame, so the run loop clears once per outer iteration. If
        you call `step` directly outside a run loop, clear events yourself.
        """
        if elapsed > self.max_accumulated:
            elapsed = self.max_accumulated
        self.accumulator += elapsed
        ticks = 0
        while self.accumulator >= self.fixed_dt:
            scheduler.tick_simulation(world, self.fixed_dt)
            world.flush()
            self.accumulator -= self.fixed_dt
            ticks += 1
        return ticks

    @property
    def alpha(self) -> float:
        """Fractional progress (0..1) into the next pending sim tick — for render interpolation."""
        if self.fixed_dt <= 0.0:
            return 0.0
        return self.accumulator / self.fixed_dt


def run_loop(window: Any, world: Any, scheduler: Scheduler) -> None:
    """Run the fixed-timestep main loop until window.should_close becomes True."""
    rs = world.get_resource(RenderState)
    if rs is None:
        rs = RenderState()
        world.insert_resource(rs)

    # Local import to avoid a circular dependency at module load time.
    from .input import InputState as _InputState
    input_state = world.get_resource(_InputState)

    driver = FixedStepDriver()
    last = time.perf_counter()

    while not window.should_close:
        # Events live for one visual frame: clear what arrived during the
        # previous frame's GLFW callbacks and any sim-emitted events that
        # weren't consumed, then poll for new input which becomes visible
        # to every system this frame (across all sim ticks).
        world.events.clear()
        # Snapshot input BEFORE polling so this frame's rising-/falling-edge
        # helpers compare against last frame's hold-state.
        if input_state is not None:
            input_state.begin_frame()
        window.swap_and_poll()

        now = time.perf_counter()
        elapsed = now - last
        last = now

        # Optional per-frame profiler markers — only fire if a FrameProfiler
        # has been attached to the scheduler (via setup_profiler).
        profiler = getattr(scheduler, "_profiler", None)
        if profiler is not None:
            profiler.begin("__frame__")

        driver.step(world, scheduler, elapsed)

        rs.alpha = driver.alpha
        rs.frame_dt = elapsed
        scheduler.tick_render(world, elapsed)

        if profiler is not None:
            profiler.end("__frame__")

        if not getattr(window, "vsync", True):
            frame_time = time.perf_counter() - now
            slack = FIXED_DT - frame_time
            if slack > 0.0:
                time.sleep(slack)
