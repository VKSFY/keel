"""Public audio surface: setup_audio + module-level play/stop helpers.

Helpers look up AudioEngine from the world resource setup_audio inserted,
and raise RuntimeError if setup_audio was never called.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..core import Phase
from .audio_engine import AudioEngine, SoundHandle
from .components import AudioSource


__all__ = [
    "AudioEngine",
    "AudioSetup",
    "AudioSource",
    "SoundHandle",
    "play_music",
    "play_sound",
    "set_volume",
    "setup_audio",
    "stop_music",
    "stop_sound",
]


@dataclass
class AudioSetup:
    """Returned by setup_audio — carries the wired AudioEngine."""
    engine: AudioEngine


def setup_audio(app: Any) -> AudioSetup:
    """Create AudioEngine, register PRE_UPDATE update + shutdown hook. Idempotent."""
    existing = getattr(app, "_pyge_audio", None)
    if existing is not None:
        return existing

    engine = AudioEngine()
    app.world.insert_resource(engine, type_=AudioEngine)

    @app.system(Phase.PRE_UPDATE)
    def audio_update(world: Any, dt: float, eng: AudioEngine) -> None:
        eng.update(dt)

    add_hook = getattr(app, "add_shutdown_hook", None)
    if add_hook is not None:
        add_hook(engine.shutdown)

    setup = AudioSetup(engine=engine)
    app._pyge_audio = setup
    return setup


def _engine(app: Any) -> AudioEngine:
    eng = app.world.get_resource(AudioEngine)
    if eng is None:
        raise RuntimeError(
            "audio: call setup_audio(app) before using play_sound/play_music."
        )
    return eng


def play_sound(
    app: Any,
    path: str,
    volume: float = 1.0,
    loop: bool = False,
) -> SoundHandle:
    """Play a sound effect. Returns a SoundHandle for later stop()."""
    return _engine(app).play(path, volume=volume, loop=loop)


def stop_sound(app: Any, handle: SoundHandle) -> None:
    """Stop the sound referenced by `handle`."""
    _engine(app).stop(handle)


def play_music(
    app: Any,
    path: str,
    loop: bool = True,
    fade_in: float = 0.0,
) -> None:
    """Stream `path` as music, replacing any active track."""
    _engine(app).play_music(path, loop=loop, fade_in=fade_in)


def stop_music(app: Any, fade_out: float = 0.0) -> None:
    """Stop the active music stream, optionally ramping down."""
    _engine(app).stop_music(fade_out=fade_out)


def set_volume(
    app: Any,
    master: Optional[float] = None,
    sfx: Optional[float] = None,
    music: Optional[float] = None,
) -> None:
    """Update any combination of (master, sfx, music). None means leave unchanged."""
    eng = _engine(app)
    if master is not None:
        eng.set_master_volume(master)
    if sfx is not None:
        eng.set_sfx_volume(sfx)
    if music is not None:
        eng.set_music_volume(music)
