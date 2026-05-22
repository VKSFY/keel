"""Audio components.

v0.1 ships only AudioSource. Positional/spatialized 3D audio (an
AudioListener + per-source position attenuation) is out of scope for
this release; most games can call `play_sound(app, path)` directly from
their gameplay systems instead of attaching components.
"""
from __future__ import annotations

from ..core import component


@component
class AudioSource:
    """Optional component that ties a sound to an entity.

    `sound_id` is an index into a user-managed sound path registry — most
    users do not need this; calling `play_sound(app, path)` from a system
    is simpler. AudioSource exists for the case where a sound's lifetime
    is tied to an entity (a pickup that plays on collect, a positional
    emitter once spatialization lands in v0.2).
    """
    sound_id: int = 0
    volume: float = 1.0
    loop: bool = False
    auto_play: bool = False
    playing: bool = False
