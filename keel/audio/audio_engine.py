"""Audio engine: one stereo miniaudio PlaybackDevice + a small mixer.

SFX decode once into _sounds and are referenced by SoundHandle id. play()
fills a _PlayState under _lock; the mixer generator (on miniaudio's thread)
reads it; update() runs on the main thread and drains finished handles.
"""
from __future__ import annotations

import array
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import miniaudio
except ImportError as e:  # pragma: no cover - import-time guard
    raise ImportError("install miniaudio: pip install miniaudio") from e


@dataclass(frozen=True)
class SoundHandle:
    """Opaque reference to a playing sound. Hashable + equality by id only."""
    id: int
    path: str = ""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SoundHandle) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


# Stereo int16 @ 44.1 kHz — decoding all SFX into this format avoids per-mix
# resampling so the mixer hot path stays on int16.
_OUTPUT_CHANNELS = 2
_OUTPUT_SAMPLE_RATE = 44100


@dataclass
class _LoadedSound:
    samples: "array.array"
    nchannels: int
    sample_rate: int


@dataclass
class _PlayState:
    handle_id: int
    path: str
    sound: _LoadedSound
    position: int = 0
    volume: float = 1.0
    loop: bool = False
    finished: bool = False


@dataclass
class _MusicState:
    path: str
    stream: Any
    loop: bool
    volume: float = 1.0
    # Linear fade ramp advanced by AudioEngine.update().
    current_gain: float = 1.0
    target_gain: float = 1.0
    fade_remaining: float = 0.0
    fade_total: float = 0.0
    stop_when_faded: bool = False
    finished: bool = False


def _clamp01(v: float) -> float:
    """Clamp `v` to [0, 1]."""
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return float(v)


class AudioEngine:
    """Playback device, SFX cache, active-sound table, and at-most-one music stream."""

    def __init__(self) -> None:
        try:
            self._device: Optional[Any] = miniaudio.PlaybackDevice(
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=_OUTPUT_CHANNELS,
                sample_rate=_OUTPUT_SAMPLE_RATE,
            )
        except Exception:
            # No backend available (headless CI). State queries still work;
            # the mixer just never runs.
            self._device = None

        self._sounds: Dict[str, _LoadedSound] = {}
        self._playing: Dict[int, _PlayState] = {}
        self._music_stream: Optional[_MusicState] = None

        self._master_volume = 1.0
        self._sfx_volume = 1.0
        self._music_volume = 1.0

        self._next_id = 1
        self._lock = threading.RLock()
        self._device_started = False
        self._shutdown_called = False
        self._last_update_dt = 0.0

    @property
    def master_volume(self) -> float:
        return self._master_volume

    @property
    def sfx_volume(self) -> float:
        return self._sfx_volume

    @property
    def music_volume(self) -> float:
        """Gain applied to the active music stream only — does not affect SFX."""
        return self._music_volume

    def set_master_volume(self, volume: float) -> None:
        """Clamp `volume` to [0, 1] and set the master gain."""
        self._master_volume = _clamp01(volume)

    def set_sfx_volume(self, volume: float) -> None:
        """Clamp `volume` to [0, 1] and set the SFX gain."""
        self._sfx_volume = _clamp01(volume)

    def set_music_volume(self, volume: float) -> None:
        """Clamp to [0, 1] and update the music gain (re-aiming any active fade)."""
        self._music_volume = _clamp01(volume)
        with self._lock:
            ms = self._music_stream
            if ms is not None and ms.fade_remaining <= 0.0:
                ms.target_gain = self._music_volume
                ms.current_gain = self._music_volume

    def load(self, path: str) -> None:
        """Decode `path` into the SFX cache. Idempotent."""
        if path in self._sounds:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        decoded = miniaudio.decode_file(
            path,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=_OUTPUT_CHANNELS,
            sample_rate=_OUTPUT_SAMPLE_RATE,
        )
        self._sounds[path] = _LoadedSound(
            samples=decoded.samples,
            nchannels=getattr(decoded, "nchannels", _OUTPUT_CHANNELS),
            sample_rate=getattr(decoded, "sample_rate", _OUTPUT_SAMPLE_RATE),
        )

    def play(
        self,
        path: str,
        volume: float = 1.0,
        loop: bool = False,
    ) -> SoundHandle:
        """Play a sound effect; auto-loads on first use."""
        if path not in self._sounds:
            # Raise on the main thread, never silently inside the audio callback.
            self.load(path)
        sound = self._sounds[path]
        with self._lock:
            handle_id = self._next_id
            self._next_id += 1
            self._playing[handle_id] = _PlayState(
                handle_id=handle_id,
                path=path,
                sound=sound,
                position=0,
                volume=_clamp01(volume),
                loop=bool(loop),
                finished=False,
            )
        self._ensure_device_started()
        return SoundHandle(id=handle_id, path=path)

    def stop(self, handle: SoundHandle) -> None:
        """Stop a specific handle. No-op if unknown."""
        with self._lock:
            state = self._playing.pop(handle.id, None)
            if state is not None:
                state.finished = True

    def stop_all(self) -> None:
        """Stop every active SFX and the music stream."""
        with self._lock:
            for state in self._playing.values():
                state.finished = True
            self._playing.clear()
            ms = self._music_stream
            if ms is not None:
                ms.finished = True
                self._music_stream = None

    def is_playing(self, handle: SoundHandle) -> bool:
        """True iff the handle is still in `_playing` and not finished."""
        with self._lock:
            state = self._playing.get(handle.id)
            return state is not None and not state.finished


    def play_music(
        self,
        path: str,
        loop: bool = True,
        fade_in: float = 0.0,
    ) -> None:
        """Stream `path` as music, replacing any current track."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        stream = miniaudio.stream_file(
            path,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=_OUTPUT_CHANNELS,
            sample_rate=_OUTPUT_SAMPLE_RATE,
        )
        with self._lock:
            if self._music_stream is not None:
                self._music_stream.finished = True
                self._music_stream = None
            fade_seconds = max(0.0, float(fade_in))
            target = self._music_volume
            self._music_stream = _MusicState(
                path=path,
                stream=stream,
                loop=bool(loop),
                volume=self._music_volume,
                current_gain=0.0 if fade_seconds > 0.0 else target,
                target_gain=target,
                fade_remaining=fade_seconds,
                fade_total=fade_seconds,
                stop_when_faded=False,
                finished=False,
            )
        self._ensure_device_started()

    def stop_music(self, fade_out: float = 0.0) -> None:
        """Stop the active music, optionally ramping down over `fade_out` seconds."""
        with self._lock:
            ms = self._music_stream
            if ms is None:
                return
            fade_seconds = max(0.0, float(fade_out))
            if fade_seconds <= 0.0:
                ms.finished = True
                self._music_stream = None
                return
            ms.target_gain = 0.0
            ms.fade_remaining = fade_seconds
            ms.fade_total = fade_seconds
            ms.stop_when_faded = True

    def is_music_playing(self) -> bool:
        """True while music is active (including during fade-out)."""
        with self._lock:
            return self._music_stream is not None and not self._music_stream.finished

    def update(self, dt: float = 0.0) -> None:
        """PRE_UPDATE tick: drain finished SFX, advance the music fade. Non-blocking."""
        with self._lock:
            self._last_update_dt = float(dt)
            for hid in [hid for hid, s in self._playing.items() if s.finished]:
                self._playing.pop(hid, None)
            ms = self._music_stream
            if ms is not None and ms.fade_remaining > 0.0:
                ms.fade_remaining -= float(dt)
                if ms.fade_remaining <= 0.0:
                    ms.fade_remaining = 0.0
                    ms.current_gain = ms.target_gain
                    if ms.stop_when_faded:
                        ms.finished = True
                        self._music_stream = None
                else:
                    progress = 1.0 - (ms.fade_remaining / ms.fade_total)
                    start = 0.0 if ms.target_gain > 0.0 else ms.volume
                    ms.current_gain = start + (ms.target_gain - start) * progress
            elif ms is not None and ms.finished:
                self._music_stream = None

    def shutdown(self) -> None:
        """Stop everything and close the device. Idempotent."""
        if self._shutdown_called:
            return
        self._shutdown_called = True
        with self._lock:
            for state in self._playing.values():
                state.finished = True
            self._playing.clear()
            if self._music_stream is not None:
                self._music_stream.finished = True
                self._music_stream = None
        if self._device is not None:
            for method in ("stop", "close"):
                fn = getattr(self._device, method, None)
                if fn is None:
                    continue
                try:
                    fn()
                except Exception:
                    pass
        self._device = None
        self._device_started = False

    def _ensure_device_started(self) -> None:
        """Start the device on first play. No-op for mocked or missing devices."""
        if self._device is None or self._device_started or self._shutdown_called:
            return
        try:
            generator = self._mixer()
            next(generator)  # prime — required by miniaudio.
            self._device.start(generator)
            self._device_started = True
        except Exception:
            # If start() fails (no backend, mocked device that raises), we
            # silently degrade: state stays correct, sound just won't play.
            self._device_started = False

    def _mixer(self):
        """Generator that yields mixed int16 stereo frames to miniaudio.

        The first send() on this generator is the priming next(), which
        miniaudio expects to return either b"" or an empty array. After
        priming, each .send(framecount) yields the next chunk.
        """
        required: int = (yield b"")  # priming yield; receives no data
        while not self._shutdown_called:
            if required is None or required <= 0:
                required = (yield b"")
                continue
            chunk = self._mix_chunk(int(required))
            required = (yield chunk)

    def _mix_chunk(self, framecount: int) -> bytes:
        """Mix one chunk of `framecount` stereo frames. Pure-Python int math."""
        # We avoid numpy here on purpose: the audio thread is hot and
        # `array.array('i', ...)` plus a tight Python loop is fast enough
        # for typical SFX counts (< 16 simultaneous sounds) and avoids
        # crossing the numpy boundary on every callback.
        out_samples = framecount * _OUTPUT_CHANNELS
        mix = array.array("i", [0] * out_samples)

        with self._lock:
            master = self._master_volume
            sfx_gain = self._sfx_volume * master
            music_gain = self._music_volume * master

            # SFX -----------------------------------------------------
            for state in self._playing.values():
                if state.finished:
                    continue
                gain = state.volume * sfx_gain
                if gain <= 0.0:
                    continue
                pcm = state.sound.samples
                pos = state.position
                remaining = len(pcm) - pos
                if remaining <= 0:
                    if state.loop:
                        state.position = 0
                        pos = 0
                        remaining = len(pcm)
                    else:
                        state.finished = True
                        continue
                copy = min(out_samples, remaining)
                # Multiply once into the mix buffer; explicit loop keeps
                # us within int range with int32 accumulator.
                for i in range(copy):
                    mix[i] += int(pcm[pos + i] * gain)
                pos += copy
                if pos >= len(pcm):
                    if state.loop:
                        state.position = 0
                    else:
                        state.position = pos
                        state.finished = True
                else:
                    state.position = pos

            # Music ---------------------------------------------------
            ms = self._music_stream
            if ms is not None and not ms.finished:
                gain = ms.current_gain * music_gain
                if gain > 0.0:
                    try:
                        chunk = ms.stream.send(framecount)
                    except (StopIteration, AttributeError, TypeError):
                        chunk = None
                    if chunk is None:
                        if ms.loop:
                            # Re-open the stream from the top.
                            try:
                                ms.stream = miniaudio.stream_file(
                                    ms.path,
                                    output_format=miniaudio.SampleFormat.SIGNED16,
                                    nchannels=_OUTPUT_CHANNELS,
                                    sample_rate=_OUTPUT_SAMPLE_RATE,
                                )
                            except Exception:
                                ms.finished = True
                        else:
                            ms.finished = True
                    else:
                        for i in range(min(len(chunk), out_samples)):
                            mix[i] += int(chunk[i] * gain)

        # Clip int32 accumulator back to int16 and emit raw bytes.
        out16 = array.array("h", [0] * out_samples)
        for i in range(out_samples):
            v = mix[i]
            if v > 32767:
                v = 32767
            elif v < -32768:
                v = -32768
            out16[i] = v
        return out16.tobytes()
