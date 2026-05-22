"""Audio engine + convenience helpers. miniaudio is fully patched out."""
from __future__ import annotations

import array
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import pyge
from pyge import Phase, Scheduler, World
from pyge.audio import (
    AudioEngine,
    AudioSetup,
    AudioSource,
    SoundHandle,
    play_music,
    play_sound,
    set_volume,
    setup_audio,
    stop_music,
    stop_sound,
)


# --- Helpers --------------------------------------------------------------

def _fake_app():
    """Minimal app shape compatible with setup_audio (no GL, no window)."""
    sched = Scheduler()
    world = World()
    shutdown_hooks: list = []

    app = SimpleNamespace(
        world=world,
        _scheduler=sched,
        scheduler=sched,
        shutdown_hooks=shutdown_hooks,
    )

    def system(phase: Phase):
        def deco(fn):
            sched.register(phase, fn)
            return fn
        return deco

    def add_shutdown_hook(hook):
        shutdown_hooks.append(hook)

    app.system = system
    app.add_shutdown_hook = add_shutdown_hook
    return app


def _fake_decoded_sound(num_samples: int = 2048):
    """Mimic miniaudio.DecodedSoundFile: .samples (array.array of int16) + metadata."""
    samples = array.array("h", [0] * num_samples)
    return SimpleNamespace(
        samples=samples,
        nchannels=2,
        sample_rate=44100,
    )


def _fake_music_stream(frame_count: int = 4096):
    """A generator-like object the engine drives via .send(framecount)."""
    stream = MagicMock()
    stream.send = MagicMock(return_value=array.array("h", [0] * frame_count))
    return stream


@pytest.fixture(autouse=True)
def _patch_playback_device():
    """Every test in this module runs with a mocked PlaybackDevice."""
    with patch("pyge.audio.audio_engine.miniaudio.PlaybackDevice") as mock_dev_cls:
        mock_dev_cls.return_value = MagicMock(spec=["start", "stop", "close"])
        yield mock_dev_cls


@pytest.fixture
def engine():
    """A fresh AudioEngine — PlaybackDevice already mocked by the autouse fixture."""
    return AudioEngine()


# --- AudioEngine init -----------------------------------------------------

def test_audio_engine_initializes(engine):
    assert engine is not None


def test_default_volumes_are_one(engine):
    assert engine.master_volume == 1.0
    assert engine.sfx_volume == 1.0
    assert engine.music_volume == 1.0


# --- Sound loading --------------------------------------------------------

def test_load_caches_decoded_file(tmp_path, engine):
    sound_path = tmp_path / "boom.wav"
    sound_path.write_bytes(b"\x00")  # contents irrelevant — decode is mocked
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        engine.load(str(sound_path))
        engine.load(str(sound_path))
        assert decode.call_count == 1


def test_load_missing_file_raises(engine, tmp_path):
    with pytest.raises(FileNotFoundError):
        engine.load(str(tmp_path / "does_not_exist.wav"))


@pytest.mark.parametrize("ext", [".wav", ".mp3", ".ogg", ".flac"])
def test_load_accepts_supported_extensions(engine, tmp_path, ext):
    p = tmp_path / f"clip{ext}"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        engine.load(str(p))
        decode.assert_called_once()


# --- Sound playback -------------------------------------------------------

def test_play_returns_handle_with_positive_id(tmp_path, engine):
    p = tmp_path / "a.wav"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        handle = engine.play(str(p))
    assert isinstance(handle, SoundHandle)
    assert handle.id > 0


def test_play_two_sounds_assigns_distinct_ids(tmp_path, engine):
    p = tmp_path / "a.wav"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        h1 = engine.play(str(p))
        h2 = engine.play(str(p))
    assert h1.id != h2.id


def test_play_auto_loads_unprelaoded(tmp_path, engine):
    p = tmp_path / "a.wav"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        # No prior load() — play() must do it.
        engine.play(str(p))
        assert decode.call_count == 1


def test_stop_known_handle_is_a_noop_then_idempotent(tmp_path, engine):
    p = tmp_path / "a.wav"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        handle = engine.play(str(p))
    engine.stop(handle)


def test_stop_already_stopped_handle_is_idempotent(tmp_path, engine):
    p = tmp_path / "a.wav"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        handle = engine.play(str(p))
    engine.stop(handle)
    engine.stop(handle)  # should not raise


def test_stop_all_clears_playing_dict(tmp_path, engine):
    p = tmp_path / "a.wav"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        engine.play(str(p))
        engine.play(str(p))
    assert len(engine._playing) == 2
    engine.stop_all()
    assert len(engine._playing) == 0


def test_is_playing_true_after_play_false_after_stop(tmp_path, engine):
    p = tmp_path / "a.wav"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        h = engine.play(str(p))
    assert engine.is_playing(h) is True
    engine.stop(h)
    assert engine.is_playing(h) is False


# --- Music ----------------------------------------------------------------

def test_play_music_sets_music_stream(tmp_path, engine):
    p = tmp_path / "song.ogg"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.stream_file") as stream_file:
        stream_file.return_value = _fake_music_stream()
        engine.play_music(str(p), loop=True)
    assert engine._music_stream is not None
    assert engine._music_stream.path == str(p)


def test_play_music_twice_replaces_stream(tmp_path, engine):
    p = tmp_path / "song.ogg"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.stream_file") as stream_file:
        s1 = _fake_music_stream()
        s2 = _fake_music_stream()
        stream_file.side_effect = [s1, s2]
        engine.play_music(str(p))
        first_state = engine._music_stream
        engine.play_music(str(p))
    assert first_state.finished is True
    assert engine._music_stream.stream is s2


def test_stop_music_immediately_clears(tmp_path, engine):
    p = tmp_path / "song.ogg"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.stream_file") as stream_file:
        stream_file.return_value = _fake_music_stream()
        engine.play_music(str(p))
    engine.stop_music()
    assert engine._music_stream is None


def test_is_music_playing_reflects_state(tmp_path, engine):
    p = tmp_path / "song.ogg"
    p.write_bytes(b"\x00")
    assert engine.is_music_playing() is False
    with patch("pyge.audio.audio_engine.miniaudio.stream_file") as stream_file:
        stream_file.return_value = _fake_music_stream()
        engine.play_music(str(p))
    assert engine.is_music_playing() is True
    engine.stop_music()
    assert engine.is_music_playing() is False


def test_fade_in_starts_at_zero_and_advances(tmp_path, engine):
    p = tmp_path / "song.ogg"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.stream_file") as stream_file:
        stream_file.return_value = _fake_music_stream()
        engine.play_music(str(p), fade_in=2.0)
    ms = engine._music_stream
    assert ms.current_gain == 0.0
    engine.update(dt=1.0)
    # Halfway through the fade — gain should be ~half of the target.
    assert 0.0 < ms.current_gain <= ms.target_gain
    engine.update(dt=1.5)
    assert ms.current_gain == ms.target_gain


def test_stop_music_with_fade_out_ramps_then_stops(tmp_path, engine):
    p = tmp_path / "song.ogg"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.stream_file") as stream_file:
        stream_file.return_value = _fake_music_stream()
        engine.play_music(str(p))
    engine.stop_music(fade_out=0.5)
    # Still playing while the ramp is in flight.
    assert engine.is_music_playing() is True
    engine.update(dt=0.5)
    # Now the ramp is done; stream torn down.
    assert engine.is_music_playing() is False


# --- Volume --------------------------------------------------------------

def test_set_master_volume(engine):
    engine.set_master_volume(0.5)
    assert engine.master_volume == 0.5


def test_set_sfx_volume(engine):
    engine.set_sfx_volume(0.0)
    assert engine.sfx_volume == 0.0


def test_set_music_volume(engine):
    engine.set_music_volume(0.8)
    assert engine.music_volume == 0.8


def test_volume_clamped_above_one(engine):
    engine.set_master_volume(1.5)
    engine.set_sfx_volume(2.0)
    engine.set_music_volume(99.0)
    assert engine.master_volume == 1.0
    assert engine.sfx_volume == 1.0
    assert engine.music_volume == 1.0


def test_volume_clamped_below_zero(engine):
    engine.set_master_volume(-0.1)
    engine.set_sfx_volume(-99.0)
    engine.set_music_volume(-1.0)
    assert engine.master_volume == 0.0
    assert engine.sfx_volume == 0.0
    assert engine.music_volume == 0.0


# --- update() -----------------------------------------------------------

def test_update_removes_finished_handles(tmp_path, engine):
    p = tmp_path / "a.wav"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        h = engine.play(str(p))
    # Pretend the mixer finished this handle.
    engine._playing[h.id].finished = True
    engine.update(dt=0.016)
    assert h.id not in engine._playing


def test_update_is_fast(engine):
    # Cold update with no handles must be sub-millisecond.
    t0 = time.perf_counter()
    engine.update(dt=0.016)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.005  # Generous on slow CI; the spec says <1ms.


# --- Shutdown -----------------------------------------------------------

def test_shutdown_calls_device_close(engine):
    device = engine._device
    engine.shutdown()
    assert device.close.called or device.stop.called


def test_shutdown_idempotent(engine):
    engine.shutdown()
    engine.shutdown()  # second call must not raise


# --- Setup --------------------------------------------------------------

def test_setup_audio_returns_setup():
    app = _fake_app()
    setup = setup_audio(app)
    assert isinstance(setup, AudioSetup)
    assert isinstance(setup.engine, AudioEngine)


def test_setup_audio_inserts_world_resource():
    app = _fake_app()
    setup_audio(app)
    assert app.world.has_resource(AudioEngine)


def test_setup_audio_registers_pre_update_system():
    app = _fake_app()
    setup_audio(app)
    systems = app.scheduler._systems[Phase.PRE_UPDATE]
    assert len(systems) >= 1


def test_setup_audio_registers_shutdown_hook():
    app = _fake_app()
    setup = setup_audio(app)
    assert setup.engine.shutdown in app.shutdown_hooks


def test_setup_audio_idempotent():
    app = _fake_app()
    s1 = setup_audio(app)
    s2 = setup_audio(app)
    assert s1 is s2


# --- Convenience helpers ------------------------------------------------

def test_play_sound_delegates_to_engine(tmp_path):
    app = _fake_app()
    setup_audio(app)
    p = tmp_path / "a.wav"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        handle = play_sound(app, str(p), volume=0.5)
    assert isinstance(handle, SoundHandle)


def test_stop_sound_delegates_to_engine(tmp_path):
    app = _fake_app()
    setup = setup_audio(app)
    p = tmp_path / "a.wav"
    p.write_bytes(b"\x00")
    with patch("pyge.audio.audio_engine.miniaudio.decode_file") as decode:
        decode.return_value = _fake_decoded_sound()
        handle = play_sound(app, str(p))
    stop_sound(app, handle)
    assert setup.engine.is_playing(handle) is False


def test_set_volume_master_only_leaves_others():
    app = _fake_app()
    setup = setup_audio(app)
    setup.engine.set_sfx_volume(0.3)
    setup.engine.set_music_volume(0.7)
    set_volume(app, master=0.5)
    assert setup.engine.master_volume == 0.5
    assert setup.engine.sfx_volume == 0.3
    assert setup.engine.music_volume == 0.7


def test_set_volume_sfx_and_music_leaves_master():
    app = _fake_app()
    setup = setup_audio(app)
    setup.engine.set_master_volume(0.42)
    set_volume(app, sfx=0.3, music=0.7)
    assert setup.engine.master_volume == 0.42
    assert setup.engine.sfx_volume == 0.3
    assert setup.engine.music_volume == 0.7


def test_play_sound_without_setup_raises_runtime_error(tmp_path):
    app = _fake_app()
    # NOTE: deliberately not calling setup_audio.
    with pytest.raises(RuntimeError):
        play_sound(app, str(tmp_path / "missing.wav"))


def test_play_sound_missing_file_raises_immediately(tmp_path):
    app = _fake_app()
    setup_audio(app)
    with pytest.raises(FileNotFoundError):
        play_sound(app, str(tmp_path / "missing.wav"))


# --- SoundHandle --------------------------------------------------------

def test_sound_handle_hashable():
    h = SoundHandle(id=5, path="x")
    {h}  # must not raise


def test_sound_handle_equality_by_id_only():
    h1 = SoundHandle(id=7, path="a.wav")
    h2 = SoundHandle(id=7, path="b.wav")
    h3 = SoundHandle(id=8, path="a.wav")
    assert h1 == h2
    assert h1 != h3
    assert hash(h1) == hash(h2)


# --- AudioSource component ---------------------------------------------

def test_audio_source_component_has_expected_fields():
    src = AudioSource(sound_id=3, volume=0.5, loop=True, auto_play=False, playing=False)
    assert src.sound_id == 3
    assert src.volume == 0.5
    assert src.loop is True
    assert src.auto_play is False
    assert src.playing is False
