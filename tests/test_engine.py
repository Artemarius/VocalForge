"""Tests for PlaybackEngine — state logic and callback behavior.

Uses synthetic numpy arrays only (no real audio devices).
"""

import numpy as np
import pytest
import sounddevice as sd

from vocalforge.audio.engine import PlaybackEngine


@pytest.fixture
def engine():
    return PlaybackEngine()


@pytest.fixture
def mono_signal():
    """1 second of mono sine wave at 44100 Hz."""
    sr = 44100
    t = np.linspace(0, 1, sr, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t), sr


@pytest.fixture
def stereo_signal():
    """1 second of stereo sine wave at 44100 Hz."""
    sr = 44100
    t = np.linspace(0, 1, sr, dtype=np.float32)
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    return np.column_stack([left, right]).astype(np.float32), sr


class TestLoadSetsState:
    def test_total_frames(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        assert engine.total_frames == len(data)

    def test_sample_rate(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        assert engine.sample_rate == sr

    def test_position_reset(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        engine._position = 1000
        engine.load(data, sr)
        assert engine.position == 0


class TestVolumeClamping:
    def test_clamp_above_one(self, engine):
        engine.volume = 1.5
        assert engine.volume == 1.0

    def test_clamp_below_zero(self, engine):
        engine.volume = -0.3
        assert engine.volume == 0.0

    def test_valid_value(self, engine):
        engine.volume = 0.5
        assert engine.volume == 0.5


class TestStopResetsPosition:
    def test_stop_resets(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        engine._position = 5000
        engine._playing = True
        engine.stop()
        assert engine.position == 0
        assert not engine.is_playing


class TestPlayWithoutData:
    def test_no_crash(self, engine):
        engine.play()
        assert not engine.is_playing


class TestCallbackMonoToStereo:
    def test_mono_duplicated_to_stereo(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        engine._playing = True

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._callback(outdata, frames, None, None)

        # Both channels should be identical
        np.testing.assert_array_equal(outdata[:, 0], outdata[:, 1])
        # Should match source data
        np.testing.assert_array_almost_equal(outdata[:, 0], data[:frames])


class TestCallbackEndOfAudio:
    def test_short_buffer_raises_callback_stop(self, engine):
        sr = 44100
        data = np.ones(100, dtype=np.float32) * 0.5
        engine.load(data, sr)
        engine._playing = True

        frames = 256
        outdata = np.zeros((frames, 2), dtype=np.float32)

        with pytest.raises(sd.CallbackStop):
            engine._callback(outdata, frames, None, None)

        # First 100 frames should have data, rest should be zero
        assert np.all(outdata[:100, 0] != 0)
        assert np.all(outdata[100:, :] == 0)


class TestCallbackAppliesVolume:
    def test_half_volume(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        engine.volume = 0.5
        engine._playing = True

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._callback(outdata, frames, None, None)

        expected = data[:frames] * 0.5
        np.testing.assert_array_almost_equal(outdata[:, 0], expected)

    def test_zero_volume(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        engine.volume = 0.0
        engine._playing = True

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._callback(outdata, frames, None, None)

        assert np.all(outdata == 0)


class TestCallbackStereo:
    def test_stereo_passthrough(self, engine, stereo_signal):
        data, sr = stereo_signal
        engine.load(data, sr)
        engine._playing = True

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._callback(outdata, frames, None, None)

        np.testing.assert_array_almost_equal(outdata[:, 0], data[:frames, 0])
        np.testing.assert_array_almost_equal(outdata[:, 1], data[:frames, 1])


class TestSeek:
    def test_seek_within_range(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        engine.seek(1000)
        assert engine.position == 1000

    def test_seek_clamped(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        engine.seek(999999)
        assert engine.position == engine.total_frames

    def test_seek_negative_clamped(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        engine.seek(-100)
        assert engine.position == 0
