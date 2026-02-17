"""Tests for AudioEngine — state logic, callback behavior, and recording.

Uses synthetic numpy arrays only (no real audio devices).
"""

import numpy as np
import pytest
import sounddevice as sd

from vocalforge.audio.engine import AudioEngine, EngineState, PlaybackEngine


@pytest.fixture
def engine():
    return AudioEngine()


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


class TestBackwardCompatAlias:
    def test_alias_is_same_class(self):
        assert PlaybackEngine is AudioEngine


class TestEngineState:
    def test_initial_state_is_idle(self, engine):
        assert engine.state == EngineState.IDLE

    def test_not_playing_initially(self, engine):
        assert not engine.is_playing

    def test_not_paused_initially(self, engine):
        assert not engine.is_paused

    def test_not_recording_initially(self, engine):
        assert not engine.is_recording

    def test_playback_ended_false_initially(self, engine):
        assert not engine.playback_ended

    def test_state_after_load(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        assert engine.state == EngineState.IDLE


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
        engine._state = EngineState.PLAYING
        engine.stop()
        assert engine.position == 0
        assert not engine.is_playing
        assert engine.state == EngineState.IDLE


class TestPlayWithoutData:
    def test_no_crash(self, engine):
        engine.play()
        assert not engine.is_playing


class TestCallbackMonoToStereo:
    def test_mono_duplicated_to_stereo(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        engine._state = EngineState.PLAYING

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._playback_callback(outdata, frames, None, None)

        # Both channels should be identical
        np.testing.assert_array_equal(outdata[:, 0], outdata[:, 1])
        # Should match source data
        np.testing.assert_array_almost_equal(outdata[:, 0], data[:frames])


class TestCallbackEndOfAudio:
    def test_short_buffer_raises_callback_stop(self, engine):
        sr = 44100
        data = np.ones(100, dtype=np.float32) * 0.5
        engine.load(data, sr)
        engine._state = EngineState.PLAYING

        frames = 256
        outdata = np.zeros((frames, 2), dtype=np.float32)

        with pytest.raises(sd.CallbackStop):
            engine._playback_callback(outdata, frames, None, None)

        # First 100 frames should have data, rest should be zero
        assert np.all(outdata[:100, 0] != 0)
        assert np.all(outdata[100:, :] == 0)


class TestCallbackAppliesVolume:
    def test_half_volume(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        engine.volume = 0.5
        engine._state = EngineState.PLAYING

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._playback_callback(outdata, frames, None, None)

        expected = data[:frames] * 0.5
        np.testing.assert_array_almost_equal(outdata[:, 0], expected)

    def test_zero_volume(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        engine.volume = 0.0
        engine._state = EngineState.PLAYING

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._playback_callback(outdata, frames, None, None)

        assert np.all(outdata == 0)


class TestCallbackStereo:
    def test_stereo_passthrough(self, engine, stereo_signal):
        data, sr = stereo_signal
        engine.load(data, sr)
        engine._state = EngineState.PLAYING

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._playback_callback(outdata, frames, None, None)

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


class TestRecordingCallback:
    def test_writes_to_buffer(self, engine):
        """Recording callback writes channel 0 of input to ring buffer."""
        engine._rec_max_frames = 4096
        engine._rec_buffer = np.zeros(4096, dtype=np.float32)
        engine._rec_position = 0

        # Simulate stereo input (2 channels)
        frames = 512
        indata = np.random.randn(frames, 2).astype(np.float32)
        engine._recording_callback(indata, frames, None, None)

        assert engine._rec_position == frames
        np.testing.assert_array_equal(
            engine._rec_buffer[:frames], indata[:, 0]
        )

    def test_multiple_writes_advance_position(self, engine):
        engine._rec_max_frames = 4096
        engine._rec_buffer = np.zeros(4096, dtype=np.float32)
        engine._rec_position = 0

        frames = 256
        indata = np.ones((frames, 1), dtype=np.float32) * 0.5
        engine._recording_callback(indata, frames, None, None)
        engine._recording_callback(indata, frames, None, None)

        assert engine._rec_position == 512

    def test_overflow_raises_callback_stop(self, engine):
        """When buffer is full, callback raises CallbackStop."""
        engine._rec_max_frames = 100
        engine._rec_buffer = np.zeros(100, dtype=np.float32)
        engine._rec_position = 100  # Buffer full

        frames = 256
        indata = np.ones((frames, 1), dtype=np.float32)

        with pytest.raises(sd.CallbackStop):
            engine._recording_callback(indata, frames, None, None)

    def test_partial_write_at_end(self, engine):
        """When buffer has less room than frames, writes what fits."""
        engine._rec_max_frames = 100
        engine._rec_buffer = np.zeros(100, dtype=np.float32)
        engine._rec_position = 80

        frames = 64
        indata = np.ones((frames, 1), dtype=np.float32) * 0.7
        engine._recording_callback(indata, frames, None, None)

        # Only 20 frames should have been written
        assert engine._rec_position == 100
        np.testing.assert_array_almost_equal(
            engine._rec_buffer[80:100], np.full(20, 0.7, dtype=np.float32)
        )


class TestLatencyCompensation:
    def _make_recording(self, engine, n_frames=1000, sr=44100):
        """Helper: simulate a recording of n_frames samples."""
        data = np.arange(n_frames, dtype=np.float32)
        engine._sample_rate = sr
        engine._rec_buffer = data.copy()
        engine._rec_position = n_frames
        engine._rec_max_frames = n_frames
        engine._state = EngineState.RECORDING
        # Mock audio data so finish_recording doesn't fail
        engine._audio_data = np.zeros(n_frames, dtype=np.float32)
        engine._total_frames = n_frames

    def test_positive_offset_trims_start(self, engine):
        self._make_recording(engine, 1000, 44100)
        engine.latency_offset_ms = 10.0  # ~441 samples at 44100

        result = engine.finish_recording()
        assert result is not None
        data, sr = result
        offset_samples = int(10.0 * 44100 / 1000.0)
        assert len(data) == 1000 - offset_samples

    def test_negative_offset_pads_start(self, engine):
        self._make_recording(engine, 1000, 44100)
        engine.latency_offset_ms = -10.0  # pad ~441 samples

        result = engine.finish_recording()
        assert result is not None
        data, sr = result
        pad_samples = int(10.0 * 44100 / 1000.0)
        assert len(data) == 1000 + pad_samples
        # First pad_samples should be zeros
        assert np.all(data[:pad_samples] == 0.0)

    def test_zero_offset_no_change(self, engine):
        self._make_recording(engine, 1000, 44100)
        engine.latency_offset_ms = 0.0

        result = engine.finish_recording()
        assert result is not None
        data, sr = result
        assert len(data) == 1000

    def test_clamping_above_500(self, engine):
        engine.latency_offset_ms = 999
        assert engine.latency_offset_ms == 500.0

    def test_clamping_below_minus_500(self, engine):
        engine.latency_offset_ms = -999
        assert engine.latency_offset_ms == -500.0


class TestStopRecordingDiscards:
    def test_buffer_freed_and_state_reset(self, engine, mono_signal):
        data, sr = mono_signal
        engine.load(data, sr)
        # Manually put engine into recording state
        engine._state = EngineState.RECORDING
        engine._rec_buffer = np.zeros(1000, dtype=np.float32)
        engine._rec_position = 500

        engine.stop_recording()

        assert engine._rec_buffer is None
        assert engine._rec_position == 0
        assert engine.state == EngineState.IDLE

    def test_stop_recording_when_not_recording_is_noop(self, engine):
        engine.stop_recording()
        assert engine.state == EngineState.IDLE


class TestOnPlaybackFinished:
    def test_sets_playback_ended(self, engine):
        engine._state = EngineState.RECORDING
        engine._on_playback_finished()
        assert engine.playback_ended is True
        # State should stay RECORDING (not auto-switch)
        assert engine.state == EngineState.RECORDING

    def test_sets_idle_when_playing(self, engine):
        engine._state = EngineState.PLAYING
        engine._on_playback_finished()
        assert engine.playback_ended is True
        assert engine.state == EngineState.IDLE


# ---- Multi-track tests ----


class TestLoadMultiTotalFrames:
    """Verify total_frames = max(track_length + offset) across tracks."""

    def test_same_length_no_offsets(self, engine):
        sr = 44100
        t1 = np.zeros(1000, dtype=np.float32)
        t2 = np.zeros(2000, dtype=np.float32)
        engine.load_multi([t1, t2], sr)
        assert engine.total_frames == 2000

    def test_offset_extends_total(self, engine):
        sr = 44100
        t1 = np.zeros(1000, dtype=np.float32)
        t2 = np.zeros(1000, dtype=np.float32)
        engine.load_multi([t1, t2], sr, offsets=[0, 500])
        # t2 ends at 500 + 1000 = 1500
        assert engine.total_frames == 1500

    def test_shorter_track_with_large_offset(self, engine):
        sr = 44100
        t1 = np.zeros(3000, dtype=np.float32)
        t2 = np.zeros(500, dtype=np.float32)
        engine.load_multi([t1, t2], sr, offsets=[0, 4000])
        # t2 ends at 4000 + 500 = 4500 > 3000
        assert engine.total_frames == 4500


class TestLoadMultiWithMutes:
    """Verify muted tracks don't produce audio but total_frames includes them."""

    def test_muted_tracks_still_count_in_total_frames(self, engine):
        sr = 44100
        t1 = np.zeros(1000, dtype=np.float32)
        t2 = np.zeros(2000, dtype=np.float32)
        engine.load_multi([t1, t2], sr, mutes=[False, True])
        # t2 is muted but still counts for total_frames
        assert engine.total_frames == 2000

    def test_muted_track_produces_silence(self, engine):
        sr = 44100
        t1 = np.ones(1024, dtype=np.float32) * 0.5
        engine.load_multi([t1], sr, mutes=[True])

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._position = 0
        engine._multi_playback_callback(outdata, frames, None, None)
        assert np.all(outdata == 0)

    def test_unmuted_track_produces_audio(self, engine):
        sr = 44100
        t1 = np.ones(1024, dtype=np.float32) * 0.5
        engine.load_multi([t1], sr, mutes=[False])

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._position = 0
        engine._multi_playback_callback(outdata, frames, None, None)
        # Mono track duplicated to both channels
        np.testing.assert_array_almost_equal(outdata[:, 0], 0.5)
        np.testing.assert_array_almost_equal(outdata[:, 1], 0.5)


class TestMultiTrackSeek:
    """Verify seek is respected in multi-track mode."""

    def test_seek_sets_position(self, engine):
        sr = 44100
        t1 = np.zeros(4000, dtype=np.float32)
        engine.load_multi([t1], sr)
        engine.seek(2000)
        assert engine.position == 2000

    def test_seek_clamped_to_total(self, engine):
        sr = 44100
        t1 = np.zeros(1000, dtype=np.float32)
        engine.load_multi([t1], sr)
        engine.seek(9999)
        assert engine.position == 1000

    def test_seek_affects_callback_output(self, engine):
        sr = 44100
        # Ramp signal: each sample = its index
        t1 = np.arange(2048, dtype=np.float32)
        engine.load_multi([t1], sr)
        engine.seek(1000)

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._multi_playback_callback(outdata, frames, None, None)
        # Output should be samples 1000..1511
        expected = np.arange(1000, 1512, dtype=np.float32)
        np.testing.assert_array_almost_equal(outdata[:, 0], expected)


class TestMultiTrackCallback:
    """Verify offset logic in multi-track callback."""

    def test_offset_track_silent_before_offset(self, engine):
        sr = 44100
        t1 = np.ones(1000, dtype=np.float32) * 0.8
        engine.load_multi([t1], sr, offsets=[500])
        engine._position = 0

        # Read first 500 frames — track hasn't started yet
        frames = 500
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._multi_playback_callback(outdata, frames, None, None)
        assert np.all(outdata == 0)

    def test_offset_track_audible_after_offset(self, engine):
        sr = 44100
        t1 = np.ones(1000, dtype=np.float32) * 0.8
        engine.load_multi([t1], sr, offsets=[500])

        # Seek past offset and read
        engine._position = 500
        frames = 256
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._multi_playback_callback(outdata, frames, None, None)
        np.testing.assert_array_almost_equal(outdata[:, 0], 0.8)

    def test_two_tracks_summed(self, engine):
        sr = 44100
        t1 = np.ones(1024, dtype=np.float32) * 0.3
        t2 = np.ones(1024, dtype=np.float32) * 0.2
        engine.load_multi([t1, t2], sr)
        engine._position = 0

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._multi_playback_callback(outdata, frames, None, None)
        np.testing.assert_array_almost_equal(outdata[:, 0], 0.5)

    def test_gain_applied(self, engine):
        sr = 44100
        t1 = np.ones(1024, dtype=np.float32) * 1.0
        engine.load_multi([t1], sr, gains=[0.5])
        engine._position = 0

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._multi_playback_callback(outdata, frames, None, None)
        np.testing.assert_array_almost_equal(outdata[:, 0], 0.5)

    def test_stereo_track(self, engine):
        sr = 44100
        left = np.ones(1024, dtype=np.float32) * 0.3
        right = np.ones(1024, dtype=np.float32) * 0.7
        stereo = np.column_stack([left, right]).astype(np.float32)
        engine.load_multi([stereo], sr)
        engine._position = 0

        frames = 512
        outdata = np.zeros((frames, 2), dtype=np.float32)
        engine._multi_playback_callback(outdata, frames, None, None)
        np.testing.assert_array_almost_equal(outdata[:, 0], 0.3)
        np.testing.assert_array_almost_equal(outdata[:, 1], 0.7)
