"""Tests for vocalforge.utils.audio_io module."""

import numpy as np
import pytest
import soundfile as sf

from vocalforge.utils.audio_io import get_audio_info, load_audio, save_audio


def test_load_save_roundtrip_mono(tmp_path):
    path = tmp_path / "mono.wav"
    original = np.clip(
        np.random.randn(44100).astype(np.float32) * 0.5, -1.0, 1.0
    )
    save_audio(str(path), original, 44100)

    loaded, sr = load_audio(str(path))
    assert sr == 44100
    assert loaded.ndim == 1
    np.testing.assert_allclose(loaded, original, atol=1e-4)


def test_load_save_roundtrip_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    original = np.clip(
        np.random.randn(44100, 2).astype(np.float32) * 0.5, -1.0, 1.0
    )
    save_audio(str(path), original, 44100)

    loaded, sr = load_audio(str(path))
    assert sr == 44100
    assert loaded.ndim == 2
    assert loaded.shape[1] == 2
    np.testing.assert_allclose(loaded, original, atol=1e-4)


def test_load_preserves_sample_rate(tmp_path):
    path = tmp_path / "rate48k.wav"
    data = np.zeros(4800, dtype=np.float32)
    save_audio(str(path), data, 48000)

    _, sr = load_audio(str(path))
    assert sr == 48000


def test_load_empty_file_raises(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(str(path), np.array([], dtype=np.float32), 44100)

    with pytest.raises(ValueError, match="empty"):
        load_audio(str(path))


def test_load_nonexistent_file_raises():
    with pytest.raises(Exception):
        load_audio("/nonexistent/path/file.wav")


def test_get_audio_info(tmp_path):
    path = tmp_path / "info_test.wav"
    data = np.random.randn(88200, 2).astype(np.float32) * 0.5
    save_audio(str(path), data, 44100)

    info = get_audio_info(str(path))
    assert info["frames"] == 88200
    assert info["channels"] == 2
    assert info["sample_rate"] == 44100
    assert abs(info["duration"] - 2.0) < 0.01
    assert info["format"] == "WAV"
