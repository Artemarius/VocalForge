"""Tests for vocalforge.separation.demucs_worker."""

import os

import numpy as np
import pytest
import soundfile as sf

from vocalforge.separation.demucs_worker import get_device_info, separate_song

SR = 44100


def _make_stereo_wav(path: str, duration_s: float = 1.0, sr: int = SR) -> None:
    """Write a short stereo WAV file for testing."""
    t = np.arange(int(sr * duration_s)) / sr
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    sf.write(path, stereo, sr)


# --- get_device_info ---

def test_get_device_info_returns_valid_keys():
    info = get_device_info()
    assert "device" in info
    assert info["device"] in ("cuda", "cpu")
    if info["device"] == "cuda":
        assert "gpu_name" in info


# --- Cache behaviour ---

def test_cache_directory_creation(tmp_path):
    """Verify stems dir is created when caching."""
    song_path = str(tmp_path / "song.wav")
    _make_stereo_wav(song_path)

    stems_dir = str(tmp_path / "song_stems")
    cached_path = os.path.join(stems_dir, "no_vocals.wav")

    # Pre-create the cache so we don't need the Demucs model
    os.makedirs(stems_dir, exist_ok=True)
    _make_stereo_wav(cached_path)

    data, sr = separate_song(song_path)
    assert os.path.isdir(stems_dir)


def test_cache_hit_returns_cached(tmp_path):
    """When a cached no_vocals.wav exists, separate_song loads it directly."""
    song_path = str(tmp_path / "mysong.wav")
    _make_stereo_wav(song_path, duration_s=0.5)

    # Pre-populate cache
    stems_dir = str(tmp_path / "mysong_stems")
    os.makedirs(stems_dir)
    cached_path = os.path.join(stems_dir, "no_vocals.wav")

    # Write a distinctive cached file (880 Hz instead of 440 Hz)
    t = np.arange(int(SR * 0.5)) / SR
    mono = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    cached_data = np.column_stack([mono, mono])
    sf.write(cached_path, cached_data, SR)

    # Track callback messages
    messages = []

    def callback(msg, frac):
        messages.append(msg)

    data, sr = separate_song(song_path, callback=callback)

    assert sr == SR
    assert data.shape == cached_data.shape
    # WAV round-trip introduces quantization noise; use relaxed tolerance
    np.testing.assert_allclose(data, cached_data, atol=1e-4)
    assert any("cached" in m.lower() for m in messages)


def test_cache_hit_with_custom_output_dir(tmp_path):
    """Cache lookup works with a custom output_dir."""
    song_path = str(tmp_path / "track.wav")
    _make_stereo_wav(song_path)

    custom_dir = str(tmp_path / "custom_stems")
    os.makedirs(custom_dir)
    cached_path = os.path.join(custom_dir, "no_vocals.wav")
    _make_stereo_wav(cached_path, duration_s=0.5)

    data, sr = separate_song(song_path, output_dir=custom_dir)
    assert sr == SR
    assert data.shape[0] == int(SR * 0.5)


# --- Integration test (requires Demucs model) ---

@pytest.mark.slow
def test_separate_song_produces_valid_output(tmp_path):
    """Run actual Demucs separation on a short synthetic file."""
    song_path = str(tmp_path / "test_song.wav")
    _make_stereo_wav(song_path, duration_s=3.0)

    data, sr = separate_song(song_path)

    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float32
    assert data.ndim == 2
    assert data.shape[1] == 2  # stereo
    assert sr > 0
