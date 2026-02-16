"""Tests for vocalforge.audio.mixer."""

import numpy as np
import pytest

from vocalforge.audio.mixer import (
    ensure_stereo,
    match_lengths,
    measure_lufs,
    mix_tracks,
    normalize_to_lufs,
    peak_limit,
)

SR = 44100


def _make_tone(duration_s=2.0, freq=440, sr=SR):
    """Generate a stereo sine tone loud enough for LUFS measurement."""
    t = np.arange(int(sr * duration_s)) / sr
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.column_stack([mono, mono])


# --- measure_lufs ---

def test_measure_lufs_silence_returns_neg_inf():
    silence = np.zeros((SR * 2, 2), dtype=np.float32)
    result = measure_lufs(silence, SR)
    assert result == float("-inf")


def test_measure_lufs_mono_accepted():
    mono = 0.5 * np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR).astype(np.float32)
    result = measure_lufs(mono, SR)
    assert np.isfinite(result)


def test_measure_lufs_stereo_accepted():
    stereo = _make_tone()
    result = measure_lufs(stereo, SR)
    assert np.isfinite(result)


# --- normalize_to_lufs ---

def test_normalize_to_lufs_hits_target():
    tone = _make_tone()
    target = -14.0
    normed = normalize_to_lufs(tone, SR, target)
    measured = measure_lufs(normed, SR)
    assert abs(measured - target) < 0.5


def test_normalize_to_lufs_silence_unchanged():
    silence = np.zeros((SR * 2, 2), dtype=np.float32)
    result = normalize_to_lufs(silence, SR, -14.0)
    np.testing.assert_array_equal(result, silence)


def test_normalize_to_lufs_preserves_shape():
    tone = _make_tone()
    normed = normalize_to_lufs(tone, SR, -14.0)
    assert normed.shape == tone.shape


# --- peak_limit ---

def test_peak_limit_clips():
    data = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
    result = peak_limit(data, ceiling=1.0)
    assert result.max() <= 1.0
    assert result.min() >= -1.0


def test_peak_limit_no_change_within_range():
    data = np.array([0.5, -0.3, 0.0], dtype=np.float32)
    result = peak_limit(data, ceiling=1.0)
    np.testing.assert_array_equal(result, data)


# --- ensure_stereo ---

def test_ensure_stereo_from_mono():
    mono = np.random.randn(1000).astype(np.float32)
    stereo = ensure_stereo(mono)
    assert stereo.shape == (1000, 2)
    np.testing.assert_array_equal(stereo[:, 0], mono)
    np.testing.assert_array_equal(stereo[:, 1], mono)


def test_ensure_stereo_passthrough():
    stereo = np.random.randn(1000, 2).astype(np.float32)
    result = ensure_stereo(stereo)
    np.testing.assert_array_equal(result, stereo)


# --- match_lengths ---

def test_match_lengths_pads_shorter():
    a = np.ones((100, 2), dtype=np.float32)
    b = np.ones((200, 2), dtype=np.float32)
    a2, b2 = match_lengths(a, b)
    assert a2.shape[0] == 200
    assert b2.shape[0] == 200
    # Padded region should be zeros
    np.testing.assert_array_equal(a2[100:], 0)


def test_match_lengths_equal_unchanged():
    a = np.ones((100, 2), dtype=np.float32)
    b = np.ones((100, 2), dtype=np.float32)
    a2, b2 = match_lengths(a, b)
    np.testing.assert_array_equal(a2, a)
    np.testing.assert_array_equal(b2, b)


# --- mix_tracks ---

def test_mix_tracks_output_is_stereo():
    minus = _make_tone()
    vocal = _make_tone(freq=880)
    mix, info = mix_tracks(minus, vocal, SR)
    assert mix.ndim == 2
    assert mix.shape[1] == 2


def test_mix_tracks_lufs_near_target():
    minus = _make_tone()
    vocal = _make_tone(freq=880)
    target = -14.0
    mix, info = mix_tracks(minus, vocal, SR, target_lufs=target)
    measured = measure_lufs(mix, SR)
    assert abs(measured - target) < 1.0


def test_mix_tracks_vocal_gain_zero_mutes_vocal():
    minus = _make_tone()
    vocal = _make_tone(freq=880)
    mix_muted, _ = mix_tracks(minus, vocal, SR, vocal_gain=0.0)
    mix_full, _ = mix_tracks(minus, vocal, SR, vocal_gain=1.0)
    # Muted mix should differ from full mix
    assert not np.allclose(mix_muted, mix_full, atol=1e-3)


def test_mix_tracks_no_samples_exceed_ceiling():
    minus = _make_tone()
    vocal = _make_tone(freq=880)
    mix, _ = mix_tracks(minus, vocal, SR)
    assert mix.max() <= 1.0
    assert mix.min() >= -1.0
