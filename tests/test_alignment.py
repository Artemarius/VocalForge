"""Tests for vocalforge.audio.alignment."""

import numpy as np
import pytest

from vocalforge.audio.alignment import (
    align_tracks,
    compute_lag,
    resample_if_needed,
    to_mono,
)

SR = 44100


# --- to_mono ---

def test_to_mono_passthrough_mono():
    mono = np.random.randn(1000).astype(np.float32)
    result = to_mono(mono)
    np.testing.assert_array_equal(result, mono)


def test_to_mono_averages_stereo():
    stereo = np.random.randn(1000, 2).astype(np.float32)
    result = to_mono(stereo)
    expected = stereo.mean(axis=1)
    np.testing.assert_allclose(result, expected)


# --- resample_if_needed ---

def test_resample_passthrough_same_rate():
    data = np.random.randn(1000).astype(np.float32)
    result = resample_if_needed(data, 44100, 44100)
    np.testing.assert_array_equal(result, data)


def test_resample_changes_length():
    data = np.random.randn(44100).astype(np.float32)
    result = resample_if_needed(data, 44100, 22050)
    assert abs(len(result) - 22050) <= 1


def test_resample_stereo():
    data = np.random.randn(44100, 2).astype(np.float32)
    result = resample_if_needed(data, 44100, 22050)
    assert result.ndim == 2
    assert result.shape[1] == 2
    assert abs(result.shape[0] - 22050) <= 1


# --- compute_lag ---

def _make_sine(length, freq=440, sr=SR):
    t = np.arange(length) / sr
    return (np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_compute_lag_zero_offset():
    sig = _make_sine(SR)  # 1 second
    info = compute_lag(sig, sig, SR)
    assert abs(info["lag_samples"]) <= 1
    assert abs(info["lag_ms"]) < 0.1


def test_compute_lag_positive_offset():
    """Target is delayed by 100 samples (target starts late)."""
    ref = _make_sine(SR)
    offset = 100
    target = np.concatenate([np.zeros(offset, dtype=np.float32), ref])
    info = compute_lag(ref, target, SR)
    assert abs(info["lag_samples"] - offset) <= 1


def test_compute_lag_negative_offset():
    """Target starts early — reference has silence prepended."""
    ref_core = _make_sine(SR)
    offset = 200
    ref = np.concatenate([np.zeros(offset, dtype=np.float32), ref_core])
    target = ref_core.copy()
    info = compute_lag(ref, target, SR)
    assert abs(info["lag_samples"] - (-offset)) <= 1


def test_compute_lag_large_offset():
    """1 second offset (44100 samples)."""
    ref = _make_sine(SR * 2)
    offset = SR  # 1 second
    target = np.concatenate([np.zeros(offset, dtype=np.float32), ref])
    info = compute_lag(ref, target, SR)
    assert abs(info["lag_samples"] - offset) <= 1
    assert abs(info["lag_ms"] - 1000.0) < 0.1


# --- align_tracks ---

def test_align_tracks_mono_positive_shift():
    minus = _make_sine(SR)
    offset = 100
    vocal = np.concatenate([np.zeros(offset, dtype=np.float32), minus])
    aligned, info = align_tracks(minus, vocal, SR)
    assert aligned.shape[0] == minus.shape[0]
    assert aligned.ndim == 1


def test_align_tracks_stereo_vocal():
    minus = _make_sine(SR)
    offset = 50
    vocal_mono = np.concatenate([np.zeros(offset, dtype=np.float32), minus])
    vocal = np.column_stack([vocal_mono, vocal_mono])
    aligned, info = align_tracks(minus, vocal, SR)
    assert aligned.shape[0] == minus.shape[0]
    assert aligned.ndim == 2
    assert aligned.shape[1] == 2


def test_align_tracks_output_length_matches_minus():
    minus = _make_sine(SR * 2)
    vocal = _make_sine(SR)  # shorter
    aligned, info = align_tracks(minus, vocal, SR)
    assert aligned.shape[0] == minus.shape[0]


def test_align_tracks_info_populated():
    minus = _make_sine(SR)
    vocal = minus.copy()
    _, info = align_tracks(minus, vocal, SR)
    assert "lag_samples" in info
    assert "lag_ms" in info
    assert "correlation_peak" in info
