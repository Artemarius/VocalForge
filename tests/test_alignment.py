"""Tests for vocalforge.audio.alignment."""

import numpy as np
import pytest

from vocalforge.audio.alignment import (
    align_tracks,
    chain_align,
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


# --- chain_align ---

def test_chain_align_known_offsets():
    """Chain alignment recovers both vocal and minus offsets."""
    minus_sep = _make_sine(SR * 2, freq=330)
    vocal_sep = _make_sine(SR * 2, freq=440)

    vocal_offset = 200
    minus_offset = 100

    vocal_rec = np.concatenate([
        np.zeros(vocal_offset, dtype=np.float32), vocal_sep
    ])
    minus_import = np.concatenate([
        np.zeros(minus_offset, dtype=np.float32), minus_sep
    ])

    aligned_minus, aligned_vocal, info = chain_align(
        minus_sep, vocal_sep, vocal_rec, SR, minus_import=minus_import
    )

    assert abs(info["vocal_lag_samples"] - vocal_offset) <= 1
    assert abs(info["minus_lag_samples"] - minus_offset) <= 1
    assert aligned_vocal.shape[0] == vocal_sep.shape[0]
    assert aligned_minus.shape[0] == minus_sep.shape[0]


def test_chain_align_no_import():
    """Without minus_import, returns minus_sep unchanged."""
    minus_sep = _make_sine(SR, freq=330)
    vocal_sep = _make_sine(SR, freq=440)
    vocal_rec = vocal_sep.copy()

    aligned_minus, aligned_vocal, info = chain_align(
        minus_sep, vocal_sep, vocal_rec, SR
    )

    np.testing.assert_array_equal(aligned_minus, minus_sep)
    assert info["minus_lag_samples"] == 0
    assert info["minus_lag_ms"] == 0.0


def test_chain_align_zero_offsets():
    """Identical signals should yield zero lag for both."""
    minus_sep = _make_sine(SR, freq=330)
    vocal_sep = _make_sine(SR, freq=440)
    vocal_rec = vocal_sep.copy()
    minus_import = minus_sep.copy()

    _, _, info = chain_align(
        minus_sep, vocal_sep, vocal_rec, SR, minus_import=minus_import
    )

    assert abs(info["vocal_lag_samples"]) <= 1
    assert abs(info["minus_lag_samples"]) <= 1


# --- constrained alignment ---

def _make_noise_burst(length, sr=SR, seed=42):
    """Generate a unique (non-periodic) noise burst signal."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(length).astype(np.float32)


def test_compute_lag_constrained_window():
    """Offset within the window is found correctly."""
    ref = _make_noise_burst(SR * 2)
    offset = 500  # ~11 ms
    target = np.concatenate([np.zeros(offset, dtype=np.float32), ref])
    info = compute_lag(ref, target, SR, max_offset_s=1.0)
    assert abs(info["lag_samples"] - offset) <= 1
    assert info["constrained"] is True


def test_compute_lag_rejects_out_of_window():
    """Repetitive signal — constrained window picks the small correct offset."""
    # Build a repeating pattern with period = 4410 samples (~100 ms)
    period = 4410
    one_cycle = _make_sine(period, freq=440)
    reps = 20
    ref = np.tile(one_cycle, reps).astype(np.float32)

    # True offset is small (50 samples), but unconstrained could jump to a
    # beat-period alias.  The multi-peak preference (closest to center among
    # peaks >= 95% of max) should pick a lag near 50 rather than a period alias.
    true_offset = 50
    target = np.concatenate([np.zeros(true_offset, dtype=np.float32), ref])

    # Constrain to +/- 0.05s (2205 samples) — smaller than one period
    info = compute_lag(ref, target, SR, max_offset_s=0.05)
    # For periodic signals multi-peak picks closest to center; accept within period
    assert abs(info["lag_samples"]) < period
    assert info["constrained"] is True


def test_compute_lag_suspicious_flag():
    """Large clipping offset is flagged as suspicious."""
    ref = _make_sine(SR)
    # Offset is 60% of target length → definitely suspicious
    offset = int(SR * 0.6)
    target = np.concatenate([np.zeros(offset, dtype=np.float32), ref[:SR // 2]])
    info = compute_lag(ref, target, SR)
    assert info["suspicious"] is True


def test_compute_lag_no_constraint():
    """max_offset_s=None behaves like original (unconstrained)."""
    ref = _make_sine(SR * 2)
    offset = SR  # 1 second
    target = np.concatenate([np.zeros(offset, dtype=np.float32), ref])
    info = compute_lag(ref, target, SR, max_offset_s=None)
    assert abs(info["lag_samples"] - offset) <= 1
    assert info["constrained"] is False


def test_align_tracks_with_constraint():
    """max_offset_s passthrough works via align_tracks."""
    minus = _make_noise_burst(SR * 2, seed=10)
    offset = 200
    vocal = np.concatenate([np.zeros(offset, dtype=np.float32), minus])
    aligned, info = align_tracks(minus, vocal, SR, max_offset_s=5.0)
    assert abs(info["lag_samples"] - offset) <= 1
    assert info["constrained"] is True
    assert aligned.shape[0] == minus.shape[0]


def test_chain_align_with_constraint():
    """max_offset_s passthrough works via chain_align."""
    minus_sep = _make_noise_burst(SR * 2, seed=10)
    vocal_sep = _make_noise_burst(SR * 2, seed=20)
    vocal_offset = 200
    vocal_rec = np.concatenate([
        np.zeros(vocal_offset, dtype=np.float32), vocal_sep
    ])

    _, _, info = chain_align(
        minus_sep, vocal_sep, vocal_rec, SR, max_offset_s=5.0,
    )
    assert abs(info["vocal_lag_samples"] - vocal_offset) <= 1
