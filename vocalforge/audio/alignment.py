"""Alignment — FFT cross-correlation to align vocal recording to minus track."""

import numpy as np
from scipy.signal import fftconvolve, resample_poly
from math import gcd


def to_mono(data: np.ndarray) -> np.ndarray:
    """Convert audio to mono by averaging channels. Passthrough if already mono."""
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


def resample_if_needed(
    data: np.ndarray, from_sr: int, to_sr: int
) -> np.ndarray:
    """Resample audio if sample rates differ. Returns unchanged if rates match."""
    if from_sr == to_sr:
        return data
    g = gcd(from_sr, to_sr)
    up = to_sr // g
    down = from_sr // g
    if data.ndim == 1:
        return resample_poly(data, up, down).astype(np.float32)
    # Resample each channel independently
    channels = [
        resample_poly(data[:, ch], up, down).astype(np.float32)
        for ch in range(data.shape[1])
    ]
    return np.column_stack(channels)


def compute_lag(
    reference: np.ndarray, target: np.ndarray, sample_rate: int
) -> dict:
    """Compute the lag of target relative to reference via FFT cross-correlation.

    Both inputs must be 1-D (mono). The reference is the minus track,
    the target is the vocal recording.

    Returns:
        Dict with lag_samples (int), lag_ms (float), correlation_peak (float).
        Positive lag means the target is late (starts after reference).
    """
    corr = fftconvolve(target, reference[::-1], mode="full")
    peak_index = int(np.argmax(corr))
    lag_samples = peak_index - (len(reference) - 1)
    lag_ms = lag_samples / sample_rate * 1000.0
    return {
        "lag_samples": lag_samples,
        "lag_ms": lag_ms,
        "correlation_peak": float(corr[peak_index]),
    }


def align_tracks(
    minus_data: np.ndarray,
    vocal_data: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, dict]:
    """Align a vocal recording to a minus track.

    Converts to mono internally for correlation, then shifts the original
    vocal data (preserving its channel layout) to match the minus track.
    The output is trimmed or padded to match the minus track length.

    Args:
        minus_data: Minus track, shape (N,) or (N, C).
        vocal_data: Vocal recording, shape (M,) or (M, C).
        sample_rate: Sample rate shared by both tracks.

    Returns:
        (aligned_vocal, info_dict) where aligned_vocal has the same number
        of channels as the input vocal_data and the same number of samples
        as minus_data.
    """
    mono_minus = to_mono(minus_data)
    mono_vocal = to_mono(vocal_data)

    info = compute_lag(mono_minus, mono_vocal, sample_rate)
    lag = info["lag_samples"]

    target_len = minus_data.shape[0]

    if lag > 0:
        # Vocal is late — trim from start
        shifted = vocal_data[lag:] if lag < len(vocal_data) else vocal_data[:0]
    elif lag < 0:
        # Vocal is early — prepend zeros
        pad_len = -lag
        if vocal_data.ndim == 1:
            pad = np.zeros(pad_len, dtype=vocal_data.dtype)
        else:
            pad = np.zeros((pad_len, vocal_data.shape[1]), dtype=vocal_data.dtype)
        shifted = np.concatenate([pad, vocal_data], axis=0)
    else:
        shifted = vocal_data

    # Trim or pad to match minus length
    if shifted.shape[0] > target_len:
        aligned = shifted[:target_len]
    elif shifted.shape[0] < target_len:
        deficit = target_len - shifted.shape[0]
        if shifted.ndim == 1:
            pad = np.zeros(deficit, dtype=shifted.dtype)
        else:
            pad = np.zeros((deficit, shifted.shape[1]), dtype=shifted.dtype)
        aligned = np.concatenate([shifted, pad], axis=0)
    else:
        aligned = shifted

    return aligned, info
