"""Alignment — FFT cross-correlation to align vocal recording to minus track."""

import numpy as np
from scipy.signal import fftconvolve, resample_poly
from math import gcd


def pad_or_trim(vocal_data: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with silence or trim vocal to match target length."""
    current_length = vocal_data.shape[0]
    if current_length > target_length:
        return vocal_data[:target_length]
    elif current_length < target_length:
        deficit = target_length - current_length
        if vocal_data.ndim == 1:
            pad = np.zeros(deficit, dtype=vocal_data.dtype)
        else:
            pad = np.zeros((deficit, vocal_data.shape[1]), dtype=vocal_data.dtype)
        return np.concatenate([vocal_data, pad], axis=0)
    return vocal_data


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
    reference: np.ndarray, target: np.ndarray, sample_rate: int,
    max_offset_s: float | None = None,
) -> dict:
    """Compute the lag of target relative to reference via FFT cross-correlation.

    Both inputs must be 1-D (mono). The reference is the minus track,
    the target is the vocal recording.

    Args:
        reference: 1-D reference signal.
        target: 1-D target signal.
        sample_rate: Sample rate in Hz.
        max_offset_s: If set, restrict the search window to +/- this many
            seconds around zero lag.  Among peaks >= 95% of the window max,
            the one closest to zero lag is preferred (avoids false peaks at
            beat-period offsets in repetitive songs).

    Returns:
        Dict with lag_samples, lag_ms, correlation_peak, constrained (bool),
        suspicious (bool).
    """
    corr = fftconvolve(target, reference[::-1], mode="full")
    center = len(reference) - 1  # zero-lag index

    constrained = max_offset_s is not None
    if constrained:
        max_offset_samples = int(max_offset_s * sample_rate)
        lo = max(0, center - max_offset_samples)
        hi = min(len(corr) - 1, center + max_offset_samples)
        window = corr[lo:hi + 1]
        window_max = float(np.max(window))

        # Multi-peak preference: among peaks >= 95% of max, pick closest to center
        threshold = 0.95 * window_max
        candidates = np.where(window >= threshold)[0]  # indices within window
        # Convert to global indices and compute lag for each
        best_idx = None
        best_abs_lag = None
        for c in candidates:
            global_idx = lo + int(c)
            lag = global_idx - center
            abs_lag = abs(lag)
            if best_abs_lag is None or abs_lag < best_abs_lag:
                best_abs_lag = abs_lag
                best_idx = global_idx
        peak_index = best_idx
    else:
        peak_index = int(np.argmax(corr))

    lag_samples = peak_index - center
    lag_ms = lag_samples / sample_rate * 1000.0

    # Sanity check: how much of the target or reference gets clipped
    ref_len = len(reference)
    tgt_len = len(target)
    if lag_samples > 0:
        clipped_start_pct = lag_samples / tgt_len if tgt_len > 0 else 0.0
        remaining = tgt_len - lag_samples
        clipped_end_pct = max(0, remaining - ref_len) / tgt_len if tgt_len > 0 else 0.0
    elif lag_samples < 0:
        pad = -lag_samples
        clipped_start_pct = 0.0
        clipped_end_pct = max(0, (pad + tgt_len) - ref_len) / tgt_len if tgt_len > 0 else 0.0
    else:
        clipped_start_pct = 0.0
        clipped_end_pct = max(0, tgt_len - ref_len) / tgt_len if tgt_len > 0 else 0.0

    suspicious = clipped_start_pct > 0.10 or clipped_end_pct > 0.10

    return {
        "lag_samples": lag_samples,
        "lag_ms": lag_ms,
        "correlation_peak": float(corr[peak_index]),
        "constrained": constrained,
        "suspicious": suspicious,
    }


def align_tracks(
    minus_data: np.ndarray,
    vocal_data: np.ndarray,
    sample_rate: int,
    max_offset_s: float | None = None,
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

    info = compute_lag(mono_minus, mono_vocal, sample_rate,
                       max_offset_s=max_offset_s)
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


def chain_align(
    minus_sep: np.ndarray,
    vocal_sep: np.ndarray,
    vocal_rec: np.ndarray,
    sample_rate: int,
    minus_import: np.ndarray | None = None,
    max_offset_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Chain alignment through separated stems as a common reference.

    Aligns a vocal recording to the vocal-sep stem, and optionally aligns
    an imported minus track to the minus-sep stem.  This allows indirect
    alignment when the imported minus differs from the original song.

    Args:
        minus_sep: Separated instrumental (no_vocals) stem.
        vocal_sep: Separated vocal stem.
        vocal_rec: User's vocal recording.
        sample_rate: Sample rate shared by all tracks.
        minus_import: Optional imported karaoke/minus track to align
            against minus_sep.

    Returns:
        (aligned_minus, aligned_vocal, info) where:
        - aligned_minus is the aligned minus_import (or minus_sep if no import).
        - aligned_vocal is the vocal recording aligned to the vocal stem.
        - info dict with vocal_lag_samples, vocal_lag_ms, minus_lag_samples,
          minus_lag_ms.
    """
    # Align vocal recording to the separated vocal stem
    aligned_vocal, vocal_info = align_tracks(
        vocal_sep, vocal_rec, sample_rate, max_offset_s=max_offset_s,
    )

    info = {
        "vocal_lag_samples": vocal_info["lag_samples"],
        "vocal_lag_ms": vocal_info["lag_ms"],
        "vocal_suspicious": vocal_info.get("suspicious", False),
        "minus_lag_samples": 0,
        "minus_lag_ms": 0.0,
        "minus_suspicious": False,
    }

    if minus_import is not None:
        # Align imported minus to the separated instrumental stem
        aligned_minus, minus_info = align_tracks(
            minus_sep, minus_import, sample_rate, max_offset_s=max_offset_s,
        )
        info["minus_lag_samples"] = minus_info["lag_samples"]
        info["minus_lag_ms"] = minus_info["lag_ms"]
        info["minus_suspicious"] = minus_info.get("suspicious", False)
    else:
        aligned_minus = minus_sep

    # For the mix info dict, expose the vocal lag as the primary lag
    info["lag_samples"] = info["vocal_lag_samples"]
    info["lag_ms"] = info["vocal_lag_ms"]

    return aligned_minus, aligned_vocal, info
