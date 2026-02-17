"""Mixer — LUFS normalization and track mixing."""

import numpy as np
import pyloudnorm as pyln


def measure_lufs(data: np.ndarray, sample_rate: int) -> float:
    """Measure integrated loudness in LUFS.

    Reshapes mono (N,) to (N, 1) for pyloudnorm compatibility.
    Returns -inf for silence or very quiet signals.
    """
    meter = pyln.Meter(sample_rate)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    loudness = meter.integrated_loudness(data)
    return loudness


def normalize_to_lufs(
    data: np.ndarray, sample_rate: int, target_lufs: float = -14.0
) -> np.ndarray:
    """Normalize audio to a target LUFS level.

    Returns the data unchanged if it is silence (loudness == -inf).
    Preserves the original shape and dtype.
    """
    current = measure_lufs(data, sample_rate)
    if not np.isfinite(current):
        return data
    gain_db = target_lufs - current
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (data * gain_linear).astype(data.dtype)


def ensure_stereo(data: np.ndarray) -> np.ndarray:
    """Convert mono (N,) to stereo (N, 2) by duplication. Passthrough if already 2-D."""
    if data.ndim == 1:
        return np.column_stack([data, data])
    return data


def match_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad the shorter array to match the longer one. Returns (a, b)."""
    la, lb = a.shape[0], b.shape[0]
    if la == lb:
        return a, b
    if la < lb:
        pad_shape = list(a.shape)
        pad_shape[0] = lb - la
        a = np.concatenate([a, np.zeros(pad_shape, dtype=a.dtype)], axis=0)
    else:
        pad_shape = list(b.shape)
        pad_shape[0] = la - lb
        b = np.concatenate([b, np.zeros(pad_shape, dtype=b.dtype)], axis=0)
    return a, b


def peak_limit(data: np.ndarray, ceiling: float = 1.0) -> np.ndarray:
    """Hard-clip samples to [-ceiling, +ceiling]."""
    return np.clip(data, -ceiling, ceiling)


def mix_tracks(
    minus: np.ndarray,
    vocal: np.ndarray,
    sr: int,
    vocal_gain: float = 1.0,
    instrumental_gain: float = 1.0,
    target_lufs: float = -14.0,
    apply_limiter: bool = True,
) -> tuple[np.ndarray, dict]:
    """Full mixing pipeline: normalize, gain, sum, final normalize, limit.

    Args:
        minus: Minus/instrumental track.
        vocal: Vocal track (already aligned to minus).
        sr: Sample rate.
        vocal_gain: Linear gain for vocal (1.0 = unity).
        instrumental_gain: Linear gain for instrumental (1.0 = unity).
        target_lufs: Target integrated loudness for the final mix.
        apply_limiter: If True (default), apply peak_limit() to the mix.
            Set to False when the caller applies effects.limiter() instead.

    Returns:
        (mix_array, info_dict) where mix_array is float32 stereo.
    """
    minus_s = ensure_stereo(minus)
    vocal_s = ensure_stereo(vocal)
    minus_s, vocal_s = match_lengths(minus_s, vocal_s)

    # Normalize each to target LUFS independently
    minus_n = normalize_to_lufs(minus_s, sr, target_lufs)
    vocal_n = normalize_to_lufs(vocal_s, sr, target_lufs)

    # Apply gains and sum
    mix = minus_n * instrumental_gain + vocal_n * vocal_gain

    # Final LUFS normalization of the combined mix
    mix = normalize_to_lufs(mix, sr, target_lufs)

    # Peak limit
    if apply_limiter:
        mix = peak_limit(mix)

    info = {
        "minus_lufs": measure_lufs(minus_s, sr),
        "vocal_lufs": measure_lufs(vocal_s, sr),
        "mix_lufs": measure_lufs(mix, sr),
        "vocal_gain": vocal_gain,
        "instrumental_gain": instrumental_gain,
        "target_lufs": target_lufs,
    }
    return mix.astype(np.float32), info
