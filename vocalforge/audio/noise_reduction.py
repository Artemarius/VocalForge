"""Noise reduction — spectral gating for vocal recordings."""

import numpy as np


def estimate_noise_profile(
    data: np.ndarray,
    sample_rate: int,
    duration_s: float = 0.5,
    strategy: str = "start",
) -> np.ndarray:
    """Extract a noise-only clip from the audio for use as a noise profile.

    Args:
        data: Audio array, shape (samples,) or (samples, channels), float32.
        sample_rate: Sample rate in Hz.
        duration_s: Length of the noise clip in seconds.
        strategy: One of "start", "end", or "quietest".

    Returns:
        A slice of data containing mostly noise.
    """
    n_samples = int(sample_rate * duration_s)
    n_samples = min(n_samples, data.shape[0])

    if n_samples == 0:
        return data

    if strategy == "end":
        return data[-n_samples:]
    elif strategy == "quietest":
        # Slide a window and return the segment with lowest RMS energy
        step = max(1, n_samples // 2)
        best_start = 0
        best_rms = np.inf
        for start in range(0, data.shape[0] - n_samples + 1, step):
            segment = data[start : start + n_samples]
            rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2))
            if rms < best_rms:
                best_rms = rms
                best_start = start
        return data[best_start : best_start + n_samples]
    else:
        # Default: "start"
        return data[:n_samples]


def reduce_noise(
    data: np.ndarray,
    sample_rate: int,
    noise_clip: np.ndarray | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply spectral-gating noise reduction to audio.

    Args:
        data: Audio array, shape (samples,) or (samples, channels), float32.
        sample_rate: Sample rate in Hz.
        noise_clip: Optional noise profile array (same channel layout as data).
            If None, auto-estimated from the start of data.
        strength: Noise reduction intensity from 0.0 (none) to 1.0 (maximum).
            Maps to the ``prop_decrease`` parameter of noisereduce.

    Returns:
        Noise-reduced audio, same shape and dtype (float32) as input.
    """
    if strength == 0.0:
        return data

    # All-zeros or empty input — nothing to do
    if data.size == 0 or not np.any(data):
        return data.copy()

    import noisereduce as nr

    if noise_clip is None:
        noise_clip = estimate_noise_profile(data, sample_rate)

    is_mono = data.ndim == 1

    # noisereduce expects (channels, samples) for multichannel
    if is_mono:
        y = data
        y_noise = noise_clip
    else:
        y = data.T  # (channels, samples)
        y_noise = noise_clip.T

    reduced = nr.reduce_noise(
        y=y,
        sr=sample_rate,
        y_noise=y_noise,
        prop_decrease=float(strength),
    )

    # Transpose back for multichannel
    if not is_mono:
        reduced = reduced.T

    return reduced.astype(np.float32)
