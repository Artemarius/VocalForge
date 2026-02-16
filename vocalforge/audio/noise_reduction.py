"""Noise reduction — spectral gating and high-pass filtering for vocal recordings."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def high_pass_filter(
    data: np.ndarray, sample_rate: int, cutoff_hz: float = 80.0, order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter.

    Args:
        data: Audio array, shape (samples,) or (samples, channels), float32.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Cutoff frequency in Hz. If <= 0, returns input unchanged.
        order: Filter order (default 4).

    Returns:
        Filtered audio, same shape and dtype (float32) as input.
    """
    if cutoff_hz <= 0 or data.size == 0:
        return data

    nyq = sample_rate / 2.0
    if cutoff_hz >= nyq:
        return data

    sos = butter(order, cutoff_hz / nyq, btype="high", output="sos")

    if data.ndim == 1:
        return sosfiltfilt(sos, data).astype(np.float32)

    # Stereo / multichannel: filter each channel independently
    channels = []
    for ch in range(data.shape[1]):
        channels.append(sosfiltfilt(sos, data[:, ch]).astype(np.float32))
    return np.column_stack(channels)


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


def estimate_noise_from_stem(
    vocal_rec: np.ndarray,
    vocal_sep: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_duration_s: float = 0.5,
) -> np.ndarray | None:
    """Extract a noise profile from vocal_rec using vocal_sep as a guide.

    Finds regions where the separated vocal stem is silent (singer not
    singing) and extracts the corresponding audio from the raw vocal
    recording — which should contain only room noise at those points.

    Args:
        vocal_rec: Raw vocal recording, shape (samples,) or (samples, channels).
        vocal_sep: Separated vocal stem (same length or longer), any layout.
        sample_rate: Sample rate in Hz.
        threshold_db: RMS threshold in dB below which a frame is "silent".
        min_duration_s: Minimum total noise duration required.

    Returns:
        A noise-only slice from vocal_rec, or None if insufficient silence.
    """
    # Work in mono for energy analysis
    sep_mono = vocal_sep.mean(axis=1) if vocal_sep.ndim > 1 else vocal_sep

    frame_samples = int(0.05 * sample_rate)  # 50 ms frames
    if frame_samples == 0:
        return None

    n_frames = len(sep_mono) // frame_samples
    if n_frames == 0:
        return None

    # Convert threshold from dB to linear RMS
    threshold_linear = 10.0 ** (threshold_db / 20.0)

    # Find silent frames in the separated vocal stem
    silent_indices = []
    for i in range(n_frames):
        start = i * frame_samples
        end = start + frame_samples
        frame = sep_mono[start:end]
        rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
        if rms < threshold_linear:
            silent_indices.append(i)

    if not silent_indices:
        return None

    # Extract corresponding regions from vocal_rec
    min_samples = int(min_duration_s * sample_rate)
    rec_len = vocal_rec.shape[0]
    segments = []
    collected = 0
    for i in silent_indices:
        start = i * frame_samples
        end = start + frame_samples
        if end > rec_len:
            break
        segments.append(vocal_rec[start:end])
        collected += frame_samples
        if collected >= min_samples:
            break  # enough noise collected

    if collected < min_samples:
        return None

    return np.concatenate(segments, axis=0)


def reduce_noise(
    data: np.ndarray,
    sample_rate: int,
    noise_clip: np.ndarray | None = None,
    strength: float = 1.0,
    guide_stem: np.ndarray | None = None,
    hpf_cutoff_hz: float = 0.0,
) -> np.ndarray:
    """Apply spectral-gating noise reduction to audio.

    Args:
        data: Audio array, shape (samples,) or (samples, channels), float32.
        sample_rate: Sample rate in Hz.
        noise_clip: Optional noise profile array (same channel layout as data).
            If None, auto-estimated from the start of data.
        strength: Noise reduction intensity from 0.0 (none) to 1.0 (maximum).
            Maps to the ``prop_decrease`` parameter of noisereduce.
        guide_stem: Optional separated vocal stem used to find silent regions
            for noise profiling.  Only used when noise_clip is None.
        hpf_cutoff_hz: High-pass filter cutoff in Hz.  0 = disabled.

    Returns:
        Noise-reduced audio, same shape and dtype (float32) as input.
    """
    if strength == 0.0 and hpf_cutoff_hz <= 0:
        return data

    # All-zeros or empty input — nothing to do
    if data.size == 0 or not np.any(data):
        return data.copy()

    # Apply high-pass filter before spectral gating
    if hpf_cutoff_hz > 0:
        data = high_pass_filter(data, sample_rate, cutoff_hz=hpf_cutoff_hz)

    if strength == 0.0:
        return data

    import noisereduce as nr

    if noise_clip is None and guide_stem is not None:
        noise_clip = estimate_noise_from_stem(data, guide_stem, sample_rate)

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
