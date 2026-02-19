"""Effects chain — 9-stage vocal processing pipeline.

Each effect function takes (data, sr, **params) and returns a same-shape
float32 array. The chain order follows VOCAL_ENHANCEMENT.md:

    1. Noise Gate        (RMS-envelope gating)
    2. Spectral NR       (wraps noise_reduction.reduce_noise)
    3. De-reverb         (spectral subtraction via transient detection)
    4. High-Pass Filter  (wraps noise_reduction.high_pass_filter)
    5. Parametric EQ     (biquad cascaded EQ with presets)
    6. Compressor        (RMS envelope, soft-knee, auto makeup)
    7. De-Esser          (stub)
    8. Reverb            (stub)
    9. Limiter           (real — brick-wall peak limiter)
"""

import copy

import numpy as np

# Processing chain order
CHAIN_ORDER = [
    "noise_gate",
    "spectral_noise_reduction",
    "dereverb",
    "highpass_filter",
    "parametric_eq",
    "compressor",
    "de_esser",
    "reverb",
    "limiter",
]

DEFAULT_CONFIG = {
    "noise_gate": {
        "enabled": False,
        "stub": False,
        "threshold_db": -35.0,
        "attack_ms": 2.0,
        "release_ms": 100.0,
        "hold_ms": 50.0,
        "reduction_db": -40.0,
    },
    "spectral_noise_reduction": {
        "enabled": True,
        "stub": False,
        "strength": 0.75,
        "guide_stem": None,
        "mode": "auto",
        "n_std_thresh": 1.5,
        "use_torch": None,
        "freq_smooth_hz": 500,
        "time_smooth_ms": 50,
    },
    "dereverb": {
        "enabled": False,
        "stub": False,
        "strength": 0.4,
        "frame_size": 2048,
    },
    "highpass_filter": {
        "enabled": True,
        "stub": False,
        "cutoff_hz": 100.0,
    },
    "parametric_eq": {
        "enabled": False,
        "stub": False,
        "preset": "bright",
        "bands": None,  # filled from EQ_PRESETS at import time (see below)
    },
    "compressor": {
        "enabled": False,
        "stub": False,
        "threshold_db": -18.0,
        "ratio": 3.0,
        "attack_ms": 15.0,
        "release_ms": 200.0,
        "knee_db": 6.0,
        "makeup_db": None,
        "mix": 1.0,
    },
    "de_esser": {
        "enabled": False,
        "stub": True,
    },
    "reverb": {
        "enabled": False,
        "stub": True,
    },
    "limiter": {
        "enabled": True,
        "stub": False,
        "ceiling_db": -0.7,
        "release_ms": 50.0,
    },
}


PRESET_CONFIGS = {
    "Raw": {name: {"enabled": False} for name in CHAIN_ORDER},
    "Clean": {
        "noise_gate": {"enabled": True, "threshold_db": -35.0, "attack_ms": 2.0,
                        "release_ms": 100.0, "hold_ms": 50.0, "reduction_db": -40.0},
        "spectral_noise_reduction": {"enabled": True, "strength": 0.75, "mode": "auto"},
        "dereverb": {"enabled": False},
        "highpass_filter": {"enabled": True, "cutoff_hz": 100.0},
        "parametric_eq": {"enabled": False},
        "compressor": {"enabled": False},
        "de_esser": {"enabled": False},
        "reverb": {"enabled": False},
        "limiter": {"enabled": True, "ceiling_db": -0.7},
    },
    "Enhanced": {
        "noise_gate": {"enabled": True, "threshold_db": -35.0, "attack_ms": 2.0,
                        "release_ms": 100.0, "hold_ms": 50.0, "reduction_db": -40.0},
        "spectral_noise_reduction": {"enabled": True, "strength": 0.75, "mode": "auto"},
        "dereverb": {"enabled": True, "strength": 0.5},
        "highpass_filter": {"enabled": True, "cutoff_hz": 100.0},
        "parametric_eq": {"enabled": True, "preset": "bright"},
        "compressor": {"enabled": True, "threshold_db": -18.0, "ratio": 3.0},
        "de_esser": {"enabled": False},
        "reverb": {"enabled": False},
        "limiter": {"enabled": True, "ceiling_db": -0.7},
    },
}


# --- EQ presets ---

EQ_PRESETS = {
    "clean_up": [
        {"freq_hz": 250.0, "gain_db": -2.5, "q": 1.5, "type": "peak"},
        {"freq_hz": 3500.0, "gain_db": 1.5, "q": 1.2, "type": "peak"},
        {"freq_hz": 10000.0, "gain_db": 1.0, "q": 0.7, "type": "high_shelf"},
    ],
    "warm": [
        {"freq_hz": 200.0, "gain_db": 1.5, "q": 0.7, "type": "low_shelf"},
        {"freq_hz": 800.0, "gain_db": -2.0, "q": 1.5, "type": "peak"},
        {"freq_hz": 4000.0, "gain_db": 1.0, "q": 1.2, "type": "peak"},
    ],
    "bright": [
        {"freq_hz": 300.0, "gain_db": -3.0, "q": 1.5, "type": "peak"},
        {"freq_hz": 3000.0, "gain_db": 2.5, "q": 1.2, "type": "peak"},
        {"freq_hz": 8000.0, "gain_db": 2.0, "q": 0.7, "type": "high_shelf"},
    ],
}

# Patch DEFAULT_CONFIG with actual EQ bands now that EQ_PRESETS is defined
DEFAULT_CONFIG["parametric_eq"]["bands"] = EQ_PRESETS["bright"]


def _merge_config(defaults: dict, overrides: dict | None) -> dict:
    """Deep-merge overrides into a copy of defaults."""
    if overrides is None:
        return copy.deepcopy(defaults)
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            for k2, v2 in value.items():
                result[key][k2] = v2
        else:
            result[key] = value
    return result


# --- Effect functions ---


def noise_gate(data: np.ndarray, sr: int, **params) -> np.ndarray:
    """RMS-envelope noise gate with hold time and attack/release smoothing.

    Silences the signal during pauses between vocal phrases by comparing
    RMS envelope to a threshold. Uses vectorized stride tricks for the
    RMS computation and an exponential envelope follower for smooth
    gain transitions.

    Args:
        data: Audio array, shape (samples,) or (samples, channels), float32.
        sr: Sample rate in Hz.
        threshold_db: Level below which the gate closes (dB).
        attack_ms: Gate opening speed (ms).
        release_ms: Gate closing speed (ms).
        hold_ms: Minimum time gate stays open after triggering (ms).
        reduction_db: Attenuation applied when gate is closed (dB).

    Returns:
        Gated audio, same shape and dtype as input.
    """
    threshold_db = params.get("threshold_db", -35.0)
    attack_ms = params.get("attack_ms", 2.0)
    release_ms = params.get("release_ms", 100.0)
    hold_ms = params.get("hold_ms", 50.0)
    reduction_db = params.get("reduction_db", -40.0)

    if data.size == 0:
        return data

    threshold_linear = 10.0 ** (threshold_db / 20.0)
    reduction_linear = 10.0 ** (reduction_db / 20.0)

    # --- Compute RMS envelope in ~20 ms windows (vectorized) ---
    window_size = max(1, int(sr * 0.02))  # 20 ms
    n_samples = data.shape[0]

    # Get mono signal for envelope detection
    if data.ndim == 1:
        mono = data
    else:
        mono = np.abs(data).max(axis=1)

    # Pad to make length divisible by window_size
    pad_len = (window_size - (n_samples % window_size)) % window_size
    padded = np.concatenate([mono, np.zeros(pad_len, dtype=mono.dtype)])

    # Block-level RMS via reshape (no Python loop)
    blocks = padded.reshape(-1, window_size)
    block_rms = np.sqrt(np.mean(blocks ** 2, axis=1))

    # Interpolate block-level RMS back to sample-level
    from scipy.interpolate import interp1d

    block_centers = np.arange(len(block_rms)) * window_size + window_size // 2
    block_centers = np.clip(block_centers, 0, n_samples - 1)
    interp_fn = interp1d(
        block_centers, block_rms, kind="linear",
        bounds_error=False, fill_value=(block_rms[0], block_rms[-1]),
    )
    rms_envelope = interp_fn(np.arange(n_samples)).astype(np.float64)

    # --- Gate open/close decision ---
    gate_open = rms_envelope >= threshold_linear  # bool array

    # --- Apply hold time ---
    hold_samples = max(1, int(hold_ms / 1000.0 * sr))
    # After each True, keep True for hold_samples
    if hold_samples > 1:
        from scipy.ndimage import maximum_filter1d
        gate_open = maximum_filter1d(
            gate_open.astype(np.float64), size=hold_samples,
            origin=-(hold_samples // 2),
        ) > 0.5

    # --- Build raw gain: 1.0 where open, reduction_linear where closed ---
    raw_gain = np.where(gate_open, 1.0, reduction_linear)

    # --- Smooth with attack/release envelope follower ---
    attack_samples = max(1, int(attack_ms / 1000.0 * sr))
    release_samples = max(1, int(release_ms / 1000.0 * sr))
    alpha_attack = 1.0 - np.exp(-1.0 / attack_samples)
    alpha_release = 1.0 - np.exp(-1.0 / release_samples)

    smoothed = np.empty(n_samples, dtype=np.float64)
    smoothed[0] = raw_gain[0]
    for i in range(1, n_samples):
        if raw_gain[i] > smoothed[i - 1]:
            # Opening (attack)
            smoothed[i] = smoothed[i - 1] + alpha_attack * (raw_gain[i] - smoothed[i - 1])
        else:
            # Closing (release)
            smoothed[i] = smoothed[i - 1] + alpha_release * (raw_gain[i] - smoothed[i - 1])

    # --- Apply gain ---
    if data.ndim == 1:
        result = data * smoothed.astype(np.float32)
    else:
        result = data * smoothed.astype(np.float32)[:, np.newaxis]

    return result.astype(np.float32)


def spectral_noise_reduction(data: np.ndarray, sr: int, **params) -> np.ndarray:
    """Spectral noise reduction wrapping noise_reduction.reduce_noise().

    Calls reduce_noise with hpf_cutoff_hz=0.0 so HPF is handled separately.
    """
    strength = params.get("strength", 0.75)
    guide_stem = params.get("guide_stem", None)
    mode = params.get("mode", "auto")
    n_std_thresh = params.get("n_std_thresh", 1.5)
    use_torch = params.get("use_torch", None)
    freq_smooth_hz = params.get("freq_smooth_hz", 500)
    time_smooth_ms = params.get("time_smooth_ms", 50)

    if strength == 0.0:
        return data

    from vocalforge.audio.noise_reduction import reduce_noise

    return reduce_noise(
        data, sr,
        strength=strength,
        guide_stem=guide_stem,
        hpf_cutoff_hz=0.0,
        mode=mode,
        n_std_thresh=n_std_thresh,
        use_torch=use_torch,
        freq_smooth_hz=freq_smooth_hz,
        time_smooth_ms=time_smooth_ms,
    )


def dereverb(data: np.ndarray, sr: int, **params) -> np.ndarray:
    """Spectral de-reverb using transient-to-steady-state ratio.

    Reduces room reverb by analyzing spectral flux (transient indicator)
    and building a mask that preserves transient energy while attenuating
    sustained/decaying reverb tails.

    Args:
        data: Audio array, shape (samples,) or (samples, channels), float32.
        sr: Sample rate in Hz.
        strength: De-reverb intensity, 0.0 (bypass) to 1.0 (maximum).
        frame_size: STFT frame size in samples.

    Returns:
        De-reverbed audio, same shape and dtype as input.
    """
    strength = params.get("strength", 0.4)
    frame_size = params.get("frame_size", 2048)

    if data.size == 0:
        return data

    if strength == 0.0:
        return data

    from scipy.ndimage import uniform_filter1d
    from scipy.signal import istft, stft

    hop_size = frame_size // 4
    original_len = data.shape[0]

    def _process_channel(signal: np.ndarray) -> np.ndarray:
        """Process a single mono channel through STFT de-reverb."""
        _f, _t, Zxx = stft(
            signal, fs=sr, nperseg=frame_size, noverlap=frame_size - hop_size,
        )
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Spectral flux: positive frame-to-frame magnitude change
        flux = np.zeros_like(magnitude)
        flux[:, 1:] = np.maximum(0, magnitude[:, 1:] - magnitude[:, :-1])

        # Normalize flux per frequency band
        flux_max = flux.max(axis=1, keepdims=True)
        flux_max = np.where(flux_max > 0, flux_max, 1.0)  # avoid div-by-zero
        flux_normalized = flux / flux_max

        # Build mask: preserve transients, attenuate sustained reverb
        mask = (1.0 - strength) + strength * flux_normalized
        mask = np.clip(mask, 0.1, 1.0)

        # Smooth mask temporally to avoid artifacts
        mask = uniform_filter1d(mask, size=3, axis=1)

        # Apply mask and reconstruct
        Zxx_processed = magnitude * mask * np.exp(1j * phase)
        _, audio_out = istft(
            Zxx_processed, fs=sr,
            nperseg=frame_size, noverlap=frame_size - hop_size,
        )
        return audio_out[:len(signal)].astype(np.float32)

    if data.ndim == 1:
        return _process_channel(data)

    # Stereo: process each channel independently
    channels = []
    for ch in range(data.shape[1]):
        channels.append(_process_channel(data[:, ch]))
    result = np.column_stack(channels)
    return result[:original_len].astype(np.float32)


def highpass_filter(data: np.ndarray, sr: int, **params) -> np.ndarray:
    """High-pass filter delegating to noise_reduction.high_pass_filter()."""
    cutoff_hz = params.get("cutoff_hz", 80.0)

    if cutoff_hz <= 0:
        return data

    from vocalforge.audio.noise_reduction import high_pass_filter as _hpf

    return _hpf(data, sr, cutoff_hz=cutoff_hz)


def _biquad_peak(freq_hz: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """Audio EQ Cookbook peaking EQ filter → SOS row shape (1, 6)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _biquad_shelf(freq_hz: float, gain_db: float, q: float, sr: int,
                  shelf_type: str) -> np.ndarray:
    """Audio EQ Cookbook low/high shelf filter → SOS row shape (1, 6)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    two_sqrt_A_alpha = 2.0 * np.sqrt(A) * alpha

    if shelf_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = 2.0 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = -2.0 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha
    else:  # high_shelf
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha)
        b1 = -2.0 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha
        a1 = 2.0 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def parametric_eq(data: np.ndarray, sr: int, **params) -> np.ndarray:
    """Parametric EQ — cascaded biquad filters with zero-phase filtering.

    Builds a second-order sections (SOS) array from all active bands and
    applies them via ``scipy.signal.sosfiltfilt`` for zero-phase response.

    Args:
        data: Audio array, shape (samples,) or (samples, channels), float32.
        sr: Sample rate in Hz.
        bands: List of band dicts with keys freq_hz, gain_db, q, type.
        preset: Name of the active preset (informational, not used by DSP).

    Returns:
        EQ'd audio, same shape and dtype as input.
    """
    bands = params.get("bands", [])

    if data.size == 0 or not bands:
        return data

    # Build cascaded SOS array from all active bands
    sos_sections = []
    for band in bands:
        gain_db = band.get("gain_db", 0.0)
        if gain_db == 0.0:
            continue
        freq_hz = band["freq_hz"]
        q = band.get("q", 1.0)
        band_type = band.get("type", "peak")

        if band_type == "peak":
            sos_sections.append(_biquad_peak(freq_hz, gain_db, q, sr))
        elif band_type in ("low_shelf", "high_shelf"):
            sos_sections.append(_biquad_shelf(freq_hz, gain_db, q, sr, band_type))

    if not sos_sections:
        return data

    from scipy.signal import sosfiltfilt

    sos = np.vstack(sos_sections)

    if data.ndim == 1:
        return sosfiltfilt(sos, data).astype(np.float32)

    # Stereo: filter each channel independently
    channels = []
    for ch in range(data.shape[1]):
        channels.append(sosfiltfilt(sos, data[:, ch]).astype(np.float32))
    return np.column_stack(channels)


def compressor(data: np.ndarray, sr: int, **params) -> np.ndarray:
    """Dynamic range compressor with soft knee and auto makeup gain.

    Uses RMS envelope detection in 10ms windows, a soft-knee gain curve,
    and exponential attack/release smoothing. Supports parallel compression
    via the ``mix`` parameter.

    Args:
        data: Audio array, shape (samples,) or (samples, channels), float32.
        sr: Sample rate in Hz.
        threshold_db: Level above which compression starts (dB).
        ratio: Compression ratio (e.g. 3.0 = 3:1).
        attack_ms: Attack time (ms) — how fast gain reduction engages.
        release_ms: Release time (ms) — how fast gain reduction releases.
        knee_db: Soft knee width in dB (0 = hard knee).
        makeup_db: Manual makeup gain (None = auto).
        mix: Dry/wet blend (1.0 = fully compressed, 0.0 = bypass).

    Returns:
        Compressed audio, same shape and dtype as input.
    """
    threshold_db = params.get("threshold_db", -18.0)
    ratio = params.get("ratio", 3.0)
    attack_ms = params.get("attack_ms", 15.0)
    release_ms = params.get("release_ms", 200.0)
    knee_db = params.get("knee_db", 6.0)
    makeup_db = params.get("makeup_db", None)
    mix = params.get("mix", 1.0)

    if data.size == 0 or ratio <= 1.0:
        return data

    from scipy.interpolate import interp1d

    n_samples = data.shape[0]

    # --- Compute RMS envelope in 10ms windows ---
    window_size = max(1, int(sr * 0.01))

    # Mono envelope from peak across channels
    if data.ndim == 1:
        mono = np.abs(data)
    else:
        mono = np.abs(data).max(axis=1)

    # Pad to make length divisible by window_size
    pad_len = (window_size - (n_samples % window_size)) % window_size
    padded = np.concatenate([mono, np.zeros(pad_len, dtype=mono.dtype)])

    # Block-level RMS
    blocks = padded.reshape(-1, window_size)
    block_rms = np.sqrt(np.mean(blocks ** 2, axis=1))

    # Convert to dB (floor at -120 dB to avoid log(0))
    block_db = 20.0 * np.log10(np.maximum(block_rms, 1e-6))

    # --- Soft-knee gain curve (dB domain) ---
    half_knee = knee_db / 2.0
    gain_reduction_db = np.zeros_like(block_db)

    for i, level_db in enumerate(block_db):
        if level_db < threshold_db - half_knee:
            # Below knee — no compression
            gain_reduction_db[i] = 0.0
        elif level_db > threshold_db + half_knee:
            # Above knee — full compression
            over = level_db - threshold_db
            gain_reduction_db[i] = over - over / ratio
        else:
            # In knee — quadratic interpolation
            x = level_db - (threshold_db - half_knee)
            gain_reduction_db[i] = ((1.0 / ratio - 1.0) * x * x) / (2.0 * knee_db) if knee_db > 0 else 0.0

    # --- Attack/release smoothing on block-level gain reduction ---
    block_duration_s = window_size / sr
    alpha_attack = 1.0 - np.exp(-block_duration_s / max(attack_ms / 1000.0, 1e-6))
    alpha_release = 1.0 - np.exp(-block_duration_s / max(release_ms / 1000.0, 1e-6))

    smoothed_gr = np.empty_like(gain_reduction_db)
    smoothed_gr[0] = gain_reduction_db[0]
    for i in range(1, len(gain_reduction_db)):
        if gain_reduction_db[i] > smoothed_gr[i - 1]:
            # Gain reduction increasing (attack)
            smoothed_gr[i] = smoothed_gr[i - 1] + alpha_attack * (
                gain_reduction_db[i] - smoothed_gr[i - 1])
        else:
            # Gain reduction decreasing (release)
            smoothed_gr[i] = smoothed_gr[i - 1] + alpha_release * (
                gain_reduction_db[i] - smoothed_gr[i - 1])

    # --- Interpolate block-level gain to sample-level ---
    block_centers = np.arange(len(smoothed_gr)) * window_size + window_size // 2
    block_centers = np.clip(block_centers, 0, n_samples - 1)
    interp_fn = interp1d(
        block_centers, smoothed_gr, kind="linear",
        bounds_error=False, fill_value=(smoothed_gr[0], smoothed_gr[-1]),
    )
    sample_gr_db = interp_fn(np.arange(n_samples))

    # --- Auto makeup gain: compensate 70% of average gain reduction ---
    if makeup_db is None:
        avg_gr = np.mean(sample_gr_db)
        makeup_db = avg_gr * 0.7
    else:
        makeup_db = float(makeup_db)

    # --- Apply gain ---
    total_gain_db = -sample_gr_db + makeup_db
    gain_linear = (10.0 ** (total_gain_db / 20.0)).astype(np.float32)

    if data.ndim == 1:
        compressed = data * gain_linear
    else:
        compressed = data * gain_linear[:, np.newaxis]

    # --- Parallel compression (dry/wet blend) ---
    if mix < 1.0:
        result = data * (1.0 - mix) + compressed * mix
    else:
        result = compressed

    return result.astype(np.float32)


def de_esser(data: np.ndarray, sr: int, **params) -> np.ndarray:
    """De-esser (stub — pass-through)."""
    return data


def reverb(data: np.ndarray, sr: int, **params) -> np.ndarray:
    """Reverb (stub — pass-through)."""
    return data


def limiter(data: np.ndarray, sr: int, **params) -> np.ndarray:
    """Brick-wall peak limiter with exponential release smoothing.

    Uses scipy.ndimage.maximum_filter1d for efficient peak detection,
    then applies gain reduction with exponential release to avoid
    hard clipping artifacts.

    Args:
        data: Audio array, shape (samples,) or (samples, channels), float32.
        sr: Sample rate in Hz.
        ceiling_db: Maximum output level in dB (e.g. -1.0).
        release_ms: Release time constant in ms.

    Returns:
        Limited audio, same shape and dtype as input.
    """
    ceiling_db = params.get("ceiling_db", -1.0)
    release_ms = params.get("release_ms", 50.0)

    ceiling_linear = 10.0 ** (ceiling_db / 20.0)

    if data.size == 0:
        return data

    from scipy.ndimage import maximum_filter1d

    # Work with absolute peak across channels
    if data.ndim == 1:
        abs_data = np.abs(data)
    else:
        abs_data = np.abs(data).max(axis=1)

    # Lookahead window: ~5ms for transparent limiting
    lookahead_samples = max(1, int(0.005 * sr))
    peak_env = maximum_filter1d(abs_data, size=lookahead_samples * 2 + 1)

    # Compute required gain reduction
    gain = np.ones(len(peak_env), dtype=np.float64)
    mask = peak_env > ceiling_linear
    gain[mask] = ceiling_linear / peak_env[mask]

    # Smooth the gain envelope with exponential release
    release_samples = max(1, int(release_ms / 1000.0 * sr))
    alpha = 1.0 - np.exp(-1.0 / release_samples)

    smoothed = np.empty_like(gain)
    smoothed[0] = gain[0]
    for i in range(1, len(gain)):
        if gain[i] < smoothed[i - 1]:
            # Attack: instant
            smoothed[i] = gain[i]
        else:
            # Release: exponential
            smoothed[i] = smoothed[i - 1] + alpha * (gain[i] - smoothed[i - 1])

    # Apply gain
    if data.ndim == 1:
        result = data * smoothed.astype(np.float32)
    else:
        result = data * smoothed.astype(np.float32)[:, np.newaxis]

    return result.astype(np.float32)


# --- Effect dispatch table ---

_EFFECT_FUNCS = {
    "noise_gate": noise_gate,
    "spectral_noise_reduction": spectral_noise_reduction,
    "dereverb": dereverb,
    "highpass_filter": highpass_filter,
    "parametric_eq": parametric_eq,
    "compressor": compressor,
    "de_esser": de_esser,
    "reverb": reverb,
    "limiter": limiter,
}


def process_vocal(data: np.ndarray, sr: int, config: dict | None = None) -> np.ndarray:
    """Run all enabled effects in chain order.

    Args:
        data: Vocal audio, float32.
        sr: Sample rate in Hz.
        config: Override dict merged on top of DEFAULT_CONFIG. Use
            ``{"effect_name": {"enabled": False}}`` to disable a stage.

    Returns:
        Processed audio, same shape as input.
    """
    cfg = _merge_config(DEFAULT_CONFIG, config)

    result = data
    for name in CHAIN_ORDER:
        effect_cfg = cfg.get(name, {})
        if not effect_cfg.get("enabled", False):
            continue
        func = _EFFECT_FUNCS[name]
        # Build kwargs: everything except 'enabled' and 'stub'
        params = {k: v for k, v in effect_cfg.items() if k not in ("enabled", "stub")}
        result = func(result, sr, **params)

    # Final NaN/inf sanitization (guards against broken effect outputs)
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result
