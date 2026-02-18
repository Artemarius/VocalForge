"""Effects chain — 9-stage vocal processing pipeline.

Each effect function takes (data, sr, **params) and returns a same-shape
float32 array. The chain order follows VOCAL_ENHANCEMENT.md:

    1. Noise Gate        (RMS-envelope gating)
    2. Spectral NR       (wraps noise_reduction.reduce_noise)
    3. De-reverb         (spectral subtraction via transient detection)
    4. High-Pass Filter  (wraps noise_reduction.high_pass_filter)
    5. Parametric EQ     (stub)
    6. Compressor        (stub)
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
        "cutoff_hz": 80.0,
    },
    "parametric_eq": {
        "enabled": False,
        "stub": True,
    },
    "compressor": {
        "enabled": False,
        "stub": True,
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
        "ceiling_db": -1.0,
        "release_ms": 50.0,
    },
}


PRESET_CONFIGS = {
    "Raw": {name: {"enabled": False} for name in CHAIN_ORDER},
    "Clean": {
        "noise_gate": {"enabled": True, "threshold_db": -35.0, "attack_ms": 2.0,
                        "release_ms": 100.0, "hold_ms": 50.0, "reduction_db": -40.0},
        "spectral_noise_reduction": {"enabled": True, "strength": 0.75},
        "dereverb": {"enabled": False},
        "highpass_filter": {"enabled": True, "cutoff_hz": 80.0},
        "parametric_eq": {"enabled": False},
        "compressor": {"enabled": False},
        "de_esser": {"enabled": False},
        "reverb": {"enabled": False},
        "limiter": {"enabled": True, "ceiling_db": -1.0},
    },
    "Enhanced": {
        "noise_gate": {"enabled": True, "threshold_db": -35.0, "attack_ms": 2.0,
                        "release_ms": 100.0, "hold_ms": 50.0, "reduction_db": -40.0},
        "spectral_noise_reduction": {"enabled": True, "strength": 0.75},
        "dereverb": {"enabled": True, "strength": 0.5},
        "highpass_filter": {"enabled": True, "cutoff_hz": 80.0},
        "parametric_eq": {"enabled": False},
        "compressor": {"enabled": False},
        "de_esser": {"enabled": False},
        "reverb": {"enabled": False},
        "limiter": {"enabled": True, "ceiling_db": -1.0},
    },
}


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

    if strength == 0.0:
        return data

    from vocalforge.audio.noise_reduction import reduce_noise

    return reduce_noise(
        data, sr,
        strength=strength,
        guide_stem=guide_stem,
        hpf_cutoff_hz=0.0,
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


def parametric_eq(data: np.ndarray, sr: int, **params) -> np.ndarray:
    """Parametric EQ (stub — pass-through)."""
    return data


def compressor(data: np.ndarray, sr: int, **params) -> np.ndarray:
    """Compressor (stub — pass-through)."""
    return data


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
