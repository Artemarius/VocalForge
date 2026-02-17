"""Effects chain — 9-stage vocal processing pipeline.

Each effect function takes (data, sr, **params) and returns a same-shape
float32 array. The chain order follows VOCAL_ENHANCEMENT.md:

    1. Noise Gate        (stub)
    2. Spectral NR       (wraps noise_reduction.reduce_noise)
    3. De-reverb         (stub)
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
        "stub": True,
    },
    "spectral_noise_reduction": {
        "enabled": True,
        "stub": False,
        "strength": 0.75,
        "guide_stem": None,
    },
    "dereverb": {
        "enabled": False,
        "stub": True,
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
    """Noise gate (stub — pass-through)."""
    return data


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
    """De-reverb (stub — pass-through)."""
    return data


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
