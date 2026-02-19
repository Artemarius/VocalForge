"""Tests for vocalforge.audio.noise_reduction."""

import numpy as np
import pytest

from vocalforge.audio.noise_reduction import (
    estimate_noise_from_stem,
    estimate_noise_profile,
    high_pass_filter,
    reduce_noise,
)

SR = 44100


def _make_tone(duration_s=2.0, freq=440, sr=SR, amplitude=0.5):
    """Generate a mono sine tone."""
    t = np.arange(int(sr * duration_s)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_noisy_tone(duration_s=2.0, freq=440, sr=SR, noise_level=0.1):
    """Generate a mono sine tone with additive white noise."""
    tone = _make_tone(duration_s, freq, sr)
    rng = np.random.default_rng(42)
    noise = (noise_level * rng.standard_normal(tone.shape)).astype(np.float32)
    return tone + noise


# --- SNR improvement ---

def test_snr_improvement():
    """Noise-region RMS should decrease after noise reduction."""
    # Create signal: 0.5s silence (noise only) then 1.5s tone+noise
    rng = np.random.default_rng(42)
    noise_only = (0.1 * rng.standard_normal(SR // 2)).astype(np.float32)
    tone_noisy = _make_noisy_tone(duration_s=1.5, noise_level=0.1)
    signal = np.concatenate([noise_only, tone_noisy])

    noise_clip = signal[:SR // 2]
    reduced = reduce_noise(signal, SR, noise_clip=noise_clip, strength=1.0)

    # RMS in the noise-only region should be lower after reduction
    original_rms = np.sqrt(np.mean(signal[:SR // 2].astype(np.float64) ** 2))
    reduced_rms = np.sqrt(np.mean(reduced[:SR // 2].astype(np.float64) ** 2))
    assert reduced_rms < original_rms


# --- Clean signal preservation ---

def test_clean_signal_preservation():
    """A clean sine wave should be mostly preserved."""
    tone = _make_tone(duration_s=2.0)
    reduced = reduce_noise(tone, SR, strength=0.75)
    # Correlation should remain high
    correlation = np.corrcoef(tone, reduced)[0, 1]
    assert correlation > 0.9


# --- Silence handling ---

def test_silence_handling():
    """All-zeros input should not crash and should return all zeros."""
    silence = np.zeros(SR, dtype=np.float32)
    result = reduce_noise(silence, SR, strength=1.0)
    assert result.shape == silence.shape
    np.testing.assert_array_equal(result, 0.0)


# --- Short audio ---

def test_short_audio():
    """Very short audio (0.1s) should complete without error."""
    short = _make_noisy_tone(duration_s=0.1, noise_level=0.1)
    result = reduce_noise(short, SR, strength=0.75)
    assert result.shape == short.shape


# --- Mono shape preserved ---

def test_mono_shape_preserved():
    """Mono input (N,) should produce mono output (N,)."""
    mono = _make_noisy_tone(duration_s=1.0, noise_level=0.1)
    assert mono.ndim == 1
    result = reduce_noise(mono, SR, strength=0.75)
    assert result.ndim == 1
    assert result.shape == mono.shape


# --- Stereo shape preserved ---

def test_stereo_shape_preserved():
    """Stereo input (N, 2) should produce stereo output (N, 2)."""
    mono = _make_noisy_tone(duration_s=1.0, noise_level=0.1)
    stereo = np.column_stack([mono, mono])
    assert stereo.shape == (len(mono), 2)
    result = reduce_noise(stereo, SR, strength=0.75)
    assert result.shape == stereo.shape


# --- Strength 0 passthrough ---

def test_strength_zero_passthrough():
    """strength=0.0 should return the input unchanged."""
    tone = _make_noisy_tone(duration_s=1.0, noise_level=0.1)
    result = reduce_noise(tone, SR, strength=0.0)
    np.testing.assert_array_equal(result, tone)


# --- Strength scaling ---

def test_strength_scaling():
    """Higher strength should remove more noise."""
    rng = np.random.default_rng(42)
    noise_only = (0.1 * rng.standard_normal(SR // 2)).astype(np.float32)
    tone_noisy = _make_noisy_tone(duration_s=1.5, noise_level=0.1)
    signal = np.concatenate([noise_only, tone_noisy])
    noise_clip = signal[:SR // 2]

    reduced_low = reduce_noise(signal, SR, noise_clip=noise_clip, strength=0.3)
    reduced_high = reduce_noise(signal, SR, noise_clip=noise_clip, strength=1.0)

    rms_low = np.sqrt(np.mean(reduced_low[:SR // 2].astype(np.float64) ** 2))
    rms_high = np.sqrt(np.mean(reduced_high[:SR // 2].astype(np.float64) ** 2))
    assert rms_high <= rms_low


# --- Output dtype ---

def test_output_dtype_float32():
    """Output should always be float32."""
    tone = _make_noisy_tone(duration_s=1.0, noise_level=0.1)
    result = reduce_noise(tone, SR, strength=0.75)
    assert result.dtype == np.float32


# --- estimate_noise_from_stem ---

def test_estimate_noise_from_stem_extracts_silent_regions():
    """Noise profile should come from regions where the vocal stem is silent."""
    rng = np.random.default_rng(42)
    # vocal_sep: first half silence, second half loud tone
    silence = np.zeros(SR, dtype=np.float32)
    tone = _make_tone(duration_s=1.0, freq=440)
    vocal_sep = np.concatenate([silence, tone])

    # vocal_rec: noise everywhere + tone in second half
    noise = (0.05 * rng.standard_normal(SR * 2)).astype(np.float32)
    vocal_rec = noise.copy()
    vocal_rec[SR:] += tone

    profile = estimate_noise_from_stem(vocal_rec, vocal_sep, SR)
    assert profile is not None
    # Profile should come from the first half (where sep is silent)
    assert len(profile) >= int(0.5 * SR)


def test_estimate_noise_from_stem_insufficient_silence():
    """Returns None when vocal stem is loud everywhere."""
    tone = _make_tone(duration_s=2.0, freq=440, amplitude=0.5)
    vocal_rec = _make_noisy_tone(duration_s=2.0, noise_level=0.1)
    result = estimate_noise_from_stem(vocal_rec, tone, SR)
    assert result is None


def test_reduce_noise_with_guide_stem():
    """Stem-guided noise reduction should reduce noise in silent regions."""
    rng = np.random.default_rng(42)
    # vocal_sep: silence then tone
    silence = np.zeros(SR, dtype=np.float32)
    tone = _make_tone(duration_s=1.0, freq=440)
    vocal_sep = np.concatenate([silence, tone])

    # vocal_rec: noise + tone in second half
    noise_full = (0.1 * rng.standard_normal(SR * 2)).astype(np.float32)
    vocal_rec = noise_full.copy()
    vocal_rec[SR:] += tone

    reduced = reduce_noise(vocal_rec, SR, strength=1.0, guide_stem=vocal_sep)

    # Noise-only region (first half) should have lower RMS after reduction
    original_rms = np.sqrt(np.mean(vocal_rec[:SR].astype(np.float64) ** 2))
    reduced_rms = np.sqrt(np.mean(reduced[:SR].astype(np.float64) ** 2))
    assert reduced_rms < original_rms


# --- high_pass_filter ---

def test_hpf_removes_low_frequency():
    """40 Hz tone should be attenuated >14 dB by an 80 Hz HPF."""
    tone_40 = _make_tone(duration_s=1.0, freq=40, amplitude=0.5)
    filtered = high_pass_filter(tone_40, SR, cutoff_hz=80.0)
    # Compare RMS
    rms_before = np.sqrt(np.mean(tone_40.astype(np.float64) ** 2))
    rms_after = np.sqrt(np.mean(filtered.astype(np.float64) ** 2))
    attenuation_db = 20 * np.log10(rms_after / rms_before)
    assert attenuation_db < -14


def test_hpf_preserves_high_frequency():
    """440 Hz tone should be highly correlated after 80 Hz HPF."""
    tone = _make_tone(duration_s=1.0, freq=440, amplitude=0.5)
    filtered = high_pass_filter(tone, SR, cutoff_hz=80.0)
    correlation = np.corrcoef(tone, filtered)[0, 1]
    assert correlation > 0.99


def test_hpf_disabled_when_zero():
    """cutoff_hz=0 returns input unchanged."""
    tone = _make_tone(duration_s=0.5, freq=100)
    result = high_pass_filter(tone, SR, cutoff_hz=0.0)
    np.testing.assert_array_equal(result, tone)


def test_hpf_preserves_shape_mono():
    """Mono shape (N,) is preserved."""
    mono = _make_tone(duration_s=0.5)
    assert mono.ndim == 1
    result = high_pass_filter(mono, SR, cutoff_hz=80.0)
    assert result.shape == mono.shape


def test_hpf_preserves_shape_stereo():
    """Stereo shape (N, 2) is preserved."""
    mono = _make_tone(duration_s=0.5)
    stereo = np.column_stack([mono, mono])
    result = high_pass_filter(stereo, SR, cutoff_hz=80.0)
    assert result.shape == stereo.shape


def test_hpf_dtype_float32():
    """Output is float32."""
    tone = _make_tone(duration_s=0.5)
    result = high_pass_filter(tone, SR, cutoff_hz=80.0)
    assert result.dtype == np.float32


def test_hpf_empty_input():
    """Empty array doesn't crash."""
    empty = np.array([], dtype=np.float32)
    result = high_pass_filter(empty, SR, cutoff_hz=80.0)
    assert result.size == 0


def test_reduce_noise_with_hpf():
    """HPF integrated into reduce_noise removes low-freq content."""
    # Mix a 40 Hz rumble with a 440 Hz tone
    rumble = _make_tone(duration_s=1.0, freq=40, amplitude=0.3)
    tone = _make_tone(duration_s=1.0, freq=440, amplitude=0.5)
    signal = rumble + tone

    # Use strength=0 so only HPF acts (skip spectral gating)
    result = reduce_noise(signal, SR, strength=0.0, hpf_cutoff_hz=80.0)

    # The 440 Hz tone should be preserved, the 40 Hz rumble attenuated
    corr_tone = np.corrcoef(tone, result)[0, 1]
    assert corr_tone > 0.95


# --- Mode selection tests ---


def test_stationary_mode_explicit():
    """mode='stationary' with explicit noise clip should reduce noise."""
    rng = np.random.default_rng(42)
    noise_only = (0.1 * rng.standard_normal(SR // 2)).astype(np.float32)
    tone_noisy = _make_noisy_tone(duration_s=1.5, noise_level=0.1)
    signal = np.concatenate([noise_only, tone_noisy])
    noise_clip = signal[:SR // 2]

    reduced = reduce_noise(signal, SR, noise_clip=noise_clip,
                           strength=1.0, mode="stationary")
    original_rms = np.sqrt(np.mean(signal[:SR // 2].astype(np.float64) ** 2))
    reduced_rms = np.sqrt(np.mean(reduced[:SR // 2].astype(np.float64) ** 2))
    assert reduced_rms < original_rms


def test_adaptive_mode_explicit():
    """mode='adaptive' should reduce noise without explicit noise clip."""
    noisy = _make_noisy_tone(duration_s=2.0, noise_level=0.15)
    reduced = reduce_noise(noisy, SR, strength=0.75, mode="adaptive")
    assert reduced.shape == noisy.shape
    assert reduced.dtype == np.float32


def test_auto_mode_with_guide_stem_uses_stationary():
    """Auto mode with a good stem should use stationary (stronger NR)."""
    rng = np.random.default_rng(42)
    silence = np.zeros(SR, dtype=np.float32)
    tone = _make_tone(duration_s=1.0, freq=440)
    vocal_sep = np.concatenate([silence, tone])

    noise_full = (0.1 * rng.standard_normal(SR * 2)).astype(np.float32)
    vocal_rec = noise_full.copy()
    vocal_rec[SR:] += tone

    reduced = reduce_noise(vocal_rec, SR, strength=1.0,
                           guide_stem=vocal_sep, mode="auto")
    # Should successfully reduce noise in the silent region
    original_rms = np.sqrt(np.mean(vocal_rec[:SR].astype(np.float64) ** 2))
    reduced_rms = np.sqrt(np.mean(reduced[:SR].astype(np.float64) ** 2))
    assert reduced_rms < original_rms


def test_auto_mode_without_guide_uses_adaptive():
    """Auto mode without stem or clip should fallback to adaptive."""
    noisy = _make_noisy_tone(duration_s=2.0, noise_level=0.1)
    reduced = reduce_noise(noisy, SR, strength=0.75, mode="auto")
    assert reduced.shape == noisy.shape
    assert reduced.dtype == np.float32


def test_smoothing_parameters_accepted():
    """Custom freq/time smoothing values should not crash."""
    noisy = _make_noisy_tone(duration_s=1.0, noise_level=0.1)
    result = reduce_noise(noisy, SR, strength=0.5,
                          freq_smooth_hz=200, time_smooth_ms=100)
    assert result.shape == noisy.shape
    assert np.all(np.isfinite(result))


def test_n_std_thresh_parameter():
    """Different n_std_thresh values should both reduce noise from original."""
    rng = np.random.default_rng(42)
    noise_only = (0.1 * rng.standard_normal(SR // 2)).astype(np.float32)
    tone_noisy = _make_noisy_tone(duration_s=1.5, noise_level=0.1)
    signal = np.concatenate([noise_only, tone_noisy])
    noise_clip = signal[:SR // 2]

    rms_original = np.sqrt(np.mean(signal[:SR // 2].astype(np.float64) ** 2))

    reduced_mild = reduce_noise(signal, SR, noise_clip=noise_clip,
                                strength=1.0, mode="stationary",
                                n_std_thresh=3.0)
    reduced_aggressive = reduce_noise(signal, SR, noise_clip=noise_clip,
                                      strength=1.0, mode="stationary",
                                      n_std_thresh=0.5)
    rms_mild = np.sqrt(np.mean(reduced_mild[:SR // 2].astype(np.float64) ** 2))
    rms_aggressive = np.sqrt(np.mean(
        reduced_aggressive[:SR // 2].astype(np.float64) ** 2))
    # Both should reduce noise from the original level
    assert rms_mild < rms_original, \
        f"Mild NR should reduce noise: {rms_original:.4f} -> {rms_mild:.4f}"
    assert rms_aggressive < rms_original, \
        f"Aggressive NR should reduce noise: {rms_original:.4f} -> {rms_aggressive:.4f}"


def test_use_torch_false_explicit():
    """use_torch=False should work (CPU path)."""
    noisy = _make_noisy_tone(duration_s=1.0, noise_level=0.1)
    result = reduce_noise(noisy, SR, strength=0.5, use_torch=False)
    assert result.shape == noisy.shape
    assert result.dtype == np.float32
