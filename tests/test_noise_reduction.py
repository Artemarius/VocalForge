"""Tests for vocalforge.audio.noise_reduction."""

import numpy as np
import pytest

from vocalforge.audio.noise_reduction import estimate_noise_profile, reduce_noise

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
