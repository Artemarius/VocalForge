"""Tests for vocalforge.audio.effects."""

import numpy as np
import pytest

from vocalforge.audio.effects import (
    CHAIN_ORDER,
    DEFAULT_CONFIG,
    PRESET_CONFIGS,
    _merge_config,
    compressor,
    de_esser,
    dereverb,
    highpass_filter,
    limiter,
    noise_gate,
    parametric_eq,
    process_vocal,
    reverb,
    spectral_noise_reduction,
)

SR = 44100


def _make_tone(duration_s=1.0, freq=440, amplitude=0.5, sr=SR):
    t = np.arange(int(sr * duration_s)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_loud_signal(duration_s=0.5, sr=SR):
    """Signal with peaks exceeding 1.0."""
    t = np.arange(int(sr * duration_s)) / sr
    return (2.0 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# --- Stub pass-through tests ---


class TestStubs:
    """Each stubbed effect returns input unchanged."""

    @pytest.mark.parametrize("func", [parametric_eq, compressor,
                                       de_esser, reverb])
    def test_stub_passthrough_mono(self, func):
        data = _make_tone()
        result = func(data, SR)
        np.testing.assert_array_equal(result, data)

    @pytest.mark.parametrize("func", [parametric_eq, compressor,
                                       de_esser, reverb])
    def test_stub_passthrough_stereo(self, func):
        mono = _make_tone()
        data = np.column_stack([mono, mono])
        result = func(data, SR)
        np.testing.assert_array_equal(result, data)


# --- Noise Gate tests ---


class TestNoiseGate:

    def test_silences_gaps(self):
        """Gate should attenuate silent regions below threshold."""
        # Build signal: 0.2s loud burst, 0.3s silence, 0.2s loud burst
        burst = _make_tone(duration_s=0.2, amplitude=0.5)
        silence = np.zeros(int(SR * 0.3), dtype=np.float32)
        data = np.concatenate([burst, silence, burst])

        result = noise_gate(data, SR, threshold_db=-30.0, reduction_db=-60.0,
                            hold_ms=10.0, release_ms=20.0)

        # The silent gap region (middle third) should be heavily attenuated
        gap_start = len(burst) + int(SR * 0.05)  # skip transition
        gap_end = len(burst) + len(silence) - int(SR * 0.05)
        gap_rms = np.sqrt(np.mean(result[gap_start:gap_end] ** 2))
        assert gap_rms < 0.01, f"Gap RMS {gap_rms} should be near zero"

    def test_preserves_loud_signal(self):
        """Continuous loud tone should pass through mostly unchanged."""
        data = _make_tone(duration_s=0.5, amplitude=0.5)
        result = noise_gate(data, SR, threshold_db=-40.0)
        # Output should be close to input (gate stays open)
        corr = np.corrcoef(data, result)[0, 1]
        assert corr > 0.99, f"Correlation {corr} too low — gate damaged signal"

    def test_preserves_shape_mono(self):
        data = _make_tone(duration_s=0.3)
        result = noise_gate(data, SR, threshold_db=-35.0)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_preserves_shape_stereo(self):
        mono = _make_tone(duration_s=0.3)
        data = np.column_stack([mono, mono])
        result = noise_gate(data, SR, threshold_db=-35.0)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_empty_input(self):
        data = np.array([], dtype=np.float32)
        result = noise_gate(data, SR)
        assert len(result) == 0

    def test_full_mute_reduction(self):
        """Very low reduction_db should make silent parts near zero."""
        silence = np.full(int(SR * 0.2), 1e-5, dtype=np.float32)
        result = noise_gate(silence, SR, threshold_db=-20.0,
                            reduction_db=-80.0, release_ms=5.0)
        assert np.abs(result).max() < 0.001


# --- De-Reverb tests ---


class TestDereverb:

    def test_reduces_reverb_tail(self):
        """Impulse with simulated decay tail should have tail energy reduced."""
        n = int(SR * 0.5)
        signal = np.zeros(n, dtype=np.float32)
        # Sharp impulse at the start
        signal[100:200] = 0.8
        # Simulated reverb tail: exponential decay
        t_tail = np.arange(n - 200) / SR
        signal[200:] = 0.3 * np.exp(-5.0 * t_tail).astype(np.float32)

        result = dereverb(signal, SR, strength=0.7, frame_size=1024)

        # Tail energy should be reduced
        tail_start = int(SR * 0.1)
        tail_energy_before = np.sum(signal[tail_start:] ** 2)
        tail_energy_after = np.sum(result[tail_start:] ** 2)
        assert tail_energy_after < tail_energy_before, \
            "De-reverb should reduce tail energy"

    def test_passthrough_at_zero_strength(self):
        """strength=0 should return input unchanged."""
        data = _make_tone(duration_s=0.5)
        result = dereverb(data, SR, strength=0.0)
        np.testing.assert_array_equal(result, data)

    def test_preserves_shape_mono(self):
        data = _make_tone(duration_s=0.3)
        result = dereverb(data, SR, strength=0.4)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_preserves_shape_stereo(self):
        mono = _make_tone(duration_s=0.3)
        data = np.column_stack([mono, mono])
        result = dereverb(data, SR, strength=0.4)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_empty_input(self):
        data = np.array([], dtype=np.float32)
        result = dereverb(data, SR, strength=0.5)
        assert len(result) == 0


# --- Preset tests ---


class TestPresets:

    def test_preset_definitions_valid(self):
        """All presets must contain all 9 chain keys with an 'enabled' key."""
        for name, preset in PRESET_CONFIGS.items():
            for chain_key in CHAIN_ORDER:
                assert chain_key in preset, \
                    f"Preset '{name}' missing key '{chain_key}'"
                assert "enabled" in preset[chain_key], \
                    f"Preset '{name}' key '{chain_key}' missing 'enabled'"


# --- Limiter tests ---


class TestLimiter:

    def test_prevents_clipping(self):
        data = _make_loud_signal()
        ceiling_db = -1.0
        ceiling_lin = 10.0 ** (ceiling_db / 20.0)
        result = limiter(data, SR, ceiling_db=ceiling_db)
        assert result.max() <= ceiling_lin + 1e-4
        assert result.min() >= -ceiling_lin - 1e-4

    def test_passthrough_below_ceiling(self):
        data = _make_tone(amplitude=0.1)
        result = limiter(data, SR, ceiling_db=0.0)
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_preserves_shape_mono(self):
        data = _make_loud_signal()
        result = limiter(data, SR)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_preserves_shape_stereo(self):
        mono = _make_loud_signal()
        data = np.column_stack([mono, mono])
        result = limiter(data, SR)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_empty_input(self):
        data = np.array([], dtype=np.float32)
        result = limiter(data, SR)
        assert len(result) == 0

    def test_ceiling_zero_db(self):
        """0 dB ceiling = 1.0 linear."""
        data = _make_loud_signal()
        result = limiter(data, SR, ceiling_db=0.0)
        assert result.max() <= 1.0 + 1e-4


# --- Highpass filter tests ---


class TestHighpassFilter:

    def test_removes_low_freq(self):
        low = _make_tone(freq=30, amplitude=0.5)
        result = highpass_filter(low, SR, cutoff_hz=100.0)
        # Significant attenuation of sub-cutoff content
        assert np.abs(result).max() < np.abs(low).max() * 0.5

    def test_zero_cutoff_passthrough(self):
        data = _make_tone()
        result = highpass_filter(data, SR, cutoff_hz=0.0)
        np.testing.assert_array_equal(result, data)


# --- process_vocal chain tests ---


class TestProcessVocal:

    def test_default_config_runs(self):
        data = _make_tone(duration_s=2.0)
        result = process_vocal(data, SR)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_all_disabled(self):
        data = _make_tone()
        config = {name: {"enabled": False} for name in CHAIN_ORDER}
        result = process_vocal(data, SR, config=config)
        np.testing.assert_array_equal(result, data)

    def test_partial_config_merge(self):
        """Overriding one effect preserves defaults for others."""
        config = {
            "highpass_filter": {"enabled": False},
        }
        merged = _merge_config(DEFAULT_CONFIG, config)
        assert merged["highpass_filter"]["enabled"] is False
        assert merged["spectral_noise_reduction"]["enabled"] is True
        assert merged["limiter"]["enabled"] is True

    def test_limiter_only(self):
        data = _make_loud_signal()
        config = {name: {"enabled": False} for name in CHAIN_ORDER}
        config["limiter"] = {"enabled": True, "ceiling_db": -1.0}
        result = process_vocal(data, SR, config=config)
        ceiling_lin = 10.0 ** (-1.0 / 20.0)
        assert result.max() <= ceiling_lin + 1e-4

    def test_chain_order_count(self):
        assert len(CHAIN_ORDER) == 9

    def test_all_effects_in_dispatch(self):
        from vocalforge.audio.effects import _EFFECT_FUNCS
        for name in CHAIN_ORDER:
            assert name in _EFFECT_FUNCS
