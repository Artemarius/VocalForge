"""Tests for vocalforge.audio.effects."""

import numpy as np
import pytest

from vocalforge.audio.effects import (
    CHAIN_ORDER,
    DEFAULT_CONFIG,
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

    @pytest.mark.parametrize("func", [noise_gate, dereverb, parametric_eq,
                                       compressor, de_esser, reverb])
    def test_stub_passthrough_mono(self, func):
        data = _make_tone()
        result = func(data, SR)
        np.testing.assert_array_equal(result, data)

    @pytest.mark.parametrize("func", [noise_gate, dereverb, parametric_eq,
                                       compressor, de_esser, reverb])
    def test_stub_passthrough_stereo(self, func):
        mono = _make_tone()
        data = np.column_stack([mono, mono])
        result = func(data, SR)
        np.testing.assert_array_equal(result, data)


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
