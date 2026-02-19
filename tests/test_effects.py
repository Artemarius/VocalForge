"""Tests for vocalforge.audio.effects."""

import numpy as np
import pytest

from vocalforge.audio.effects import (
    CHAIN_ORDER,
    DEFAULT_CONFIG,
    EQ_PRESETS,
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

    @pytest.mark.parametrize("func", [de_esser, reverb])
    def test_stub_passthrough_mono(self, func):
        data = _make_tone()
        result = func(data, SR)
        np.testing.assert_array_equal(result, data)

    @pytest.mark.parametrize("func", [de_esser, reverb])
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


# --- Parametric EQ tests ---


class TestParametricEQ:

    def test_boosts_target_frequency(self):
        """6dB peak boost at 3500Hz should increase amplitude of 3500Hz tone."""
        data = _make_tone(duration_s=1.0, freq=3500, amplitude=0.3)
        bands = [{"freq_hz": 3500.0, "gain_db": 6.0, "q": 1.0, "type": "peak"}]
        result = parametric_eq(data, SR, bands=bands)
        # RMS should increase
        rms_before = np.sqrt(np.mean(data ** 2))
        rms_after = np.sqrt(np.mean(result ** 2))
        assert rms_after > rms_before * 1.3, \
            f"Expected boost: rms {rms_before:.4f} -> {rms_after:.4f}"

    def test_cuts_target_frequency(self):
        """−6dB peak cut at 250Hz should decrease amplitude of 250Hz tone."""
        data = _make_tone(duration_s=1.0, freq=250, amplitude=0.5)
        bands = [{"freq_hz": 250.0, "gain_db": -6.0, "q": 1.0, "type": "peak"}]
        result = parametric_eq(data, SR, bands=bands)
        rms_before = np.sqrt(np.mean(data ** 2))
        rms_after = np.sqrt(np.mean(result ** 2))
        assert rms_after < rms_before * 0.8, \
            f"Expected cut: rms {rms_before:.4f} -> {rms_after:.4f}"

    def test_high_shelf_boost(self):
        """High shelf at 8kHz should boost a 10kHz tone."""
        data = _make_tone(duration_s=1.0, freq=10000, amplitude=0.3)
        bands = [{"freq_hz": 8000.0, "gain_db": 6.0, "q": 0.7, "type": "high_shelf"}]
        result = parametric_eq(data, SR, bands=bands)
        rms_before = np.sqrt(np.mean(data ** 2))
        rms_after = np.sqrt(np.mean(result ** 2))
        assert rms_after > rms_before * 1.3

    def test_low_shelf_boost(self):
        """Low shelf at 200Hz should boost a 150Hz tone."""
        data = _make_tone(duration_s=1.0, freq=150, amplitude=0.3)
        bands = [{"freq_hz": 200.0, "gain_db": 6.0, "q": 0.7, "type": "low_shelf"}]
        result = parametric_eq(data, SR, bands=bands)
        rms_before = np.sqrt(np.mean(data ** 2))
        rms_after = np.sqrt(np.mean(result ** 2))
        assert rms_after > rms_before * 1.3

    def test_passthrough_zero_gain(self):
        """All bands with gain_db=0 should return signal unchanged."""
        data = _make_tone(duration_s=0.5)
        bands = [
            {"freq_hz": 250.0, "gain_db": 0.0, "q": 1.0, "type": "peak"},
            {"freq_hz": 3500.0, "gain_db": 0.0, "q": 1.0, "type": "peak"},
        ]
        result = parametric_eq(data, SR, bands=bands)
        np.testing.assert_array_equal(result, data)

    def test_passthrough_empty_bands(self):
        """Empty bands list should return signal unchanged."""
        data = _make_tone(duration_s=0.5)
        result = parametric_eq(data, SR, bands=[])
        np.testing.assert_array_equal(result, data)

    def test_preserves_shape_mono(self):
        data = _make_tone(duration_s=0.5)
        bands = [{"freq_hz": 1000.0, "gain_db": 3.0, "q": 1.0, "type": "peak"}]
        result = parametric_eq(data, SR, bands=bands)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_preserves_shape_stereo(self):
        mono = _make_tone(duration_s=0.5)
        data = np.column_stack([mono, mono])
        bands = [{"freq_hz": 1000.0, "gain_db": 3.0, "q": 1.0, "type": "peak"}]
        result = parametric_eq(data, SR, bands=bands)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_empty_input(self):
        data = np.array([], dtype=np.float32)
        bands = [{"freq_hz": 1000.0, "gain_db": 3.0, "q": 1.0, "type": "peak"}]
        result = parametric_eq(data, SR, bands=bands)
        assert len(result) == 0

    def test_multi_band_cascade(self):
        """Two bands active simultaneously should both take effect."""
        # Mix of low and high tones
        low = _make_tone(duration_s=1.0, freq=250, amplitude=0.3)
        high = _make_tone(duration_s=1.0, freq=4000, amplitude=0.3)
        data = low + high
        bands = [
            {"freq_hz": 250.0, "gain_db": -6.0, "q": 1.5, "type": "peak"},
            {"freq_hz": 4000.0, "gain_db": 6.0, "q": 1.5, "type": "peak"},
        ]
        result = parametric_eq(data, SR, bands=bands)
        # Output should differ from input
        assert not np.allclose(result, data, atol=1e-3)

    def test_preset_bands_valid(self):
        """All EQ_PRESETS should produce valid (finite) output."""
        data = _make_tone(duration_s=0.5)
        for preset_name, bands in EQ_PRESETS.items():
            result = parametric_eq(data, SR, bands=bands)
            assert result.shape == data.shape, f"Preset {preset_name} shape mismatch"
            assert np.all(np.isfinite(result)), f"Preset {preset_name} produced non-finite values"


# --- Compressor tests ---


class TestCompressor:

    def test_reduces_dynamic_range(self):
        """Loud signal should have its peak reduced by compression."""
        data = _make_loud_signal(duration_s=1.0)
        result = compressor(data, SR, threshold_db=-6.0, ratio=4.0,
                            knee_db=0.0, makeup_db=0.0)
        assert np.abs(result).max() < np.abs(data).max(), \
            "Compressor should reduce peak level"

    def test_quiet_signal_uncompressed(self):
        """Signal well below threshold should pass through mostly unchanged."""
        data = _make_tone(duration_s=1.0, amplitude=0.01)
        result = compressor(data, SR, threshold_db=-6.0, ratio=4.0,
                            makeup_db=0.0)
        np.testing.assert_allclose(result, data, atol=1e-3)

    def test_ratio_one_passthrough(self):
        """Ratio of 1:1 means no compression — output equals input."""
        data = _make_loud_signal()
        result = compressor(data, SR, ratio=1.0)
        np.testing.assert_array_equal(result, data)

    def test_higher_ratio_more_compression(self):
        """Ratio 8:1 should compress more than 2:1."""
        data = _make_loud_signal(duration_s=1.0)
        r2 = compressor(data, SR, threshold_db=-6.0, ratio=2.0,
                        knee_db=0.0, makeup_db=0.0)
        r8 = compressor(data, SR, threshold_db=-6.0, ratio=8.0,
                        knee_db=0.0, makeup_db=0.0)
        # Higher ratio => lower peak
        assert np.abs(r8).max() < np.abs(r2).max(), \
            "Higher ratio should compress more"

    def test_preserves_shape_mono(self):
        data = _make_tone(duration_s=0.5, amplitude=0.5)
        result = compressor(data, SR, threshold_db=-10.0, ratio=3.0)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_preserves_shape_stereo(self):
        mono = _make_tone(duration_s=0.5, amplitude=0.5)
        data = np.column_stack([mono, mono])
        result = compressor(data, SR, threshold_db=-10.0, ratio=3.0)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_empty_input(self):
        data = np.array([], dtype=np.float32)
        result = compressor(data, SR)
        assert len(result) == 0

    def test_parallel_compression_mix(self):
        """mix=0.5 should produce output closer to original than mix=1.0."""
        data = _make_loud_signal(duration_s=1.0)
        full = compressor(data, SR, threshold_db=-6.0, ratio=4.0,
                          knee_db=0.0, makeup_db=0.0, mix=1.0)
        half = compressor(data, SR, threshold_db=-6.0, ratio=4.0,
                          knee_db=0.0, makeup_db=0.0, mix=0.5)
        # Distance from original should be smaller with mix=0.5
        dist_full = np.sqrt(np.mean((data - full) ** 2))
        dist_half = np.sqrt(np.mean((data - half) ** 2))
        assert dist_half < dist_full, \
            "Parallel compression (mix=0.5) should be closer to original"

    def test_auto_makeup_gain(self):
        """Auto makeup should prevent output from being drastically quieter."""
        data = _make_loud_signal(duration_s=1.0)
        result = compressor(data, SR, threshold_db=-6.0, ratio=4.0)
        rms_before = np.sqrt(np.mean(data ** 2))
        rms_after = np.sqrt(np.mean(result ** 2))
        # With auto makeup, output should be at least 50% of original RMS
        assert rms_after > rms_before * 0.5, \
            f"Auto makeup too weak: {rms_before:.4f} -> {rms_after:.4f}"


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

    def test_eq_compressor_limiter_integration(self):
        """EQ + compressor + limiter enabled together produce valid output."""
        data = _make_loud_signal(duration_s=2.0)
        config = {name: {"enabled": False} for name in CHAIN_ORDER}
        config["parametric_eq"] = {
            "enabled": True,
            "bands": EQ_PRESETS["clean_up"],
        }
        config["compressor"] = {
            "enabled": True,
            "threshold_db": -12.0,
            "ratio": 3.0,
        }
        config["limiter"] = {"enabled": True, "ceiling_db": -1.0}
        result = process_vocal(data, SR, config=config)
        assert result.shape == data.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))


# --- Spectral Noise Reduction wrapper tests ---


class TestSpectralNoiseReduction:

    def test_mode_param_accepted(self):
        """Wrapper should accept mode kwarg without error."""
        data = _make_tone(duration_s=2.0)
        result = spectral_noise_reduction(data, SR, strength=0.5, mode="adaptive")
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_advanced_params_accepted(self):
        """All new kwargs should pass through to reduce_noise."""
        data = _make_tone(duration_s=2.0)
        result = spectral_noise_reduction(
            data, SR, strength=0.5, mode="stationary",
            n_std_thresh=2.0, use_torch=False,
            freq_smooth_hz=300, time_smooth_ms=80,
        )
        assert result.shape == data.shape
        assert np.all(np.isfinite(result))

    def test_default_config_includes_mode(self):
        """DEFAULT_CONFIG should have mode='auto' for spectral_noise_reduction."""
        snr_cfg = DEFAULT_CONFIG["spectral_noise_reduction"]
        assert snr_cfg["mode"] == "auto"
        assert "n_std_thresh" in snr_cfg
        assert "use_torch" in snr_cfg
        assert "freq_smooth_hz" in snr_cfg
        assert "time_smooth_ms" in snr_cfg
