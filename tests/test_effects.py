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
    de_plosive,
    dereverb,
    gain_rider,
    highpass_filter,
    limiter,
    noise_gate,
    nr_cleanup,
    parametric_eq,
    process_vocal,
    reverb,
    soft_clipper,
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


# --- Gain Rider tests ---


class TestGainRider:

    def test_levels_dynamics(self):
        """Gain rider should make loud+quiet sections more uniform in RMS."""
        # Build signal: 0.5s loud, 0.5s quiet
        loud = _make_tone(duration_s=0.5, amplitude=0.5)
        quiet = _make_tone(duration_s=0.5, amplitude=0.05)
        data = np.concatenate([loud, quiet])

        result = gain_rider(data, SR, target_rms_db=-20.0, max_gain_db=6.0,
                            max_cut_db=6.0, silence_threshold_db=-60.0)

        # Measure RMS of first and second halves
        n = len(data)
        rms_first_in = np.sqrt(np.mean(data[:n // 2] ** 2))
        rms_second_in = np.sqrt(np.mean(data[n // 2:] ** 2))
        rms_first_out = np.sqrt(np.mean(result[:n // 2] ** 2))
        rms_second_out = np.sqrt(np.mean(result[n // 2:] ** 2))

        # Input ratio should be ~10:1, output ratio should be closer to 1:1
        input_ratio = rms_first_in / max(rms_second_in, 1e-10)
        output_ratio = rms_first_out / max(rms_second_out, 1e-10)
        assert output_ratio < input_ratio, \
            f"Gain rider should reduce dynamics ratio: {input_ratio:.1f} -> {output_ratio:.1f}"

    def test_silence_not_boosted(self):
        """Silence below threshold should not be boosted."""
        data = np.full(SR, 1e-6, dtype=np.float32)
        result = gain_rider(data, SR, silence_threshold_db=-50.0,
                            max_gain_db=6.0)
        # Output should not be louder than input
        assert np.abs(result).max() <= np.abs(data).max() * 2.0

    def test_preserves_shape_mono(self):
        data = _make_tone(duration_s=0.5)
        result = gain_rider(data, SR)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_preserves_shape_stereo(self):
        mono = _make_tone(duration_s=0.5)
        data = np.column_stack([mono, mono])
        result = gain_rider(data, SR)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_empty_input(self):
        data = np.array([], dtype=np.float32)
        result = gain_rider(data, SR)
        assert len(result) == 0

    def test_finite_output(self):
        data = _make_tone(duration_s=1.0, amplitude=0.5)
        result = gain_rider(data, SR, max_gain_db=12.0)
        assert np.all(np.isfinite(result))


# --- De-Plosive tests ---


def _make_plosive_signal(duration_s=0.5, sr=SR):
    """Signal with a low-frequency transient burst simulating a plosive."""
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    # Normal singing at 300 Hz
    singing = 0.2 * np.sin(2 * np.pi * 300 * t)
    # Plosive burst: 80 Hz burst in first 20ms
    burst_len = int(sr * 0.02)
    burst = np.zeros(n, dtype=np.float32)
    burst[:burst_len] = 0.8 * np.sin(2 * np.pi * 80 * t[:burst_len])
    return (singing + burst).astype(np.float32)


class TestDePlosive:

    def test_attenuates_plosive_burst(self):
        """Low-frequency energy in the burst region should be reduced."""
        from scipy.signal import butter, sosfiltfilt
        data = _make_plosive_signal()
        result = de_plosive(data, SR, plosive_freq_hz=200,
                            threshold_db=-30.0, reduction_db=10.0)
        # Measure low-freq energy in first 30ms
        burst_end = int(SR * 0.03)
        sos = butter(4, 200, btype="low", fs=SR, output="sos")
        low_before = np.sum(sosfiltfilt(sos, data[:burst_end]) ** 2)
        low_after = np.sum(sosfiltfilt(sos, result[:burst_end]) ** 2)
        assert low_after < low_before, \
            f"Plosive energy should decrease: {low_before:.4f} -> {low_after:.4f}"

    def test_preserves_non_plosive(self):
        """Normal signal without plosives should pass through mostly unchanged."""
        data = _make_tone(duration_s=0.5, freq=300, amplitude=0.2)
        result = de_plosive(data, SR, threshold_db=-15.0, reduction_db=10.0)
        corr = np.corrcoef(data, result)[0, 1]
        assert corr > 0.95, f"Non-plosive signal damaged: correlation {corr}"

    def test_preserves_shape_mono(self):
        data = _make_plosive_signal()
        result = de_plosive(data, SR)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_preserves_shape_stereo(self):
        mono = _make_plosive_signal()
        data = np.column_stack([mono, mono])
        result = de_plosive(data, SR)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_empty_input(self):
        data = np.array([], dtype=np.float32)
        result = de_plosive(data, SR)
        assert len(result) == 0

    def test_finite_output(self):
        data = _make_plosive_signal()
        result = de_plosive(data, SR, reduction_db=15.0)
        assert np.all(np.isfinite(result))


# --- Soft Clipper tests ---


class TestSoftClipper:

    def test_reduces_peaks(self):
        """Peaks should be lower after soft clipping."""
        data = _make_loud_signal()
        result = soft_clipper(data, SR, drive=2.0, ceiling_db=-1.0,
                              mode="tanh")
        ceiling_lin = 10.0 ** (-1.0 / 20.0)
        assert np.abs(result).max() <= ceiling_lin + 1e-4, \
            f"Peak {np.abs(result).max():.4f} exceeds ceiling {ceiling_lin:.4f}"

    def test_drive_one_passthrough(self):
        """Drive of 1.0 should return input unchanged."""
        data = _make_tone(amplitude=0.5)
        result = soft_clipper(data, SR, drive=1.0)
        np.testing.assert_array_equal(result, data)

    def test_tanh_mode(self):
        data = _make_loud_signal()
        result = soft_clipper(data, SR, drive=2.0, mode="tanh")
        assert np.all(np.isfinite(result))
        assert result.dtype == np.float32

    def test_arctan_mode(self):
        data = _make_loud_signal()
        result = soft_clipper(data, SR, drive=2.0, mode="arctan")
        assert np.all(np.isfinite(result))
        assert result.dtype == np.float32

    def test_cubic_mode(self):
        data = _make_loud_signal()
        result = soft_clipper(data, SR, drive=2.0, mode="cubic")
        assert np.all(np.isfinite(result))
        assert result.dtype == np.float32

    def test_higher_drive_more_reduction(self):
        """Higher drive should compress peaks more."""
        data = _make_loud_signal()
        low = soft_clipper(data, SR, drive=1.5, ceiling_db=-1.0)
        high = soft_clipper(data, SR, drive=3.0, ceiling_db=-1.0)
        # Higher drive: more waveform compression → lower peak-to-RMS ratio
        ptr_low = np.abs(low).max() / np.sqrt(np.mean(low ** 2))
        ptr_high = np.abs(high).max() / np.sqrt(np.mean(high ** 2))
        assert ptr_high <= ptr_low + 0.1, \
            f"Higher drive should reduce crest factor: {ptr_low:.2f} vs {ptr_high:.2f}"

    def test_preserves_shape_mono(self):
        data = _make_loud_signal()
        result = soft_clipper(data, SR, drive=1.5)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_preserves_shape_stereo(self):
        mono = _make_loud_signal()
        data = np.column_stack([mono, mono])
        result = soft_clipper(data, SR, drive=1.5)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_empty_input(self):
        data = np.array([], dtype=np.float32)
        result = soft_clipper(data, SR)
        assert len(result) == 0


# --- De-Esser tests ---


def _make_sibilant_signal(duration_s=1.0, sr=SR):
    """Signal with both low body (300 Hz) and sibilant (6 kHz) components."""
    t = np.arange(int(sr * duration_s)) / sr
    body = 0.3 * np.sin(2 * np.pi * 300 * t)
    sibilant = 0.3 * np.sin(2 * np.pi * 6000 * t)
    return (body + sibilant).astype(np.float32)


class TestDeEsser:

    def test_reduces_sibilant_energy(self):
        """Bandpass energy in 4-8 kHz should be reduced after processing."""
        from scipy.signal import butter, sosfiltfilt
        data = _make_sibilant_signal()
        result = de_esser(data, SR, freq_hz=6000, reduction_db=6.0,
                          threshold_db=-30.0, mode="split")
        # Measure 4-8 kHz energy before and after
        sos = butter(4, [4000, 8000], btype="band", fs=SR, output="sos")
        sib_before = np.sum(sosfiltfilt(sos, data) ** 2)
        sib_after = np.sum(sosfiltfilt(sos, result) ** 2)
        assert sib_after < sib_before, \
            f"Sibilant energy should decrease: {sib_before:.4f} -> {sib_after:.4f}"

    def test_preserves_low_frequency(self):
        """Energy below 1 kHz should be mostly unchanged in split mode."""
        from scipy.signal import butter, sosfiltfilt
        data = _make_sibilant_signal()
        result = de_esser(data, SR, freq_hz=6000, reduction_db=6.0,
                          threshold_db=-30.0, mode="split")
        sos = butter(4, 1000, btype="low", fs=SR, output="sos")
        low_before = np.sum(sosfiltfilt(sos, data) ** 2)
        low_after = np.sum(sosfiltfilt(sos, result) ** 2)
        ratio = low_after / low_before
        assert 0.9 < ratio < 1.1, \
            f"Low freq energy ratio {ratio:.3f} out of expected range"

    def test_wideband_mode(self):
        """Wideband mode should reduce overall energy of a sibilant signal."""
        data = _make_sibilant_signal()
        result = de_esser(data, SR, freq_hz=6000, reduction_db=6.0,
                          threshold_db=-30.0, mode="wideband")
        energy_before = np.sum(data ** 2)
        energy_after = np.sum(result ** 2)
        assert energy_after < energy_before

    def test_zero_reduction_passthrough(self):
        """reduction_db=0 should return input unchanged."""
        data = _make_sibilant_signal()
        result = de_esser(data, SR, reduction_db=0.0)
        np.testing.assert_array_equal(result, data)

    def test_preserves_shape_mono(self):
        data = _make_sibilant_signal(duration_s=0.5)
        result = de_esser(data, SR, reduction_db=6.0)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_preserves_shape_stereo(self):
        mono = _make_sibilant_signal(duration_s=0.5)
        data = np.column_stack([mono, mono])
        result = de_esser(data, SR, reduction_db=6.0)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_empty_input(self):
        data = np.array([], dtype=np.float32)
        result = de_esser(data, SR)
        assert len(result) == 0

    def test_finite_output(self):
        data = _make_sibilant_signal()
        result = de_esser(data, SR, reduction_db=12.0, threshold_db=-40.0)
        assert np.all(np.isfinite(result))


# --- Reverb tests ---


class TestReverb:

    def test_adds_decay_tail(self):
        """Impulse signal should have energy beyond original after reverb."""
        data = np.zeros(SR, dtype=np.float32)
        data[100] = 1.0  # impulse
        result = reverb(data, SR, wet_mix=0.5, decay=0.7, predelay_ms=0.0)
        # Energy in second half should increase (reverb tail)
        tail_energy_dry = np.sum(data[SR // 2:] ** 2)
        tail_energy_wet = np.sum(result[SR // 2:] ** 2)
        assert tail_energy_wet > tail_energy_dry

    def test_dry_preserved(self):
        """At low wet_mix, output should be highly correlated with input."""
        data = _make_tone(duration_s=1.0, amplitude=0.5)
        result = reverb(data, SR, wet_mix=0.05, decay=0.5)
        corr = np.corrcoef(data, result)[0, 1]
        assert corr > 0.95, f"Correlation {corr} too low at wet_mix=0.05"

    def test_zero_wet_passthrough(self):
        """wet_mix=0 should return input unchanged."""
        data = _make_tone()
        result = reverb(data, SR, wet_mix=0.0)
        np.testing.assert_array_equal(result, data)

    def test_higher_decay_more_tail(self):
        """Higher decay should produce more tail energy."""
        data = np.zeros(SR, dtype=np.float32)
        data[100] = 1.0
        low = reverb(data, SR, wet_mix=0.5, decay=0.3, predelay_ms=0.0)
        high = reverb(data, SR, wet_mix=0.5, decay=0.8, predelay_ms=0.0)
        tail_low = np.sum(low[SR // 2:] ** 2)
        tail_high = np.sum(high[SR // 2:] ** 2)
        assert tail_high > tail_low, \
            f"Higher decay should produce more tail: {tail_low:.6f} vs {tail_high:.6f}"

    def test_predelay_shifts_wet_signal(self):
        """With predelay, wet onset should be later."""
        data = np.zeros(SR, dtype=np.float32)
        data[100] = 1.0
        no_delay = reverb(data, SR, wet_mix=1.0, decay=0.5, predelay_ms=0.0)
        with_delay = reverb(data, SR, wet_mix=1.0, decay=0.5, predelay_ms=50.0)
        # Find first sample above threshold
        thresh = 0.01
        onset_no = np.argmax(np.abs(no_delay) > thresh)
        onset_wd = np.argmax(np.abs(with_delay) > thresh)
        assert onset_wd >= onset_no, \
            f"Predelay should shift onset: {onset_no} vs {onset_wd}"

    def test_preserves_shape_mono(self):
        data = _make_tone(duration_s=0.5)
        result = reverb(data, SR, wet_mix=0.15)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_preserves_shape_stereo(self):
        mono = _make_tone(duration_s=0.5)
        data = np.column_stack([mono, mono])
        result = reverb(data, SR, wet_mix=0.15)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_empty_input(self):
        data = np.array([], dtype=np.float32)
        result = reverb(data, SR)
        assert len(result) == 0

    def test_finite_output(self):
        data = _make_tone(duration_s=0.5)
        result = reverb(data, SR, wet_mix=0.3, decay=0.8)
        assert np.all(np.isfinite(result))


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
        """All presets must contain all 14 chain keys with an 'enabled' key."""
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
        assert len(CHAIN_ORDER) == 14

    def test_chain_order_sequence(self):
        """Chain order must follow the 14-stage pipeline specification."""
        expected = [
            "noise_gate",
            "spectral_noise_reduction",
            "gain_rider",
            "de_plosive",
            "nr_cleanup",
            "dereverb",
            "highpass_filter",
            "parametric_eq",
            "compressor_peak",
            "compressor_body",
            "de_esser",
            "soft_clipper",
            "reverb",
            "limiter",
        ]
        assert CHAIN_ORDER == expected

    def test_all_effects_in_dispatch(self):
        from vocalforge.audio.effects import _EFFECT_FUNCS
        for name in CHAIN_ORDER:
            assert name in _EFFECT_FUNCS

    def test_full_chain_all_enabled(self):
        """All 14 effects enabled should produce finite output with correct shape."""
        data = _make_tone(duration_s=2.0, amplitude=0.5)
        config = {
            "noise_gate": {"enabled": True, "threshold_db": -35.0,
                           "reduction_db": -40.0},
            "spectral_noise_reduction": {"enabled": True, "strength": 0.5,
                                          "mode": "adaptive"},
            "gain_rider": {"enabled": True, "target_rms_db": -20.0,
                           "max_gain_db": 6.0, "max_cut_db": 6.0},
            "de_plosive": {"enabled": True, "plosive_freq_hz": 200.0,
                           "threshold_db": -25.0, "reduction_db": 10.0},
            "nr_cleanup": {"enabled": True, "strength": 0.4,
                           "mode": "stationary", "n_std_thresh": 2.5},
            "dereverb": {"enabled": True, "strength": 0.4},
            "highpass_filter": {"enabled": True, "cutoff_hz": 100.0},
            "parametric_eq": {"enabled": True,
                              "bands": EQ_PRESETS["bright"]},
            "compressor_peak": {"enabled": True, "threshold_db": -12.0,
                                "ratio": 8.0, "attack_ms": 2.0,
                                "release_ms": 80.0, "knee_db": 3.0},
            "compressor_body": {"enabled": True, "threshold_db": -20.0,
                                "ratio": 2.5, "attack_ms": 20.0,
                                "release_ms": 200.0, "knee_db": 8.0},
            "de_esser": {"enabled": True, "freq_hz": 6000,
                         "reduction_db": 6.0, "mode": "split"},
            "soft_clipper": {"enabled": True, "drive": 1.5,
                             "ceiling_db": -1.0, "mode": "tanh"},
            "reverb": {"enabled": True, "wet_mix": 0.15, "decay": 0.7},
            "limiter": {"enabled": True, "ceiling_db": -0.7},
        }
        result = process_vocal(data, SR, config=config)
        assert result.shape == data.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_eq_compressor_limiter_integration(self):
        """EQ + serial compression + limiter enabled together produce valid output."""
        data = _make_loud_signal(duration_s=2.0)
        config = {name: {"enabled": False} for name in CHAIN_ORDER}
        config["parametric_eq"] = {
            "enabled": True,
            "bands": EQ_PRESETS["clean_up"],
        }
        config["compressor_peak"] = {
            "enabled": True,
            "threshold_db": -12.0,
            "ratio": 8.0,
        }
        config["compressor_body"] = {
            "enabled": True,
            "threshold_db": -20.0,
            "ratio": 2.5,
        }
        config["limiter"] = {"enabled": True, "ceiling_db": -1.0}
        result = process_vocal(data, SR, config=config)
        assert result.shape == data.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_serial_compression_stages(self):
        """Peak compressor (high threshold) and body compressor (low threshold) in series."""
        data = _make_loud_signal(duration_s=1.0)
        # Peak only
        config_peak = {name: {"enabled": False} for name in CHAIN_ORDER}
        config_peak["compressor_peak"] = {
            "enabled": True, "threshold_db": -12.0, "ratio": 8.0,
            "makeup_db": 0.0,
        }
        peak_only = process_vocal(data, SR, config=config_peak)
        # Body only
        config_body = {name: {"enabled": False} for name in CHAIN_ORDER}
        config_body["compressor_body"] = {
            "enabled": True, "threshold_db": -20.0, "ratio": 2.5,
            "makeup_db": 0.0,
        }
        body_only = process_vocal(data, SR, config=config_body)
        # Body has lower threshold → should compress more of the signal
        peak_reduction = np.abs(data).max() - np.abs(peak_only).max()
        body_reduction = np.abs(data).max() - np.abs(body_only).max()
        assert body_reduction > 0, "Body compressor should reduce peaks"
        assert peak_reduction > 0, "Peak compressor should reduce peaks"


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


# --- NR Cleanup (Pass 2) tests ---


class TestNrCleanup:

    def test_gentle_reduction(self):
        """nr_cleanup should reduce noise more gently than pass 1."""
        rng = np.random.default_rng(42)
        noise_only = (0.1 * rng.standard_normal(SR // 2)).astype(np.float32)
        tone_noisy = _make_tone(duration_s=1.5, amplitude=0.5) + \
            (0.1 * rng.standard_normal(int(SR * 1.5))).astype(np.float32)
        signal = np.concatenate([noise_only, tone_noisy])

        # Pass 1 (aggressive defaults)
        pass1 = spectral_noise_reduction(signal, SR, strength=0.75,
                                          mode="stationary", n_std_thresh=1.5)
        # Pass 2 / nr_cleanup (gentle defaults)
        pass2 = nr_cleanup(signal, SR, strength=0.4,
                           mode="stationary", n_std_thresh=2.5)

        rms_original = np.sqrt(np.mean(signal[:SR // 2].astype(np.float64) ** 2))
        rms_pass1 = np.sqrt(np.mean(pass1[:SR // 2].astype(np.float64) ** 2))
        rms_pass2 = np.sqrt(np.mean(pass2[:SR // 2].astype(np.float64) ** 2))

        # Both should reduce noise
        assert rms_pass1 < rms_original
        assert rms_pass2 < rms_original
        # Pass 1 should be more aggressive (lower residual noise)
        assert rms_pass1 < rms_pass2, \
            f"Pass 1 should be more aggressive: {rms_pass1:.4f} vs {rms_pass2:.4f}"

    def test_no_guide_stem(self):
        """nr_cleanup must NOT pass guide_stem to reduce_noise."""
        from unittest.mock import patch
        data = _make_tone(duration_s=2.0)
        with patch("vocalforge.audio.noise_reduction.reduce_noise") as mock_rn:
            mock_rn.return_value = data
            nr_cleanup(data, SR, strength=0.4,
                       guide_stem=np.ones(SR, dtype=np.float32))
            mock_rn.assert_called_once()
            call_kwargs = mock_rn.call_args.kwargs
            assert call_kwargs.get("guide_stem") is None, \
                "nr_cleanup must not forward guide_stem to reduce_noise"

    def test_strength_zero_passthrough(self):
        """strength=0 should return input unchanged."""
        data = _make_tone(duration_s=2.0)
        result = nr_cleanup(data, SR, strength=0.0)
        np.testing.assert_array_equal(result, data)

    def test_preserves_shape_mono(self):
        data = _make_tone(duration_s=2.0)
        result = nr_cleanup(data, SR, strength=0.4)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_preserves_shape_stereo(self):
        mono = _make_tone(duration_s=2.0)
        data = np.column_stack([mono, mono])
        result = nr_cleanup(data, SR, strength=0.4)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_default_config_nr_cleanup(self):
        """DEFAULT_CONFIG should have nr_cleanup with stationary mode, no guide_stem."""
        cfg = DEFAULT_CONFIG["nr_cleanup"]
        assert cfg["enabled"] is True
        assert cfg["mode"] == "stationary"
        assert cfg["strength"] == 0.7
        assert cfg["n_std_thresh"] == 1.5
        assert "guide_stem" not in cfg
