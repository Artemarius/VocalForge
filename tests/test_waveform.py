"""Tests for WaveformWidget — mono display, offset visualization.

Headless tests: only verify internal state, no window needed.
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication exists for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def widget(qapp):
    from vocalforge.ui.waveform import WaveformWidget
    w = WaveformWidget()
    # Force a fixed width for deterministic envelope computation
    w.resize(200, 60)
    return w


class TestMonoEnvelopeFromStereo:
    """Stereo audio should produce a single mono envelope."""

    def test_stereo_produces_single_channel_envelope(self, widget):
        sr = 44100
        n = 44100
        left = np.sin(np.linspace(0, 2 * np.pi * 440, n)).astype(np.float32)
        right = np.sin(np.linspace(0, 2 * np.pi * 880, n)).astype(np.float32)
        stereo = np.column_stack([left, right])

        widget.set_audio(stereo, sr)

        assert widget._envelope is not None
        assert len(widget._envelope) == 1  # single mono channel

    def test_mono_also_produces_single_channel_envelope(self, widget):
        sr = 44100
        mono = np.sin(np.linspace(0, 2 * np.pi * 440, sr)).astype(np.float32)

        widget.set_audio(mono, sr)

        assert widget._envelope is not None
        assert len(widget._envelope) == 1

    def test_stereo_envelope_is_mean_of_channels(self, widget):
        """Verify the mono downmix is the average of left and right."""
        sr = 44100
        n = 44100
        left = np.ones(n, dtype=np.float32) * 0.4
        right = np.ones(n, dtype=np.float32) * 0.8
        stereo = np.column_stack([left, right])

        widget.set_audio(stereo, sr)

        # The mono signal should be 0.6 everywhere
        mins, maxs = widget._envelope[0]
        # Each bin should contain the constant 0.6
        np.testing.assert_array_almost_equal(mins, 0.6, decimal=2)
        np.testing.assert_array_almost_equal(maxs, 0.6, decimal=2)


class TestOffsetVisualization:
    """Verify set_offset() changes the envelope pixel range."""

    def test_offset_creates_silent_prefix(self, widget):
        sr = 44100
        mono = np.ones(22050, dtype=np.float32) * 0.5  # 0.5s
        widget.set_audio(mono, sr)

        # Place this track starting at halfway in a 44100-sample timeline
        widget.set_offset(22050, 44100)

        assert widget._envelope is not None
        mins, maxs = widget._envelope[0]
        # First half of pixels should be silent (zero)
        half = len(mins) // 2
        assert np.all(mins[:half] == 0)
        assert np.all(maxs[:half] == 0)
        # Second half should have signal
        assert np.any(maxs[half:] != 0)

    def test_clear_offset_restores_full_range(self, widget):
        sr = 44100
        mono = np.ones(44100, dtype=np.float32) * 0.5
        widget.set_audio(mono, sr)

        widget.set_offset(22050, 44100)
        widget.clear_offset()

        assert widget._offset_samples == 0
        assert widget._total_duration_samples == 0

        # Envelope should fill the full width
        mins, maxs = widget._envelope[0]
        assert np.all(maxs != 0)

    def test_no_offset_fills_full_width(self, widget):
        sr = 44100
        mono = np.ones(44100, dtype=np.float32) * 0.5
        widget.set_audio(mono, sr)

        mins, maxs = widget._envelope[0]
        # All bins should have signal
        assert np.all(maxs != 0)


class TestClear:
    def test_clear_resets_state(self, widget):
        mono = np.ones(1000, dtype=np.float32)
        widget.set_audio(mono, 44100)
        widget.clear()

        assert widget._data is None
        assert widget._envelope is None
        assert widget._sample_rate == 0


class TestDisplayGain:
    def test_display_gain_stored(self, widget):
        widget.set_display_gain(2.0)
        assert widget._display_gain == 2.0

    def test_display_gain_clamps_negative(self, widget):
        widget.set_display_gain(-1.0)
        assert widget._display_gain == 0.0
