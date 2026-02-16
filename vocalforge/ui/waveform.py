"""Waveform display widget — QPainter-based waveform rendering."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

_BG_COLOR = QColor("#2D2D2D")
_CENTER_COLOR = QColor("#555555")
_WAVE_COLOR = QColor("#3DD5F3")
_SEPARATOR_COLOR = QColor("#555555")


class WaveformWidget(QWidget):
    """Displays an audio waveform using min/max envelope rendering."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: np.ndarray | None = None
        self._sample_rate: int = 0
        self._channels: int = 0
        self._envelope: list[tuple[np.ndarray, np.ndarray]] | None = None
        self.setMinimumHeight(60)

    def set_audio(self, data: np.ndarray, sample_rate: int) -> None:
        """Load audio data and trigger a repaint.

        Args:
            data: float32 array, shape (samples,) for mono or (samples, channels).
            sample_rate: Sample rate in Hz.
        """
        self._data = data
        self._sample_rate = sample_rate
        self._channels = 1 if data.ndim == 1 else data.shape[1]
        self._compute_envelope()
        self.update()

    def clear(self) -> None:
        """Reset to blank state."""
        self._data = None
        self._sample_rate = 0
        self._channels = 0
        self._envelope = None
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._data is not None:
            self._compute_envelope()

    def _compute_envelope(self) -> None:
        """Compute min/max envelope per pixel column for each channel."""
        if self._data is None:
            self._envelope = None
            return

        width = self.width()
        if width < 1:
            self._envelope = None
            return

        data = self._data
        if data.ndim == 1:
            channels = [data]
        else:
            channels = [data[:, ch] for ch in range(data.shape[1])]

        envelope = []
        for ch_data in channels:
            n_samples = len(ch_data)
            if n_samples <= width:
                # Fewer samples than pixels — one sample per column
                envelope.append((ch_data.copy(), ch_data.copy()))
            else:
                # Bin samples into pixel columns
                bin_size = n_samples // width
                usable = bin_size * width
                reshaped = ch_data[:usable].reshape(width, bin_size)
                mins = reshaped.min(axis=1)
                maxs = reshaped.max(axis=1)
                envelope.append((mins, maxs))

        self._envelope = envelope

    def paintEvent(self, event) -> None:
        if self.width() < 1 or self.height() < 1:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Background
        painter.fillRect(self.rect(), _BG_COLOR)

        if self._envelope is None:
            painter.end()
            return

        n_channels = len(self._envelope)
        total_height = self.height()

        for ch_idx, (mins, maxs) in enumerate(self._envelope):
            if n_channels == 1:
                y_offset = 0
                ch_height = total_height
            else:
                ch_height = total_height // n_channels
                y_offset = ch_idx * ch_height

            center_y = y_offset + ch_height // 2

            # Center line
            painter.setPen(QPen(_CENTER_COLOR, 1))
            painter.drawLine(0, center_y, self.width(), center_y)

            # Separator between channels
            if n_channels > 1 and ch_idx > 0:
                painter.setPen(QPen(_SEPARATOR_COLOR, 1))
                painter.drawLine(0, y_offset, self.width(), y_offset)

            # Waveform
            painter.setPen(QPen(_WAVE_COLOR, 1))
            half_height = ch_height / 2.0
            n_cols = len(mins)
            for x in range(n_cols):
                y_min = int(center_y - maxs[x] * half_height)
                y_max = int(center_y - mins[x] * half_height)
                painter.drawLine(x, y_min, x, y_max)

        painter.end()
