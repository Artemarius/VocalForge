"""Waveform display widget — QPainter-based waveform rendering."""

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

_BG_COLOR = QColor("#2D2D2D")
_CENTER_COLOR = QColor("#555555")
_WAVE_COLOR = QColor("#3DD5F3")
_SEPARATOR_COLOR = QColor("#555555")
_CURSOR_COLOR = QColor("#FF6B6B")


class WaveformWidget(QWidget):
    """Displays an audio waveform using min/max envelope rendering."""

    seek_requested = Signal(float)  # emits normalized position 0.0–1.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: np.ndarray | None = None
        self._sample_rate: int = 0
        self._channels: int = 0
        self._envelope: list[tuple[np.ndarray, np.ndarray]] | None = None
        self._cursor_position: float = -1.0  # normalized 0.0–1.0, -1.0 = hidden
        # Offset visualization: place this waveform at a position in a global timeline
        self._offset_samples: int = 0
        self._total_duration_samples: int = 0
        # Display gain: visual scaling applied in paintEvent (no data recompute)
        self._display_gain: float = 1.0
        self.setMinimumHeight(60)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

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

    def set_offset(self, offset_samples: int, total_duration_samples: int) -> None:
        """Set this waveform's position within a global timeline.

        When set, the envelope is recomputed to show the waveform at its
        offset within the total duration. All waveforms sharing the same
        total_duration_samples will have a consistent timeline scale.

        Args:
            offset_samples: Start offset of this track in the global timeline.
            total_duration_samples: Total duration of the global timeline.
        """
        self._offset_samples = offset_samples
        self._total_duration_samples = total_duration_samples
        if self._data is not None:
            self._compute_envelope()
            self.update()

    def clear_offset(self) -> None:
        """Reset to local-only timeline (no global offset)."""
        self._offset_samples = 0
        self._total_duration_samples = 0
        if self._data is not None:
            self._compute_envelope()
            self.update()

    def set_display_gain(self, gain: float) -> None:
        """Set visual gain multiplier for the waveform. Triggers repaint only."""
        self._display_gain = max(0.0, gain)
        self.update()

    def clear(self) -> None:
        """Reset to blank state."""
        self._data = None
        self._sample_rate = 0
        self._channels = 0
        self._envelope = None
        self._cursor_position = -1.0
        self._offset_samples = 0
        self._total_duration_samples = 0
        self._display_gain = 1.0
        self.update()

    def set_cursor_position(self, normalized: float) -> None:
        """Set the playback cursor position (0.0–1.0). Triggers repaint."""
        self._cursor_position = normalized
        self.update()

    def clear_cursor(self) -> None:
        """Hide the playback cursor."""
        self._cursor_position = -1.0
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.position().y() > self.height() - 6:
            super().mousePressEvent(event)
            return
        if self._data is not None and self.width() > 0:
            normalized = max(0.0, min(1.0, event.position().x() / self.width()))
            self.seek_requested.emit(normalized)
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._data is not None:
            self._compute_envelope()

    def _compute_envelope(self) -> None:
        """Compute min/max envelope per pixel column for each channel.

        When offset visualization is active (total_duration_samples > 0),
        the envelope is computed within the global timeline so that the
        waveform occupies only its proportional pixel range.
        """
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

        use_global = (
            self._total_duration_samples > 0
            and self._total_duration_samples >= data.shape[0]
        )

        envelope = []
        for ch_data in channels:
            n_samples = len(ch_data)

            if use_global:
                total = self._total_duration_samples
                offset = self._offset_samples

                # Initialize full-width silent envelope
                mins = np.zeros(width, dtype=np.float32)
                maxs = np.zeros(width, dtype=np.float32)

                # Compute pixel range for this track's data
                px_start = int(offset / total * width) if total > 0 else 0
                px_end = int((offset + n_samples) / total * width) if total > 0 else width
                px_start = max(0, min(px_start, width))
                px_end = max(px_start, min(px_end, width))

                track_width = px_end - px_start
                if track_width > 0 and n_samples > 0:
                    if n_samples <= track_width:
                        mins[px_start:px_start + n_samples] = ch_data
                        maxs[px_start:px_start + n_samples] = ch_data
                    else:
                        bin_size = n_samples // track_width
                        usable = bin_size * track_width
                        reshaped = ch_data[:usable].reshape(track_width, bin_size)
                        mins[px_start:px_end] = reshaped.min(axis=1)
                        maxs[px_start:px_end] = reshaped.max(axis=1)

                envelope.append((mins, maxs))
            else:
                if n_samples <= width:
                    envelope.append((ch_data.copy(), ch_data.copy()))
                else:
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

            # Waveform (scaled by display gain)
            painter.setPen(QPen(_WAVE_COLOR, 1))
            half_height = ch_height / 2.0
            g = self._display_gain
            n_cols = len(mins)
            for x in range(n_cols):
                y_min = int(center_y - maxs[x] * g * half_height)
                y_max = int(center_y - mins[x] * g * half_height)
                painter.drawLine(x, y_min, x, y_max)

        # Playback cursor
        if self._cursor_position >= 0:
            cursor_x = int(self._cursor_position * self.width())
            painter.setPen(QPen(_CURSOR_COLOR, 2))
            painter.drawLine(cursor_x, 0, cursor_x, self.height())

        painter.end()
