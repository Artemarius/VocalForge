"""Import panel — song loading with waveform display (5-track layout)."""

import functools
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vocalforge.ui.waveform import WaveformWidget
from vocalforge.utils.audio_io import get_audio_info, load_audio

_FILE_FILTER = "Audio Files (*.wav *.flac *.ogg);;All Files (*)"

_SLOT_NAMES = ("song", "minus_sep", "vocal_sep", "minus_import", "vocal")


class ImportPanel(QWidget):
    """Panel with five track slots: Song, Minus-sep, Vocal-sep, Minus (import), Vocal."""

    track_loaded = Signal(str)  # emits slot name on successful load
    separate_requested = Signal(str)  # emits song file path

    def __init__(self, parent=None):
        super().__init__(parent)

        self._tracks: dict[str, tuple | None] = {s: None for s in _SLOT_NAMES}
        self._paths: dict[str, str | None] = {s: None for s in _SLOT_NAMES}
        self._waveforms: dict[str, WaveformWidget] = {}
        self._file_labels: dict[str, QLabel] = {}
        self._info_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)

        # --- Source group ---
        layout.addWidget(
            self._create_track_group(
                "song", "Song (Plus Track)", load_button="Load Song"
            )
        )

        # --- Separated group ---
        sep_group = QGroupBox("Separated Stems")
        sep_layout = QVBoxLayout(sep_group)
        sep_layout.addWidget(
            self._create_track_row("minus_sep", "Minus (sep)")
        )
        sep_layout.addWidget(
            self._create_track_row("vocal_sep", "Vocal (sep)")
        )
        layout.addWidget(sep_group)

        # --- Working group ---
        layout.addWidget(
            self._create_track_group(
                "minus_import", "Minus (Import)", load_button="Load Minus"
            )
        )
        layout.addWidget(
            self._create_track_group(
                "vocal", "Vocal Track", load_button="Load Vocal"
            )
        )

        layout.addStretch()

    def _create_track_group(
        self, slot_name: str, title: str, load_button: str | None = None
    ) -> QGroupBox:
        """Create a group box with optional load button, labels, and waveform."""
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)

        # Top row: load button + filename
        top_row = QHBoxLayout()

        if load_button:
            btn = QPushButton(load_button)
            btn.clicked.connect(
                functools.partial(self._on_load_clicked, slot_name)
            )
            top_row.addWidget(btn)

        # Separate button (song slot only)
        if slot_name == "song":
            self._separate_btn = QPushButton("Separate")
            self._separate_btn.setEnabled(False)
            self._separate_btn.clicked.connect(self._on_separate_clicked)
            top_row.addWidget(self._separate_btn)

        file_label = QLabel("No file loaded")
        self._file_labels[slot_name] = file_label
        top_row.addWidget(file_label, stretch=1)
        group_layout.addLayout(top_row)

        # Info label
        info_label = QLabel("")
        self._info_labels[slot_name] = info_label
        group_layout.addWidget(info_label)

        # Separation progress label (song slot only)
        if slot_name == "song":
            self._separate_progress_label = QLabel("")
            self._separate_progress_label.setVisible(False)
            group_layout.addWidget(self._separate_progress_label)

        # Waveform
        waveform = WaveformWidget()
        self._waveforms[slot_name] = waveform
        group_layout.addWidget(waveform, stretch=1)

        return group

    def _create_track_row(self, slot_name: str, label_text: str) -> QWidget:
        """Create a compact track row (no load button) inside a parent group."""
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Top row: label + filename
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel(f"{label_text}:"))

        file_label = QLabel("—")
        self._file_labels[slot_name] = file_label
        top_row.addWidget(file_label, stretch=1)
        container_layout.addLayout(top_row)

        # Info label
        info_label = QLabel("")
        self._info_labels[slot_name] = info_label
        container_layout.addWidget(info_label)

        # Waveform
        waveform = WaveformWidget()
        self._waveforms[slot_name] = waveform
        container_layout.addWidget(waveform, stretch=1)

        return container

    def _on_load_clicked(self, slot_name: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {slot_name.capitalize()} Track", "", _FILE_FILTER
        )
        if not path:
            return
        self._load_track(slot_name, path)

    def _load_track(self, slot_name: str, path: str) -> None:
        try:
            data, sample_rate = load_audio(path)
            info = get_audio_info(path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Load Error",
                f"Could not load file:\n{path}\n\n{exc}",
            )
            return

        self._tracks[slot_name] = (data, sample_rate)
        self._paths[slot_name] = path

        # Update labels
        filename = os.path.basename(path)
        self._file_labels[slot_name].setText(filename)

        duration = info["duration"]
        channels = info["channels"]
        sr = info["sample_rate"]
        ch_str = "mono" if channels == 1 else f"{channels}ch"
        self._info_labels[slot_name].setText(
            f"{duration:.1f}s | {ch_str} | {sr} Hz"
        )

        # Update waveform
        self._waveforms[slot_name].set_audio(data, sample_rate)

        # Enable Separate button when song is loaded
        if slot_name == "song":
            self._separate_btn.setEnabled(True)

        self.track_loaded.emit(slot_name)

    def _on_separate_clicked(self) -> None:
        song_path = self._paths.get("song")
        if song_path:
            self.separate_requested.emit(song_path)

    # --- Public API for separation ---

    def set_separate_enabled(self, enabled: bool) -> None:
        """Enable/disable the Separate button."""
        self._separate_btn.setEnabled(enabled)

    def set_separate_progress(self, text: str) -> None:
        """Update the separation progress label. Empty string hides it."""
        if text:
            self._separate_progress_label.setText(text)
            self._separate_progress_label.setVisible(True)
        else:
            self._separate_progress_label.setText("")
            self._separate_progress_label.setVisible(False)

    def _set_track_data(
        self, slot_name: str, data, sample_rate: int,
        path: str | None, default_label: str,
    ) -> None:
        """Internal helper to populate any track slot programmatically."""
        self._tracks[slot_name] = (data, sample_rate)
        self._paths[slot_name] = path

        if path:
            filename = os.path.basename(path)
        else:
            filename = default_label
        self._file_labels[slot_name].setText(filename)

        duration = len(data) / sample_rate
        ch_str = "mono" if data.ndim == 1 else f"{data.shape[1]}ch"
        self._info_labels[slot_name].setText(
            f"{duration:.1f}s | {ch_str} | {sample_rate} Hz"
        )

        self._waveforms[slot_name].set_audio(data, sample_rate)
        self.track_loaded.emit(slot_name)

    def set_minus_sep_track(
        self, data, sample_rate: int, path: str | None = None
    ) -> None:
        """Populate the minus-sep slot (after Demucs separation)."""
        self._set_track_data(
            "minus_sep", data, sample_rate, path, "Separated instrumental"
        )

    def set_vocal_sep_track(
        self, data, sample_rate: int, path: str | None = None
    ) -> None:
        """Populate the vocal-sep slot (after Demucs separation)."""
        self._set_track_data(
            "vocal_sep", data, sample_rate, path, "Separated vocal"
        )

    def set_vocal_track(
        self, data, sample_rate: int, path: str | None = None
    ) -> None:
        """Programmatically populate the vocal slot (e.g. after recording)."""
        self._set_track_data("vocal", data, sample_rate, path, "Recording")

    # --- Existing public API ---

    def get_track_path(self, slot_name: str) -> str | None:
        """Return the file path for a given slot, or None."""
        return self._paths.get(slot_name)

    def get_waveform(self, slot_name: str) -> WaveformWidget | None:
        """Return the WaveformWidget for a given slot, or None."""
        return self._waveforms.get(slot_name)

    @property
    def song_track(self) -> tuple | None:
        return self._tracks["song"]

    @property
    def minus_sep_track(self) -> tuple | None:
        return self._tracks["minus_sep"]

    @property
    def vocal_sep_track(self) -> tuple | None:
        return self._tracks["vocal_sep"]

    @property
    def minus_import_track(self) -> tuple | None:
        return self._tracks["minus_import"]

    @property
    def minus_track(self) -> tuple | None:
        """Convenience: returns minus_import if loaded, else minus_sep."""
        return self._tracks["minus_import"] or self._tracks["minus_sep"]

    @property
    def vocal_track(self) -> tuple | None:
        return self._tracks["vocal"]
