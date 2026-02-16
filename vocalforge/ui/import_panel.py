"""Import panel — song loading with waveform display."""

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


class ImportPanel(QWidget):
    """Panel with three track slots: Song, Minus, and Vocal."""

    track_loaded = Signal(str)  # emits slot name on successful load

    def __init__(self, parent=None):
        super().__init__(parent)

        self._tracks: dict[str, tuple | None] = {
            "song": None,
            "minus": None,
            "vocal": None,
        }
        self._waveforms: dict[str, WaveformWidget] = {}
        self._file_labels: dict[str, QLabel] = {}
        self._info_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)

        layout.addWidget(
            self._create_track_group("song", "Song (Plus Track)", "Load Song")
        )
        layout.addWidget(
            self._create_track_group("minus", "Minus Track", "Load Minus")
        )
        layout.addWidget(
            self._create_track_group("vocal", "Vocal Track", "Load Vocal")
        )

        layout.addStretch()

    def _create_track_group(
        self, slot_name: str, title: str, button_text: str
    ) -> QGroupBox:
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)

        # Top row: load button + filename
        top_row = QHBoxLayout()
        btn = QPushButton(button_text)
        btn.clicked.connect(functools.partial(self._on_load_clicked, slot_name))
        top_row.addWidget(btn)

        file_label = QLabel("No file loaded")
        self._file_labels[slot_name] = file_label
        top_row.addWidget(file_label, stretch=1)
        group_layout.addLayout(top_row)

        # Info label
        info_label = QLabel("")
        self._info_labels[slot_name] = info_label
        group_layout.addWidget(info_label)

        # Waveform
        waveform = WaveformWidget()
        self._waveforms[slot_name] = waveform
        group_layout.addWidget(waveform, stretch=1)

        return group

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

        self.track_loaded.emit(slot_name)

    @property
    def song_track(self) -> tuple | None:
        return self._tracks["song"]

    @property
    def minus_track(self) -> tuple | None:
        return self._tracks["minus"]

    @property
    def vocal_track(self) -> tuple | None:
        return self._tracks["vocal"]
