"""Import panel — song loading with waveform display (7-track layout)."""

import functools
import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from vocalforge.ui.waveform import WaveformWidget
from vocalforge.utils.audio_io import get_audio_info, load_audio

_FILE_FILTER = "Audio Files (*.wav *.flac *.ogg);;All Files (*)"

_SLOT_NAMES = (
    "song", "minus_sep", "vocal_sep", "minus_import",
    "vocal", "vocal_processed", "mix_result",
)

_TARGET_LUFS = -14.0


def _lufs_normalize(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Normalize audio to -14 LUFS. Returns unchanged if too quiet."""
    try:
        from vocalforge.audio.mixer import measure_lufs
        current = measure_lufs(data, sample_rate)
        if not np.isfinite(current):
            return data
        gain_db = _TARGET_LUFS - current
        gain_linear = 10.0 ** (gain_db / 20.0)
        return (data * gain_linear).astype(np.float32)
    except Exception:
        return data


class ImportPanel(QWidget):
    """Panel with seven track slots including vocal_processed."""

    track_loaded = Signal(str)  # emits slot name on successful load
    separate_requested = Signal(str)  # emits song file path
    track_gain_changed = Signal(str, float)  # slot_name, linear gain

    def __init__(self, parent=None):
        super().__init__(parent)

        self._tracks: dict[str, tuple | None] = {s: None for s in _SLOT_NAMES}
        self._paths: dict[str, str | None] = {s: None for s in _SLOT_NAMES}
        self._waveforms: dict[str, WaveformWidget] = {}
        self._file_labels: dict[str, QLabel] = {}
        self._info_labels: dict[str, QLabel] = {}
        self._vol_sliders: dict[str, QSlider] = {}
        self._vol_labels: dict[str, QLabel] = {}

        # Outer layout with scroll area
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        outer_layout.addWidget(scroll)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        scroll.setWidget(inner)

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

        # --- Vocal Processed group ---
        proc_group = QGroupBox("Vocal (Processed)")
        proc_layout = QVBoxLayout(proc_group)
        proc_layout.addWidget(
            self._create_track_row("vocal_processed", "Not yet processed")
        )
        layout.addWidget(proc_group)

        # --- Mix Result group ---
        mix_group = QGroupBox("Mix Result")
        mix_layout = QVBoxLayout(mix_group)
        mix_layout.addWidget(
            self._create_track_row("mix_result", "Not yet mixed")
        )
        layout.addWidget(mix_group)

        layout.addStretch()

    def _create_volume_row(self, slot_name: str) -> QHBoxLayout:
        """Create a volume slider row for a track."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Vol:"))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 200)
        slider.setValue(100)
        slider.setFixedHeight(18)
        row.addWidget(slider, stretch=1)
        label = QLabel("100%")
        label.setFixedWidth(38)
        row.addWidget(label)
        self._vol_sliders[slot_name] = slider
        self._vol_labels[slot_name] = label

        slider.valueChanged.connect(
            functools.partial(self._on_vol_changed, slot_name)
        )
        return row

    def _on_vol_changed(self, slot_name: str, value: int) -> None:
        pct = value
        self._vol_labels[slot_name].setText(f"{pct}%")
        gain = value / 100.0
        # Update waveform display gain
        wf = self._waveforms.get(slot_name)
        if wf is not None:
            wf.set_display_gain(gain)
        # Emit signal for engine gain update
        self.track_gain_changed.emit(slot_name, gain)

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

            # Clear button for minus_import
            if slot_name == "minus_import":
                clear_btn = QPushButton("\u2715")
                clear_btn.setFixedSize(24, 24)
                clear_btn.setToolTip("Clear minus import")
                clear_btn.clicked.connect(self._clear_minus_import)
                top_row.addWidget(clear_btn)

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

        group_layout.addSpacing(6)

        # Volume slider
        group_layout.addLayout(self._create_volume_row(slot_name))

        return group

    def _create_track_row(self, slot_name: str, label_text: str) -> QWidget:
        """Create a compact track row (no load button) inside a parent group."""
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Top row: label + filename
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel(f"{label_text}:"))

        file_label = QLabel("\u2014")
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

        container_layout.addSpacing(6)

        # Volume slider
        container_layout.addLayout(self._create_volume_row(slot_name))

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

        # LUFS normalize on load
        data = _lufs_normalize(data, sample_rate)

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

        # Reset volume slider to 100%
        slider = self._vol_sliders.get(slot_name)
        if slider:
            slider.setValue(100)

        # Enable Separate button when song is loaded
        if slot_name == "song":
            self._separate_btn.setEnabled(True)

        self.track_loaded.emit(slot_name)

    def _clear_minus_import(self) -> None:
        """Clear the minus import track slot."""
        self._tracks["minus_import"] = None
        self._paths["minus_import"] = None
        self._file_labels["minus_import"].setText("No file loaded")
        self._info_labels["minus_import"].setText("")
        self._waveforms["minus_import"].clear()
        slider = self._vol_sliders.get("minus_import")
        if slider:
            slider.setValue(100)
        self.track_loaded.emit("minus_import")

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
        normalize: bool = True,
    ) -> None:
        """Internal helper to populate any track slot programmatically."""
        if normalize:
            data = _lufs_normalize(data, sample_rate)

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

        # Reset volume slider to 100%
        slider = self._vol_sliders.get(slot_name)
        if slider:
            slider.setValue(100)

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

    def set_vocal_processed_track(
        self, data, sample_rate: int, path: str | None = None
    ) -> None:
        """Populate the vocal_processed slot (after effects chain)."""
        self._set_track_data(
            "vocal_processed", data, sample_rate, path, "Processed vocal",
            normalize=True,
        )

    def clear_slot(self, slot_name: str) -> None:
        """Reset a track slot to empty state (data, path, waveform, labels, volume)."""
        if slot_name not in _SLOT_NAMES:
            return
        if self._tracks[slot_name] is None:
            return  # already empty
        self._tracks[slot_name] = None
        self._paths[slot_name] = None
        self._file_labels[slot_name].setText(
            "No file loaded" if slot_name in ("song", "minus_import", "vocal")
            else "\u2014"
        )
        self._info_labels[slot_name].setText("")
        self._waveforms[slot_name].clear()
        slider = self._vol_sliders.get(slot_name)
        if slider:
            slider.setValue(100)
        self.track_loaded.emit(slot_name)

    # --- Volume fader access ---

    def get_track_gain(self, slot_name: str) -> float:
        """Return the current volume fader gain (0.0–2.0) for a slot."""
        slider = self._vol_sliders.get(slot_name)
        if slider:
            return slider.value() / 100.0
        return 1.0

    def set_track_gain(self, slot_name: str, gain: float) -> None:
        """Set a track's volume fader (also updates waveform display)."""
        slider = self._vol_sliders.get(slot_name)
        if slider:
            slider.setValue(int(gain * 100))

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

    @property
    def vocal_processed_track(self) -> tuple | None:
        return self._tracks["vocal_processed"]

    def set_mix_result_track(
        self, data, sample_rate: int, path: str | None = None
    ) -> None:
        """Populate the mix result slot (after mix & export)."""
        self._set_track_data(
            "mix_result", data, sample_rate, path, "Mix result",
            normalize=False,  # mix is already LUFS-normalized
        )

    @property
    def mix_result_track(self) -> tuple | None:
        return self._tracks["mix_result"]

    # --- Global cursor + offset helpers ---

    def set_all_cursors(
        self, position_samples: int, sr: int,
        offsets: dict[str, int] | None = None,
        total_duration: int = 0,
    ) -> None:
        """Set cursor on ALL non-empty waveforms at a global position."""
        if offsets is None:
            offsets = {}
        if total_duration <= 0:
            total_duration = 1

        for slot in _SLOT_NAMES:
            wf = self._waveforms.get(slot)
            if wf is None:
                continue
            track = self._tracks.get(slot)
            if track is None:
                wf.clear_cursor()
                continue
            normalized = position_samples / total_duration
            wf.set_cursor_position(max(0.0, min(1.0, normalized)))

    def apply_waveform_offsets(
        self, offsets: dict[str, int], total_duration: int,
    ) -> None:
        """Apply per-track offsets to waveform visualizations."""
        for slot in _SLOT_NAMES:
            wf = self._waveforms.get(slot)
            if wf is None:
                continue
            offset = offsets.get(slot, 0)
            wf.set_offset(offset, total_duration)
