"""Record panel — audio device selection, playback, and recording controls.

Phase 1: device selection dropdowns.
Phase 3: playback transport controls and volume slider.
Phase 4: recording controls with REC indicator and latency offset.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QGroupBox,
    QPushButton, QSlider, QDoubleSpinBox,
)

from vocalforge.audio.engine import (
    get_input_devices, get_output_devices,
    get_default_input_device, get_default_output_device,
)


class RecordPanel(QWidget):

    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    volume_changed = Signal(float)
    output_device_changed = Signal(object)
    input_device_changed = Signal(object)
    record_start_clicked = Signal()
    record_finish_clicked = Signal()
    record_stop_clicked = Signal()
    latency_offset_changed = Signal(float)
    track_selected = Signal(str)  # emits slot name: "song", "minus", "vocal"
    seek_requested = Signal(float)  # emits normalized position 0.0–1.0

    def __init__(self, parent=None):
        super().__init__(parent)

        self._selected_input = None
        self._selected_output = None

        layout = QVBoxLayout(self)

        # --- Device selection group ---
        device_group = QGroupBox("Audio Devices")
        device_layout = QVBoxLayout(device_group)

        # Input device
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Input:"))
        self._input_combo = QComboBox()
        self._input_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        input_row.addWidget(self._input_combo, stretch=1)
        device_layout.addLayout(input_row)

        # Output device
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output:"))
        self._output_combo = QComboBox()
        self._output_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        output_row.addWidget(self._output_combo, stretch=1)
        device_layout.addLayout(output_row)

        layout.addWidget(device_group)

        # --- Playback group ---
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout(playback_group)

        # Track selector
        track_row = QHBoxLayout()
        track_row.addWidget(QLabel("Track:"))
        self._track_combo = QComboBox()
        self._track_combo.addItems(
            ["Song", "Minus (sep)", "Vocal (sep)", "Minus (import)", "Vocal"]
        )
        self._track_combo.setCurrentIndex(1)  # default to Minus (sep)
        track_row.addWidget(self._track_combo, stretch=1)
        playback_layout.addLayout(track_row)

        # Transport buttons
        btn_row = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._pause_btn = QPushButton("Pause")
        self._stop_btn = QPushButton("Stop")
        self._play_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._play_btn)
        btn_row.addWidget(self._pause_btn)
        btn_row.addWidget(self._stop_btn)
        playback_layout.addLayout(btn_row)

        # Time display
        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        playback_layout.addWidget(self._time_label)

        # Seek slider
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 10000)
        self._seek_slider.setValue(0)
        playback_layout.addWidget(self._seek_slider)

        # Volume slider
        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Volume:"))
        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(100)
        vol_row.addWidget(self._volume_slider, stretch=1)
        self._volume_label = QLabel("100%")
        vol_row.addWidget(self._volume_label)
        playback_layout.addLayout(vol_row)

        layout.addWidget(playback_group)

        # --- Recording group ---
        recording_group = QGroupBox("Recording")
        recording_layout = QVBoxLayout(recording_group)

        # Record / Finish / Discard buttons
        rec_btn_row = QHBoxLayout()
        self._record_btn = QPushButton("Record")
        self._finish_btn = QPushButton("Finish")
        self._discard_btn = QPushButton("Discard")
        self._record_btn.setEnabled(False)
        self._finish_btn.setEnabled(False)
        self._discard_btn.setEnabled(False)
        rec_btn_row.addWidget(self._record_btn)
        rec_btn_row.addWidget(self._finish_btn)
        rec_btn_row.addWidget(self._discard_btn)
        recording_layout.addLayout(rec_btn_row)

        # REC indicator
        self._rec_indicator = QLabel("REC")
        self._rec_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rec_indicator.setStyleSheet(
            "color: red; font-weight: bold; font-size: 14px;"
        )
        self._rec_indicator.setVisible(False)
        recording_layout.addWidget(self._rec_indicator)

        # Latency offset
        latency_row = QHBoxLayout()
        latency_row.addWidget(QLabel("Latency offset (ms):"))
        self._latency_spin = QDoubleSpinBox()
        self._latency_spin.setRange(-500.0, 500.0)
        self._latency_spin.setValue(0.0)
        self._latency_spin.setSingleStep(1.0)
        self._latency_spin.setDecimals(1)
        latency_row.addWidget(self._latency_spin)
        recording_layout.addLayout(latency_row)

        layout.addWidget(recording_group)

        layout.addStretch()

        # Populate devices
        self._populate_devices()

        # Connect signals
        self._input_combo.currentIndexChanged.connect(self._on_input_changed)
        self._output_combo.currentIndexChanged.connect(self._on_output_changed)
        self._play_btn.clicked.connect(self.play_clicked)
        self._pause_btn.clicked.connect(self.pause_clicked)
        self._stop_btn.clicked.connect(self.stop_clicked)
        self._volume_slider.valueChanged.connect(self._on_volume_changed)
        self._record_btn.clicked.connect(self.record_start_clicked)
        self._finish_btn.clicked.connect(self.record_finish_clicked)
        self._discard_btn.clicked.connect(self.record_stop_clicked)
        self._latency_spin.valueChanged.connect(self._on_latency_changed)
        self._track_combo.currentIndexChanged.connect(self._on_track_changed)
        self._seek_slider.sliderReleased.connect(self._on_seek_released)

    def _populate_devices(self):
        default_input = get_default_input_device()
        default_output = get_default_output_device()

        self._input_devices = get_input_devices()
        self._output_devices = get_output_devices()

        select_input = 0
        for i, dev in enumerate(self._input_devices):
            label = f"{dev['name']} ({dev['channels']}ch)"
            self._input_combo.addItem(label, userData=dev["index"])
            if dev["index"] == default_input:
                select_input = i

        select_output = 0
        for i, dev in enumerate(self._output_devices):
            label = f"{dev['name']} ({dev['channels']}ch)"
            self._output_combo.addItem(label, userData=dev["index"])
            if dev["index"] == default_output:
                select_output = i

        if self._input_devices:
            self._input_combo.setCurrentIndex(select_input)
            self._selected_input = self._input_devices[select_input]
        if self._output_devices:
            self._output_combo.setCurrentIndex(select_output)
            self._selected_output = self._output_devices[select_output]

    def _on_input_changed(self, index):
        if 0 <= index < len(self._input_devices):
            self._selected_input = self._input_devices[index]
            self.input_device_changed.emit(self._selected_input)

    def _on_output_changed(self, index):
        if 0 <= index < len(self._output_devices):
            self._selected_output = self._output_devices[index]
            self.output_device_changed.emit(self._selected_output)

    def _on_volume_changed(self, value: int) -> None:
        self._volume_label.setText(f"{value}%")
        self.volume_changed.emit(value / 100.0)

    def _on_latency_changed(self, value: float) -> None:
        self.latency_offset_changed.emit(value)

    def _on_track_changed(self, index: int) -> None:
        names = ["song", "minus_sep", "vocal_sep", "minus_import", "vocal"]
        if 0 <= index < len(names):
            self.track_selected.emit(names[index])

    def _on_seek_released(self) -> None:
        self.seek_requested.emit(self._seek_slider.value() / 10000.0)

    def set_playback_enabled(self, enabled: bool) -> None:
        """Enable or disable the Play button (called when track loaded/cleared)."""
        self._play_btn.setEnabled(enabled)

    def set_record_enabled(self, enabled: bool) -> None:
        """Enable or disable the Record button."""
        self._record_btn.setEnabled(enabled)

    def update_transport_state(self, playing: bool, paused: bool) -> None:
        """Sync button enabled states with engine state."""
        self._play_btn.setEnabled(not playing or paused)
        self._pause_btn.setEnabled(playing and not paused)
        self._stop_btn.setEnabled(playing or paused)

    def update_recording_state(self, recording: bool) -> None:
        """Toggle UI state for recording mode."""
        self._rec_indicator.setVisible(recording)

        # Recording buttons
        self._record_btn.setEnabled(not recording)
        self._finish_btn.setEnabled(recording)
        self._discard_btn.setEnabled(recording)

        # Disable playback controls and device combos during recording
        self._play_btn.setEnabled(not recording)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(not recording)
        self._input_combo.setEnabled(not recording)
        self._output_combo.setEnabled(not recording)
        self._seek_slider.setEnabled(not recording)
        self._track_combo.setEnabled(not recording)

    def update_time_display(self, current_sec: float, total_sec: float) -> None:
        """Format and display the current playback time."""
        cur_m, cur_s = divmod(int(current_sec), 60)
        tot_m, tot_s = divmod(int(total_sec), 60)
        self._time_label.setText(f"{cur_m}:{cur_s:02d} / {tot_m}:{tot_s:02d}")

    def update_seek_slider(self, normalized: float) -> None:
        """Sync seek slider with playback position. Blocks signals to prevent feedback."""
        self._seek_slider.blockSignals(True)
        self._seek_slider.setValue(int(normalized * 10000))
        self._seek_slider.blockSignals(False)

    @property
    def selected_input_device(self):
        return self._selected_input

    @property
    def selected_output_device(self):
        return self._selected_output
