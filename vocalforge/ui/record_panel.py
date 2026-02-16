"""Record panel — audio device selection and recording controls.

Phase 1: device selection dropdowns.
Phase 3: playback transport controls and volume slider.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QGroupBox,
    QPushButton, QSlider,
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
        if self._output_devices:
            self._output_combo.setCurrentIndex(select_output)

    def _on_input_changed(self, index):
        if 0 <= index < len(self._input_devices):
            self._selected_input = self._input_devices[index]

    def _on_output_changed(self, index):
        if 0 <= index < len(self._output_devices):
            self._selected_output = self._output_devices[index]
            self.output_device_changed.emit(self._selected_output)

    def _on_volume_changed(self, value: int) -> None:
        self._volume_label.setText(f"{value}%")
        self.volume_changed.emit(value / 100.0)

    def set_playback_enabled(self, enabled: bool) -> None:
        """Enable or disable the Play button (called when track loaded/cleared)."""
        self._play_btn.setEnabled(enabled)

    def update_transport_state(self, playing: bool, paused: bool) -> None:
        """Sync button enabled states with engine state."""
        self._play_btn.setEnabled(not playing or paused)
        self._pause_btn.setEnabled(playing and not paused)
        self._stop_btn.setEnabled(playing or paused)

    def update_time_display(self, current_sec: float, total_sec: float) -> None:
        """Format and display the current playback time."""
        cur_m, cur_s = divmod(int(current_sec), 60)
        tot_m, tot_s = divmod(int(total_sec), 60)
        self._time_label.setText(f"{cur_m}:{cur_s:02d} / {tot_m}:{tot_s:02d}")

    @property
    def selected_input_device(self):
        return self._selected_input

    @property
    def selected_output_device(self):
        return self._selected_output
