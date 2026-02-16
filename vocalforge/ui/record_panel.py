"""Record panel — audio device selection and recording controls.

Phase 1: device selection dropdowns only.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QGroupBox,
)

from vocalforge.audio.engine import (
    get_input_devices, get_output_devices,
    get_default_input_device, get_default_output_device,
)


class RecordPanel(QWidget):

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

        # Placeholder for future recording controls
        layout.addStretch()

        # Populate devices
        self._populate_devices()

        # Connect signals
        self._input_combo.currentIndexChanged.connect(self._on_input_changed)
        self._output_combo.currentIndexChanged.connect(self._on_output_changed)

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

    @property
    def selected_input_device(self):
        return self._selected_input

    @property
    def selected_output_device(self):
        return self._selected_output
