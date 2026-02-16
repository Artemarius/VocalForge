"""Mix panel — mixing controls and export."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt


class MixPanel(QWidget):

    mix_export_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        group = QGroupBox("Mix && Export")
        group_layout = QVBoxLayout(group)

        # Alignment mode selector
        align_row = QHBoxLayout()
        align_row.addWidget(QLabel("Alignment:"))
        self._align_combo = QComboBox()
        self._align_combo.addItems(["None", "Background music", "Vocal matching"])
        self._align_combo.setCurrentIndex(1)  # default to Background music
        align_row.addWidget(self._align_combo, stretch=1)
        group_layout.addLayout(align_row)

        # Alignment info
        self._alignment_label = QLabel("Offset: not yet computed")
        group_layout.addWidget(self._alignment_label)

        # Vocal balance slider
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Vocal:"))
        self._vocal_slider = QSlider(Qt.Horizontal)
        self._vocal_slider.setRange(0, 100)
        self._vocal_slider.setValue(50)
        self._vocal_slider.setTickPosition(QSlider.TicksBelow)
        self._vocal_slider.setTickInterval(25)
        slider_row.addWidget(self._vocal_slider, stretch=1)
        self._vocal_value_label = QLabel("1.0x")
        self._vocal_slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self._vocal_value_label)
        group_layout.addLayout(slider_row)

        # Target LUFS spinner
        lufs_row = QHBoxLayout()
        lufs_row.addWidget(QLabel("Target LUFS:"))
        self._lufs_spin = QDoubleSpinBox()
        self._lufs_spin.setRange(-30.0, -5.0)
        self._lufs_spin.setValue(-14.0)
        self._lufs_spin.setSingleStep(0.5)
        self._lufs_spin.setSuffix(" LUFS")
        lufs_row.addWidget(self._lufs_spin)
        lufs_row.addStretch()
        group_layout.addLayout(lufs_row)

        # Mix & Export button
        self._mix_btn = QPushButton("Mix && Export")
        self._mix_btn.setEnabled(False)
        self._mix_btn.clicked.connect(self.mix_export_clicked)
        group_layout.addWidget(self._mix_btn)

        # Status label
        self._status_label = QLabel("Ready")
        group_layout.addWidget(self._status_label)

        layout.addWidget(group)
        layout.addStretch()

    def _on_slider_changed(self, value: int) -> None:
        gain = value / 50.0
        self._vocal_value_label.setText(f"{gain:.1f}x")

    def vocal_gain(self) -> float:
        return self._vocal_slider.value() / 50.0

    def instrumental_gain(self) -> float:
        return 1.0

    def target_lufs(self) -> float:
        return self._lufs_spin.value()

    def set_alignment_info(self, lag_ms: float, lag_samples: int) -> None:
        self._alignment_label.setText(
            f"Offset: {lag_ms:+.1f} ms ({lag_samples:+d} samples)"
        )

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def alignment_mode(self) -> str:
        """Return the selected alignment mode: 'none', 'background', or 'vocal'."""
        index = self._align_combo.currentIndex()
        return ["none", "background", "vocal"][index]

    def set_mix_enabled(self, enabled: bool) -> None:
        self._mix_btn.setEnabled(enabled)
