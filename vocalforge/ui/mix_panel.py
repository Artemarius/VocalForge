"""Mix panel — mixing controls and export."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


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
        self._align_combo.setCurrentIndex(2)  # default to Vocal matching
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
        self._vocal_slider.setValue(75)
        self._vocal_slider.setTickPosition(QSlider.TicksBelow)
        self._vocal_slider.setTickInterval(25)
        slider_row.addWidget(self._vocal_slider, stretch=1)
        self._vocal_value_label = QLabel("1.5x")
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

        # Noise reduction controls
        nr_row = QHBoxLayout()
        self._nr_checkbox = QCheckBox("Noise reduction")
        self._nr_checkbox.setChecked(True)
        nr_row.addWidget(self._nr_checkbox)
        self._nr_combo = QComboBox()
        self._nr_combo.addItems(["Subtle", "Moderate", "Aggressive"])
        self._nr_combo.setCurrentIndex(1)  # default Moderate
        nr_row.addWidget(self._nr_combo, stretch=1)
        self._nr_checkbox.toggled.connect(self._nr_combo.setEnabled)
        group_layout.addLayout(nr_row)

        # HPF cutoff spinner (tied to NR enabled state)
        hpf_row = QHBoxLayout()
        hpf_row.addWidget(QLabel("HPF cutoff:"))
        self._hpf_spin = QSpinBox()
        self._hpf_spin.setRange(0, 200)
        self._hpf_spin.setValue(80)
        self._hpf_spin.setSuffix(" Hz")
        self._hpf_spin.setToolTip("High-pass filter cutoff (0 = disabled)")
        hpf_row.addWidget(self._hpf_spin)
        hpf_row.addStretch()
        self._nr_checkbox.toggled.connect(self._hpf_spin.setEnabled)
        group_layout.addLayout(hpf_row)

        # Max alignment offset spinner
        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Max offset:"))
        self._max_offset_spin = QSpinBox()
        self._max_offset_spin.setRange(0, 300)
        self._max_offset_spin.setValue(30)
        self._max_offset_spin.setSuffix(" s")
        self._max_offset_spin.setToolTip("Max alignment search window (0 = no limit)")
        offset_row.addWidget(self._max_offset_spin)
        offset_row.addStretch()
        group_layout.addLayout(offset_row)

        # Alignment warning label
        self._alignment_warning = QLabel("")
        self._alignment_warning.setStyleSheet("color: orange; font-weight: bold;")
        self._alignment_warning.setVisible(False)
        self._alignment_warning.setWordWrap(True)
        group_layout.addWidget(self._alignment_warning)

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

    def noise_reduction_enabled(self) -> bool:
        return self._nr_checkbox.isChecked()

    def noise_reduction_strength(self) -> float:
        """Map combo selection to a prop_decrease value."""
        return {0: 0.5, 1: 0.75, 2: 1.0}[self._nr_combo.currentIndex()]

    def hpf_cutoff_hz(self) -> float:
        """Return HPF cutoff in Hz, or 0.0 if NR is disabled."""
        if not self._nr_checkbox.isChecked():
            return 0.0
        return float(self._hpf_spin.value())

    def max_offset_s(self) -> float | None:
        """Return max alignment offset in seconds, or None if 0 (no limit)."""
        val = self._max_offset_spin.value()
        return float(val) if val > 0 else None

    def set_alignment_warning(self, text: str) -> None:
        """Show or hide the alignment warning label."""
        if text:
            self._alignment_warning.setText(text)
            self._alignment_warning.setVisible(True)
        else:
            self._alignment_warning.setText("")
            self._alignment_warning.setVisible(False)

    def set_mix_enabled(self, enabled: bool) -> None:
        self._mix_btn.setEnabled(enabled)
