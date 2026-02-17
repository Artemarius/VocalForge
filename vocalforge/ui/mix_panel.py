"""Mix panel — effects chain controls, alignment, and export."""

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

# Effect display definitions: (config_key, label, stub, controls_factory)
# controls_factory returns (widget_list, value_getter_name) or None for stubs.

_NR_STRENGTHS = {"Subtle": 0.5, "Moderate": 0.75, "Aggressive": 1.0}


class MixPanel(QWidget):

    mix_export_clicked = Signal()
    auto_align_clicked = Signal()
    minus_offset_changed = Signal(float)   # ms
    vocal_offset_changed = Signal(float)   # ms
    apply_effects_clicked = Signal()
    auto_tune_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # ============================================================
        # Alignment group
        # ============================================================
        align_group = QGroupBox("Alignment")
        align_layout = QVBoxLayout(align_group)

        # Alignment mode selector
        align_row = QHBoxLayout()
        align_row.addWidget(QLabel("Mode:"))
        self._align_combo = QComboBox()
        self._align_combo.addItems(["None", "Background music", "Vocal matching"])
        self._align_combo.setCurrentIndex(2)  # default to Vocal matching
        align_row.addWidget(self._align_combo, stretch=1)
        align_layout.addLayout(align_row)

        # Minus offset spinner
        minus_off_row = QHBoxLayout()
        minus_off_row.addWidget(QLabel("Minus offset:"))
        self._minus_offset_spin = QDoubleSpinBox()
        self._minus_offset_spin.setRange(-60000.0, 60000.0)
        self._minus_offset_spin.setValue(0.0)
        self._minus_offset_spin.setSingleStep(1.0)
        self._minus_offset_spin.setDecimals(1)
        self._minus_offset_spin.setSuffix(" ms")
        minus_off_row.addWidget(self._minus_offset_spin)
        minus_off_row.addStretch()
        align_layout.addLayout(minus_off_row)

        # Vocal offset spinner
        vocal_off_row = QHBoxLayout()
        vocal_off_row.addWidget(QLabel("Vocal offset:"))
        self._vocal_offset_spin = QDoubleSpinBox()
        self._vocal_offset_spin.setRange(-60000.0, 60000.0)
        self._vocal_offset_spin.setValue(0.0)
        self._vocal_offset_spin.setSingleStep(1.0)
        self._vocal_offset_spin.setDecimals(1)
        self._vocal_offset_spin.setSuffix(" ms")
        vocal_off_row.addWidget(self._vocal_offset_spin)
        vocal_off_row.addStretch()
        align_layout.addLayout(vocal_off_row)

        # Auto-Align button
        self._auto_align_btn = QPushButton("Auto-Align")
        self._auto_align_btn.setEnabled(False)
        self._auto_align_btn.clicked.connect(self.auto_align_clicked)
        align_layout.addWidget(self._auto_align_btn)

        # Confidence / info label
        self._alignment_label = QLabel("Offset: not yet computed")
        align_layout.addWidget(self._alignment_label)

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
        align_layout.addLayout(offset_row)

        # Alignment warning label
        self._alignment_warning = QLabel("")
        self._alignment_warning.setStyleSheet("color: orange; font-weight: bold;")
        self._alignment_warning.setVisible(False)
        self._alignment_warning.setWordWrap(True)
        align_layout.addWidget(self._alignment_warning)

        layout.addWidget(align_group)

        # ============================================================
        # Effects chain group
        # ============================================================
        effects_group = QGroupBox("Vocal Effects Chain")
        effects_layout = QVBoxLayout(effects_group)

        self._effect_checkboxes: dict[str, QCheckBox] = {}
        self._effect_controls: dict[str, dict] = {}

        # 1. Noise Gate (stub)
        self._add_effect_row(effects_layout, "noise_gate", "Noise Gate",
                             stub=True)

        # 2. Noise Reduction (working)
        nr_combo = QComboBox()
        nr_combo.addItems(["Subtle", "Moderate", "Aggressive"])
        nr_combo.setCurrentIndex(1)
        self._add_effect_row(effects_layout, "spectral_noise_reduction",
                             "Noise Reduction", stub=False,
                             controls=[nr_combo])
        self._effect_controls["spectral_noise_reduction"] = {"combo": nr_combo}

        # 3. De-Reverb (stub)
        self._add_effect_row(effects_layout, "dereverb", "De-Reverb", stub=True)

        # 4. High-Pass Filter (working)
        hpf_spin = QSpinBox()
        hpf_spin.setRange(0, 200)
        hpf_spin.setValue(80)
        hpf_spin.setSuffix(" Hz")
        hpf_spin.setToolTip("High-pass filter cutoff (0 = disabled)")
        self._add_effect_row(effects_layout, "highpass_filter",
                             "High-Pass Filter", stub=False,
                             controls=[hpf_spin])
        self._effect_controls["highpass_filter"] = {"spin": hpf_spin}

        # 5. Parametric EQ (stub)
        self._add_effect_row(effects_layout, "parametric_eq", "Parametric EQ",
                             stub=True)

        # 6. Compressor (stub)
        self._add_effect_row(effects_layout, "compressor", "Compressor",
                             stub=True)

        # 7. De-Esser (stub)
        self._add_effect_row(effects_layout, "de_esser", "De-Esser", stub=True)

        # 8. Reverb (stub)
        self._add_effect_row(effects_layout, "reverb", "Reverb", stub=True)

        # 9. Limiter (working)
        limiter_spin = QDoubleSpinBox()
        limiter_spin.setRange(-3.0, 0.0)
        limiter_spin.setValue(-1.0)
        limiter_spin.setSingleStep(0.1)
        limiter_spin.setSuffix(" dB")
        limiter_spin.setToolTip("Limiter ceiling in dB")
        self._add_effect_row(effects_layout, "limiter", "Limiter", stub=False,
                             controls=[limiter_spin])
        self._effect_controls["limiter"] = {"spin": limiter_spin}

        # Apply Effects button
        self._apply_effects_btn = QPushButton("Apply Effects to Vocal")
        self._apply_effects_btn.setEnabled(False)
        self._apply_effects_btn.setToolTip(
            "Process vocal through the effects chain and load into V-proc slot"
        )
        self._apply_effects_btn.clicked.connect(self.apply_effects_clicked)
        effects_layout.addWidget(self._apply_effects_btn)

        # Effects status label
        self._effects_status = QLabel("")
        effects_layout.addWidget(self._effects_status)

        layout.addWidget(effects_group)

        # ============================================================
        # Mix & Export group
        # ============================================================
        mix_group = QGroupBox("Mix && Export")
        mix_layout = QVBoxLayout(mix_group)

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
        mix_layout.addLayout(slider_row)

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
        mix_layout.addLayout(lufs_row)

        # Auto-Tune Volume button
        self._auto_tune_btn = QPushButton("Auto-Tune Volume")
        self._auto_tune_btn.setEnabled(False)
        self._auto_tune_btn.setToolTip(
            "Automatically adjust vocal and minus volume faders for optimal balance"
        )
        self._auto_tune_btn.clicked.connect(self.auto_tune_clicked)
        mix_layout.addWidget(self._auto_tune_btn)

        # Mix & Export button
        self._mix_btn = QPushButton("Mix && Export")
        self._mix_btn.setEnabled(False)
        self._mix_btn.clicked.connect(self.mix_export_clicked)
        mix_layout.addWidget(self._mix_btn)

        # Status label
        self._status_label = QLabel("Ready")
        mix_layout.addWidget(self._status_label)

        layout.addWidget(mix_group)
        layout.addStretch()

        # Connect offset spinner signals
        self._minus_offset_spin.valueChanged.connect(self.minus_offset_changed)
        self._vocal_offset_spin.valueChanged.connect(self.vocal_offset_changed)
        self._minus_offset_spin.valueChanged.connect(
            lambda v: self._update_offset_color(self._minus_offset_spin, v)
        )
        self._vocal_offset_spin.valueChanged.connect(
            lambda v: self._update_offset_color(self._vocal_offset_spin, v)
        )

    # --- Effect row helper ---

    def _add_effect_row(
        self, parent_layout: QVBoxLayout, key: str, label: str,
        stub: bool = False, controls: list | None = None,
    ) -> None:
        row = QHBoxLayout()
        cb = QCheckBox(label)
        self._effect_checkboxes[key] = cb

        if stub:
            cb.setChecked(False)
            cb.setEnabled(False)
            cb.setToolTip("Coming soon")
            row.addWidget(cb)
            coming = QLabel("(coming soon)")
            coming.setStyleSheet("color: gray; font-style: italic;")
            row.addWidget(coming)
            row.addStretch()
        else:
            cb.setChecked(True)
            row.addWidget(cb)
            if controls:
                for ctrl in controls:
                    cb.toggled.connect(ctrl.setEnabled)
                    row.addWidget(ctrl)
            row.addStretch()

        parent_layout.addLayout(row)

    # --- Slider callback ---

    def _on_slider_changed(self, value: int) -> None:
        gain = value / 50.0
        self._vocal_value_label.setText(f"{gain:.1f}x")

    # --- Public getters ---

    def vocal_gain(self) -> float:
        return self._vocal_slider.value() / 50.0

    def instrumental_gain(self) -> float:
        return 1.0

    def target_lufs(self) -> float:
        return self._lufs_spin.value()

    def alignment_mode(self) -> str:
        """Return the selected alignment mode: 'none', 'background', or 'vocal'."""
        index = self._align_combo.currentIndex()
        return ["none", "background", "vocal"][index]

    def max_offset_s(self) -> float | None:
        """Return max alignment offset in seconds, or None if 0 (no limit)."""
        val = self._max_offset_spin.value()
        return float(val) if val > 0 else None

    def minus_offset_ms(self) -> float:
        return self._minus_offset_spin.value()

    def vocal_offset_ms(self) -> float:
        return self._vocal_offset_spin.value()

    # --- Effects config ---

    def get_effects_config(self) -> dict:
        """Read all effect controls and return config for process_vocal()."""
        config = {}

        # Noise Gate (stub)
        config["noise_gate"] = {"enabled": False}

        # Noise Reduction
        nr_cb = self._effect_checkboxes["spectral_noise_reduction"]
        nr_combo = self._effect_controls["spectral_noise_reduction"]["combo"]
        strength_map = {0: 0.5, 1: 0.75, 2: 1.0}
        config["spectral_noise_reduction"] = {
            "enabled": nr_cb.isChecked(),
            "strength": strength_map[nr_combo.currentIndex()],
        }

        # De-Reverb (stub)
        config["dereverb"] = {"enabled": False}

        # High-Pass Filter
        hpf_cb = self._effect_checkboxes["highpass_filter"]
        hpf_spin = self._effect_controls["highpass_filter"]["spin"]
        config["highpass_filter"] = {
            "enabled": hpf_cb.isChecked(),
            "cutoff_hz": float(hpf_spin.value()),
        }

        # Parametric EQ (stub)
        config["parametric_eq"] = {"enabled": False}

        # Compressor (stub)
        config["compressor"] = {"enabled": False}

        # De-Esser (stub)
        config["de_esser"] = {"enabled": False}

        # Reverb (stub)
        config["reverb"] = {"enabled": False}

        # Limiter
        lim_cb = self._effect_checkboxes["limiter"]
        lim_spin = self._effect_controls["limiter"]["spin"]
        config["limiter"] = {
            "enabled": lim_cb.isChecked(),
            "ceiling_db": lim_spin.value(),
        }

        return config

    # --- Setters ---

    def set_alignment_info(self, lag_ms: float, lag_samples: int) -> None:
        self._alignment_label.setText(
            f"Offset: {lag_ms:+.1f} ms ({lag_samples:+d} samples)"
        )

    def set_align_confidence(self, text: str) -> None:
        self._alignment_label.setText(text)

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_alignment_warning(self, text: str) -> None:
        if text:
            self._alignment_warning.setText(text)
            self._alignment_warning.setVisible(True)
        else:
            self._alignment_warning.setText("")
            self._alignment_warning.setVisible(False)

    def set_mix_enabled(self, enabled: bool) -> None:
        self._mix_btn.setEnabled(enabled)

    def set_auto_align_enabled(self, enabled: bool) -> None:
        self._auto_align_btn.setEnabled(enabled)

    def set_minus_offset(self, ms: float) -> None:
        self._minus_offset_spin.blockSignals(True)
        self._minus_offset_spin.setValue(ms)
        self._minus_offset_spin.blockSignals(False)
        self._update_offset_color(self._minus_offset_spin, ms)

    def set_vocal_offset(self, ms: float) -> None:
        self._vocal_offset_spin.blockSignals(True)
        self._vocal_offset_spin.setValue(ms)
        self._vocal_offset_spin.blockSignals(False)
        self._update_offset_color(self._vocal_offset_spin, ms)

    def set_apply_effects_enabled(self, enabled: bool) -> None:
        self._apply_effects_btn.setEnabled(enabled)

    def set_effects_status(self, text: str) -> None:
        self._effects_status.setText(text)

    def set_auto_tune_enabled(self, enabled: bool) -> None:
        self._auto_tune_btn.setEnabled(enabled)

    @staticmethod
    def _update_offset_color(spin: QDoubleSpinBox, value: float) -> None:
        if abs(value) > 0.05:
            spin.setStyleSheet("background-color: #2A4A3A;")
        else:
            spin.setStyleSheet("")
