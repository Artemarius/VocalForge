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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vocalforge.ui import JumpSlider


# Effect display definitions: (config_key, label, stub, controls_factory)
# controls_factory returns (widget_list, value_getter_name) or None for stubs.

_NR_STRENGTHS = {"Subtle": 0.5, "Moderate": 0.75, "Aggressive": 1.0}


class MixPanel(QWidget):

    mix_export_clicked = Signal()
    auto_align_clicked = Signal()
    minus_offset_changed = Signal(float)   # ms
    vocal_offset_changed = Signal(float)   # ms
    apply_effects_clicked = Signal()
    export_vocal_clicked = Signal()
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

        # Minus offset slider
        minus_off_row = QHBoxLayout()
        minus_off_row.addWidget(QLabel("Minus offset:"))
        self._minus_offset_slider = JumpSlider(Qt.Horizontal)
        self._minus_offset_slider.setRange(-30000, 30000)
        self._minus_offset_slider.setValue(0)
        self._minus_offset_slider.setSingleStep(1)
        self._minus_offset_slider.setPageStep(100)
        minus_off_row.addWidget(self._minus_offset_slider, stretch=1)
        self._minus_offset_label = QLabel("0 ms")
        self._minus_offset_label.setFixedWidth(70)
        minus_off_row.addWidget(self._minus_offset_label)
        self._minus_reset_btn = QPushButton("\u21BA")
        self._minus_reset_btn.setFixedSize(24, 24)
        self._minus_reset_btn.setToolTip("Reset minus offset to 0")
        self._minus_reset_btn.clicked.connect(self._on_reset_minus_offset)
        minus_off_row.addWidget(self._minus_reset_btn)
        align_layout.addLayout(minus_off_row)

        # Vocal offset slider
        vocal_off_row = QHBoxLayout()
        vocal_off_row.addWidget(QLabel("Vocal offset:"))
        self._vocal_offset_slider = JumpSlider(Qt.Horizontal)
        self._vocal_offset_slider.setRange(-30000, 30000)
        self._vocal_offset_slider.setValue(0)
        self._vocal_offset_slider.setSingleStep(1)
        self._vocal_offset_slider.setPageStep(100)
        vocal_off_row.addWidget(self._vocal_offset_slider, stretch=1)
        self._vocal_offset_label = QLabel("0 ms")
        self._vocal_offset_label.setFixedWidth(70)
        vocal_off_row.addWidget(self._vocal_offset_label)
        self._vocal_reset_btn = QPushButton("\u21BA")
        self._vocal_reset_btn.setFixedSize(24, 24)
        self._vocal_reset_btn.setToolTip("Reset vocal offset to 0")
        self._vocal_reset_btn.clicked.connect(self._on_reset_vocal_offset)
        vocal_off_row.addWidget(self._vocal_reset_btn)
        align_layout.addLayout(vocal_off_row)

        # Auto-Align button
        align_btn_row = QHBoxLayout()
        self._auto_align_btn = QPushButton("Auto-Align")
        self._auto_align_btn.setEnabled(False)
        self._auto_align_btn.clicked.connect(self.auto_align_clicked)
        align_btn_row.addWidget(self._auto_align_btn)

        align_layout.addLayout(align_btn_row)

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
        self._preset_updating = False  # guard against recursive signals

        # Global bypass checkbox
        self._effects_enabled_cb = QCheckBox("Enable Effects Chain")
        self._effects_enabled_cb.setChecked(True)
        self._effects_enabled_cb.setToolTip(
            "Uncheck to skip all effects (use for pre-enhanced vocals)"
        )
        self._effects_enabled_cb.toggled.connect(self._on_effects_enabled_toggled)
        effects_layout.addWidget(self._effects_enabled_cb)

        # Container widget for all effect controls (toggled by bypass checkbox)
        self._effects_container = QWidget()
        self._effects_container_layout = QVBoxLayout(self._effects_container)
        self._effects_container_layout.setContentsMargins(0, 0, 0, 0)
        effects_layout.addWidget(self._effects_container)

        # Preset row (top of effects container)
        ecl = self._effects_container_layout
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(["Custom", "Raw", "Clean", "Enhanced"])
        preset_row.addWidget(self._preset_combo, stretch=1)
        preset_row.addStretch()
        ecl.addLayout(preset_row)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        # 1. Noise Gate (working)
        gate_thresh_spin = QSpinBox()
        gate_thresh_spin.setRange(-60, -20)
        gate_thresh_spin.setValue(-35)
        gate_thresh_spin.setSuffix(" dB")
        gate_thresh_spin.setToolTip("Threshold below which the gate closes")
        gate_reduction_spin = QSpinBox()
        gate_reduction_spin.setRange(-60, 0)
        gate_reduction_spin.setValue(-40)
        gate_reduction_spin.setSuffix(" dB")
        gate_reduction_spin.setToolTip("Attenuation when gate is closed")
        self._add_effect_row(ecl, "noise_gate", "Noise Gate",
                             stub=False,
                             controls=[gate_thresh_spin, gate_reduction_spin])
        self._effect_controls["noise_gate"] = {
            "threshold_spin": gate_thresh_spin,
            "reduction_spin": gate_reduction_spin,
        }
        # Default: unchecked (matches DEFAULT_CONFIG enabled=False)
        self._effect_checkboxes["noise_gate"].setChecked(False)

        # 2. Noise Reduction (working)
        nr_combo = QComboBox()
        nr_combo.addItems(["Subtle", "Moderate", "Aggressive"])
        nr_combo.setCurrentIndex(1)
        self._add_effect_row(ecl, "spectral_noise_reduction",
                             "Noise Reduction", stub=False,
                             controls=[nr_combo])
        self._effect_controls["spectral_noise_reduction"] = {"combo": nr_combo}

        # 3. De-Reverb (working)
        dereverb_combo = QComboBox()
        dereverb_combo.addItems(["Light", "Medium", "Strong"])
        dereverb_combo.setCurrentIndex(0)
        self._add_effect_row(ecl, "dereverb", "De-Reverb",
                             stub=False,
                             controls=[dereverb_combo])
        self._effect_controls["dereverb"] = {"combo": dereverb_combo}
        # Default: unchecked (matches DEFAULT_CONFIG enabled=False)
        self._effect_checkboxes["dereverb"].setChecked(False)

        # 4. High-Pass Filter (working)
        hpf_spin = QSpinBox()
        hpf_spin.setRange(0, 200)
        hpf_spin.setValue(80)
        hpf_spin.setSuffix(" Hz")
        hpf_spin.setToolTip("High-pass filter cutoff (0 = disabled)")
        self._add_effect_row(ecl, "highpass_filter",
                             "High-Pass Filter", stub=False,
                             controls=[hpf_spin])
        self._effect_controls["highpass_filter"] = {"spin": hpf_spin}

        # 5. Parametric EQ (working)
        eq_combo = QComboBox()
        eq_combo.addItems(["Clean Up", "Warm", "Bright"])
        eq_combo.setCurrentIndex(0)
        eq_combo.setToolTip("EQ preset: shape the vocal tone")
        self._add_effect_row(ecl, "parametric_eq", "Parametric EQ",
                             stub=False, controls=[eq_combo])
        self._effect_controls["parametric_eq"] = {"combo": eq_combo}
        self._effect_checkboxes["parametric_eq"].setChecked(False)

        # 6. Compressor (working)
        comp_thresh_spin = QSpinBox()
        comp_thresh_spin.setRange(-40, 0)
        comp_thresh_spin.setValue(-18)
        comp_thresh_spin.setSuffix(" dB")
        comp_thresh_spin.setToolTip("Threshold above which compression starts")
        comp_ratio_spin = QDoubleSpinBox()
        comp_ratio_spin.setRange(1.0, 20.0)
        comp_ratio_spin.setValue(3.0)
        comp_ratio_spin.setSingleStep(0.5)
        comp_ratio_spin.setSuffix(":1")
        comp_ratio_spin.setToolTip("Compression ratio")
        self._add_effect_row(ecl, "compressor", "Compressor",
                             stub=False,
                             controls=[comp_thresh_spin, comp_ratio_spin])
        self._effect_controls["compressor"] = {
            "threshold_spin": comp_thresh_spin,
            "ratio_spin": comp_ratio_spin,
        }
        self._effect_checkboxes["compressor"].setChecked(False)

        # 7. De-Esser (stub)
        self._add_effect_row(ecl, "de_esser", "De-Esser", stub=True)

        # 8. Reverb (stub)
        self._add_effect_row(ecl, "reverb", "Reverb", stub=True)

        # 9. Limiter (working)
        limiter_spin = QDoubleSpinBox()
        limiter_spin.setRange(-3.0, 0.0)
        limiter_spin.setValue(-1.0)
        limiter_spin.setSingleStep(0.1)
        limiter_spin.setSuffix(" dB")
        limiter_spin.setToolTip("Limiter ceiling in dB")
        self._add_effect_row(ecl, "limiter", "Limiter", stub=False,
                             controls=[limiter_spin])
        self._effect_controls["limiter"] = {"spin": limiter_spin}

        # Connect all working effect controls to switch preset to "Custom"
        for key, cb in self._effect_checkboxes.items():
            cb.toggled.connect(self._on_effect_manual_change)
        for key, ctrls in self._effect_controls.items():
            for ctrl in ctrls.values():
                if isinstance(ctrl, QComboBox):
                    ctrl.currentIndexChanged.connect(self._on_effect_manual_change)
                elif isinstance(ctrl, (QSpinBox, QDoubleSpinBox)):
                    ctrl.valueChanged.connect(self._on_effect_manual_change)

        # Apply Effects button
        self._apply_effects_btn = QPushButton("Apply Effects to Vocal")
        self._apply_effects_btn.setEnabled(False)
        self._apply_effects_btn.setToolTip(
            "Process vocal through the effects chain and load into V-proc slot"
        )
        self._apply_effects_btn.clicked.connect(self.apply_effects_clicked)
        effects_layout.addWidget(self._apply_effects_btn)

        # Export Processed Vocal button
        self._export_vocal_btn = QPushButton("Export Processed Vocal")
        self._export_vocal_btn.setEnabled(False)
        self._export_vocal_btn.setToolTip(
            "Save the processed vocal track to a file"
        )
        self._export_vocal_btn.clicked.connect(self.export_vocal_clicked)
        effects_layout.addWidget(self._export_vocal_btn)

        # Effects status label
        self._effects_status = QLabel("")
        effects_layout.addWidget(self._effects_status)

        layout.addWidget(effects_group)

        # ============================================================
        # Mix & Export group
        # ============================================================
        mix_group = QGroupBox("Mix && Export")
        mix_layout = QVBoxLayout(mix_group)

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

        # Connect offset slider signals:
        # valueChanged → label update only (during drag)
        # sliderReleased → emit change signal (triggers processing)
        self._minus_offset_slider.valueChanged.connect(self._on_minus_offset_label_update)
        self._minus_offset_slider.sliderReleased.connect(self._on_minus_offset_released)
        self._vocal_offset_slider.valueChanged.connect(self._on_vocal_offset_label_update)
        self._vocal_offset_slider.sliderReleased.connect(self._on_vocal_offset_released)

        # Connect Max Offset spinner to update slider ranges dynamically
        self._max_offset_spin.valueChanged.connect(self._on_max_offset_changed)
        # Set initial slider range from default Max Offset value
        self._on_max_offset_changed(self._max_offset_spin.value())

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

    # --- Preset handling ---

    def _on_preset_changed(self, index: int) -> None:
        """Apply a preset's effect settings to all UI controls."""
        if self._preset_updating:
            return
        name = self._preset_combo.currentText()
        if name == "Custom":
            return  # user manually selected Custom, nothing to apply

        from vocalforge.audio.effects import PRESET_CONFIGS
        preset = PRESET_CONFIGS.get(name)
        if preset is None:
            return

        self._preset_updating = True
        try:
            self._apply_preset_config(preset)
        finally:
            self._preset_updating = False

    def _apply_preset_config(self, preset: dict) -> None:
        """Set all effect checkboxes and control values from a preset dict."""
        _NR_STRENGTH_INDEX = {0.5: 0, 0.75: 1, 1.0: 2}
        _DR_STRENGTH_INDEX = {0.3: 0, 0.5: 1, 0.7: 2}

        for key, cfg in preset.items():
            cb = self._effect_checkboxes.get(key)
            if cb is None:
                continue
            enabled = cfg.get("enabled", False)
            if cb.isEnabled():  # skip stub checkboxes
                cb.setChecked(enabled)

            ctrls = self._effect_controls.get(key, {})

            if key == "noise_gate":
                if "threshold_spin" in ctrls and "threshold_db" in cfg:
                    ctrls["threshold_spin"].setValue(int(cfg["threshold_db"]))
                if "reduction_spin" in ctrls and "reduction_db" in cfg:
                    ctrls["reduction_spin"].setValue(int(cfg["reduction_db"]))

            elif key == "spectral_noise_reduction":
                if "combo" in ctrls and "strength" in cfg:
                    idx = _NR_STRENGTH_INDEX.get(cfg["strength"], 1)
                    ctrls["combo"].setCurrentIndex(idx)

            elif key == "dereverb":
                if "combo" in ctrls and "strength" in cfg:
                    idx = _DR_STRENGTH_INDEX.get(cfg["strength"], 0)
                    ctrls["combo"].setCurrentIndex(idx)

            elif key == "highpass_filter":
                if "spin" in ctrls and "cutoff_hz" in cfg:
                    ctrls["spin"].setValue(int(cfg["cutoff_hz"]))

            elif key == "parametric_eq":
                if "combo" in ctrls and "preset" in cfg:
                    _EQ_PRESET_INDEX = {"clean_up": 0, "warm": 1, "bright": 2}
                    idx = _EQ_PRESET_INDEX.get(cfg["preset"], 0)
                    ctrls["combo"].setCurrentIndex(idx)

            elif key == "compressor":
                if "threshold_spin" in ctrls and "threshold_db" in cfg:
                    ctrls["threshold_spin"].setValue(int(cfg["threshold_db"]))
                if "ratio_spin" in ctrls and "ratio" in cfg:
                    ctrls["ratio_spin"].setValue(float(cfg["ratio"]))

            elif key == "limiter":
                if "spin" in ctrls and "ceiling_db" in cfg:
                    ctrls["spin"].setValue(cfg["ceiling_db"])

    def _on_effect_manual_change(self, *_args) -> None:
        """Any manual change to an effect control switches preset to Custom."""
        if self._preset_updating:
            return
        self._preset_updating = True
        self._preset_combo.setCurrentIndex(0)  # "Custom"
        self._preset_updating = False

    def _on_effects_enabled_toggled(self, enabled: bool) -> None:
        """Show/hide the effects container when the global bypass toggles."""
        self._effects_container.setVisible(enabled)

    # --- Public getters ---

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
        return float(self._minus_offset_slider.value())

    def vocal_offset_ms(self) -> float:
        return float(self._vocal_offset_slider.value())

    def effects_chain_enabled(self) -> bool:
        """Return whether the global effects chain is enabled."""
        return self._effects_enabled_cb.isChecked()

    # --- Effects config ---

    def get_effects_config(self) -> dict:
        """Read all effect controls and return config for process_vocal()."""
        config = {}

        # Noise Gate
        gate_cb = self._effect_checkboxes["noise_gate"]
        gate_ctrls = self._effect_controls["noise_gate"]
        config["noise_gate"] = {
            "enabled": gate_cb.isChecked(),
            "threshold_db": float(gate_ctrls["threshold_spin"].value()),
            "reduction_db": float(gate_ctrls["reduction_spin"].value()),
        }

        # Noise Reduction
        nr_cb = self._effect_checkboxes["spectral_noise_reduction"]
        nr_combo = self._effect_controls["spectral_noise_reduction"]["combo"]
        strength_map = {0: 0.5, 1: 0.75, 2: 1.0}
        config["spectral_noise_reduction"] = {
            "enabled": nr_cb.isChecked(),
            "strength": strength_map[nr_combo.currentIndex()],
        }

        # De-Reverb
        dr_cb = self._effect_checkboxes["dereverb"]
        dr_combo = self._effect_controls["dereverb"]["combo"]
        strength_map_dr = {0: 0.3, 1: 0.5, 2: 0.7}
        config["dereverb"] = {
            "enabled": dr_cb.isChecked(),
            "strength": strength_map_dr[dr_combo.currentIndex()],
        }

        # High-Pass Filter
        hpf_cb = self._effect_checkboxes["highpass_filter"]
        hpf_spin = self._effect_controls["highpass_filter"]["spin"]
        config["highpass_filter"] = {
            "enabled": hpf_cb.isChecked(),
            "cutoff_hz": float(hpf_spin.value()),
        }

        # Parametric EQ
        from vocalforge.audio.effects import EQ_PRESETS
        eq_cb = self._effect_checkboxes["parametric_eq"]
        eq_combo = self._effect_controls["parametric_eq"]["combo"]
        eq_preset_map = {0: "clean_up", 1: "warm", 2: "bright"}
        eq_preset_name = eq_preset_map[eq_combo.currentIndex()]
        config["parametric_eq"] = {
            "enabled": eq_cb.isChecked(),
            "preset": eq_preset_name,
            "bands": EQ_PRESETS[eq_preset_name],
        }

        # Compressor
        comp_cb = self._effect_checkboxes["compressor"]
        comp_ctrls = self._effect_controls["compressor"]
        config["compressor"] = {
            "enabled": comp_cb.isChecked(),
            "threshold_db": float(comp_ctrls["threshold_spin"].value()),
            "ratio": comp_ctrls["ratio_spin"].value(),
        }

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
        self._minus_offset_slider.blockSignals(True)
        self._minus_offset_slider.setValue(int(ms))
        self._minus_offset_slider.blockSignals(False)
        self._minus_offset_label.setText(f"{ms:+.0f} ms")
        self._update_offset_color_slider(self._minus_offset_label, ms)

    def set_vocal_offset(self, ms: float) -> None:
        self._vocal_offset_slider.blockSignals(True)
        self._vocal_offset_slider.setValue(int(ms))
        self._vocal_offset_slider.blockSignals(False)
        self._vocal_offset_label.setText(f"{ms:+.0f} ms")
        self._update_offset_color_slider(self._vocal_offset_label, ms)

    def set_apply_effects_enabled(self, enabled: bool) -> None:
        self._apply_effects_btn.setEnabled(enabled)

    def set_export_vocal_enabled(self, enabled: bool) -> None:
        self._export_vocal_btn.setEnabled(enabled)

    def set_effects_status(self, text: str) -> None:
        self._effects_status.setText(text)

    def set_auto_tune_enabled(self, enabled: bool) -> None:
        self._auto_tune_btn.setEnabled(enabled)

    def _on_minus_offset_label_update(self, value: int) -> None:
        """Update label during drag (no processing)."""
        ms = float(value)
        self._minus_offset_label.setText(f"{ms:+.0f} ms")
        self._update_offset_color_slider(self._minus_offset_label, value)

    def _on_minus_offset_released(self) -> None:
        """Emit change signal when slider is released (triggers processing)."""
        self.minus_offset_changed.emit(float(self._minus_offset_slider.value()))

    def _on_vocal_offset_label_update(self, value: int) -> None:
        """Update label during drag (no processing)."""
        ms = float(value)
        self._vocal_offset_label.setText(f"{ms:+.0f} ms")
        self._update_offset_color_slider(self._vocal_offset_label, value)

    def _on_vocal_offset_released(self) -> None:
        """Emit change signal when slider is released (triggers processing)."""
        self.vocal_offset_changed.emit(float(self._vocal_offset_slider.value()))

    def _on_max_offset_changed(self, value: int) -> None:
        """Update offset slider ranges when Max Offset spinner changes."""
        max_ms = value * 1000 if value > 0 else 60000
        self._minus_offset_slider.setRange(-max_ms, max_ms)
        self._vocal_offset_slider.setRange(-max_ms, max_ms)

    def _on_reset_minus_offset(self) -> None:
        """Reset minus offset slider to 0 and emit signal."""
        self.set_minus_offset(0.0)
        self.minus_offset_changed.emit(0.0)

    def _on_reset_vocal_offset(self) -> None:
        """Reset vocal offset slider to 0 and emit signal."""
        self.set_vocal_offset(0.0)
        self.vocal_offset_changed.emit(0.0)

    @staticmethod
    def _update_offset_color_slider(label: QLabel, value: float) -> None:
        if abs(value) > 0.5:
            label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            label.setStyleSheet("")
