"""Mix panel — effects chain controls, alignment, and export."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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


class _NRAdvancedDialog(QDialog):
    """Advanced noise reduction parameters dialog."""

    _DEFAULTS = {
        "n_std_thresh": 1.5,
        "freq_smooth_hz": 500,
        "time_smooth_ms": 50,
        "use_torch": None,
    }

    def __init__(self, current_values: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Noise Reduction — Advanced")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._thresh_spin = QDoubleSpinBox()
        self._thresh_spin.setRange(0.5, 3.0)
        self._thresh_spin.setSingleStep(0.1)
        self._thresh_spin.setValue(current_values.get("n_std_thresh", 1.5))
        self._thresh_spin.setToolTip("Stationary mode threshold sensitivity (lower = more aggressive)")
        form.addRow("Stat. threshold:", self._thresh_spin)

        self._freq_spin = QSpinBox()
        self._freq_spin.setRange(100, 2000)
        self._freq_spin.setSingleStep(50)
        self._freq_spin.setValue(current_values.get("freq_smooth_hz", 500))
        self._freq_spin.setSuffix(" Hz")
        self._freq_spin.setToolTip("Frequency mask smoothing width")
        form.addRow("Freq. smoothing:", self._freq_spin)

        self._time_spin = QSpinBox()
        self._time_spin.setRange(10, 200)
        self._time_spin.setSingleStep(5)
        self._time_spin.setValue(current_values.get("time_smooth_ms", 50))
        self._time_spin.setSuffix(" ms")
        self._time_spin.setToolTip("Temporal mask smoothing width")
        form.addRow("Time smoothing:", self._time_spin)

        self._cuda_cb = QCheckBox()
        # Auto-detect CUDA availability
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pass
        self._cuda_cb.setEnabled(cuda_available)
        use_torch_val = current_values.get("use_torch")
        if use_torch_val is None:
            self._cuda_cb.setChecked(cuda_available)
        else:
            self._cuda_cb.setChecked(bool(use_torch_val))
        self._cuda_cb.setToolTip(
            "Use GPU acceleration (CUDA)" if cuda_available
            else "CUDA not available — install torch with CUDA support"
        )
        form.addRow("Use GPU (CUDA):", self._cuda_cb)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        restore_btn = QPushButton("Restore Defaults")
        restore_btn.clicked.connect(self._restore_defaults)
        btn_layout.addWidget(restore_btn)
        btn_layout.addStretch()

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        btn_layout.addWidget(button_box)
        layout.addLayout(btn_layout)

    def _restore_defaults(self):
        self._thresh_spin.setValue(self._DEFAULTS["n_std_thresh"])
        self._freq_spin.setValue(self._DEFAULTS["freq_smooth_hz"])
        self._time_spin.setValue(self._DEFAULTS["time_smooth_ms"])
        cuda_available = self._cuda_cb.isEnabled()
        self._cuda_cb.setChecked(cuda_available)

    def values(self) -> dict:
        use_torch = self._cuda_cb.isChecked() if self._cuda_cb.isEnabled() else None
        return {
            "n_std_thresh": self._thresh_spin.value(),
            "freq_smooth_hz": self._freq_spin.value(),
            "time_smooth_ms": self._time_spin.value(),
            "use_torch": use_torch,
        }


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
        self._nr_advanced_values = {
            "n_std_thresh": 1.5, "freq_smooth_hz": 500,
            "time_smooth_ms": 50, "use_torch": None,
        }

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
        self._preset_combo.addItems(["Custom", "Raw", "Clean", "Enhanced",
                                         "Broadcast", "Warm", "Bright"])
        preset_row.addWidget(self._preset_combo, stretch=1)
        preset_row.addStretch()
        ecl.addLayout(preset_row)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        # 1. Noise Gate
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
        self._effect_checkboxes["noise_gate"].setChecked(True)

        # 2. Noise Reduction (Pass 1 — main, stem-guided)
        nr_combo = QComboBox()
        nr_combo.addItems(["Subtle", "Moderate", "Aggressive"])
        nr_combo.setCurrentIndex(1)
        nr_mode_combo = QComboBox()
        nr_mode_combo.addItems(["Auto", "Stationary", "Adaptive"])
        nr_mode_combo.setCurrentIndex(0)
        nr_mode_combo.setToolTip(
            "Auto: stationary when stem-guided profile available, adaptive otherwise\n"
            "Stationary: best for constant noise (fan, hiss) with good profile\n"
            "Adaptive: handles varying noise, less dependent on profile quality"
        )
        nr_adv_btn = QPushButton("\u2699")
        nr_adv_btn.setFixedSize(24, 24)
        nr_adv_btn.setToolTip("Advanced noise reduction parameters")
        nr_adv_btn.clicked.connect(self._on_nr_advanced)
        self._add_effect_row(ecl, "spectral_noise_reduction",
                             "Noise Reduction", stub=False,
                             controls=[nr_combo, nr_mode_combo, nr_adv_btn])
        self._effect_controls["spectral_noise_reduction"] = {
            "combo": nr_combo, "mode_combo": nr_mode_combo, "adv_btn": nr_adv_btn,
        }

        # 3. Gain Rider
        gr_target_spin = QSpinBox()
        gr_target_spin.setRange(-30, -12)
        gr_target_spin.setValue(-20)
        gr_target_spin.setSuffix(" dB")
        gr_target_spin.setToolTip("Target RMS level for gain riding")
        gr_max_gain_spin = QSpinBox()
        gr_max_gain_spin.setRange(0, 12)
        gr_max_gain_spin.setValue(4)
        gr_max_gain_spin.setSuffix(" dB")
        gr_max_gain_spin.setToolTip("Maximum boost for quiet sections")
        self._add_effect_row(ecl, "gain_rider", "Gain Rider",
                             stub=False,
                             controls=[gr_target_spin, gr_max_gain_spin])
        self._effect_controls["gain_rider"] = {
            "target_spin": gr_target_spin,
            "max_gain_spin": gr_max_gain_spin,
        }
        self._effect_checkboxes["gain_rider"].setChecked(True)

        # 4. De-Plosive
        dp_thresh_spin = QSpinBox()
        dp_thresh_spin.setRange(-40, -10)
        dp_thresh_spin.setValue(-25)
        dp_thresh_spin.setSuffix(" dB")
        dp_thresh_spin.setToolTip("Detection threshold for plosive events")
        dp_reduction_spin = QSpinBox()
        dp_reduction_spin.setRange(3, 18)
        dp_reduction_spin.setValue(10)
        dp_reduction_spin.setSuffix(" dB")
        dp_reduction_spin.setToolTip("Attenuation applied to plosive energy")
        self._add_effect_row(ecl, "de_plosive", "De-Plosive",
                             stub=False,
                             controls=[dp_thresh_spin, dp_reduction_spin])
        self._effect_controls["de_plosive"] = {
            "threshold_spin": dp_thresh_spin,
            "reduction_spin": dp_reduction_spin,
        }
        self._effect_checkboxes["de_plosive"].setChecked(True)

        # 5. NR Cleanup (Pass 2 — gentle residual noise cleanup after gain rider)
        nr2_combo = QComboBox()
        nr2_combo.addItems(["Light", "Medium", "Strong"])
        nr2_combo.setCurrentIndex(2)
        nr2_combo.setToolTip(
            "Residual noise cleanup after gain rider\n"
            "Light: barely touches signal, catches obvious residual noise\n"
            "Medium: good balance\n"
            "Strong: more aggressive, risk of artifacts"
        )
        self._add_effect_row(ecl, "nr_cleanup", "NR Cleanup",
                             stub=False,
                             controls=[nr2_combo])
        self._effect_controls["nr_cleanup"] = {"combo": nr2_combo}
        self._effect_checkboxes["nr_cleanup"].setChecked(True)

        # 6. De-Reverb
        dereverb_combo = QComboBox()
        dereverb_combo.addItems(["Light", "Medium", "Strong"])
        dereverb_combo.setCurrentIndex(1)
        self._add_effect_row(ecl, "dereverb", "De-Reverb",
                             stub=False,
                             controls=[dereverb_combo])
        self._effect_controls["dereverb"] = {"combo": dereverb_combo}
        # Default: unchecked (matches DEFAULT_CONFIG enabled=False)
        self._effect_checkboxes["dereverb"].setChecked(True)

        # 4. High-Pass Filter (working)
        hpf_spin = QSpinBox()
        hpf_spin.setRange(0, 200)
        hpf_spin.setValue(120)
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
        self._effect_checkboxes["parametric_eq"].setChecked(True)

        # 8. Compressor Peak (fast attack, catches transient spikes)
        cpeak_thresh_spin = QSpinBox()
        cpeak_thresh_spin.setRange(-40, 0)
        cpeak_thresh_spin.setValue(-12)
        cpeak_thresh_spin.setSuffix(" dB")
        cpeak_thresh_spin.setToolTip("Threshold above which peak compression starts")
        cpeak_ratio_spin = QDoubleSpinBox()
        cpeak_ratio_spin.setRange(1.0, 20.0)
        cpeak_ratio_spin.setValue(8.0)
        cpeak_ratio_spin.setSingleStep(0.5)
        cpeak_ratio_spin.setSuffix(":1")
        cpeak_ratio_spin.setToolTip("Peak compression ratio")
        self._add_effect_row(ecl, "compressor_peak", "Compressor (Peak)",
                             stub=False,
                             controls=[cpeak_thresh_spin, cpeak_ratio_spin])
        self._effect_controls["compressor_peak"] = {
            "threshold_spin": cpeak_thresh_spin,
            "ratio_spin": cpeak_ratio_spin,
        }
        self._effect_checkboxes["compressor_peak"].setChecked(True)

        # 9. Compressor Body (slow attack, smooths overall dynamics)
        cbody_thresh_spin = QSpinBox()
        cbody_thresh_spin.setRange(-40, 0)
        cbody_thresh_spin.setValue(-20)
        cbody_thresh_spin.setSuffix(" dB")
        cbody_thresh_spin.setToolTip("Threshold above which body compression starts")
        cbody_ratio_spin = QDoubleSpinBox()
        cbody_ratio_spin.setRange(1.0, 20.0)
        cbody_ratio_spin.setValue(2.5)
        cbody_ratio_spin.setSingleStep(0.5)
        cbody_ratio_spin.setSuffix(":1")
        cbody_ratio_spin.setToolTip("Body compression ratio")
        self._add_effect_row(ecl, "compressor_body", "Compressor (Body)",
                             stub=False,
                             controls=[cbody_thresh_spin, cbody_ratio_spin])
        self._effect_controls["compressor_body"] = {
            "threshold_spin": cbody_thresh_spin,
            "ratio_spin": cbody_ratio_spin,
        }
        self._effect_checkboxes["compressor_body"].setChecked(True)

        # 7. De-Esser (working)
        deesser_freq_spin = QSpinBox()
        deesser_freq_spin.setRange(3000, 10000)
        deesser_freq_spin.setValue(5000)
        deesser_freq_spin.setSuffix(" Hz")
        deesser_freq_spin.setToolTip("Center frequency for sibilance detection")
        deesser_reduction_spin = QSpinBox()
        deesser_reduction_spin.setRange(0, 12)
        deesser_reduction_spin.setValue(12)
        deesser_reduction_spin.setSuffix(" dB")
        deesser_reduction_spin.setToolTip("Maximum sibilance reduction")
        self._add_effect_row(ecl, "de_esser", "De-Esser",
                             stub=False,
                             controls=[deesser_freq_spin, deesser_reduction_spin])
        self._effect_controls["de_esser"] = {
            "freq_spin": deesser_freq_spin,
            "reduction_spin": deesser_reduction_spin,
        }
        self._effect_checkboxes["de_esser"].setChecked(True)

        # 11. Soft Clipper
        sc_drive_spin = QDoubleSpinBox()
        sc_drive_spin.setRange(1.0, 4.0)
        sc_drive_spin.setValue(1.8)
        sc_drive_spin.setSingleStep(0.1)
        sc_drive_spin.setToolTip("Saturation drive (1.0 = no effect)")
        sc_mode_combo = QComboBox()
        sc_mode_combo.addItems(["tanh", "arctan", "cubic"])
        sc_mode_combo.setCurrentIndex(0)
        sc_mode_combo.setToolTip("Waveshaping function")
        self._add_effect_row(ecl, "soft_clipper", "Soft Clipper",
                             stub=False,
                             controls=[sc_drive_spin, sc_mode_combo])
        self._effect_controls["soft_clipper"] = {
            "drive_spin": sc_drive_spin,
            "mode_combo": sc_mode_combo,
        }
        self._effect_checkboxes["soft_clipper"].setChecked(True)

        # 12. Reverb
        self._reverb_ir_path: str | None = None
        reverb_wet_spin = QDoubleSpinBox()
        reverb_wet_spin.setRange(0.0, 0.50)
        reverb_wet_spin.setValue(0.15)
        reverb_wet_spin.setSingleStep(0.05)
        reverb_wet_spin.setToolTip("Wet/dry mix ratio")
        reverb_predelay_spin = QSpinBox()
        reverb_predelay_spin.setRange(0, 80)
        reverb_predelay_spin.setValue(25)
        reverb_predelay_spin.setSuffix(" ms")
        reverb_predelay_spin.setToolTip("Pre-delay before reverb onset")
        reverb_ir_btn = QPushButton("IR...")
        reverb_ir_btn.setFixedWidth(40)
        reverb_ir_btn.setToolTip("Load impulse response file (WAV/FLAC) for convolution reverb")
        reverb_ir_btn.clicked.connect(self._on_reverb_ir_browse)
        self._add_effect_row(ecl, "reverb", "Reverb",
                             stub=False,
                             controls=[reverb_wet_spin, reverb_predelay_spin,
                                       reverb_ir_btn])
        self._effect_controls["reverb"] = {
            "wet_spin": reverb_wet_spin,
            "predelay_spin": reverb_predelay_spin,
            "ir_btn": reverb_ir_btn,
        }
        self._effect_checkboxes["reverb"].setChecked(True)

        # 9. Limiter (working)
        limiter_spin = QDoubleSpinBox()
        limiter_spin.setRange(-3.0, 0.0)
        limiter_spin.setValue(-0.7)
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

            if key == "gain_rider":
                if "target_spin" in ctrls and "target_rms_db" in cfg:
                    ctrls["target_spin"].setValue(int(cfg["target_rms_db"]))
                if "max_gain_spin" in ctrls and "max_gain_db" in cfg:
                    ctrls["max_gain_spin"].setValue(int(cfg["max_gain_db"]))

            elif key == "de_plosive":
                if "threshold_spin" in ctrls and "threshold_db" in cfg:
                    ctrls["threshold_spin"].setValue(int(cfg["threshold_db"]))
                if "reduction_spin" in ctrls and "reduction_db" in cfg:
                    ctrls["reduction_spin"].setValue(int(cfg["reduction_db"]))

            elif key == "nr_cleanup":
                if "combo" in ctrls and "strength" in cfg:
                    _nrc_idx = {0.3: 0, 0.5: 1, 0.7: 2}
                    ctrls["combo"].setCurrentIndex(_nrc_idx.get(cfg["strength"], 0))

            elif key == "noise_gate":
                if "threshold_spin" in ctrls and "threshold_db" in cfg:
                    ctrls["threshold_spin"].setValue(int(cfg["threshold_db"]))
                if "reduction_spin" in ctrls and "reduction_db" in cfg:
                    ctrls["reduction_spin"].setValue(int(cfg["reduction_db"]))

            elif key == "spectral_noise_reduction":
                if "combo" in ctrls and "strength" in cfg:
                    idx = _NR_STRENGTH_INDEX.get(cfg["strength"], 1)
                    ctrls["combo"].setCurrentIndex(idx)
                if "mode_combo" in ctrls and "mode" in cfg:
                    mode_idx = {"auto": 0, "stationary": 1, "adaptive": 2}.get(
                        cfg["mode"], 0)
                    ctrls["mode_combo"].setCurrentIndex(mode_idx)
                self._nr_advanced_values = {
                    "n_std_thresh": cfg.get("n_std_thresh", 1.5),
                    "freq_smooth_hz": cfg.get("freq_smooth_hz", 500),
                    "time_smooth_ms": cfg.get("time_smooth_ms", 50),
                    "use_torch": cfg.get("use_torch", None),
                }

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

            elif key in ("compressor_peak", "compressor_body"):
                if "threshold_spin" in ctrls and "threshold_db" in cfg:
                    ctrls["threshold_spin"].setValue(int(cfg["threshold_db"]))
                if "ratio_spin" in ctrls and "ratio" in cfg:
                    ctrls["ratio_spin"].setValue(float(cfg["ratio"]))

            elif key == "de_esser":
                if "freq_spin" in ctrls and "freq_hz" in cfg:
                    ctrls["freq_spin"].setValue(int(cfg["freq_hz"]))
                if "reduction_spin" in ctrls and "reduction_db" in cfg:
                    ctrls["reduction_spin"].setValue(int(cfg["reduction_db"]))

            elif key == "soft_clipper":
                if "drive_spin" in ctrls and "drive" in cfg:
                    ctrls["drive_spin"].setValue(float(cfg["drive"]))
                if "mode_combo" in ctrls and "mode" in cfg:
                    _SC_MODE_INDEX = {"tanh": 0, "arctan": 1, "cubic": 2}
                    idx = _SC_MODE_INDEX.get(cfg["mode"], 0)
                    ctrls["mode_combo"].setCurrentIndex(idx)

            elif key == "reverb":
                if "wet_spin" in ctrls and "wet_mix" in cfg:
                    ctrls["wet_spin"].setValue(float(cfg["wet_mix"]))
                if "predelay_spin" in ctrls and "predelay_ms" in cfg:
                    ctrls["predelay_spin"].setValue(int(cfg["predelay_ms"]))
                self._reverb_ir_path = None

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

    def _on_nr_advanced(self) -> None:
        """Open the advanced NR parameters dialog."""
        dlg = _NRAdvancedDialog(self._nr_advanced_values, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._nr_advanced_values = dlg.values()
            self._on_effect_manual_change()

    def _on_reverb_ir_browse(self) -> None:
        """Open file dialog to select an impulse response file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Impulse Response",
            "", "Audio Files (*.wav *.flac);;All Files (*)",
        )
        if path:
            self._reverb_ir_path = path
            self._effect_controls["reverb"]["ir_btn"].setToolTip(f"IR: {path}")
            self._on_effect_manual_change()

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

        # Gain Rider
        gr_cb = self._effect_checkboxes["gain_rider"]
        gr_ctrls = self._effect_controls["gain_rider"]
        config["gain_rider"] = {
            "enabled": gr_cb.isChecked(),
            "target_rms_db": float(gr_ctrls["target_spin"].value()),
            "max_gain_db": float(gr_ctrls["max_gain_spin"].value()),
            "max_cut_db": float(gr_ctrls["max_gain_spin"].value()),
        }

        # De-Plosive
        dp_cb = self._effect_checkboxes["de_plosive"]
        dp_ctrls = self._effect_controls["de_plosive"]
        config["de_plosive"] = {
            "enabled": dp_cb.isChecked(),
            "threshold_db": float(dp_ctrls["threshold_spin"].value()),
            "reduction_db": float(dp_ctrls["reduction_spin"].value()),
        }

        # NR Cleanup (Pass 2)
        nr2_cb = self._effect_checkboxes["nr_cleanup"]
        nr2_combo = self._effect_controls["nr_cleanup"]["combo"]
        _nr2_params = {
            0: {"strength": 0.3, "n_std_thresh": 2.5},   # Light
            1: {"strength": 0.5, "n_std_thresh": 2.0},   # Medium
            2: {"strength": 0.7, "n_std_thresh": 1.5},   # Strong
        }
        nr2_p = _nr2_params[nr2_combo.currentIndex()]
        config["nr_cleanup"] = {
            "enabled": nr2_cb.isChecked(),
            "strength": nr2_p["strength"],
            "n_std_thresh": nr2_p["n_std_thresh"],
            "mode": "stationary",
        }

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
        nr_ctrls = self._effect_controls["spectral_noise_reduction"]
        strength_map = {0: 0.5, 1: 0.75, 2: 1.0}
        mode_map = {0: "auto", 1: "stationary", 2: "adaptive"}
        config["spectral_noise_reduction"] = {
            "enabled": nr_cb.isChecked(),
            "strength": strength_map[nr_ctrls["combo"].currentIndex()],
            "mode": mode_map[nr_ctrls["mode_combo"].currentIndex()],
            **self._nr_advanced_values,
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

        # Compressor Peak
        cpeak_cb = self._effect_checkboxes["compressor_peak"]
        cpeak_ctrls = self._effect_controls["compressor_peak"]
        config["compressor_peak"] = {
            "enabled": cpeak_cb.isChecked(),
            "threshold_db": float(cpeak_ctrls["threshold_spin"].value()),
            "ratio": cpeak_ctrls["ratio_spin"].value(),
        }

        # Compressor Body
        cbody_cb = self._effect_checkboxes["compressor_body"]
        cbody_ctrls = self._effect_controls["compressor_body"]
        config["compressor_body"] = {
            "enabled": cbody_cb.isChecked(),
            "threshold_db": float(cbody_ctrls["threshold_spin"].value()),
            "ratio": cbody_ctrls["ratio_spin"].value(),
        }

        # De-Esser
        de_cb = self._effect_checkboxes["de_esser"]
        de_ctrls = self._effect_controls["de_esser"]
        config["de_esser"] = {
            "enabled": de_cb.isChecked(),
            "freq_hz": de_ctrls["freq_spin"].value(),
            "reduction_db": float(de_ctrls["reduction_spin"].value()),
        }

        # Soft Clipper
        sc_cb = self._effect_checkboxes["soft_clipper"]
        sc_ctrls = self._effect_controls["soft_clipper"]
        sc_mode_map = {0: "tanh", 1: "arctan", 2: "cubic"}
        config["soft_clipper"] = {
            "enabled": sc_cb.isChecked(),
            "drive": sc_ctrls["drive_spin"].value(),
            "ceiling_db": -1.0,
            "mode": sc_mode_map[sc_ctrls["mode_combo"].currentIndex()],
        }

        # Reverb
        rev_cb = self._effect_checkboxes["reverb"]
        rev_ctrls = self._effect_controls["reverb"]
        config["reverb"] = {
            "enabled": rev_cb.isChecked(),
            "wet_mix": rev_ctrls["wet_spin"].value(),
            "predelay_ms": float(rev_ctrls["predelay_spin"].value()),
            "reverb_type": "convolution" if self._reverb_ir_path else "algorithmic",
            "ir_path": self._reverb_ir_path,
        }

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
