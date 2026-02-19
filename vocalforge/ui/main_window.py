"""Main window — layout and panel orchestration."""

import os
from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QWidget,
)

from vocalforge.audio.alignment import (
    align_tracks, chain_align, pad_or_trim, resample_if_needed, shift_track,
)
from vocalforge.audio.effects import process_vocal, limiter as effects_limiter
from vocalforge.audio.engine import AudioEngine
from vocalforge.audio.mixer import mix_tracks
from vocalforge.separation.demucs_worker import separate_song
from vocalforge.ui.import_panel import ImportPanel, _SLOT_NAMES
from vocalforge.ui.mix_panel import MixPanel
from vocalforge.ui.record_panel import RecordPanel
from vocalforge.utils.audio_io import save_audio


class _AlignWorker(QThread):
    """Background thread for alignment only (no NR, no mixing)."""

    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        minus_data: np.ndarray,
        minus_sr: int,
        vocal_data: np.ndarray,
        vocal_sr: int,
        alignment_mode: str = "vocal",
        vocals_stem_path: str | None = None,
        minus_sep_data: np.ndarray | None = None,
        minus_sep_sr: int | None = None,
        vocal_sep_data: np.ndarray | None = None,
        vocal_sep_sr: int | None = None,
        max_offset_s: float | None = None,
    ):
        super().__init__()
        self._minus_data = minus_data
        self._minus_sr = minus_sr
        self._vocal_data = vocal_data
        self._vocal_sr = vocal_sr
        self._alignment_mode = alignment_mode
        self._vocals_stem_path = vocals_stem_path
        self._minus_sep_data = minus_sep_data
        self._minus_sep_sr = minus_sep_sr
        self._vocal_sep_data = vocal_sep_data
        self._vocal_sep_sr = vocal_sep_sr
        self._max_offset_s = max_offset_s

    def run(self) -> None:
        try:
            sr = self._minus_sr

            vocal = self._vocal_data
            if self._vocal_sr != sr:
                self.progress.emit("Resampling vocal...")
                vocal = resample_if_needed(vocal, self._vocal_sr, sr)

            minus_sep = self._minus_sep_data
            if minus_sep is not None and self._minus_sep_sr != sr:
                minus_sep = resample_if_needed(minus_sep, self._minus_sep_sr, sr)

            vocal_sep = self._vocal_sep_data
            if vocal_sep is not None and self._vocal_sep_sr != sr:
                vocal_sep = resample_if_needed(vocal_sep, self._vocal_sep_sr, sr)

            result = {"minus_lag_samples": 0, "minus_lag_ms": 0.0,
                      "vocal_lag_samples": 0, "vocal_lag_ms": 0.0}

            if self._alignment_mode == "vocal":
                if minus_sep is not None and vocal_sep is not None:
                    self.progress.emit("Chain-aligning through stems...")
                    is_import = self._minus_data is not minus_sep
                    minus_import = self._minus_data if is_import else None
                    _, _, align_info = chain_align(
                        minus_sep, vocal_sep, vocal, sr,
                        minus_import=minus_import,
                        max_offset_s=self._max_offset_s,
                    )
                    result.update(align_info)
                elif self._vocals_stem_path is not None:
                    self.progress.emit("Aligning to vocals stem...")
                    import soundfile as sf
                    vocals_stem, stem_sr = sf.read(
                        self._vocals_stem_path, dtype="float32"
                    )
                    if stem_sr != sr:
                        vocals_stem = resample_if_needed(vocals_stem, stem_sr, sr)
                    _, align_info = align_tracks(
                        vocals_stem, vocal, sr,
                        max_offset_s=self._max_offset_s,
                    )
                    result["vocal_lag_samples"] = align_info["lag_samples"]
                    result["vocal_lag_ms"] = align_info["lag_ms"]
                    result.update({k: v for k, v in align_info.items()
                                   if k not in result})
                else:
                    raise RuntimeError(
                        "Vocal matching requires separated stems. "
                        "Run Separate first."
                    )
            elif self._alignment_mode == "background":
                self.progress.emit("Aligning to background...")
                _, align_info = align_tracks(
                    self._minus_data, vocal, sr,
                    max_offset_s=self._max_offset_s,
                )
                result["vocal_lag_samples"] = align_info["lag_samples"]
                result["vocal_lag_ms"] = align_info["lag_ms"]
                result.update({k: v for k, v in align_info.items()
                               if k not in result})

            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _MixWorker(QThread):
    """Background thread for alignment + mixing + export."""

    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        minus_data: np.ndarray,
        minus_sr: int,
        vocal_data: np.ndarray,
        vocal_sr: int,
        output_path: str,
        vocal_gain: float,
        instrumental_gain: float,
        target_lufs: float,
        alignment_mode: str = "background",
        vocals_stem_path: str | None = None,
        effects_config: dict | None = None,
        minus_sep_data: np.ndarray | None = None,
        minus_sep_sr: int | None = None,
        vocal_sep_data: np.ndarray | None = None,
        vocal_sep_sr: int | None = None,
        max_offset_s: float | None = None,
        manual_minus_offset_samples: int | None = None,
        manual_vocal_offset_samples: int | None = None,
    ):
        super().__init__()
        self._minus_data = minus_data
        self._minus_sr = minus_sr
        self._vocal_data = vocal_data
        self._vocal_sr = vocal_sr
        self._output_path = output_path
        self._vocal_gain = vocal_gain
        self._instrumental_gain = instrumental_gain
        self._target_lufs = target_lufs
        self._alignment_mode = alignment_mode
        self._vocals_stem_path = vocals_stem_path
        self._effects_config = effects_config
        self._minus_sep_data = minus_sep_data
        self._minus_sep_sr = minus_sep_sr
        self._vocal_sep_data = vocal_sep_data
        self._vocal_sep_sr = vocal_sep_sr
        self._max_offset_s = max_offset_s
        self._manual_minus_offset = manual_minus_offset_samples
        self._manual_vocal_offset = manual_vocal_offset_samples

    @staticmethod
    def _peak(data: np.ndarray) -> float:
        """Return peak absolute amplitude of audio array."""
        if data.size == 0:
            return 0.0
        return float(np.max(np.abs(data)))

    def run(self) -> None:
        try:
            sr = self._minus_sr

            # Resample vocal if sample rates differ
            vocal = self._vocal_data
            if self._vocal_sr != sr:
                self.progress.emit("Resampling vocal...")
                vocal = resample_if_needed(vocal, self._vocal_sr, sr)

            # Resample separated stems to match mix sample rate if needed
            minus_sep = self._minus_sep_data
            if minus_sep is not None and self._minus_sep_sr != sr:
                minus_sep = resample_if_needed(minus_sep, self._minus_sep_sr, sr)

            vocal_sep = self._vocal_sep_data
            if vocal_sep is not None and self._vocal_sep_sr != sr:
                vocal_sep = resample_if_needed(vocal_sep, self._vocal_sep_sr, sr)

            target_len = self._minus_data.shape[0]
            align_info = {"lag_samples": 0, "lag_ms": 0.0}
            minus_for_mix = self._minus_data

            self.progress.emit(
                f"Input: minus {self._minus_data.shape} peak={self._peak(self._minus_data):.3f}, "
                f"vocal {vocal.shape} peak={self._peak(vocal):.3f}"
            )

            # Check for manual offsets
            has_manual = (
                self._manual_vocal_offset is not None
                or self._manual_minus_offset is not None
            )

            if has_manual:
                vocal_offset = self._manual_vocal_offset or 0
                minus_offset = self._manual_minus_offset or 0
                # Compute net vocal offset relative to the minus track.
                # Both offsets are relative to the separated stems timeline.
                # For mixing we keep the minus track unshifted and only
                # shift the vocal by the difference.
                net_vocal_offset = vocal_offset - minus_offset
                self.progress.emit(
                    f"Manual offsets: vocal={vocal_offset}, minus={minus_offset}, "
                    f"net vocal-to-minus={net_vocal_offset}"
                )
                aligned_vocal = shift_track(vocal, net_vocal_offset, target_len)
                # minus_for_mix stays as self._minus_data (unshifted)
                align_info["lag_samples"] = net_vocal_offset
                align_info["lag_ms"] = net_vocal_offset / sr * 1000.0
            elif self._alignment_mode == "none":
                self.progress.emit("Padding/trimming (no alignment)...")
                aligned_vocal = pad_or_trim(vocal, target_len)
            elif self._alignment_mode == "vocal":
                # Chain alignment through stems when available
                if minus_sep is not None and vocal_sep is not None:
                    self.progress.emit("Chain-aligning through stems...")
                    is_import = self._minus_data is not minus_sep
                    minus_import = self._minus_data if is_import else None
                    _, _, align_info = chain_align(
                        minus_sep, vocal_sep, vocal, sr,
                        minus_import=minus_import,
                        max_offset_s=self._max_offset_s,
                    )
                    # Keep the full minus track; shift vocal by the net offset
                    net_lag = (align_info.get("vocal_lag_samples", 0)
                               - align_info.get("minus_lag_samples", 0))
                    aligned_vocal = shift_track(vocal, net_lag, target_len)
                    align_info["lag_samples"] = net_lag
                    align_info["lag_ms"] = net_lag / sr * 1000.0
                    # minus_for_mix stays as self._minus_data (unshifted)
                elif self._vocals_stem_path is not None:
                    self.progress.emit("Loading vocals stem...")
                    import soundfile as sf
                    vocals_stem, stem_sr = sf.read(
                        self._vocals_stem_path, dtype="float32"
                    )
                    if stem_sr != sr:
                        vocals_stem = resample_if_needed(vocals_stem, stem_sr, sr)
                    self.progress.emit("Aligning to vocals stem...")
                    aligned_vocal, align_info = align_tracks(
                        vocals_stem, vocal, sr,
                        max_offset_s=self._max_offset_s,
                    )
                else:
                    raise RuntimeError(
                        "Vocal matching requires separated stems or a Demucs "
                        "vocals stem. Run Separate first."
                    )
            else:
                # Default: background music alignment
                self.progress.emit("Aligning...")
                aligned_vocal, align_info = align_tracks(
                    self._minus_data, vocal, sr,
                    max_offset_s=self._max_offset_s,
                )

            self.progress.emit(
                f"After align: vocal peak={self._peak(aligned_vocal):.3f}, "
                f"minus peak={self._peak(minus_for_mix):.3f}"
            )

            # Apply vocal effects chain
            self.progress.emit("Processing vocal effects...")
            effects_cfg = self._effects_config or {}
            # Inject guide_stem for NR if available
            if vocal_sep is not None:
                nr_cfg = effects_cfg.get("spectral_noise_reduction", {})
                nr_cfg["guide_stem"] = vocal_sep
                effects_cfg["spectral_noise_reduction"] = nr_cfg
            # Disable limiter on vocal — we apply it to the mix instead
            effects_cfg_vocal = dict(effects_cfg)
            effects_cfg_vocal["limiter"] = {"enabled": False}
            aligned_vocal = process_vocal(aligned_vocal, sr, config=effects_cfg_vocal)

            self.progress.emit(
                f"After FX: vocal peak={self._peak(aligned_vocal):.3f}"
            )

            # Mix (without built-in peak limiter — we use effects.limiter)
            self.progress.emit("Mixing...")
            mix_data, mix_info = mix_tracks(
                minus_for_mix,
                aligned_vocal,
                sr,
                vocal_gain=self._vocal_gain,
                instrumental_gain=self._instrumental_gain,
                target_lufs=self._target_lufs,
                apply_limiter=False,
            )

            self.progress.emit(
                f"After mix: peak={self._peak(mix_data):.3f}, "
                f"LUFS={mix_info.get('mix_lufs', '?')}"
            )

            # Apply limiter to the final mix
            limiter_cfg = effects_cfg.get("limiter", {})
            if limiter_cfg.get("enabled", True):
                self.progress.emit("Limiting...")
                mix_data = effects_limiter(
                    mix_data, sr,
                    ceiling_db=limiter_cfg.get("ceiling_db", -1.0),
                )

            # Save
            self.progress.emit(
                f"Saving ({mix_data.shape}, peak={self._peak(mix_data):.3f})..."
            )
            save_audio(self._output_path, mix_data, sr)

            result = {**align_info, **mix_info, "output_path": self._output_path}
            self.finished.emit(result)

        except Exception as exc:
            self.error.emit(str(exc))


class _SeparationWorker(QThread):
    """Background thread for Demucs source separation."""

    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, song_path: str):
        super().__init__()
        self._song_path = song_path

    def run(self) -> None:
        try:
            def _callback(message: str, fraction: float) -> None:
                if fraction < 1.0:
                    pct = int(fraction * 100)
                    self.progress.emit(f"{message} {pct}%")
                else:
                    self.progress.emit(message)

            data, sr = separate_song(
                self._song_path,
                callback=_callback,
            )

            # Build the cached paths so MainWindow can pass them along
            song_dir = os.path.dirname(os.path.abspath(self._song_path))
            song_name = os.path.splitext(os.path.basename(self._song_path))[0]
            stems_dir = os.path.join(song_dir, f"{song_name}_stems")
            cached_path = os.path.join(stems_dir, "no_vocals.wav")
            vocals_path = os.path.join(stems_dir, "vocals.wav")

            result = {
                "data": data,
                "sr": sr,
                "path": cached_path,
            }

            # Also load and return the vocals stem data
            if os.path.isfile(vocals_path):
                import soundfile as sf
                vocals_data, vocals_sr = sf.read(vocals_path, dtype="float32")
                result["vocals_path"] = vocals_path
                result["vocals_data"] = vocals_data
                result["vocals_sr"] = vocals_sr

            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _EffectsWorker(QThread):
    """Background thread for applying vocal effects chain."""

    progress = Signal(str)
    finished = Signal(object, int)  # processed_data, sample_rate
    error = Signal(str)

    def __init__(
        self,
        vocal_data: np.ndarray,
        vocal_sr: int,
        effects_config: dict,
        vocal_sep_data: np.ndarray | None = None,
        vocal_sep_sr: int | None = None,
    ):
        super().__init__()
        self._vocal_data = vocal_data
        self._vocal_sr = vocal_sr
        self._effects_config = effects_config
        self._vocal_sep_data = vocal_sep_data
        self._vocal_sep_sr = vocal_sep_sr

    def run(self) -> None:
        try:
            sr = self._vocal_sr
            cfg = dict(self._effects_config)

            # Inject guide stem for NR
            if self._vocal_sep_data is not None:
                vocal_sep = self._vocal_sep_data
                if self._vocal_sep_sr != sr:
                    vocal_sep = resample_if_needed(vocal_sep, self._vocal_sep_sr, sr)
                nr_cfg = cfg.get("spectral_noise_reduction", {})
                nr_cfg["guide_stem"] = vocal_sep
                cfg["spectral_noise_reduction"] = nr_cfg

            self.progress.emit("Processing vocal effects...")
            result = process_vocal(self._vocal_data, sr, config=cfg)
            self.finished.emit(result, sr)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("VocalForge")
        self.setMinimumSize(800, 400)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)

        self._import_panel = ImportPanel()
        self._record_panel = RecordPanel()
        self._mix_panel = MixPanel()

        layout.addWidget(self._import_panel, stretch=1)
        layout.addWidget(self._record_panel, stretch=1)
        layout.addWidget(self._mix_panel, stretch=1)

        # Audio engine
        self._engine = AudioEngine()

        # Track state
        self._vocals_stem_path: str | None = None

        # Manual offset state (in samples, at minus_sr)
        self._minus_offset_samples: int = 0
        self._vocal_offset_samples: int = 0

        # Waveform total frames (includes offsets — may differ from engine total)
        self._waveform_total_frames: int = 0
        self._track_offsets: dict[str, int] = {}  # per-track offsets for meters

        # Worker references (prevent GC while running)
        self._mix_worker: _MixWorker | None = None
        self._separation_worker: _SeparationWorker | None = None
        self._align_worker: _AlignWorker | None = None
        self._effects_worker: _EffectsWorker | None = None

        # Position polling timer (30ms ~ 33 fps cursor updates)
        self._position_timer = QTimer(self)
        self._position_timer.setInterval(30)
        self._position_timer.timeout.connect(self._on_position_tick)

        # Connect signals — playback
        self._import_panel.track_loaded.connect(self._on_track_loaded)
        self._record_panel.play_clicked.connect(self._on_play)
        self._record_panel.pause_clicked.connect(self._on_pause)
        self._record_panel.stop_clicked.connect(self._on_stop)
        self._record_panel.volume_changed.connect(self._on_volume_changed)
        self._record_panel.output_device_changed.connect(self._on_output_device_changed)

        # Connect signals — recording
        self._record_panel.input_device_changed.connect(self._on_input_device_changed)
        self._record_panel.record_start_clicked.connect(self._on_record_start)
        self._record_panel.record_finish_clicked.connect(self._on_record_finish)
        self._record_panel.record_stop_clicked.connect(self._on_record_stop)
        self._record_panel.latency_offset_changed.connect(self._on_latency_offset_changed)

        # Connect signals — seek
        self._record_panel.seek_requested.connect(self._on_seek)
        for slot in _SLOT_NAMES:
            wf = self._import_panel.get_waveform(slot)
            if wf is not None:
                wf.seek_requested.connect(self._on_seek)

        # Connect signals — separation
        self._import_panel.separate_requested.connect(self._on_separate_start)

        # Connect signals — mix & export
        self._import_panel.track_loaded.connect(self._update_mix_ready)
        self._mix_panel.mix_export_clicked.connect(self._on_mix_export)

        # Connect signals — alignment
        self._mix_panel.auto_align_clicked.connect(self._on_auto_align)
        self._mix_panel.minus_offset_changed.connect(self._on_minus_offset_changed)
        self._mix_panel.vocal_offset_changed.connect(self._on_vocal_offset_changed)

        # Connect signals — mute & gain
        self._record_panel.mute_state_changed.connect(self._on_mute_state_changed)
        self._import_panel.track_gain_changed.connect(self._on_track_gain_changed)

        # Connect signals — effects & auto-tune
        self._mix_panel.apply_effects_clicked.connect(self._on_apply_effects)
        self._mix_panel.export_vocal_clicked.connect(self._on_export_vocal)
        self._mix_panel.auto_tune_clicked.connect(self._on_auto_tune)

    # --- Track loading ---

    def _on_track_loaded(self, slot_name: str) -> None:
        track = self._get_track(slot_name)

        # Handle cleared tracks: reset offsets and stale state
        if track is None:
            if slot_name == "minus_import":
                self._minus_offset_samples = 0
                self._mix_panel.set_minus_offset(0.0)
            elif slot_name == "vocal":
                self._vocal_offset_samples = 0
                self._mix_panel.set_vocal_offset(0.0)
                # Clear stale processed vocal
                self._import_panel.clear_slot("vocal_processed")
            elif slot_name == "song":
                self._vocals_stem_path = None
        else:
            # Auto-check the mute checkbox only when a track is loaded (not cleared)
            self._record_panel.set_mute_checked(slot_name, True)

            # Clear stale downstream slots on new source load
            if slot_name == "vocal":
                self._import_panel.clear_slot("vocal_processed")
            elif slot_name == "song":
                self._vocals_stem_path = None

        # Always recompute waveform offsets (keeps _waveform_total_frames in sync)
        self._update_waveform_offsets()

        # Update track highlighting for mix
        self._import_panel.update_mix_highlights()

        # Enable record only if a minus track actually exists
        if slot_name in ("minus_sep", "minus_import"):
            has_minus = self._import_panel.minus_track is not None
            self._record_panel.set_record_enabled(has_minus)

        # Enable auto-align when we have minus + vocal
        self._update_auto_align_ready()
        self._update_effects_ready()
        self._update_auto_tune_ready()

        # Rebuild multi-track playback
        self._update_multi_playback()

    def _get_track(self, slot_name: str) -> tuple | None:
        """Return (data, sr) for a given slot name."""
        if slot_name == "song":
            return self._import_panel.song_track
        elif slot_name == "minus_sep":
            return self._import_panel.minus_sep_track
        elif slot_name == "vocal_sep":
            return self._import_panel.vocal_sep_track
        elif slot_name == "minus_import":
            return self._import_panel.minus_import_track
        elif slot_name == "vocal":
            return self._import_panel.vocal_track
        elif slot_name == "vocal_processed":
            return self._import_panel.vocal_processed_track
        elif slot_name == "mix_result":
            return self._import_panel.mix_result_track
        return None

    # --- Mix readiness ---

    def _update_mix_ready(self, _slot_name: str = "") -> None:
        """Enable mix button when at least one minus and vocal are loaded."""
        has_minus = self._import_panel.minus_track is not None
        has_vocal = self._import_panel.vocal_track is not None
        self._mix_panel.set_mix_enabled(has_minus and has_vocal)

    def _update_auto_align_ready(self) -> None:
        """Enable auto-align when we have at least minus and vocal tracks."""
        has_minus = self._import_panel.minus_track is not None
        has_vocal = self._import_panel.vocal_track is not None
        self._mix_panel.set_auto_align_enabled(has_minus and has_vocal)

    def _update_effects_ready(self) -> None:
        """Enable Apply Effects when vocal track is loaded."""
        has_vocal = self._import_panel.vocal_track is not None
        self._mix_panel.set_apply_effects_enabled(has_vocal)

    def _update_auto_tune_ready(self) -> None:
        """Enable Auto-Tune Volume when we have minus and some vocal."""
        has_minus = self._import_panel.minus_track is not None
        has_vocal = (
            self._import_panel.vocal_processed_track is not None
            or self._import_panel.vocal_track is not None
        )
        self._mix_panel.set_auto_tune_enabled(has_minus and has_vocal)

    # --- Auto-Align ---

    def _on_auto_align(self) -> None:
        """Run alignment in background to populate offset spinners."""
        minus = self._import_panel.minus_track
        vocal = self._import_panel.vocal_track
        if minus is None or vocal is None:
            return

        minus_data, minus_sr = minus
        vocal_data, vocal_sr = vocal

        self._mix_panel.set_auto_align_enabled(False)
        self._mix_panel.set_align_confidence("Aligning...")

        # Gather stems
        minus_sep_data = minus_sep_sr = None
        vocal_sep_data = vocal_sep_sr = None
        sep = self._import_panel.minus_sep_track
        if sep is not None:
            minus_sep_data, minus_sep_sr = sep
        sep = self._import_panel.vocal_sep_track
        if sep is not None:
            vocal_sep_data, vocal_sep_sr = sep

        self._align_worker = _AlignWorker(
            minus_data, minus_sr,
            vocal_data, vocal_sr,
            alignment_mode=self._mix_panel.alignment_mode(),
            vocals_stem_path=self._vocals_stem_path,
            minus_sep_data=minus_sep_data,
            minus_sep_sr=minus_sep_sr,
            vocal_sep_data=vocal_sep_data,
            vocal_sep_sr=vocal_sep_sr,
            max_offset_s=self._mix_panel.max_offset_s(),
        )
        self._align_worker.progress.connect(self._mix_panel.set_align_confidence)
        self._align_worker.finished.connect(self._on_auto_align_finished)
        self._align_worker.error.connect(self._on_auto_align_error)
        self._align_worker.start()

    def _on_auto_align_finished(self, result: dict) -> None:
        minus_lag_ms = result.get("minus_lag_ms", 0.0)
        vocal_lag_ms = result.get("vocal_lag_ms", 0.0)

        # Populate offset spinners
        self._mix_panel.set_minus_offset(minus_lag_ms)
        self._mix_panel.set_vocal_offset(vocal_lag_ms)

        # Store samples
        minus = self._import_panel.minus_track
        sr = minus[1] if minus else 44100
        self._minus_offset_samples = int(result.get("minus_lag_samples", 0))
        self._vocal_offset_samples = int(result.get("vocal_lag_samples", 0))

        # Show confidence info
        warnings = []
        if result.get("suspicious") or result.get("vocal_suspicious"):
            warnings.append("Vocal alignment may be off")
        if result.get("minus_suspicious"):
            warnings.append("Minus alignment may be off")
        self._mix_panel.set_alignment_warning("; ".join(warnings))

        info_text = (
            f"Vocal: {vocal_lag_ms:+.1f} ms, "
            f"Minus: {minus_lag_ms:+.1f} ms"
        )
        self._mix_panel.set_align_confidence(info_text)

        # Update waveform offsets
        self._update_waveform_offsets()

        self._mix_panel.set_auto_align_enabled(True)
        self._align_worker = None

    def _on_auto_align_error(self, msg: str) -> None:
        self._mix_panel.set_align_confidence("Alignment failed")
        self._mix_panel.set_auto_align_enabled(True)
        self._align_worker = None
        QMessageBox.warning(self, "Alignment Error", f"Alignment failed:\n{msg}")

    # --- Manual offset changes ---

    def _on_minus_offset_changed(self, ms: float) -> None:
        minus = self._import_panel.minus_track
        sr = minus[1] if minus else 44100
        self._minus_offset_samples = int(ms * sr / 1000.0)
        self._update_waveform_offsets()
        self._update_multi_playback()

    def _on_vocal_offset_changed(self, ms: float) -> None:
        minus = self._import_panel.minus_track
        sr = minus[1] if minus else 44100
        self._vocal_offset_samples = int(ms * sr / 1000.0)
        self._update_waveform_offsets()
        self._update_multi_playback()

    # --- Waveform offset helpers ---

    def _update_waveform_offsets(self) -> None:
        """Recompute global timeline and update all waveform offset visuals.

        Converts lag values (positive = trim from start) to delay values
        (positive = starts later) using: delay = max_lag - track_lag.
        """
        minus_lag = self._minus_offset_samples
        vocal_lag = self._vocal_offset_samples
        max_lag = max(0, minus_lag, vocal_lag)

        offsets = {}
        total = 0
        for slot in _SLOT_NAMES:
            track = self._get_track(slot)
            if track is None:
                continue
            data, sr = track
            track_len = data.shape[0]
            if slot == "minus_import":
                off = max_lag - minus_lag
            elif slot in ("vocal", "vocal_processed"):
                off = max_lag - vocal_lag
            elif slot == "mix_result":
                # Mix has alignment baked in — its time 0 corresponds
                # to the alignment point (stems time 0) which sits at
                # timeline position max_lag.
                off = max_lag
            else:
                # song, minus_sep, vocal_sep: reference tracks (lag=0)
                off = max_lag
            offsets[slot] = off
            end = track_len + off
            if end > total:
                total = end

        self._waveform_total_frames = total
        self._track_offsets = offsets
        self._import_panel.apply_waveform_offsets(offsets, total)

    # --- Multi-track playback ---

    def _on_mute_state_changed(self, state: dict) -> None:
        self._update_multi_playback()

    def _update_multi_playback(self) -> None:
        """Load ALL tracks into engine, using mute flags for audibility.

        This keeps engine.total_frames constant regardless of which tracks
        are muted, preventing seek/cursor misalignment when toggling mutes.
        """
        mute_states = self._record_panel.get_mute_states()

        # Determine common sample rate from first available track
        target_sr = None
        for slot in _SLOT_NAMES:
            track = self._get_track(slot)
            if track is not None:
                target_sr = track[1]
                break
        if target_sr is None:
            return

        # Convert lag values to delay values for multi-track playback
        minus_lag = self._minus_offset_samples
        vocal_lag = self._vocal_offset_samples
        max_lag = max(0, minus_lag, vocal_lag)

        tracks = []
        offsets = []
        gains = []
        mutes = []

        for slot in _SLOT_NAMES:
            track = self._get_track(slot)
            if track is None:
                continue
            data, sr = track
            if sr != target_sr:
                data = resample_if_needed(data, sr, target_sr)

            if slot == "minus_import":
                off = max_lag - minus_lag
            elif slot in ("vocal", "vocal_processed"):
                off = max_lag - vocal_lag
            elif slot == "mix_result":
                # Mix has alignment baked in — its time 0 corresponds
                # to the alignment point (stems time 0) which sits at
                # timeline position max_lag.
                off = max_lag
            else:
                # song, minus_sep, vocal_sep: reference tracks (lag=0)
                off = max_lag

            gain = self._import_panel.get_track_gain(slot)
            is_muted = not mute_states.get(slot, False)
            tracks.append(data)
            offsets.append(off)
            gains.append(gain)
            mutes.append(is_muted)

        if not tracks:
            return

        was_playing = self._engine.is_playing
        pos = self._engine.position

        self._engine.load_multi(tracks, target_sr, offsets, gains, mutes)
        self._sync_output_device()
        self._sync_input_device()

        # Enable playback controls
        self._record_panel.set_playback_enabled(True)
        total_sec = self._engine.total_frames / target_sr
        if not was_playing:
            self._record_panel.update_transport_state(False, False)
            self._record_panel.update_time_display(pos / target_sr, total_sec)
            self._record_panel.update_seek_slider(
                pos / self._engine.total_frames if self._engine.total_frames > 0 else 0
            )

        if was_playing:
            self._engine.seek(pos)
            self._engine.play()
            self._position_timer.start()

    # --- Mix & Export ---

    def _on_mix_export(self) -> None:
        minus = self._import_panel.minus_track  # prefers import, falls back to sep
        vocal = self._import_panel.vocal_track
        if minus is None or vocal is None:
            return

        minus_data, minus_sr = minus[0].copy(), minus[1]
        vocal_data, vocal_sr = vocal[0].copy(), vocal[1]

        # Choose output path
        default_dir = ""
        minus_path = (
            self._import_panel.get_track_path("minus_import")
            or self._import_panel.get_track_path("minus_sep")
        )
        if minus_path:
            default_dir = os.path.dirname(minus_path)

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Mixed Track",
            os.path.join(default_dir, "mix.wav"),
            "WAV Files (*.wav);;FLAC Files (*.flac);;All Files (*)",
        )
        if not output_path:
            return

        # Disable button and show progress
        self._mix_panel.set_mix_enabled(False)
        self._mix_panel.set_status("Starting...")

        alignment_mode = self._mix_panel.alignment_mode()

        # Check vocal alignment prerequisites
        if alignment_mode == "vocal":
            has_stems = (
                self._import_panel.minus_sep_track is not None
                and self._import_panel.vocal_sep_track is not None
            )
            if not has_stems and self._vocals_stem_path is None:
                QMessageBox.warning(
                    self,
                    "Missing Vocals Stem",
                    "Vocal matching alignment requires separated stems.\n"
                    "Run Separate on the song first.",
                )
                self._mix_panel.set_mix_enabled(True)
                return

        # Gather separated stem data for chain alignment and NR (defensive copy)
        minus_sep_data = minus_sep_sr = None
        vocal_sep_data = vocal_sep_sr = None
        sep = self._import_panel.minus_sep_track
        if sep is not None:
            minus_sep_data, minus_sep_sr = sep[0].copy(), sep[1]
        sep = self._import_panel.vocal_sep_track
        if sep is not None:
            vocal_sep_data, vocal_sep_sr = sep[0].copy(), sep[1]

        # Build effects config from UI (bypass if chain disabled)
        if self._mix_panel.effects_chain_enabled():
            effects_config = self._mix_panel.get_effects_config()
        else:
            from vocalforge.audio.effects import CHAIN_ORDER
            effects_config = {name: {"enabled": False} for name in CHAIN_ORDER}

        # Check for manual offsets from spinners
        manual_minus = None
        manual_vocal = None
        if (self._mix_panel.minus_offset_ms() != 0.0
                or self._mix_panel.vocal_offset_ms() != 0.0):
            manual_minus = self._minus_offset_samples
            manual_vocal = self._vocal_offset_samples

        self._mix_worker = _MixWorker(
            minus_data,
            minus_sr,
            vocal_data,
            vocal_sr,
            output_path,
            vocal_gain=self._import_panel.get_track_gain("vocal"),
            instrumental_gain=self._import_panel.get_track_gain(
                "minus_import" if self._import_panel.minus_import_track is not None
                else "minus_sep"
            ),
            target_lufs=self._mix_panel.target_lufs(),
            alignment_mode=alignment_mode,
            vocals_stem_path=self._vocals_stem_path,
            effects_config=effects_config,
            minus_sep_data=minus_sep_data,
            minus_sep_sr=minus_sep_sr,
            vocal_sep_data=vocal_sep_data,
            vocal_sep_sr=vocal_sep_sr,
            max_offset_s=self._mix_panel.max_offset_s(),
            manual_minus_offset_samples=manual_minus,
            manual_vocal_offset_samples=manual_vocal,
        )
        self._mix_worker.progress.connect(self._mix_panel.set_status)
        self._mix_worker.finished.connect(self._on_mix_finished)
        self._mix_worker.error.connect(self._on_mix_error)
        self._mix_worker.start()

    def _on_mix_finished(self, result: dict) -> None:
        lag_ms = result.get("lag_ms", 0.0)
        lag_samples = result.get("lag_samples", 0)
        self._mix_panel.set_alignment_info(lag_ms, lag_samples)

        # Show detailed mix info in status
        vocal_lufs = result.get("vocal_lufs", float("nan"))
        minus_lufs = result.get("minus_lufs", float("nan"))
        mix_lufs = result.get("mix_lufs", float("nan"))
        self._mix_panel.set_status(
            f"Done — {os.path.basename(result['output_path'])} | "
            f"V:{vocal_lufs:.1f} M:{minus_lufs:.1f} Mix:{mix_lufs:.1f} LUFS"
        )

        # Check for suspicious alignment
        warnings = []
        if result.get("suspicious") or result.get("vocal_suspicious"):
            warnings.append("Vocal alignment may be incorrect (large offset)")
        if result.get("minus_suspicious"):
            warnings.append("Minus alignment may be incorrect (large offset)")
        self._mix_panel.set_alignment_warning("; ".join(warnings))

        # Load mix result into import panel for playback
        output_path = result["output_path"]
        try:
            from vocalforge.utils.audio_io import load_audio
            mix_data, mix_sr = load_audio(output_path)
            # Validate loaded data
            if mix_data.size == 0:
                raise ValueError("Mix file is empty")
            peak = float(np.max(np.abs(mix_data)))
            if peak < 1e-6:
                raise ValueError(
                    f"Mix file has near-zero amplitude (peak={peak:.2e})"
                )
            self._import_panel.set_mix_result_track(mix_data, mix_sr, output_path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Mix Result",
                f"Mix exported successfully but could not load for preview:\n{exc}",
            )

        self._mix_panel.set_mix_enabled(True)
        self._mix_worker = None

    def _on_mix_error(self, msg: str) -> None:
        self._mix_panel.set_status("Error")
        self._mix_panel.set_mix_enabled(True)
        self._mix_worker = None
        QMessageBox.warning(self, "Mix Error", f"Mixing failed:\n{msg}")

    # --- Separation ---

    def _on_separate_start(self, song_path: str) -> None:
        self._import_panel.set_separate_enabled(False)
        self._import_panel.set_separate_progress("Starting separation...")

        self._separation_worker = _SeparationWorker(song_path)
        self._separation_worker.progress.connect(self._on_separate_progress)
        self._separation_worker.finished.connect(self._on_separate_finished)
        self._separation_worker.error.connect(self._on_separate_error)
        self._separation_worker.start()

    def _on_separate_progress(self, msg: str) -> None:
        self._import_panel.set_separate_progress(msg)

    def _on_separate_finished(self, result: dict) -> None:
        # Populate minus-sep slot
        self._import_panel.set_minus_sep_track(
            result["data"], result["sr"], result["path"]
        )
        # Populate vocal-sep slot if available
        if "vocals_data" in result:
            self._import_panel.set_vocal_sep_track(
                result["vocals_data"], result["vocals_sr"],
                result.get("vocals_path"),
            )
        self._vocals_stem_path = result.get("vocals_path")
        self._import_panel.set_separate_enabled(True)
        self._import_panel.set_separate_progress("")
        self._separation_worker = None

    def _on_separate_error(self, msg: str) -> None:
        self._import_panel.set_separate_enabled(True)
        self._import_panel.set_separate_progress("")
        self._separation_worker = None
        QMessageBox.warning(self, "Separation Error", f"Source separation failed:\n{msg}")

    # --- Device syncing ---

    def _sync_output_device(self) -> None:
        """Push the currently selected output device to the engine."""
        dev = self._record_panel.selected_output_device
        if dev is not None:
            self._engine.set_device(dev["index"], dev["channels"])

    def _sync_input_device(self) -> None:
        """Push the currently selected input device to the engine."""
        dev = self._record_panel.selected_input_device
        if dev is not None:
            self._engine.set_input_device(dev["index"], dev["channels"])

    def _on_output_device_changed(self, device: dict) -> None:
        self._engine.set_device(device["index"], device["channels"])
        # Restart stream if currently playing so device change takes effect
        if self._engine.is_playing:
            pos = self._engine.position
            self._engine.stop()
            self._engine.seek(pos)
            self._engine.play()

    def _on_input_device_changed(self, device: dict) -> None:
        self._engine.set_input_device(device["index"], device["channels"])

    def _on_latency_offset_changed(self, value: float) -> None:
        self._engine.latency_offset_ms = value

    # --- Seek ---

    def _on_seek(self, normalized: float) -> None:
        """Handle seek from slider or waveform click."""
        if self._engine.is_recording:
            return
        engine_total = self._engine.total_frames
        if engine_total == 0:
            return

        # Use waveform total for cursor positioning (includes offsets)
        wf_total = self._waveform_total_frames or engine_total
        frame = int(normalized * wf_total)
        # Clamp to engine range so we don't seek past actual audio
        frame = max(0, min(frame, engine_total - 1))
        self._engine.seek(frame)
        sr = self._engine.sample_rate
        self._record_panel.update_time_display(frame / sr, wf_total / sr)
        self._record_panel.update_seek_slider(
            frame / wf_total if wf_total > 0 else 0
        )
        # Update all waveform cursors
        self._import_panel.set_all_cursors(frame, sr, total_duration=wf_total)

    # --- Playback controls ---

    def _on_play(self) -> None:
        self._sync_output_device()
        try:
            self._engine.play()
        except Exception as exc:
            QMessageBox.warning(self, "Playback Error", str(exc))
            return
        self._position_timer.start()
        self._record_panel.update_transport_state(True, False)

    def _on_pause(self) -> None:
        self._engine.pause()
        self._position_timer.stop()
        self._record_panel.update_transport_state(False, True)

    def _on_stop(self) -> None:
        self._engine.stop()
        self._position_timer.stop()
        self._record_panel.update_transport_state(False, False)
        self._record_panel.set_playback_enabled(True)
        total_sec = self._engine.total_frames / self._engine.sample_rate
        self._record_panel.update_time_display(0.0, total_sec)
        self._record_panel.update_seek_slider(0.0)
        # Clear all waveform cursors and level meters
        for s in _SLOT_NAMES:
            wf = self._import_panel.get_waveform(s)
            if wf is not None:
                wf.clear_cursor()
        self._import_panel.reset_level_meters()

    def _on_volume_changed(self, value: float) -> None:
        self._engine.volume = value

    # --- Recording controls ---

    def _on_record_start(self) -> None:
        # Recording uses single-track mode — load minus track for playback
        minus = self._import_panel.minus_track
        if minus is None:
            QMessageBox.warning(self, "Recording Error", "No minus track loaded.")
            return
        minus_data, minus_sr = minus
        self._engine.load(minus_data, minus_sr)

        self._sync_output_device()
        self._sync_input_device()
        try:
            self._engine.start_recording()
        except Exception as exc:
            QMessageBox.warning(self, "Recording Error", str(exc))
            return
        self._position_timer.start()
        self._record_panel.update_recording_state(True)

    def _on_record_finish(self) -> None:
        result = self._engine.finish_recording()
        self._position_timer.stop()
        self._record_panel.update_recording_state(False)
        self._record_panel.set_record_enabled(True)

        if result is None:
            return

        data, sr = result

        # Auto-save WAV alongside the minus file
        save_path = self._generate_recording_path()
        if save_path:
            try:
                save_audio(save_path, data, sr)
            except Exception as exc:
                QMessageBox.warning(self, "Save Error", f"Could not save recording:\n{exc}")
                save_path = None

        # Populate vocal slot (triggers _on_track_loaded → auto-check mute, rebuild multi)
        self._import_panel.set_vocal_track(data, sr, path=save_path)

    def _on_record_stop(self) -> None:
        self._engine.stop_recording()
        self._position_timer.stop()
        self._record_panel.update_recording_state(False)
        self._record_panel.set_record_enabled(True)

        # Rebuild multi-track to restore normal playback
        self._update_multi_playback()

    def _generate_recording_path(self) -> str | None:
        """Generate a path for the recording WAV file next to the minus track."""
        minus_path = (
            self._import_panel.get_track_path("minus_import")
            or self._import_panel.get_track_path("minus_sep")
        )
        if minus_path is None:
            return None
        directory = os.path.dirname(minus_path)
        minus_name = os.path.splitext(os.path.basename(minus_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(directory, f"{minus_name}_vocal_{timestamp}.wav")

    # --- Position tick ---

    def _on_position_tick(self) -> None:
        total = self._engine.total_frames
        if total == 0:
            return

        pos = self._engine.position
        sr = self._engine.sample_rate

        # Use waveform total for cursor positioning (includes offsets)
        wf_total = self._waveform_total_frames or total
        normalized = pos / wf_total

        # Update ALL waveform cursors (global timeline)
        self._import_panel.set_all_cursors(pos, sr, total_duration=wf_total)

        # Update level meters
        self._import_panel.update_level_meters(pos, wf_total, self._track_offsets)

        # Update time display and seek slider
        self._record_panel.update_time_display(pos / sr, wf_total / sr)
        self._record_panel.update_seek_slider(normalized)

        # Detect natural end of playback
        if self._engine.is_recording and self._engine.playback_ended:
            # Minus track ended during recording → auto-finish
            self._on_record_finish()
            return

        if not self._engine.is_playing and not self._engine.is_paused and not self._engine.is_recording:
            self._position_timer.stop()
            self._record_panel.update_transport_state(False, False)
            self._record_panel.set_playback_enabled(True)
            self._import_panel.reset_level_meters()

    # --- Apply Effects ---

    def _on_apply_effects(self) -> None:
        """Run the vocal effects chain in background and populate V-proc slot."""
        vocal = self._import_panel.vocal_track
        if vocal is None:
            return

        vocal_data, vocal_sr = vocal[0].copy(), vocal[1]

        # If effects chain is bypassed, copy vocal directly to V-proc slot
        if not self._mix_panel.effects_chain_enabled():
            self._import_panel.set_vocal_processed_track(vocal_data, vocal_sr)
            self._mix_panel.set_effects_status("Bypassed (copied raw vocal)")
            self._mix_panel.set_export_vocal_enabled(True)
            self._update_auto_tune_ready()
            return

        self._mix_panel.set_apply_effects_enabled(False)
        self._mix_panel.set_effects_status("Processing...")

        # Gather vocal separation stem for NR guide (defensive copy)
        vocal_sep_data = vocal_sep_sr = None
        sep = self._import_panel.vocal_sep_track
        if sep is not None:
            vocal_sep_data, vocal_sep_sr = sep[0].copy(), sep[1]

        effects_config = self._mix_panel.get_effects_config()

        self._effects_worker = _EffectsWorker(
            vocal_data, vocal_sr, effects_config,
            vocal_sep_data=vocal_sep_data,
            vocal_sep_sr=vocal_sep_sr,
        )
        self._effects_worker.progress.connect(self._mix_panel.set_effects_status)
        self._effects_worker.finished.connect(self._on_effects_finished)
        self._effects_worker.error.connect(self._on_effects_error)
        self._effects_worker.start()

    def _on_effects_finished(self, data: np.ndarray, sr: int) -> None:
        self._import_panel.set_vocal_processed_track(data, sr)
        self._mix_panel.set_effects_status("Done")
        self._mix_panel.set_apply_effects_enabled(True)
        self._mix_panel.set_export_vocal_enabled(True)
        self._update_auto_tune_ready()
        self._effects_worker = None

    def _on_effects_error(self, msg: str) -> None:
        self._mix_panel.set_effects_status("Error")
        self._mix_panel.set_apply_effects_enabled(True)
        self._effects_worker = None
        QMessageBox.warning(self, "Effects Error", f"Vocal processing failed:\n{msg}")

    def _on_export_vocal(self) -> None:
        """Export the processed vocal track to a file, with alignment baked in."""
        proc = self._import_panel.vocal_processed_track
        if proc is None:
            return

        data, sr = proc

        # Apply alignment offset so the exported vocal is time-aligned
        net_offset = self._vocal_offset_samples - self._minus_offset_samples
        if net_offset != 0:
            minus = self._import_panel.minus_track
            target_len = minus[0].shape[0] if minus else data.shape[0]
            data = shift_track(data, net_offset, target_len)

        # Suggest path next to the vocal source
        default_dir = ""
        vocal_path = self._import_panel.get_track_path("vocal")
        if vocal_path:
            default_dir = os.path.dirname(vocal_path)

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Processed Vocal",
            os.path.join(default_dir, "vocal_processed.wav"),
            "WAV Files (*.wav);;FLAC Files (*.flac);;All Files (*)",
        )
        if not output_path:
            return

        try:
            save_audio(output_path, data, sr)
            self._mix_panel.set_effects_status(
                f"Exported: {os.path.basename(output_path)}"
            )
        except Exception as exc:
            QMessageBox.warning(
                self, "Export Error",
                f"Could not save processed vocal:\n{exc}",
            )

    # --- Auto-Tune Volume ---

    def _on_auto_tune(self) -> None:
        """Automatically adjust volume faders for optimal vocal/minus balance."""
        vocal_track = (
            self._import_panel.vocal_processed_track
            or self._import_panel.vocal_track
        )
        minus_track = self._import_panel.minus_track
        if vocal_track is None or minus_track is None:
            return

        from vocalforge.audio.mixer import measure_lufs

        vocal_data, vocal_sr = vocal_track
        minus_data, minus_sr = minus_track

        vocal_lufs = measure_lufs(vocal_data, vocal_sr)
        minus_lufs = measure_lufs(minus_data, minus_sr)

        if not np.isfinite(vocal_lufs) or not np.isfinite(minus_lufs):
            QMessageBox.warning(
                self, "Auto-Tune Volume",
                "Could not measure loudness of one or both tracks.",
            )
            return

        # Target: vocal ~3dB above minus in the mix
        target_diff = 3.0
        diff = vocal_lufs - minus_lufs
        needed = target_diff - diff

        # Split adjustment: boost vocal half, attenuate minus half
        vocal_gain = 10.0 ** (needed / 2.0 / 20.0)
        minus_gain = 10.0 ** (-needed / 2.0 / 20.0)

        # Clamp to fader range (0.0 – 2.0)
        vocal_gain = max(0.0, min(2.0, vocal_gain))
        minus_gain = max(0.0, min(2.0, minus_gain))

        # Apply to the right faders
        vocal_slot = (
            "vocal_processed"
            if self._import_panel.vocal_processed_track is not None
            else "vocal"
        )
        minus_slot = (
            "minus_import"
            if self._import_panel.minus_import_track is not None
            else "minus_sep"
        )

        self._import_panel.set_track_gain(vocal_slot, vocal_gain)
        self._import_panel.set_track_gain(minus_slot, minus_gain)

        # Rebuild multi-track with new gains
        self._update_multi_playback()

        self._mix_panel.set_status(
            f"Auto-tune: vocal {vocal_gain:.0%}, minus {minus_gain:.0%}"
        )

    # --- Track gain changed ---

    def _on_track_gain_changed(self, slot_name: str, gain: float) -> None:
        """Handle per-track volume fader changes — rebuild multi-track."""
        self._update_multi_playback()

    # --- Keyboard shortcuts ---

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            # Skip when focused widget is a text-entry control
            focused = self.focusWidget()
            if isinstance(focused, (QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit)):
                super().keyPressEvent(event)
                return
            # Skip during recording
            if self._engine.is_recording:
                super().keyPressEvent(event)
                return
            # Toggle play/pause
            if self._engine.is_playing:
                self._on_pause()
            else:
                self._on_play()
            return
        super().keyPressEvent(event)
