"""Main window — layout and panel orchestration."""

import os
from datetime import datetime

import numpy as np
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from vocalforge.audio.alignment import align_tracks, pad_or_trim, resample_if_needed
from vocalforge.audio.engine import AudioEngine
from vocalforge.audio.mixer import mix_tracks
from vocalforge.separation.demucs_worker import separate_song
from vocalforge.ui.import_panel import ImportPanel
from vocalforge.ui.mix_panel import MixPanel
from vocalforge.ui.record_panel import RecordPanel
from vocalforge.utils.audio_io import save_audio


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

    def run(self) -> None:
        try:
            sr = self._minus_sr

            # Resample vocal if sample rates differ
            vocal = self._vocal_data
            if self._vocal_sr != sr:
                self.progress.emit("Resampling vocal...")
                vocal = resample_if_needed(vocal, self._vocal_sr, sr)

            target_len = self._minus_data.shape[0]
            align_info = {"lag_samples": 0, "lag_ms": 0.0}

            if self._alignment_mode == "none":
                self.progress.emit("Padding/trimming (no alignment)...")
                aligned_vocal = pad_or_trim(vocal, target_len)
            elif self._alignment_mode == "vocal":
                # Align against Demucs vocals stem
                if self._vocals_stem_path is None:
                    raise RuntimeError(
                        "Vocal matching requires a Demucs vocals stem. "
                        "Run Separate first."
                    )
                self.progress.emit("Loading vocals stem...")
                import soundfile as sf
                vocals_stem, stem_sr = sf.read(self._vocals_stem_path, dtype="float32")
                if stem_sr != sr:
                    vocals_stem = resample_if_needed(vocals_stem, stem_sr, sr)
                self.progress.emit("Aligning to vocals stem...")
                aligned_vocal, align_info = align_tracks(vocals_stem, vocal, sr)
            else:
                # Default: background music alignment
                self.progress.emit("Aligning...")
                aligned_vocal, align_info = align_tracks(self._minus_data, vocal, sr)

            # Mix
            self.progress.emit("Mixing...")
            mix_data, mix_info = mix_tracks(
                self._minus_data,
                aligned_vocal,
                sr,
                vocal_gain=self._vocal_gain,
                instrumental_gain=self._instrumental_gain,
                target_lufs=self._target_lufs,
            )

            # Save
            self.progress.emit("Saving...")
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
            if os.path.isfile(vocals_path):
                result["vocals_path"] = vocals_path

            self.finished.emit(result)
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
        self._active_track_slot: str = "minus"
        self._vocals_stem_path: str | None = None

        # Worker references (prevent GC while running)
        self._mix_worker: _MixWorker | None = None
        self._separation_worker: _SeparationWorker | None = None

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

        # Connect signals — track selector & seek
        self._record_panel.track_selected.connect(self._on_track_selected)
        self._record_panel.seek_requested.connect(self._on_seek)
        for slot in ("song", "minus", "vocal"):
            wf = self._import_panel.get_waveform(slot)
            if wf is not None:
                wf.seek_requested.connect(self._on_seek)

        # Connect signals — separation
        self._import_panel.separate_requested.connect(self._on_separate_start)

        # Connect signals — mix & export
        self._import_panel.track_loaded.connect(self._update_mix_ready)
        self._mix_panel.mix_export_clicked.connect(self._on_mix_export)

    # --- Track loading ---

    def _on_track_loaded(self, slot_name: str) -> None:
        # Always enable record when minus is loaded
        if slot_name == "minus":
            self._record_panel.set_record_enabled(True)

        # Only load into engine if the loaded track matches the active selector
        if slot_name != self._active_track_slot:
            return
        track = self._get_track(slot_name)
        if track is None:
            return
        data, sr = track
        self._engine.load(data, sr)
        self._sync_output_device()
        self._sync_input_device()
        self._record_panel.set_playback_enabled(True)
        self._record_panel.update_transport_state(False, False)
        total_sec = self._engine.total_frames / self._engine.sample_rate
        self._record_panel.update_time_display(0.0, total_sec)
        self._record_panel.update_seek_slider(0.0)

    def _get_track(self, slot_name: str) -> tuple | None:
        """Return (data, sr) for a given slot name."""
        if slot_name == "song":
            return self._import_panel.song_track
        elif slot_name == "minus":
            return self._import_panel.minus_track
        elif slot_name == "vocal":
            return self._import_panel.vocal_track
        return None

    def _on_track_selected(self, slot_name: str) -> None:
        """Handle track selector combo change."""
        self._active_track_slot = slot_name
        track = self._get_track(slot_name)
        if track is None:
            return
        # Stop current playback and load selected track
        self._engine.stop()
        self._position_timer.stop()
        data, sr = track
        self._engine.load(data, sr)
        self._sync_output_device()
        self._record_panel.set_playback_enabled(True)
        self._record_panel.update_transport_state(False, False)
        total_sec = self._engine.total_frames / self._engine.sample_rate
        self._record_panel.update_time_display(0.0, total_sec)
        self._record_panel.update_seek_slider(0.0)
        # Clear all waveform cursors
        for s in ("song", "minus", "vocal"):
            wf = self._import_panel.get_waveform(s)
            if wf is not None:
                wf.clear_cursor()

    # --- Mix readiness ---

    def _update_mix_ready(self, _slot_name: str = "") -> None:
        """Enable mix button when both minus and vocal tracks are loaded."""
        has_minus = self._import_panel.minus_track is not None
        has_vocal = self._import_panel.vocal_track is not None
        self._mix_panel.set_mix_enabled(has_minus and has_vocal)

    # --- Mix & Export ---

    def _on_mix_export(self) -> None:
        minus = self._import_panel.minus_track
        vocal = self._import_panel.vocal_track
        if minus is None or vocal is None:
            return

        minus_data, minus_sr = minus
        vocal_data, vocal_sr = vocal

        # Choose output path
        default_dir = ""
        minus_path = self._import_panel.get_track_path("minus")
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

        if alignment_mode == "vocal" and self._vocals_stem_path is None:
            QMessageBox.warning(
                self,
                "Missing Vocals Stem",
                "Vocal matching alignment requires a Demucs vocals stem.\n"
                "Run Separate on the song first.",
            )
            self._mix_panel.set_mix_enabled(True)
            return

        self._mix_worker = _MixWorker(
            minus_data,
            minus_sr,
            vocal_data,
            vocal_sr,
            output_path,
            vocal_gain=self._mix_panel.vocal_gain(),
            instrumental_gain=self._mix_panel.instrumental_gain(),
            target_lufs=self._mix_panel.target_lufs(),
            alignment_mode=alignment_mode,
            vocals_stem_path=self._vocals_stem_path,
        )
        self._mix_worker.progress.connect(self._mix_panel.set_status)
        self._mix_worker.finished.connect(self._on_mix_finished)
        self._mix_worker.error.connect(self._on_mix_error)
        self._mix_worker.start()

    def _on_mix_finished(self, result: dict) -> None:
        lag_ms = result.get("lag_ms", 0.0)
        lag_samples = result.get("lag_samples", 0)
        self._mix_panel.set_alignment_info(lag_ms, lag_samples)
        self._mix_panel.set_status(
            f"Done — saved to {os.path.basename(result['output_path'])}"
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
        self._import_panel.set_minus_track(
            result["data"], result["sr"], result["path"]
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
        total = self._engine.total_frames
        if total == 0:
            return
        frame = int(normalized * total)
        self._engine.seek(frame)
        sr = self._engine.sample_rate
        self._record_panel.update_time_display(frame / sr, total / sr)
        self._record_panel.update_seek_slider(normalized)
        waveform = self._import_panel.get_waveform(self._active_track_slot)
        if waveform is not None:
            waveform.set_cursor_position(normalized)

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
        waveform = self._import_panel.get_waveform(self._active_track_slot)
        if waveform is not None:
            waveform.clear_cursor()

    def _on_volume_changed(self, value: float) -> None:
        self._engine.volume = value

    # --- Recording controls ---

    def _on_record_start(self) -> None:
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

        # Populate vocal slot
        self._import_panel.set_vocal_track(data, sr, path=save_path)

        # Reset time display
        total_sec = self._engine.total_frames / self._engine.sample_rate
        self._record_panel.update_time_display(0.0, total_sec)
        self._record_panel.update_seek_slider(0.0)
        waveform = self._import_panel.get_waveform(self._active_track_slot)
        if waveform is not None:
            waveform.clear_cursor()

    def _on_record_stop(self) -> None:
        self._engine.stop_recording()
        self._position_timer.stop()
        self._record_panel.update_recording_state(False)
        self._record_panel.set_record_enabled(True)

        # Reset time display
        total_sec = self._engine.total_frames / self._engine.sample_rate
        self._record_panel.update_time_display(0.0, total_sec)
        self._record_panel.update_seek_slider(0.0)
        waveform = self._import_panel.get_waveform(self._active_track_slot)
        if waveform is not None:
            waveform.clear_cursor()

    def _generate_recording_path(self) -> str | None:
        """Generate a path for the recording WAV file next to the minus track."""
        minus_path = self._import_panel.get_track_path("minus")
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
        normalized = pos / total

        # Update cursor on active track's waveform
        waveform = self._import_panel.get_waveform(self._active_track_slot)
        if waveform is not None:
            waveform.set_cursor_position(normalized)

        # Update time display and seek slider
        self._record_panel.update_time_display(pos / sr, total / sr)
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
