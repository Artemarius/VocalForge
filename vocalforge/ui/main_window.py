"""Main window — layout and panel orchestration."""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QMessageBox

from vocalforge.audio.engine import PlaybackEngine
from vocalforge.ui.import_panel import ImportPanel
from vocalforge.ui.record_panel import RecordPanel
from vocalforge.ui.mix_panel import MixPanel


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

        # Playback engine
        self._engine = PlaybackEngine()

        # Position polling timer (30ms ≈ 33 fps cursor updates)
        self._position_timer = QTimer(self)
        self._position_timer.setInterval(30)
        self._position_timer.timeout.connect(self._on_position_tick)

        # Connect signals
        self._import_panel.track_loaded.connect(self._on_track_loaded)
        self._record_panel.play_clicked.connect(self._on_play)
        self._record_panel.pause_clicked.connect(self._on_pause)
        self._record_panel.stop_clicked.connect(self._on_stop)
        self._record_panel.volume_changed.connect(self._on_volume_changed)
        self._record_panel.output_device_changed.connect(self._on_output_device_changed)

    def _on_track_loaded(self, slot_name: str) -> None:
        if slot_name != "minus":
            return
        track = self._import_panel.minus_track
        if track is None:
            return
        data, sr = track
        self._engine.load(data, sr)
        self._sync_device()
        self._record_panel.set_playback_enabled(True)
        self._record_panel.update_transport_state(False, False)
        total_sec = self._engine.total_frames / self._engine.sample_rate
        self._record_panel.update_time_display(0.0, total_sec)

    def _sync_device(self) -> None:
        """Push the currently selected output device to the engine."""
        dev = self._record_panel.selected_output_device
        if dev is not None:
            self._engine.set_device(dev["index"], dev["channels"])

    def _on_play(self) -> None:
        self._sync_device()
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
        waveform = self._import_panel.get_waveform("minus")
        if waveform is not None:
            waveform.clear_cursor()

    def _on_volume_changed(self, value: float) -> None:
        self._engine.volume = value

    def _on_output_device_changed(self, device: dict) -> None:
        self._engine.set_device(device["index"], device["channels"])

    def _on_position_tick(self) -> None:
        total = self._engine.total_frames
        if total == 0:
            return

        pos = self._engine.position
        sr = self._engine.sample_rate

        # Update cursor
        waveform = self._import_panel.get_waveform("minus")
        if waveform is not None:
            waveform.set_cursor_position(pos / total)

        # Update time display
        self._record_panel.update_time_display(pos / sr, total / sr)

        # Detect natural end of playback
        if not self._engine.is_playing and not self._engine.is_paused:
            self._position_timer.stop()
            self._record_panel.update_transport_state(False, False)
            self._record_panel.set_playback_enabled(True)
