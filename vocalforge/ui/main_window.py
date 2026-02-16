"""Main window — layout and panel orchestration."""

from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout

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
