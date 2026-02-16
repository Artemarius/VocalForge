"""Import panel — song loading and separation trigger.

Phase 1: placeholder only.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox


class ImportPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        group = QGroupBox("Import")
        group_layout = QVBoxLayout(group)
        group_layout.addWidget(QLabel("Song loading will be available in Phase 2."))
        layout.addWidget(group)

        layout.addStretch()
