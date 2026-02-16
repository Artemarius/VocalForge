"""Mix panel — mixing controls and export.

Phase 1: placeholder only.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox


class MixPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        group = QGroupBox("Mix & Export")
        group_layout = QVBoxLayout(group)
        group_layout.addWidget(QLabel("Mixing controls will be available in Phase 5."))
        layout.addWidget(group)

        layout.addStretch()
