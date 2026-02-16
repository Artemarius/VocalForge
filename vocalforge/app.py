"""QApplication setup and main entry point."""

import sys

from PySide6.QtWidgets import QApplication

from vocalforge.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
