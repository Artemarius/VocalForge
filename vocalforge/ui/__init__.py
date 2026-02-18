"""Shared UI widgets for VocalForge panels."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider


class JumpSlider(QSlider):
    """QSlider that jumps to the clicked position instead of page-stepping."""

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            groove = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, opt,
                QStyle.SubControl.SC_SliderGroove, self,
            )
            if self.orientation() == Qt.Orientation.Horizontal:
                pos = event.position().x()
                val = QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(),
                    int(pos - groove.x()), groove.width(),
                )
            else:
                pos = event.position().y()
                val = QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(),
                    int(pos - groove.y()), groove.height(),
                    upsideDown=True,
                )
            self.setValue(val)
        super().mousePressEvent(event)
