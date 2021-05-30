from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtCore import pyqtSignal


class SpinBox(QDoubleSpinBox):

    buttonClicked = pyqtSignal()

    def stepBy(self, step):
        value = self.value()
        super(SpinBox, self).stepBy(step)
        if self.value() != value:
            self.buttonClicked.emit()
