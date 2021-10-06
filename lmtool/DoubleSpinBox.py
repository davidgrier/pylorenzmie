from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtCore import (pyqtSignal, pyqtSlot)


class DoubleSpinBox(QDoubleSpinBox):
    '''QDoubleSpinBox with buttonClicked signal'''

    buttonClicked = pyqtSignal(float)
    editingFinished = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super(DoubleSpinBox, self).__init__(*args, **kwargs)
        super(DoubleSpinBox, self).editingFinished.connect(
            self._reemitEditingFinished)

    def stepBy(self, step):
        value = self.value()
        super(DoubleSpinBox, self).stepBy(step)
        if self.value() != value:
            self.buttonClicked.emit(self.value())

    @pyqtSlot(int)
    def setDisabled(self, state):
        super(DoubleSpinBox, self).setDisabled(bool(state))

    @pyqtSlot()
    def _reemitEditingFinished(self):
        self.editingFinished[float].emit(self.value())
