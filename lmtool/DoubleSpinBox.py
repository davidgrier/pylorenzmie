from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtCore import (pyqtSignal, pyqtSlot)


class DoubleSpinBox(QDoubleSpinBox):
    '''QDoubleSpinBox with buttonClicked signal'''

    buttonClicked = pyqtSignal(float)
    editingFinished = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().editingFinished.connect(self._editingFinished)

    def stepBy(self, step):
        value = self.value()
        super().stepBy(step)
        if self.value() != value:
            self.buttonClicked.emit(self.value())

    @pyqtSlot()
    def _editingFinished(self):
        self.editingFinished[float].emit(self.value())


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication

    def report(value):
        print(f'{value:.2f}', end='\r')

    app = QApplication([])
    widget = DoubleSpinBox()
    widget.show()
    widget.valueChanged[float].connect(report)
    app.exec_()
