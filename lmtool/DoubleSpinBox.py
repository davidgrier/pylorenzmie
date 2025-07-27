from pyqtgraph.Qt.QtWidgets import QDoubleSpinBox
from pyqtgraph.Qt.QtCore import (pyqtSignal, pyqtSlot)


class DoubleSpinBox(QDoubleSpinBox):
    '''QDoubleSpinBox with buttonClicked signal'''

    buttonClicked = pyqtSignal(float)
    editingFinished = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().editingFinished.connect(self._editingFinished)

    def stepBy(self, step: float) -> None:
        value = self.value()
        super().stepBy(step)
        if self.value() != value:
            self.buttonClicked.emit(self.value())

    @pyqtSlot()
    def _editingFinished(self) -> None:
        self.editingFinished[float].emit(self.value())

    @classmethod
    def example(cls) -> None:
        from pyqtgraph import mkQApp

        def report(value):
            print(f'{value:.2f}', end='\r')

        app = mkQApp()
        widget = cls()
        widget.show()
        widget.valueChanged[float].connect(report)
        app.exec()


if __name__ == '__main__':
    DoubleSpinBox.example()
