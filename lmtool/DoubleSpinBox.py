from qtpy.QtWidgets import QDoubleSpinBox
from qtpy.QtCore import Signal, Slot

from pylorenzmie.lmtool.required import required


class DoubleSpinBox(QDoubleSpinBox):
    '''QDoubleSpinBox with float-valued editingFinished and buttonClicked
    signals.'''

    buttonClicked = Signal(float)
    editingFinished = Signal(float)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # the signal of the base class is hidden by the float signal that
        # this class declares, so listen to the line edit, which has the
        # same triggers: the Return key, and the loss of focus
        required(self.lineEdit()).editingFinished.connect(
            self._editingFinished)

    def stepBy(self, step: int) -> None:
        value = self.value()
        super().stepBy(step)
        if self.value() != value:
            self.buttonClicked.emit(self.value())

    @Slot()
    def _editingFinished(self) -> None:
        self.editingFinished.emit(self.value())

    @classmethod
    def example(cls) -> None:
        '''Run a demonstration of the widget, for use as a script.'''
        from pyqtgraph import mkQApp

        def report(value):
            print(f'{value:.2f}', end='\r')

        app = mkQApp()
        widget = cls()
        widget.show()
        widget.valueChanged[float].connect(report)
        app.exec()


if __name__ == '__main__':  # pragma: no cover
    DoubleSpinBox.example()
