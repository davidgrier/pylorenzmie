import logging

import numpy as np
from pyqtgraph.Qt.QtCore import (pyqtSignal, pyqtSlot)
from pyqtgraph.Qt.QtWidgets import QSlider


logger = logging.getLogger(__name__)


class DoubleSlider(QSlider):
    '''QSlider with a floating-point value range.

    Maps the integer slider position to a float in ``[minimum, maximum]``
    and emits ``valueChanged[float]`` rather than ``valueChanged[int]``.

    Parameters
    ----------
    minimum : float, optional
        Lower bound of the float range. Default 0.0.
    maximum : float, optional
        Upper bound of the float range. Default 100.0.
    '''

    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._imin = 0
        self._imax = 10000
        super().setMinimum(self._imin)
        super().setMaximum(self._imax)
        self._minimum = 0.
        self._maximum = 100.
        super().valueChanged[int].connect(self._valueChanged)

    def _i2f(self, ivalue: int) -> float:
        '''Convert integer slider value to floating point'''
        frac = (ivalue - self._imin) / (self._imax - self._imin)
        return frac * (self._maximum - self._minimum) + self._minimum

    def _f2i(self, value: float) -> int:
        '''Convert floating point value to integer slider value'''
        frac = (value - self._minimum) / (self._maximum - self._minimum)
        return int(frac * (self._imax - self._imin) + self._imin)

    @pyqtSlot(int)
    def _valueChanged(self, value: int) -> None:
        '''Overload valueChanged signal'''
        self.valueChanged[float].emit(self._i2f(value))

    def value(self) -> float:
        return self._i2f(super().value())

    @pyqtSlot(float)
    def setValue(self, value: float) -> None:
        value = float(np.clip(value, self._minimum, self._maximum))
        super().setValue(self._f2i(value))

    def minimum(self) -> float:
        return self._minimum

    def setMinimum(self, minimum: float) -> None:
        if minimum > self._maximum:
            logger.warning(f'{minimum} > maximum ({self._maximum})')
            return
        self._minimum = minimum
        self.setValue(self.value())

    def maximum(self) -> float:
        return self._maximum

    def setMaximum(self, maximum: float) -> None:
        if maximum < self._minimum:
            logger.warning(f'{maximum} < minimum ({self._minimum})')
            return
        self._maximum = maximum
        self.setValue(self.value())

    def setRange(self, minimum: float, maximum: float) -> None:
        self.setMinimum(minimum)
        self.setMaximum(maximum)

    def setSingleStep(self, value: float) -> None:
        span = self._maximum - self._minimum
        int_step = max(1, int(value / span * (self._imax - self._imin)))
        super().setSingleStep(int_step)

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


if __name__ == '__main__':  # pragma: no cover
    DoubleSlider.example()
