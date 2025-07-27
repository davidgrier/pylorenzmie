from pyqtgraph.Qt.QtWidgets import QSlider
from pyqtgraph.Qt.QtCore import (pyqtSignal, pyqtSlot)
import logging
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class DoubleSlider(QSlider):
    '''Slider widget for double-precision floating point values

    ...

    Methods
    -------
    value() : float
        Returns current floating point value
    setMinimum(value) :
        Set minimum floating point value
    setMaximum(value) :
        Set maximum floating point value
    setRange(minimum, maximum) :
        Set floating point range of slider
    setSingleStep(value) :
        Set value change associated with single slider step

    Signals
    -------
    valueChanged[float] :
        Overloaded signal containing current value

    Slots
    -----
    setValue(value: float) :
        Overloaded slot for setting current value
    '''

    __pyqtSignals__ = ('valueChanged(float)',)

    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._imin = 0
        self._imax = 10000
        super().setMinimum(self._imin)
        super().setMaximum(self._imax)
        self._minimum = 0.
        self._maximum = 100.
        self._value = 0.
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
        '''Current floating point value

        Returns
        -------
        value : floateditingFinished
        '''
        return self._i2f(self._value)

    @pyqtSlot(float)
    def setValue(self, value: float) -> None:
        '''Set slider value programmatically

        Parameters
        ----------
        value : float
        '''
        self._value = np.clip(value, self._minimum, self._maximum)
        super().setValue(self._f2i(value))

    def minimum(self) -> float:
        return self._minimum

    def setMinimum(self, minimum: float) -> None:
        '''Set minimum end of slider range

        Parameters
        ----------
        minimum : float
        '''
        if minimum > self._maximum:
            logger.warn(f'{minimum} > maximum ({self._max})')
            return
        self._minimum = minimum
        self.setValue(self._value)

    def maximum(self) -> float:
        return self._maximum

    def setMaximum(self, maximum: float) -> None:
        '''Set maximum end of slider range

        Parameters
        ----------
        maximum : float
        '''
        if maximum < self._minimum:
            logger.warn(f'{maximum} < minimum ({self._min})')
            return
        self._maximum = maximum
        self.setValue(self._value)

    def setRange(self, minimum: float, maximum: float) -> None:
        '''Set minimum and maximum of slider range

        Parameters
        ----------
        minimum : float
        maximum : float
        '''
        self.setMinimum(minimum)
        self.setMaximum(maximum)

    def setSingleStep(self, value: float) -> None:
        '''Set value change associated with single slider step

        Parameters
        ----------
        value : float
        '''
        super().setSingleStep(self._f2i(value))

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
    DoubleSlider.example()
