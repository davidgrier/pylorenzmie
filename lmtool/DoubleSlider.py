from typing import Any
import logging

import numpy as np
from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import QAbstractSlider, QSlider


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

    valueChanged = Signal(float)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._imin = 0
        self._imax = 10000
        super().setMinimum(self._imin)
        super().setMaximum(self._imax)
        self._minimum = 0.
        self._maximum = 100.

    def _i2f(self, ivalue: int) -> float:
        '''Convert integer slider value to floating point'''
        frac = (ivalue - self._imin) / (self._imax - self._imin)
        return frac * (self._maximum - self._minimum) + self._minimum

    def _f2i(self, value: float) -> int:
        '''Convert floating point value to the nearest slider value'''
        frac = (value - self._minimum) / (self._maximum - self._minimum)
        return round(frac * (self._imax - self._imin) + self._imin)

    def sliderChange(self, change: QAbstractSlider.SliderChange) -> None:
        '''Emit ``valueChanged`` with the float value.

        The signal of the base class, ``valueChanged(int)``, is hidden by
        the float signal that this class declares, so the change of the
        value is caught here instead.
        '''
        super().sliderChange(change)
        if change == QAbstractSlider.SliderChange.SliderValueChange:
            self.valueChanged.emit(self.value())

    def value(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._i2f(super().value())

    @Slot(float)
    def setValue(self, value: float) -> None:
        value = float(np.clip(value, self._minimum, self._maximum))
        super().setValue(self._f2i(value))

    def minimum(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._minimum

    def setMinimum(self, minimum: float) -> None:
        if minimum > self._maximum:
            logger.warning(f'{minimum} > maximum ({self._maximum})')
            return
        self._minimum = minimum
        self.setValue(self.value())

    def maximum(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
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
        '''Run a demonstration of the widget, for use as a script.'''
        from pyqtgraph import mkQApp

        def report(value: float) -> None:
            print(f'{value:.2f}', end='\r')

        app = mkQApp()
        widget = cls()
        widget.show()
        widget.valueChanged[float].connect(report)
        app.exec()


if __name__ == '__main__':  # pragma: no cover
    DoubleSlider.example()
