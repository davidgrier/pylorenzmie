# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import (pyqtSignal, pyqtSlot)


class QDoubleSlider(QSlider):

    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super(QDoubleSlider, self).__init__(*args, **kwargs)
        self._imin = 0.
        self._imax = 10000.
        super(QDoubleSlider, self).setMinimum(int(self._imin))
        super(QDoubleSlider, self).setMaximum(int(self._imax))
        self._min = 0.
        self._max = 100.
        super(QDoubleSlider, self).valueChanged[int].connect(
            self.reemitValueChanged)

    def _convert_i2f(self, ivalue):
        frac = float(ivalue - self._imin) / (self._imax - self._imin)
        return frac * (self._max - self._min) + self._min

    def _convert_f2i(self, value):
        frac = (value - self._min) / (self._max - self._min)
        return int(frac * (self._imax - self._imin) + self._imin)

    @pyqtSlot(int)
    def reemitValueChanged(self, ivalue):
        value = self._convert_i2f(ivalue)
        self.valueChanged[float].emit(value)

    def value(self):
        ivalue = float(super(QDoubleSlider, self).value())
        return self._convert_i2f(ivalue)

    @pyqtSlot(float)
    def setValue(self, value):
        ivalue = self._convert_f2i(value)
        super(QDoubleSlider, self).setValue(ivalue)

    def setMinimum(self, value):
        self.setRange(value, self._max)

    def setMaximum(self, value):
        self.setRange(self._min, value)

    def setRange(self, minimum, maximum):
        ovalue = self.value()
        self._min = minimum
        self._max = maximum
        self.setValue(ovalue)

    def setSingleStep(self, value):
        ivalue = self._convert_f2i(value)
        super(QDoubleSlider, self).setSingleStep(ivalue)
