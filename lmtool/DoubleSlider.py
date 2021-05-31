# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import (pyqtSignal, pyqtSlot)


class DoubleSlider(QSlider):
    '''
    Slider widget for double-precision floating point values

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
    valueChanged(float) :
        Overloaded signal containing current value

    Slots
    -----
    setValue(float) :
        Overloaded slot for setting current value
    '''

    __pyqtSignals__ = ('valueChanged(float)',)
    
    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super(DoubleSlider, self).__init__(*args, **kwargs)
        self._imin = 0.
        self._imax = 10000.
        super(DoubleSlider, self).setMinimum(int(self._imin))
        super(DoubleSlider, self).setMaximum(int(self._imax))
        self._min = 0.
        self._max = 100.
        super(DoubleSlider, self).valueChanged[int].connect(
            self._reemitValueChanged)

    def _convert_i2f(self, ivalue):
        '''Convert integer slider value to floating point'''
        frac = float(ivalue - self._imin) / (self._imax - self._imin)
        return frac * (self._max - self._min) + self._min

    def _convert_f2i(self, value):
        '''Convert floating point value to integer slider value'''
        frac = (value - self._min) / (self._max - self._min)
        return int(frac * (self._imax - self._imin) + self._imin)

    @pyqtSlot(int)
    def _reemitValueChanged(self, ivalue):
        '''Overload valueChanged signal'''
        value = self._convert_i2f(ivalue)
        self.valueChanged[float].emit(value)

    def value(self):
        '''Current floating point value

        Returns
        -------
        value : floateditingFinished
        '''
        ivalue = float(super(DoubleSlider, self).value())
        return self._convert_i2f(ivalue)

    @pyqtSlot(float)
    def setValue(self, value):
        '''Set slider value programmatically

        Parameters
        ----------
        value : float
        '''
        ivalue = self._convert_f2i(value)
        super(DoubleSlider, self).setValue(ivalue)

    def setMinimum(self, minimum):
        '''Set minimum end of slider range

        Parameters
        ----------
        minimum : float
        '''
        self.setRange(minimum, self._max)

    def setMaximum(self, maximum):
        '''Set maximum end of slider range

        Parameters
        ----------
        maximum : float
        '''
        self.setRange(self._min, maximum)

    def setRange(self, minimum, maximum):
        '''Set minimum and maximum of slider range

        Parameters
        ----------
        minimum : float
        maximum : float
        '''
        ovalue = self.value()
        self._min = minimum
        self._max = maximum
        self.setValue(ovalue)

    def setSingleStep(self, value):
        '''Set value change associated with single slider step

        Parameters
        ----------
        value : float
        '''
        ivalue = self._convert_f2i(value)
        super(DoubleSlider, self).setSingleStep(ivalue)

    @pyqtSlot(int)
    def setDisabled(self, state):
        super(DoubleSlider, self).setDisabled(bool(state))
