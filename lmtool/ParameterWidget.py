#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5 import (QtWidgets, QtCore, uic)


class ParameterWidget(QtWidgets.QFrame):

    def __init__(self, *args,
                 text='parameter',
                 minimum=0,
                 maximum=100,
                 value=50,
                 decimals=3,
                 **kwargs):
        super(ParameterWidget, self).__init__(*args, **kwargs)
        uic.loadUi('ParameterWidget.ui', self)
        self.setupAPI()
        self.setText(text)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setValue(value)
        self.setDecimals(decimals)

    def setObjectName(self, name):
        super().setObjectName(name)
        for child in self.children():
            child.setObjectName(name)

    def setupAPI(self):
        self.text = self.label.text
        self.setText = self.label.setText
        self.minimum = self.spinbox.minimum
        self.maximum = self.spinbox.maximum
        self.decimals = self.spinbox.decimals
        self.setDecimals = self.spinbox.setDecimals
        self.setSuffix = self.spinbox.setSuffix
        self.value = self.spinbox.value

        # Slots
        self.setValue = self.slider.setValue

        # Signals
        self.valueChanged = self.slider.valueChanged

    @QtCore.pyqtProperty(bool)
    def fixed(self):
        '''Parameter cannot be changed if True'''
        return self.checkbox.isChecked()

    @fixed.setter
    def fixed(self, state):
        self.checkbox.setChecked(state)

    def setFixed(self, state):
        self.fixed = bool(state)

    def setMinimum(self, min):
        '''Set minimum end of value range

        Parameters
        ----------
        min : float
        '''
        self.spinbox.setMinimum(min)
        self.slider.setMinimum(min)

    def setMaximum(self, max):
        '''Set maximum end of value range

        Parameters
        ----------
        max : float
        '''
        self.spinbox.setMaximum(max)
        self.slider.setMaximum(max)

    def setRange(self, range):
        '''Set range of values

        Parameters
        ----------
        range : tuple
            (min, max)
        '''
        self.spinbox.setRange(*range)
        self.slider.setRange(*range)

    def setStep(self, value):
        '''Set value change associated with single step

        Parameters
        ----------
        value : float
        '''
        self.spinbox.setSingleStep(value)
        self.slider.setSingleStep(value)


def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    param = ParameterWidget(text='a<sub>p</sub>')
    param.setRange((0.3, 10))
    param.setValue(0.75)
    param.setDecimals(3)
    param.setSuffix(' μm')
    param.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
