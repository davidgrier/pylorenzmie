#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtWidgets
from QDoubleSlider import QDoubleSlider


class QParameterWidget(QtWidgets.QFrame):

    '''Widget for adjusting a floating-point parameter

    QParameterWidget combines a QDoubleSpinBox with
    a custom QDoubleSlider to set and adjust the value
    of a parameter.  The name of the parameter is displayed
    with a QLabel.  The parameter can be fixed with a
    QCheckBox, which also disables the controls.

    ...

    Attributes
    ----------
    spinbox : QDoubleSpinBox
        Widget that displays and controls the parameter
    slider : QDoubleSlider
        Widget for continuously adjusting the parameter
    checkbox : QCheckBox
        Checkbox for fixing the parameter
    fixed : bool
        True if parameter is fixed

    Methods
    -------
    minimum() : float
        Minimum value for parameter
    maximum() : float
        Maximum value for parameter
    value() : float
        Current value of parameter
    decimals() : int
        Number of decimal places to display
    setMinimum(minimum) :
        Set minimum end of value range
    setMaximum(maximum) :
        Set maximum end of value range
    setRange(minimum, maximum) :
        Set value range
    setSingleStep(value) :
        Set value change associated with single widget step

    Slots
    -----
    setValue(float) :
        Sets the parameter value

    Signals
    -------
    valueChanged(float) :
        Emitted when parameter value is changed
    '''

    def __init__(self,
                 parent=None,
                 text='parameter',
                 minimum=0,
                 maximum=100,
                 value=50,
                 decimals=3, **kwargs):
        super(QParameterWidget, self).__init__(parent, **kwargs)
        self.setupUI()
        self.setupAPI()
        self.setText(text)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setValue(value)
        self.setDecimals(decimals)

    def setupUI(self):
        self.label = QtWidgets.QLabel(self)
        self.spinbox = QtWidgets.QDoubleSpinBox(self)
        self.checkbox = QtWidgets.QCheckBox(self)
        self.slider = QDoubleSlider(self)
        self.slider.setOrientation(QtCore.Qt.Horizontal)

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setHorizontalSpacing(4)
        self.layout.setVerticalSpacing(0)
        self.layout.addWidget(self.label, 0, 0)
        self.layout.addWidget(self.spinbox, 0, 1)
        self.layout.addWidget(self.checkbox, 0, 2)
        self.layout.addWidget(self.slider, 1, 0, 1, 3)

        self.spinbox.editingFinished.connect(self.updateValues)
        self.slider.valueChanged['double'].connect(self.spinbox.setValue)
        self.checkbox.stateChanged.connect(self.fixValue)

    @QtCore.pyqtSlot()
    def updateValues(self):
        self.slider.setValue(self.spinbox.value())

    def setupAPI(self):
        # Methods
        self.text = self.label.text
        self.setText = self.label.setText
        self.minimum = self.spinbox.minimum
        self.maximum = self.spinbox.maximum
        self.decimals = self.spinbox.decimals
        self.setDecimals = self.spinbox.setDecimals
        self.value = self.spinbox.value

        # Slots
        self.setValue = self.slider.setValue

        # Signals
        self.valueChanged = self.slider.valueChanged

    @QtCore.pyqtSlot(int)
    def fixValue(self, state):
        self.spinbox.setDisabled(state)
        self.slider.setDisabled(state)

    @property
    def fixed(self):
        '''Parameter cannot be changed if True'''
        return self.checkbox.isChecked()

    @fixed.setter
    def fixed(self, state):
        self.checkbox.setChecked(state)

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

    def setRange(self, min, max):
        '''Set range of values

        Parameters
        ----------
        min : float
        max : float
        '''
        self.spinbox.setRange(min, max)
        self.slider.setRange(min, max)

    def setSingleStep(self, value):
        '''Set value change associated with single step

        Parameters
        ----------
        value : float
        '''
        self.spinbox.setSingleStep(value)
        self.slider.setSingleStep(value)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    param = QParameterWidget(text='a<sub>p</sub>')
    param.setRange(0.3, 10)
    param.setValue(0.75)
    param.setDecimals(3)
    param.spinbox.setSuffix(' μm')
    param.show()
    sys.exit(app.exec_())
