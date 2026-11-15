from collections.abc import Sequence
from pathlib import Path

from qtpy import uic
from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import QCheckBox, QFrame, QLabel

from pylorenzmie.lmtool.DoubleSlider import DoubleSlider
from pylorenzmie.lmtool.DoubleSpinBox import DoubleSpinBox

_DIR = Path(__file__).parent


class ParameterWidget(QFrame):
    '''Compound widget combining a label, spinbox, slider, and lock checkbox.

    Exposes a unified interface for reading and setting a single floating-point
    parameter value.  The slider and spinbox are kept in sync via Qt Designer
    connections in ``ParameterWidget.ui``.
    '''

    # what the layout in ParameterWidget.ui makes
    label: QLabel
    spinbox: DoubleSpinBox
    checkbox: QCheckBox
    slider: DoubleSlider

    #: Emitted with the new value when the slider or the spinbox changes.
    valueChanged = Signal(float)
    #: Emitted when the lock checkbox changes.
    fixedChanged = Signal(bool)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        uic.loadUi(_DIR / 'ParameterWidget.ui', self)
        self._connectSignals()

    def _connectSignals(self) -> None:
        self.slider.valueChanged.connect(self._reportChange)
        self.checkbox.toggled.connect(self.fixedChanged)

    @Slot(float)
    def _reportChange(self, value: float) -> None:
        self.valueChanged.emit(value)

    # ------------------------------------------------------------------
    # Value interface
    # ------------------------------------------------------------------

    def value(self) -> float:
        '''Return the value shown in the spinbox.'''
        return self.spinbox.value()

    def setValue(self, value: float) -> None:
        '''Set the value, moving the slider (the spinbox follows).'''
        self.slider.setValue(value)

    # ------------------------------------------------------------------
    # Range / step
    # ------------------------------------------------------------------

    def minimum(self) -> float:
        '''Return the lower limit of the value.'''
        return self.slider.minimum()

    def maximum(self) -> float:
        '''Return the upper limit of the value.'''
        return self.slider.maximum()

    def range(self) -> list[float]:
        '''Return the limits of the value as ``[minimum, maximum]``.'''
        return [self.minimum(), self.maximum()]

    def setRange(self, limits: Sequence[float]) -> None:
        '''Set the limits of the slider and the spinbox.'''
        self.slider.setRange(*limits)
        self.spinbox.setRange(*limits)

    def step(self) -> float:
        '''Return the single step of the spinbox.'''
        return self.spinbox.singleStep()

    def setStep(self, value: float) -> None:
        '''Set the single step of the slider and the spinbox.'''
        self.slider.setSingleStep(value)
        self.spinbox.setSingleStep(value)

    # ------------------------------------------------------------------
    # Display formatting
    # ------------------------------------------------------------------

    def decimals(self) -> int:
        '''Return the number of decimals that the spinbox shows.'''
        return self.spinbox.decimals()

    def setDecimals(self, decimals: int) -> None:
        '''Set the number of decimals that the spinbox shows.'''
        self.spinbox.setDecimals(decimals)

    def prefix(self) -> str:
        '''Return the text that the spinbox shows before the value.'''
        return self.spinbox.prefix()

    def setPrefix(self, prefix: str) -> None:
        '''Set the text that the spinbox shows before the value.'''
        self.spinbox.setPrefix(prefix)

    def suffix(self) -> str:
        '''Return the text that the spinbox shows after the value.'''
        return self.spinbox.suffix()

    def setSuffix(self, suffix: str) -> None:
        '''Set the text that the spinbox shows after the value.'''
        self.spinbox.setSuffix(suffix)

    def text(self) -> str:
        '''Return the text of the label.'''
        return self.label.text()

    def setText(self, text: str) -> None:
        '''Set the text of the label.'''
        self.label.setText(text)

    # ------------------------------------------------------------------
    # Lock (fix parameter during optimization)
    # ------------------------------------------------------------------

    def fixed(self) -> bool:
        '''Return True if the parameter is held fixed in a fit.'''
        return self.checkbox.isChecked()

    def setFixed(self, fixed: bool) -> None:
        '''Hold the parameter fixed in a fit, or release it.'''
        self.checkbox.setChecked(fixed)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def settings(self) -> dict:
        '''Return the text, range, value, step, fixed state, decimals,
        prefix and suffix as a dict.
        '''
        s = dict(text=self.text(),
                 range=self.range(),
                 value=self.value(),
                 step=self.step(),
                 fixed=self.fixed(),
                 decimals=self.decimals())
        if self.prefix():
            s['prefix'] = self.prefix()
        if self.suffix():
            s['suffix'] = self.suffix()
        return s

    def setSettings(self, settings: dict) -> None:
        '''Apply a dict of the form that :meth:`settings` returns.

        A key without a matching setter (``set`` and the capitalized key)
        is ignored.
        '''
        for setting, value in settings.items():
            setter = f'set{setting.capitalize()}'
            if hasattr(self, setter):
                getattr(self, setter)(value)

    @classmethod
    def example(cls) -> None:
        '''Run a demonstration of the widget, for use as a script.'''
        from pyqtgraph import mkQApp

        def report(value):
            print(f'{value:.2f}', end='\r')

        app = mkQApp()
        widget = cls()
        widget.setRange([3, 11])
        widget.show()
        widget.valueChanged.connect(report)
        app.exec()


if __name__ == '__main__':  # pragma: no cover
    ParameterWidget.example()
