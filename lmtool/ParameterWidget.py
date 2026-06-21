from pathlib import Path
from pyqtgraph.Qt.QtWidgets import QFrame
from pyqtgraph.Qt import uic
from pyqtgraph.Qt.QtCore import (pyqtSignal, pyqtSlot)

_DIR = Path(__file__).parent


class ParameterWidget(QFrame):
    '''Compound widget combining a label, spinbox, slider, and lock checkbox.

    Exposes a unified interface for reading and setting a single floating-point
    parameter value.  The slider and spinbox are kept in sync via Qt Designer
    connections in ``ParameterWidget.ui``.
    '''

    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        uic.loadUi(_DIR / 'ParameterWidget.ui', self)
        self._connectSignals()

    def _connectSignals(self) -> None:
        self.slider.valueChanged.connect(self._reportChange)

    @pyqtSlot(float)
    def _reportChange(self, value: float) -> None:
        self.valueChanged.emit(value)

    # ------------------------------------------------------------------
    # Value interface
    # ------------------------------------------------------------------

    def value(self) -> float:
        return self.spinbox.value()

    def setValue(self, value: float) -> None:
        self.slider.setValue(value)

    # ------------------------------------------------------------------
    # Range / step
    # ------------------------------------------------------------------

    def minimum(self) -> float:
        return self.slider.minimum()

    def maximum(self) -> float:
        return self.slider.maximum()

    def range(self) -> list[float]:
        return [self.minimum(), self.maximum()]

    def setRange(self, range: list[float]) -> None:
        self.slider.setRange(*range)
        self.spinbox.setRange(*range)

    def step(self) -> float:
        return self.spinbox.singleStep()

    def setStep(self, value: float) -> None:
        self.slider.setSingleStep(value)
        self.spinbox.setSingleStep(value)

    # ------------------------------------------------------------------
    # Display formatting
    # ------------------------------------------------------------------

    def decimals(self) -> int:
        return self.spinbox.decimals()

    def setDecimals(self, decimals: int) -> None:
        self.spinbox.setDecimals(decimals)

    def prefix(self) -> str:
        return self.spinbox.prefix()

    def setPrefix(self, prefix: str) -> None:
        self.spinbox.setPrefix(prefix)

    def suffix(self) -> str:
        return self.spinbox.suffix()

    def setSuffix(self, suffix: str) -> None:
        self.spinbox.setSuffix(suffix)

    def text(self) -> str:
        return self.label.text()

    def setText(self, text: str) -> None:
        self.label.setText(text)

    # ------------------------------------------------------------------
    # Lock (fix parameter during optimization)
    # ------------------------------------------------------------------

    def fixed(self) -> bool:
        return self.checkbox.isChecked()

    def setFixed(self, fixed: bool) -> None:
        self.checkbox.setChecked(fixed)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def settings(self) -> dict:
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
        for setting, value in settings.items():
            setter = f'set{setting.capitalize()}'
            if hasattr(self, setter):
                getattr(self, setter)(value)

    @classmethod
    def example(cls) -> None:
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
