from PyQt5.QtWidgets import QFrame
from PyQt5 import uic
from PyQt5.QtCore import (pyqtSignal, pyqtSlot)
from typing import (List, Dict)


class ParameterWidget(QFrame):

    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        uic.loadUi('ParameterWidget.ui', self)
        self._addMethods()
        self._connectSignals()

    def _addMethods(self) -> None:
        self.value = self.spinbox.value
        self.setValue = self.slider.setValue
        self.minimum = self.slider.minimum
        self.maximum = self.slider.maximum
        self.decimals = self.spinbox.decimals
        self.setDecimals = self.spinbox.setDecimals
        self.step = self.spinbox.singleStep
        self.prefix = self.spinbox.prefix
        self.setPrefix = self.spinbox.setPrefix
        self.suffix = self.spinbox.suffix
        self.setSuffix = self.spinbox.setSuffix
        self.text = self.label.text
        self.setText = self.label.setText
        self.fixed = self.checkbox.isChecked
        self.setFixed = self.checkbox.setChecked

    def _connectSignals(self):
        self.slider.valueChanged.connect(self._reportChange)

    @pyqtSlot(float)
    def _reportChange(self, value):
        self.valueChanged.emit(value)

    def range(self) -> List[float]:
        return [self.minimum(), self.maximum()]

    def setRange(self, range: List[float]) -> None:
        self.slider.setRange(*range)
        self.spinbox.setRange(*range)

    def setStep(self, value: float) -> None:
        self.slider.setSingleStep(value)
        self.spinbox.setSingleStep(value)

    def settings(self) -> Dict:
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
            if hasattr(self, setting):
                method = f'set{setting.capitalize()}'
                if hasattr(self, method):
                    setter = getattr(self, method)
                    setter(value)

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication

    def report(value):
        print(f'{value:.2f}', end='\r')

    app = QApplication([])
    widget = ParameterWidget()
    widget.setRange([3, 11])
    widget.show()
    widget.valueChanged.connect(report)
    app.exec_()
