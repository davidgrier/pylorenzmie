import json
from pathlib import Path

from pyqtgraph.Qt import uic
from pyqtgraph.Qt.QtCore import (pyqtSignal, pyqtProperty)
from pyqtgraph.Qt.QtWidgets import QFrame

from pylorenzmie.lmtool.ParameterWidget import ParameterWidget
from pylorenzmie.theory import LorenzMie, best_model

_DIR = Path(__file__).parent


class LMWidget(QFrame):
    '''Parameter-control panel for a LorenzMie model.

    Loads its layout from ``uiFile`` and its parameter configuration
    from ``configFile`` (both relative to the lmtool package directory).
    Emits :attr:`propertyChanged` whenever any parameter value changes.
    '''

    cls = staticmethod(best_model)
    uiFile = 'LMWidget.ui'
    configFile = 'LMTool.json'

    propertyChanged = pyqtSignal(str, float)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        uic.loadUi(_DIR / self.uiFile, self)
        self.model = self.cls()
        self._loadConfig()
        self._connectSignals()
        self.model.properties = self.properties

    def _loadConfig(self) -> None:
        with open(_DIR / self.configFile) as f:
            config = json.load(f)
        self.setConfig(config)

    def _connectSignals(self) -> None:
        for control in self.controls:
            name = control.objectName()
            control.valueChanged.connect(
                lambda v, n=name: self._handleChange(n, v))

    @pyqtProperty(dict)
    def properties(self) -> LorenzMie.Properties:
        return {control.objectName(): control.value() for
                control in self.controls}

    @properties.setter
    def properties(self, properties: LorenzMie.Properties) -> None:
        for name, value in properties.items():
            if hasattr(self, name):
                getattr(self, name).setValue(value)

    @pyqtProperty(list)
    def fixed(self) -> list[str]:
        return [c.objectName() for c in self.controls if c.fixed()]

    @fixed.setter
    def fixed(self, fixed: list[str]) -> None:
        for control in self.controls:
            control.setFixed(control.objectName() in fixed)

    def _handleChange(self, name: str, value: float) -> None:
        self.model.properties = {name: value}
        self.propertyChanged.emit(name, value)

    def config(self) -> dict[str, LorenzMie.Properties]:
        return {c.objectName(): c.settings() for c in self.controls}

    def setConfig(self, config: dict[str, LorenzMie.Properties]) -> None:
        self.controls = []
        for control, settings in config.items():
            if hasattr(self, control):
                widget = getattr(self, control)
                self.controls.append(widget)
                widget.setSettings(settings)

    @classmethod
    def example(cls) -> None:
        from pyqtgraph import mkQApp

        def report(name, value):
            result = f'{name}: {value}'
            print(f'{result: <30}', end='\r')

        app = mkQApp()
        widget = cls()
        widget.show()
        widget.propertyChanged.connect(report)
        app.exec()


if __name__ == '__main__':  # pragma: no cover
    LMWidget.example()
