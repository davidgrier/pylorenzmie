from pyqtgraph.Qt.QtWidgets import QFrame
from pyqtgraph.Qt import uic
from pyqtgraph.Qt.QtCore import (pyqtSignal, pyqtSlot, pyqtProperty)
from pylorenzmie.lmtool.ParameterWidget import ParameterWidget
from pylorenzmie.theory import LorenzMie
import json


class LMWidget(QFrame):

    cls = LorenzMie
    uiFile = 'LMWidget.ui'
    configFile = 'LMTool.json'

    propertyChanged = pyqtSignal(str, float)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        uic.loadUi(self.uiFile, self)
        self.model = self.cls()
        self._loadConfig()
        self._connectSignals()
        self.model.properties = self.properties

    def _loadConfig(self) -> None:
        with open(self.configFile) as f:
            config = json.load(f)
        self.setConfig(config)

    def _connectSignals(self) -> None:
        for control in self.controls:
            control.valueChanged.connect(self._handleChange)

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
            control.setFixed(control.objectName in fixed)

    @pyqtSlot(float)
    def _handleChange(self, value: float) -> None:
        name = self.sender().objectName()
        self.model.properties = {name: value}
        self.propertyChanged.emit(name, value)

    def config(self) -> dict[str, LorenzMie.Properties]:
        return {n: c.settings() for n, c in self.__dict__.items()
                if isinstance(c, ParameterWidget)}

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


if __name__ == '__main__':
    LMWidget.example()
