from pathlib import Path

from pyqtgraph.Qt import uic
from pyqtgraph.Qt.QtCore import (pyqtSignal, pyqtProperty)
from pyqtgraph.Qt.QtWidgets import QFrame

from pylorenzmie.lmtool.LMSession import LMSession
from pylorenzmie.lmtool.parameters import PARAMETERS
from pylorenzmie.lmtool.ParameterWidget import ParameterWidget
from pylorenzmie.theory import LorenzMie

_DIR = Path(__file__).parent


class LMWidget(QFrame):
    '''Parameter-control panel for a LorenzMie model.

    Loads its layout from ``uiFile`` (relative to the lmtool package
    directory) and configures the controls that the layout has from
    :attr:`specs`.  A specification without a control describes a hidden
    parameter, which a fit holds fixed.

    The panel is a view of an :class:`~pylorenzmie.lmtool.LMSession.LMSession`,
    which holds the model, the value of every parameter and which parameters
    a fit holds fixed: editing a control sets the parameter in the session,
    and a change in the session updates the controls.  A panel creates a
    session of its own unless it is given one.

    Parameters
    ----------
    session : LMSession, optional
        Session to display.  Default: a new session of the model class
        :attr:`cls`.
    '''

    cls = LorenzMie
    uiFile = 'LMWidget.ui'
    specs = PARAMETERS

    #: Emitted with the name and value of a control that the user changed.
    propertyChanged = pyqtSignal(str, float)

    def __init__(self, *args, session: LMSession | None = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        uic.loadUi(_DIR / self.uiFile, self)
        self.session = session if session is not None else LMSession(self.cls)
        self._updating = False
        self._setupControls()
        self.session.setShown([c.objectName() for c in self.controls])
        self._startSession()
        self._connectSignals()

    @property
    def model(self) -> LorenzMie:
        '''The model of the session.'''
        return self.session.model

    def _setupControls(self) -> None:
        """Configure the controls that the layout has, from the specs."""
        self.setConfig({spec.name: spec.settings() for spec in self.specs})

    def _startSession(self) -> None:
        '''Give the session the values and the fixed flags of the controls.'''
        self.session.update({c.objectName(): c.value()
                             for c in self.controls})
        for control in self.controls:
            self.session.setFixed(control.objectName(), control.fixed())

    def _connectSignals(self) -> None:
        for control in self.controls:
            name = control.objectName()
            control.valueChanged.connect(
                lambda v, n=name: self._handleChange(n, v))
            control.fixedChanged.connect(
                lambda f, n=name: self._handleFixed(n, f))
        self.session.parametersChanged.connect(self._showValues)
        self.session.fixedChanged.connect(self._showFixed)

    def _handleChange(self, name: str, value: float) -> None:
        if self._updating:
            return
        self.session.set(name, value)
        self.propertyChanged.emit(name, value)

    def _handleFixed(self, name: str, fixed: bool) -> None:
        if not self._updating:
            self.session.setFixed(name, fixed)

    def _showValues(self, names: list[str]) -> None:
        '''Show values of the session in the controls.'''
        self._updating = True
        try:
            shown = {c.objectName(): c for c in self.controls}
            for name in names:
                if name in shown:
                    shown[name].setValue(self.session.get(name))
        finally:
            self._updating = False

    def _showFixed(self, fixed: list[str]) -> None:
        self._updating = True
        try:
            for control in self.controls:
                control.setFixed(control.objectName() in fixed)
        finally:
            self._updating = False

    @pyqtProperty(dict)
    def properties(self) -> LorenzMie.Properties:
        '''Values of the parameters that have controls.'''
        return {control.objectName(): self.session.get(control.objectName())
                for control in self.controls}

    @properties.setter
    def properties(self, properties: LorenzMie.Properties) -> None:
        self.session.update(properties)

    @pyqtProperty(list)
    def fixed(self) -> list[str]:
        '''Names of the properties that a fit must hold constant.

        These are the controls that are checked, and every property of the
        model that has no control: a parameter that the user cannot see
        (for example, the aberration coefficients that the panel does not
        show) must not be fitted.
        '''
        return self.session.fixed

    @fixed.setter
    def fixed(self, fixed: list[str]) -> None:
        for control in self.controls:
            self.session.setFixed(control.objectName(),
                                  control.objectName() in fixed)

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
