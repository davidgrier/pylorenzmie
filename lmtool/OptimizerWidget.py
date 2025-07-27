from pylorenzmie.lib import LMObject
from pyqtgraph.parametertree import (Parameter, ParameterTree)
from pyqtgraph.Qt.QtCore import (pyqtProperty, pyqtSlot, pyqtSignal)
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class OptimizerWidget(ParameterTree):

    settingChanged = pyqtSignal(str, object)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._buildTree()
        self._consistencyCheck()
        self._connectSignals()

    def _buildTree(self) -> None:
        p = [
            {'name': 'fraction', 'type': 'float',
             'value': 0.5, 'limits': (0, 1), 'step': 0.01,
             'tip': 'fraction of pixels to fit'},
            {'name': 'method', 'type': 'list',
             'values': {'Levenberg-Marquardt': 'lm',
                        'Dogbox': 'dogbox',
                        'Trust Region Reflective': 'trf'},
             'default': 'trf'},
            {'name': 'loss', 'type': 'list',
             'values': 'linear soft_l1 huber cauchy arctan'.split(),
             'default': 'soft_l1'},
            {'name': 'ftol', 'type': 'float',
             'value': 1e-3, 'limits': [1e-8, 1e-2], 'default': 1e-3},
            {'name': 'xtol', 'type': 'float',
             'value': 1e-6, 'limits': [1e-8, 1e-2], 'default': 1e-6},
            {'name': 'gtol', 'type': 'float',
             'value': 1e-6, 'limits': [1e-8, 1e-2], 'default': 1e-6},
            {'name': 'diff_step', 'type': 'float',
             'value': 1e-5, 'limits': [1e-8, 1e-2], 'default': 1e-5},
        ]
        self.params = Parameter.create(name='params',
                                       type='group',
                                       children=p)
        self.setParameters(self.params, showTop=False)

    def _consistencyCheck(self) -> None:
        if self.params.child('method').value() == 'lm':
            widget = self.params.child('loss')
            widget.setValue('linear')
            widget.setOpts(enabled=False)
            self.settingChanged.emit('loss', 'linear')
        else:
            self.params.child('loss').setOpts(enabled=True)

    def _connectSignals(self) -> None:
        p = self.params.child('method')
        p.sigValueChanged.connect(self._updateLoss)
        for p in self.params:
            p.sigValueChanged.connect(self._handleValueChanged)

    @pyqtSlot(object, object)
    def _updateLoss(self, widget: 'QWidget', value: LMObject.Property):
        self._consistencyCheck()

    @pyqtSlot(object, object)
    def _handleValueChanged(self, widget: 'QWidget',
                            value: LMObject.Property):
        logger.debug(f'emitting {widget.name()} {value}')
        self.settingChanged.emit(widget.name(), value)

    @pyqtProperty(dict)
    def settings(self) -> dict[str, LMObject.Property]:
        return {p.name(): p.value() for p in self.params}

    @settings.setter
    def settings(self, settings: dict[str, LMObject.Property]) -> None:
        for name, value in settings.items():
            try:
                widget = self.params.child(name)
                widget.setValue(value)
            except KeyError:
                logger.debug(f'{name} unknown')

    @classmethod
    def example(cls) -> None:
        from pyqtgraph import mkQApp

        app = mkQApp()
        widget = cls()
        widget.show()
        widget.settings = {'ftol': 1e-3, 'method': 'dogbox'}
        app.exec()


if __name__ == '__main__':
    OptimizerWidget.example()
