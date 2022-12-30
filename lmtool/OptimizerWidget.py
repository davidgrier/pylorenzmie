from pyqtgraph.parametertree import (Parameter, ParameterTree)
from PyQt5.QtCore import (pyqtProperty, pyqtSlot, pyqtSignal)
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OptimizerWidget(ParameterTree):

    settingChanged = pyqtSignal(str, object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buildTree()
        self._connectSignals()

    def _buildTree(self):
        p = [
            {'name': 'fraction', 'type': 'float',
             'value': 0.5, 'limits': (0, 1), 'step': 0.01,
             'tip': 'fraction of pixels to fit'},
            {'name': 'method', 'type': 'list',
             'values': {'Levenberg-Marquardt': 'lm',
                        'Dogbox': 'dogbox',
                        'Trust Region Reflective': 'trf'},
             'default': 'lm'},
            {'name': 'loss', 'type': 'list',
             'values': 'linear soft_l1 huber cauchy arctan'.split(),
             'default': 'linear'},
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

    def _connectSignals(self):
        p = self.params.child('method')
        p.sigValueChanged.connect(self._updateLoss)
        for p in self.params:
            p.sigValueChanged.connect(self._handleValueChanged)

    @pyqtSlot(object, object)
    def _updateLoss(self, widget, value):
        widget = self.params.child('loss')
        logger.debug('updating loss')
        if value == 'lm':
            widget.setValue('linear')
            widget.setOpts(enabled=False)
            logger.debug(f'emitting {widget.name()} {value}')
            self.settingChanged.emit('loss', 'linear')
        else:
            widget.setOpts(enabled=True)

    @pyqtSlot(object, object)
    def _handleValueChanged(self, widget, value):
        logger.debug(f'emitting {widget.name()} {value}')
        self.settingChanged.emit(widget.name(), value)

    @pyqtProperty(dict)
    def settings(self):
        return {p.name(): p.value() for p in self.params}

    @settings.setter
    def settings(self, settings):
        for name, value in settings.items():
            try:
                widget = self.params.child(name)
                widget.setValue(value)
            except KeyError:
                logger.warn(f'{name} unknown')


def example():
    from pyqtgraph import mkQApp

    app = mkQApp()
    widget = OptimizerWidget()
    widget.show()
    widget.settings = {'ftol': 1e-3, 'method': 'dogbox'}
    app.exec_()


if __name__ == '__main__':
    example()
