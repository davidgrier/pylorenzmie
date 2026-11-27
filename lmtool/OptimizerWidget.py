from typing import Any
import logging

from qtpy.QtCore import Slot, Signal
from pyqtgraph.parametertree import (Parameter, ParameterTree)

from pylorenzmie.lib import LMObject


logger = logging.getLogger(__name__)


class OptimizerWidget(ParameterTree):
    '''Parameter-tree widget for configuring scipy least_squares settings.

    Emits :attr:`settingChanged` (name, value) whenever any setting changes,
    and :attr:`robustRequested` when its button *Robust fit* is pushed.
    Enforces the constraint that Levenberg-Marquardt requires
    ``loss='linear'``.
    '''

    #: Emitted with ``(name, value)`` whenever a solver setting changes.
    settingChanged = Signal(str, object)

    #: Emitted when the button that sets up a robust fit is pushed.
    robustRequested = Signal()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._buildTree()
        self._consistencyCheck()
        self._connectSignals()

    def _buildTree(self) -> None:
        p = [
            {'name': 'robust', 'type': 'action', 'title': 'Robust fit',
             'tip': 'set up a fit that discounts what the model lacks'},
            {'name': 'fraction', 'type': 'float',
             'value': 0.25, 'limits': (0.01, 1), 'step': 0.01,
             'tip': 'fraction of pixels to fit'},
            {'name': 'method', 'type': 'list',
             'limits': {'Levenberg-Marquardt': 'lm',
                        'Dogbox': 'dogbox',
                        'Trust Region Reflective': 'trf'},
             'default': 'lm'},
            {'name': 'loss', 'type': 'list',
             'limits': 'linear soft_l1 huber cauchy arctan'.split(),
             'default': 'linear'},
            {'name': 'f_scale', 'type': 'list',
             'limits': {'auto': 'auto', '1': 1., '2': 2., '4': 4.},
             # (pyqtgraph 0.13 otherwise starts at the first of the limits)
             'value': 1., 'default': 1.,
             'tip': 'residual, in units of the noise, beyond which the loss '
                    'discounts a pixel; auto matches it to the residuals'},
            {'name': 'ftol', 'type': 'float',
             'value': 1e-7, 'limits': [1e-8, 1e-2], 'default': 1e-7},
            {'name': 'xtol', 'type': 'float',
             'value': 1e-8, 'limits': [1e-8, 1e-2], 'default': 1e-8},
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
        lm = self.params.child('method').value() == 'lm'
        if lm:
            self.params.child('loss').setValue('linear')
            self.settingChanged.emit('loss', 'linear')
        # least squares has no loss to choose, and no scale of one
        for name in ('loss', 'f_scale'):
            self.params.child(name).setOpts(enabled=not lm)

    def _connectSignals(self) -> None:
        method_param = self.params.child('method')
        method_param.sigValueChanged.connect(self._updateLoss)
        self.params.child('robust').sigActivated.connect(self._requestRobust)
        for p in self.params:
            p.sigValueChanged.connect(self._handleValueChanged)

    @Slot(object)
    def _requestRobust(self, widget: Parameter) -> None:
        self.robustRequested.emit()

    @Slot(object, object)
    def _updateLoss(self,
                    widget: Parameter,
                    value: LMObject.Property) -> None:
        self._consistencyCheck()

    @Slot(object, object)
    def _handleValueChanged(self,
                            widget: Parameter,
                            value: LMObject.Property) -> None:
        logger.debug(f'emitting {widget.name()} {value}')
        self.settingChanged.emit(widget.name(), value)

    @property
    def settings(self) -> dict[str, LMObject.Property]:
        '''The solver settings that are shown, by name.  Setting it changes
        the ones that are named, and ignores the others.'''
        return {p.name(): p.value() for p in self.params
                if p.type() != 'action'}       # a button has no value

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
        '''Run a demonstration of the widget, for use as a script.'''
        from pyqtgraph import mkQApp

        app = mkQApp()
        widget = cls()
        widget.show()
        widget.settings = {'ftol': 1e-3, 'method': 'dogbox'}
        app.exec()


if __name__ == '__main__':  # pragma: no cover
    OptimizerWidget.example()
