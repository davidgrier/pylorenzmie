'''The parameters of a fit in lmtool: one place that holds them.

An :class:`LMSession` owns the model that lmtool displays, estimates and
fits, the value of every one of its parameters and which of them a fit
holds fixed.  The widgets of the application are views of it: a widget that
the user edits calls :meth:`~LMSession.set`, and every widget and plot that
shows a parameter listens to :attr:`~LMSession.parametersChanged`.  Nothing
else changes a parameter, so there is one answer to the question of what the
value of a parameter is, and a parameter that has no widget is still part of
the session, fixed.
'''

import logging

import numpy as np
from pyqtgraph.Qt.QtCore import QObject, pyqtSignal

from pylorenzmie.theory import LorenzMie

logger = logging.getLogger(__name__)


class LMSession(QObject):
    '''Parameters and fixed flags for the model of an lmtool session.

    Parameters
    ----------
    cls : type, optional
        Class of the model, a :class:`~pylorenzmie.theory.LorenzMie.LorenzMie`
        calculator.  Default: :class:`~pylorenzmie.theory.LorenzMie.LorenzMie`.

    Notes
    -----
    A parameter has a value and a fixed flag.  A value is announced once,
    in one :attr:`parametersChanged`, however many parameters change
    together; setting a value that is already the value, to within
    ``tolerance``, announces nothing, which also ends any loop between a
    widget and the session.

    The session keeps the values that it has announced.  A background fit
    changes the model without announcing it; :meth:`adopt` takes the new
    values from the model and announces what changed, and :meth:`restore`
    puts the announced values back.
    '''

    #: Emitted with the names of the parameters whose values changed.
    parametersChanged = pyqtSignal(list)
    #: Emitted with the names of the fixed parameters when they change.
    fixedChanged = pyqtSignal(list)

    #: Values closer than this are the same value.
    tolerance = 1e-9

    def __init__(self, cls: type = LorenzMie) -> None:
        super().__init__()
        self.cls = cls
        self.model = cls()
        self._values = dict(self.model.properties)
        self._fixed: set[str] = set()
        self._shown: set[str] | None = None

    # ------------------------------------------------------------------
    # values
    # ------------------------------------------------------------------

    @property
    def properties(self) -> dict[str, float]:
        '''Value of every parameter of the model (a copy).'''
        return dict(self._values)

    def get(self, name: str) -> float:
        '''Value of the parameter ``name``.'''
        return self._values[name]

    def set(self, name: str, value: float) -> None:
        '''Set one parameter; see :meth:`update`.'''
        self.update({name: value})

    def update(self, properties: dict[str, float]) -> list[str]:
        '''Set parameters and announce the ones that changed, once.

        The model receives the new values.  Names that the model does not
        have are ignored and logged.

        Returns
        -------
        changed : list[str]
            Names of the parameters whose values changed.
        '''
        changed = {}
        for name, value in properties.items():
            if name not in self._values:
                logger.debug(f'{name!r} is not a parameter of the model')
                continue
            if not self._same(self._values[name], value):
                changed[name] = float(value)
        if changed:
            self._values.update(changed)
            self.model.properties = changed
            self.parametersChanged.emit(list(changed))
        return list(changed)

    def adopt(self) -> list[str]:
        '''Take the values from the model and announce what changed.

        A fit or an estimate changes the model directly; this brings the
        session, and so the widgets, up to date.
        '''
        current = self.model.properties
        changed = [name for name, value in current.items()
                   if name in self._values
                   and not self._same(self._values[name], value)]
        if changed:
            self._values.update({name: float(current[name])
                                 for name in changed})
            self.parametersChanged.emit(changed)
        return changed

    def restore(self) -> None:
        '''Give the model the values that the session holds.

        A fit or an estimate that does not finish, or whose result is not
        wanted, leaves the model in a state that the session never announced.
        '''
        self.model.properties = dict(self._values)

    def _same(self, a: float, b: float) -> bool:
        return bool(np.isclose(a, b, rtol=0., atol=self.tolerance))

    # ------------------------------------------------------------------
    # fixed parameters
    # ------------------------------------------------------------------

    def setShown(self, names: list[str]) -> None:
        '''Say which parameters the user can see and change.

        Every other parameter of the model is hidden, and a fit holds it
        fixed: it must not move where the user cannot see it.
        '''
        self._shown = set(names)
        self.fixedChanged.emit(self.fixed)

    @property
    def hidden(self) -> list[str]:
        '''Names of the parameters that the user cannot see.'''
        if self._shown is None:
            return []
        return [name for name in self._values if name not in self._shown]

    @property
    def fixed(self) -> list[str]:
        '''Names of the parameters that a fit holds fixed.

        These are the ones that the user fixed and the hidden ones.
        '''
        return [name for name in self._values
                if name in self._fixed or name in self.hidden]

    def isFixed(self, name: str) -> bool:
        '''True if a fit holds the parameter ``name`` fixed.'''
        return name in self._fixed or name in self.hidden

    def setFixed(self, name: str, fixed: bool = True) -> None:
        '''Fix (or free) the parameter ``name``.'''
        if (name in self._fixed) == bool(fixed):
            return
        if fixed:
            self._fixed.add(name)
        else:
            self._fixed.discard(name)
        self.fixedChanged.emit(self.fixed)
