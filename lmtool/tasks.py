'''Estimates and fits that lmtool runs in the background.

A fit takes seconds and must not freeze the window, and it must not share
anything with it: while a worker thread changes the parameters of a model,
the interface thread must not read that model.  Each task therefore works on
a :class:`Snapshot`, a frozen copy of everything that it needs, with a model
of its own, and returns a result.  Nothing is shared and nothing needs a lock;
the session takes the result on the interface thread.

A task can be stopped.  :func:`run_fit` and :func:`run_estimate` look at an
event each time they evaluate the model, and raise :class:`Cancelled` when
it is set, which unwinds the optimizer or the search and ends the task.
:class:`TaskRunner` runs one task at a time in a thread and reports with
signals.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import threading

import numpy as np
import pandas as pd
from pyqtgraph.Qt.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from pylorenzmie.analysis import DEEstimator, Optimizer
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.lib.lmtypes import Coordinates, Image

logger = logging.getLogger(__name__)


class Cancelled(Exception):
    '''Raised inside a task that has been asked to stop.'''


@dataclass(frozen=True)
class Snapshot:
    '''Everything that a task needs, copied from the state of the session.

    Parameters
    ----------
    cls : type
        Class of the model.  The task makes a model of its own.
    properties : dict
        Value of every parameter of the model.
    data : numpy.ndarray
        Normalized hologram of the region of interest, shape (ny, nx).
    coordinates : numpy.ndarray
        Pixel coordinates of the region, shape (2, ny, nx).
    fixed : tuple[str, ...], optional
        Parameters that a fit holds fixed.
    settings : dict, optional
        Settings of the optimizer (see
        :attr:`~pylorenzmie.analysis.Optimizer.Optimizer.settings`).
        Default: those of the optimizer.
    fraction : float, optional
        Fraction of the pixels that a fit uses.  Default: 0.25.
    '''

    cls: type
    properties: dict
    data: np.ndarray
    coordinates: np.ndarray
    fixed: tuple[str, ...] = ()
    settings: dict | None = None
    fraction: float = 0.25

    @classmethod
    def take(cls, model_cls: type, properties: dict, data: Image,
             coordinates: Coordinates, fixed: list[str] | None = None,
             settings: dict | None = None,
             fraction: float = 0.25) -> Snapshot:
        '''A snapshot with its own copy of every array and dictionary.'''
        return cls(model_cls, dict(properties), np.array(data, copy=True),
                   np.array(coordinates, copy=True), tuple(fixed or ()),
                   None if settings is None else dict(settings), fraction)


@dataclass
class FitResult:
    '''What a fit returns.

    Attributes
    ----------
    properties : dict
        Value of every parameter of the model after the fit.
    result : pandas.Series
        Fitted values, uncertainties and statistics.
    metadata : pandas.Series
        The fixed parameters and the settings of the fit, which go with the
        result when it is saved.
    '''

    properties: dict
    result: pd.Series
    metadata: pd.Series = field(default_factory=pd.Series)


def _model(snapshot: Snapshot, cancel: threading.Event):
    '''A model for the snapshot that stops when ``cancel`` is set.'''
    model = snapshot.cls()
    model.properties = snapshot.properties
    hologram, jac = model.hologram, getattr(model, 'jac', None)

    def checked(function: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if cancel.is_set():
                raise Cancelled
            return function(*args, **kwargs)
        return wrapper

    model.hologram = checked(hologram)
    if jac is not None:
        model.jac = checked(jac)
    return model


def _hologram(snapshot: Snapshot) -> Hologram:
    return Hologram._from_slice(snapshot.data, snapshot.coordinates)


def run_fit(snapshot: Snapshot, cancel: threading.Event) -> FitResult:
    '''Fit the model to the hologram of a snapshot.

    Raises
    ------
    Cancelled
        If ``cancel`` is set before the fit is done.
    '''
    model = _model(snapshot, cancel)
    optimizer = Optimizer(model=model, fixed=list(snapshot.fixed),
                          settings=snapshot.settings)
    optimizer.fraction = snapshot.fraction
    # a pixel at the maximum is taken to be saturated
    optimizer.mask.exclude = snapshot.data == np.max(snapshot.data)
    result = optimizer.optimize(_hologram(snapshot))
    return FitResult(dict(model.properties), result, optimizer.metadata)


def run_estimate(snapshot: Snapshot, cancel: threading.Event) -> pd.Series:
    '''Estimate the height, radius and index of the particle of a snapshot.

    Raises
    ------
    Cancelled
        If ``cancel`` is set before the search is done.
    '''
    model = _model(snapshot, cancel)
    return DEEstimator(model=model).estimate(_hologram(snapshot))


class _Worker(QObject):
    '''Runs one task in a thread and reports how it ended.'''

    finished = pyqtSignal(object)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, function: Callable, snapshot: Snapshot,
                 cancel: threading.Event) -> None:
        super().__init__()
        self._function = function
        self._snapshot = snapshot
        self._cancel = cancel

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._function(self._snapshot, self._cancel)
        except Cancelled:
            self.cancelled.emit()
        except Exception as error:
            logger.exception('Background task failed')
            self.error.emit(f'{type(error).__name__}: {error}')
        else:
            self.finished.emit(result)


class TaskRunner(QObject):
    '''Runs one task at a time in a thread of its own.

    A task is a function of a :class:`Snapshot` and an event that it looks at
    to see whether it should stop (see :func:`run_fit`).  Its result arrives
    on the thread of the runner through :attr:`finished`.
    '''

    #: Emitted with the name of the task when it starts.
    started = pyqtSignal(str)
    #: Emitted with the name of the task and its result.
    finished = pyqtSignal(str, object)
    #: Emitted with the name of the task and the reason when it fails.
    failed = pyqtSignal(str, str)
    #: Emitted with the name of the task when it stops because it was asked to.
    cancelled = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._name = ''
        self._thread = None
        self._worker = None
        self._event = threading.Event()
        self._retired = []

    @property
    def busy(self) -> bool:
        '''True while a task is running.'''
        return self._thread is not None

    def start(self, name: str, function: Callable,
              snapshot: Snapshot) -> bool:
        '''Start a task; returns False, and does nothing, if one is running.'''
        if self.busy:
            return False
        self._name = name
        self._event = threading.Event()
        worker = _Worker(function, snapshot, self._event)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._onFinished)
        worker.cancelled.connect(self._onCancelled)
        worker.error.connect(self._onError)
        for signal in (worker.finished, worker.cancelled, worker.error):
            signal.connect(thread.quit)
        thread.finished.connect(self._onThreadFinished)
        self._worker, self._thread = worker, thread
        thread.start()
        self.started.emit(name)
        return True

    def cancel(self) -> None:
        '''Ask the task that is running, if any, to stop.'''
        self._event.set()

    def shutdown(self) -> None:
        '''Stop the task that is running and wait for the threads to end.'''
        self.cancel()
        for thread in [self._thread, *self._retired]:
            if thread is not None:
                thread.quit()
                thread.wait()

    def _release(self) -> str:
        name = self._name
        if self._thread is not None:
            self._retired.append(self._thread)
        self._thread = self._worker = None
        return name

    @pyqtSlot(object)
    def _onFinished(self, result: object) -> None:
        self.finished.emit(self._release(), result)

    @pyqtSlot()
    def _onCancelled(self) -> None:
        self.cancelled.emit(self._release())

    @pyqtSlot(str)
    def _onError(self, message: str) -> None:
        self.failed.emit(self._release(), message)

    @pyqtSlot()
    def _onThreadFinished(self) -> None:
        self._retired = [t for t in self._retired if t.isRunning()]
