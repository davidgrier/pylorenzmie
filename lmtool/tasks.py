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
from enum import StrEnum
import logging
import threading
from typing import Any

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, QThread, Signal, Slot

from pylorenzmie.analysis import DEEstimator, Localizer, Mask, Optimizer
from pylorenzmie.analysis.DEEstimator import default_bounds
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.lib.lmtypes import Coordinates, Image
from pylorenzmie.theory import LorenzMie

logger = logging.getLogger(__name__)


class Task(StrEnum):
    '''The tasks that the program runs in the background.

    A member is the name that :class:`TaskRunner` reports with its signals.
    '''

    LOCALIZE = 'localize'
    ESTIMATE = 'estimate'
    OPTIMIZE = 'optimize'


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
    def take(cls, model_cls: type,
             properties: dict,
             data: Image,
             coordinates: Coordinates,
             fixed: list[str] | None = None,
             settings: dict | None = None,
             fraction: float = 0.25) -> Snapshot:
        '''A snapshot with its own copy of every array and dictionary.'''
        return cls(model_cls,
                   dict(properties),
                   np.array(data, copy=True),
                   np.array(coordinates, copy=True),
                   tuple(fixed or ()),
                   None if settings is None else dict(settings),
                   fraction)


@dataclass(frozen=True)
class LocalizeRequest:
    '''An image to look for particles in.

    Parameters
    ----------
    image : numpy.ndarray
        Normalized hologram of the whole frame.  It is only read, so a
        request does not copy it.
    maxfeatures : int, optional
        Number of candidates to keep, the strongest first.  Default: 10.
    '''

    image: np.ndarray
    maxfeatures: int = 10


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


def _model(snapshot: Snapshot, cancel: threading.Event) -> LorenzMie:
    '''A model for the snapshot that stops when ``cancel`` is set.

    The model is of a subclass of the class of the snapshot that looks at
    ``cancel`` before each evaluation of the hologram or of its Jacobian,
    which is where a fit or a search spends its time.
    '''
    class Cancellable(snapshot.cls):
        def hologram(self, *args: Any, **kwargs: Any) -> Any:
            if cancel.is_set():
                raise Cancelled
            return super().hologram(*args, **kwargs)

        def jac(self) -> Any:
            if cancel.is_set():
                raise Cancelled
            return super().jac()

    model = Cancellable()
    model.properties = snapshot.properties
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
    # a pixel at the maximum is taken to be saturated
    mask = Mask(fraction=snapshot.fraction)
    mask.exclude = snapshot.data == np.max(snapshot.data)
    optimizer = Optimizer(model=model,
                          mask=mask,
                          fixed=list(snapshot.fixed),
                          settings=snapshot.settings)
    result = optimizer.optimize(_hologram(snapshot))
    return FitResult(dict(model.properties), result, optimizer.metadata)


def run_estimate(snapshot: Snapshot, cancel: threading.Event) -> pd.Series:
    '''Estimate the height, radius and index of the particle of a snapshot.

    A parameter that the snapshot holds fixed keeps its value: the search
    covers only the others, which are estimated for that value.

    Raises
    ------
    Cancelled
        If ``cancel`` is set before the search is done.
    '''
    model = _model(snapshot, cancel)
    defaults = default_bounds(model.instrument.magnification)
    bounds = {name: limits for name, limits in defaults.items()
              if name not in snapshot.fixed}
    if not bounds:
        return pd.Series(model.single().properties)
    return DEEstimator(model=model, bounds=bounds).estimate(
        _hologram(snapshot))


def run_localize(request: LocalizeRequest,
                 cancel: threading.Event) -> pd.DataFrame:
    '''Find the particles in an image.

    Returns
    -------
    candidates : pandas.DataFrame
        The output of
        :meth:`~pylorenzmie.analysis.Localizer.Localizer.localize`.

    Raises
    ------
    Cancelled
        If ``cancel`` is set before the search is done.
    '''
    def check() -> None:
        if cancel.is_set():
            raise Cancelled

    check()
    return Localizer().localize(request.image,
                                maxfeatures=request.maxfeatures,
                                check=check)


class _Worker(QObject):
    '''Runs one task in a thread and reports how it ended.'''

    finished = Signal(object)
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, function: Callable,
                 snapshot: Snapshot | LocalizeRequest,
                 cancel: threading.Event) -> None:
        super().__init__()
        self._function = function
        self._snapshot = snapshot
        self._cancel = cancel

    @Slot()
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
    started = Signal(str)
    #: Emitted with the name of the task and its result.
    finished = Signal(str, object)
    #: Emitted with the name of the task and the reason when it fails.
    failed = Signal(str, str)
    #: Emitted with the name of the task when it stops because it was asked to.
    cancelled = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._name = ''
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._event = threading.Event()
        # A thread and its worker leave together, once the thread is done
        # (see _release).
        self._retired: list[tuple[QThread, _Worker]] = []

    @property
    def name(self) -> str:
        '''Name of the task that is running, or an empty string.'''
        return self._name if self.busy else ''

    @property
    def busy(self) -> bool:
        '''True while a task is running.'''
        return self._thread is not None

    def start(self,
              name: str,
              function: Callable,
              snapshot: Snapshot | LocalizeRequest) -> bool:
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
        threads = [thread for thread, _ in self._retired]
        if self._thread is not None:
            threads.append(self._thread)
        for thread in threads:
            thread.quit()
            thread.wait()

    def _release(self) -> str:
        '''Retire the thread and its worker; return the name of the task.

        The worker lives in its thread, which is still running when the task
        is done (it was only asked to quit).  A worker that Python lets go of
        is destroyed by the event loop of its own thread, and the destructor
        calls into Python for the GIL while it holds the lock that Qt keeps
        for its signals; a slot that the main thread runs at that moment holds
        the GIL and waits for that lock.  Neither goes on.  The worker
        therefore stays with its thread, and both go when the thread has
        finished (:meth:`_onThreadFinished`).
        '''
        name = self._name
        if self._thread is not None and self._worker is not None:
            self._retired.append((self._thread, self._worker))
        self._thread = self._worker = None
        return name

    @Slot(object)
    def _onFinished(self, result: object) -> None:
        self.finished.emit(self._release(), result)

    @Slot()
    def _onCancelled(self) -> None:
        self.cancelled.emit(self._release())

    @Slot(str)
    def _onError(self, message: str) -> None:
        self.failed.emit(self._release(), message)

    @Slot()
    def _onThreadFinished(self) -> None:
        self._retired = [pair for pair in self._retired
                         if pair[0].isRunning()]
