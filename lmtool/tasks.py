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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
import importlib.util
import logging
import threading
from typing import Any, cast

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, QThread, Signal, Slot

from pylorenzmie.analysis import DEEstimator, Localizer, Mask, Optimizer
from pylorenzmie.analysis.DEEstimator import default_bounds
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.lib.lmtypes import Coordinates, Image, Properties, Property
from pylorenzmie.theory import LorenzMie

logger = logging.getLogger(__name__)


def catch_available() -> bool:
    '''True if CATCH is installed.

    CATCH needs PyTorch, which takes seconds to import, so this only looks
    for the package: nothing is imported until a task asks for CATCH.
    '''
    spec = importlib.util.find_spec('CATCH')
    # a directory of that name without an __init__.py (a checkout of the
    # project, say) is found as a namespace package, which has no origin
    return spec is not None and spec.origin is not None


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
    catch : bool, optional
        Estimate with CATCH rather than with differential evolution.
        Default: False.
    saturated : numpy.ndarray of bool, optional
        Saturated pixels of the region, shape (ny, nx), which estimates and
        fits leave out.  Default: none.
    '''

    cls: type[LorenzMie]
    properties: Properties
    data: np.ndarray
    coordinates: np.ndarray
    fixed: tuple[str, ...] = ()
    settings: dict[str, Any] | None = None
    fraction: float = 0.25
    catch: bool = False
    saturated: np.ndarray | None = None

    @classmethod
    def take(cls, model_cls: type[LorenzMie],
             properties: Mapping[str, Property],
             data: Image,
             coordinates: Coordinates,
             fixed: list[str] | None = None,
             settings: dict[str, Any] | None = None,
             fraction: float = 0.25,
             catch: bool = False,
             saturated: np.ndarray | None = None) -> Snapshot:
        '''A snapshot with its own copy of every array and dictionary.'''
        return cls(model_cls,
                   dict(properties),
                   np.array(data, copy=True),
                   np.array(coordinates, copy=True),
                   tuple(fixed or ()),
                   None if settings is None else dict(settings),
                   fraction,
                   catch,
                   None if saturated is None else np.array(saturated,
                                                           copy=True))


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
    catch : bool, optional
        Look with CATCH rather than with the circle transform.
        Default: False.
    instrument : dict, optional
        ``wavelength``, ``magnification`` and ``n_m`` of the microscope,
        which CATCH needs.  Default: the instrument of CATCH.
    '''

    image: np.ndarray
    maxfeatures: int = 10
    catch: bool = False
    instrument: Mapping[str, float] | None = None


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

    properties: dict[str, float]
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
    return Hologram.from_coordinates(snapshot.data,
                                     snapshot.coordinates,
                                     snapshot.saturated)


def run_fit(snapshot: Snapshot, cancel: threading.Event) -> FitResult:
    '''Fit the model to the hologram of a snapshot.

    Raises
    ------
    Cancelled
        If ``cancel`` is set before the fit is done.
    '''
    model = _model(snapshot, cancel)
    hologram = _hologram(snapshot)
    mask = Mask(fraction=snapshot.fraction)
    mask.exclude = hologram.excluded
    optimizer = Optimizer(model=model,
                          mask=mask,
                          fixed=list(snapshot.fixed),
                          settings=snapshot.settings)
    result = optimizer.optimize(hologram)
    properties = cast(dict[str, float], dict(model.properties))
    return FitResult(properties, result, optimizer.metadata)


def run_estimate(snapshot: Snapshot, cancel: threading.Event) -> pd.Series:
    '''Estimate the height, radius and index of the particle of a snapshot.

    A parameter that the snapshot holds fixed keeps its value: the search
    covers only the others, which are estimated for that value.

    Raises
    ------
    Cancelled
        If ``cancel`` is set before the search is done.
    '''
    if snapshot.catch:
        return _catch_estimate(snapshot, cancel)
    model = _model(snapshot, cancel)
    defaults = default_bounds(model.instrument.magnification)
    bounds = {name: limits for name, limits in defaults.items()
              if name not in snapshot.fixed}
    if not bounds:
        return pd.Series(model.single().properties)
    hologram = _hologram(snapshot)
    estimator = DEEstimator(model=model,
                            bounds=bounds,
                            exclude=hologram.excluded)
    return estimator.estimate(hologram)


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
    if request.catch:
        return _catch_localize(request, cancel)
    return Localizer().localize(request.image,
                                maxfeatures=request.maxfeatures,
                                check=check)


_catch_lock = threading.Lock()
_catch_app: Any = None

#: How far outside the range of pixel sizes that CATCH was trained for an
#: instrument may be, as a fraction, before lmtool warns.
CATCH_MARGIN = 0.05


def _catch_warning(app: Any, network: Any) -> str:
    '''Why a network of CATCH may not suit the instrument, if it may not.

    A network was trained for pixels of a range of sizes, measured in
    wavelengths in the medium, relative to those of its reference microscope.

    Parameters
    ----------
    app : CATCH.CATCHapp.CATCHapp
        With the instrument of the model.
    network : CATCH.Localizer.Localizer or CATCH.Estimator.Estimator
        The network that is used, with its range in ``scales``.

    Returns
    -------
    warning : str
        Empty if the instrument is within the range, or close to it.
    '''
    low, high = network.scales
    if low / (1. + CATCH_MARGIN) <= app.scale <= high * (1. + CATCH_MARGIN):
        return ''
    return (f'CATCH is meant for pixels {low:g} to {high:g} times those of '
            f'its reference microscope, and these are {app.scale:.2g} times: '
            'check its results')


def _catch(instrument: Mapping[str, Property] | None) -> Any:
    '''The :class:`CATCH.CATCHapp.CATCHapp` for an instrument.

    The networks load once, on the first call, and stay.  The Estimator is
    not compiled: compiling takes longer than the single estimate that
    lmtool asks for.
    '''
    global _catch_app
    with _catch_lock:
        if _catch_app is None:
            # CATCH is optional, and CI does not install it
            from CATCH import CATCHapp  # pyright: ignore
            _catch_app = CATCHapp(compile=False)
        if instrument is not None:
            _catch_app.instrument = {
                name: float(cast(float, instrument[name]))
                for name in ('wavelength', 'magnification', 'n_m')}
        return _catch_app


def _catch_localize(request: LocalizeRequest,
                    cancel: threading.Event) -> pd.DataFrame:
    '''Find the particles in an image with CATCH.

    The result has the columns of the output of
    :meth:`~pylorenzmie.analysis.Localizer.Localizer.localize` that lmtool
    uses: ``x_p``, ``y_p`` and ``bbox``, the strongest candidate first.
    '''
    app = _catch(request.instrument)
    found = app(request.image)
    if cancel.is_set():
        raise Cancelled
    if len(found) == 0:
        found = pd.DataFrame(columns=['x_p', 'y_p', 'bbox'])
        nfound = 0
    else:
        nfound = len(found)
        found = found.sort_values('confidence', ascending=False)
        found = found.head(request.maxfeatures).reset_index(drop=True)
    found.attrs['nfound'] = nfound
    found.attrs['warning'] = _catch_warning(app, app.localizer)
    return found


def _catch_estimate(snapshot: Snapshot,
                    cancel: threading.Event) -> pd.Series:
    '''Estimate the height, radius and index of a particle with CATCH.

    CATCH measures the profile of the hologram around the position of the
    particle, which is in the region of the snapshot.
    '''
    app = _catch(snapshot.properties)
    if cancel.is_set():
        raise Cancelled
    x_0 = float(snapshot.coordinates[0].min())
    y_0 = float(snapshot.coordinates[1].min())
    x_p = float(cast(float, snapshot.properties['x_p']))
    y_p = float(cast(float, snapshot.properties['y_p']))
    estimates = app.estimator(snapshot.data,
                              [x_p - x_0],
                              [y_p - y_0],
                              app.instrument)
    if len(estimates) == 0:
        raise ValueError('CATCH has no estimate for this particle')
    result = pd.Series({name: float(estimates[name].iloc[0])
                        for name in ('z_p', 'a_p', 'n_p')})
    result.attrs['warning'] = _catch_warning(app, app.estimator)
    return result


class _Worker(QObject):
    '''Runs one task in a thread and reports how it ended.'''

    finished = Signal(object)
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, function: Callable[..., Any],
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
              function: Callable[..., Any],
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
