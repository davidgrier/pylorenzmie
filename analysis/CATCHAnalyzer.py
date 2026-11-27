'''Detection and estimation with the neural networks of CATCH.

`CATCH <https://github.com/laltman2/CATCH>`_ finds the holograms of
particles in an image and estimates each particle's height, radius and
refractive index.  A :class:`CATCHAnalyzer` gives its results in the forms
that this package uses: it is a localizer of a
:class:`~pylorenzmie.analysis.Frame.Frame`, which then starts each fit from
its estimates, and an estimator of a single
:class:`~pylorenzmie.analysis.Feature.Feature`::

    catch = CATCHAnalyzer(instrument)
    frame = Frame(instrument=instrument, localizer=catch)
    results = frame.analyze(image)

CATCH is optional.  It needs PyTorch, is installed from its repository
(``pip install git+https://github.com/laltman2/CATCH.git``; the package
``CATCH`` of PyPI is something else), and is imported only when a
:class:`CATCHAnalyzer` is first used.  Its networks load once and are shared.
'''

import importlib.util
import threading
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pylorenzmie.analysis.BaseEstimator import BaseEstimator
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.lmtypes import Image, Properties
from pylorenzmie.theory import Instrument


#: How far outside the range of pixel sizes that CATCH was trained for an
#: instrument may be, as a fraction, before a warning is given.
CATCH_MARGIN = 0.05

#: The columns that a table of detections has even if it is empty.
COLUMNS = ('x_p', 'y_p', 'bbox', 'z_p', 'a_p', 'n_p', 'confidence',
           'quality')

_OPTICS = ('wavelength', 'magnification', 'n_m')

# the instrument of the shared application is set for each call
_lock = threading.RLock()
_apps: dict[bool, Any] = {}


def catch_available() -> bool:
    '''True if CATCH is installed.

    CATCH needs PyTorch, which takes seconds to import, so this only looks
    for the package: nothing is imported.
    '''
    spec = importlib.util.find_spec('CATCH')
    # a directory of that name without an __init__.py (a checkout of the
    # project, say) is found as a namespace package, which has no origin
    return spec is not None and spec.origin is not None


def _app(compiled: bool) -> Any:
    '''The :class:`CATCH.CATCHapp.CATCHapp`, loaded on the first call.'''
    if compiled not in _apps:
        # CATCH is optional, and CI does not install it
        from CATCH import CATCHapp  # pyright: ignore
        _apps[compiled] = CATCHapp(compile=compiled)
    return _apps[compiled]


def _warning(app: Any, network: Any) -> str:
    '''Why a network of CATCH may not suit the instrument, if it may not.

    A network was trained for pixels of a range of sizes, measured in
    wavelengths in the medium, relative to those of its reference microscope.
    '''
    low, high = network.scales
    if low / (1. + CATCH_MARGIN) <= app.scale <= high * (1. + CATCH_MARGIN):
        return ''
    return (f'CATCH is meant for pixels {low:g} to {high:g} times those of '
            f'its reference microscope, and these are {app.scale:.2g} times: '
            'check its results')


@dataclass
class CATCHAnalyzer(BaseEstimator):
    '''Find and estimate particles with CATCH.

    Parameters
    ----------
    instrument : Instrument, optional
        Optics of the microscope: its wavelength [um], magnification
        [um/pixel] and medium index are given to CATCH for each image.  It
        is the object that a frame and its fits should share.
        Default: ``Instrument()``.
    compile : bool, optional
        Compile the inner loops of CATCH's estimator with
        ``torch.compile``, which pays off over many images.
        Default: False.

    Attributes
    ----------
    saturated : numpy.ndarray of bool or None
        Saturated pixels of the image that :meth:`localize` is given, which
        CATCH leaves out.  A :class:`~pylorenzmie.analysis.Frame.Frame`
        sets it.

    Notes
    -----
    CATCH leaves out pixels that are not finite.  Images must be normalized
    (background 1).  The networks were trained for a range of pixel sizes
    measured in wavelengths; outside it the results carry a warning in
    their ``attrs['warning']``.

    As an estimator (:meth:`estimate`) one analyzer serves one feature, as
    every estimator does.
    '''

    instrument: Instrument = field(default_factory=Instrument)
    compile: bool = False

    def __post_init__(self) -> None:
        self.saturated: NDArray[np.bool_] | None = None

    @LMObject.properties.getter
    def properties(self) -> Properties:
        '''Configuration of the analyzer.'''
        return dict(compile=self.compile)

    def _configured(self) -> Any:
        '''The application of CATCH, for the instrument of this analyzer.'''
        app = _app(self.compile)
        optics = self.instrument.properties
        app.instrument = {name: float(cast(float, optics[name]))
                          for name in _OPTICS}
        return app

    def localize(self, image: Image) -> pd.DataFrame:
        '''Find the particles in an image and estimate them.

        Parameters
        ----------
        image : numpy.ndarray
            Normalized hologram.

        Returns
        -------
        detections : pandas.DataFrame
            One row for each particle, with the columns that CATCH reports
            (see ``CATCHapp.analyze`` of CATCH), among them
            ``x_p``, ``y_p`` [pixels], ``bbox`` (``((x0, y0), width,
            height)``, which may cross the edge of the image), the estimates
            ``z_p`` [pixels], ``a_p`` [um] and ``n_p``, the ``confidence``
            of the detection and its ``quality`` (``''``, or why not to
            trust the particle).  ``attrs['nfound']`` is the number of
            particles and ``attrs['warning']`` a warning about the
            instrument, or ``''``.
        '''
        with _lock:
            app = self._configured()
            found = app.analyze(np.asarray(image),
                                saturation=self.saturated)
            warning = _warning(app, app.localizer)
        if len(found) == 0:
            found = pd.DataFrame(columns=list(COLUMNS))
        found.attrs['nfound'] = len(found)
        found.attrs['warning'] = warning
        return found

    def estimate(self, hologram: Hologram) -> pd.Series:
        '''Estimate the particle at the center of a crop.

        Parameters
        ----------
        hologram : Hologram
            Normalized hologram crop.  The particle is at
            :attr:`~pylorenzmie.analysis.BaseEstimator.BaseEstimator.center`,
            refined by
            :meth:`~pylorenzmie.analysis.BaseEstimator.BaseEstimator.find_center`,
            or without it at the middle of the crop.

        Returns
        -------
        result : pandas.Series
            ``x_p``, ``y_p``, ``z_p`` [pixels], ``a_p`` [um] and ``n_p``;
            ``attrs['warning']`` as for :meth:`localize`.

        Raises
        ------
        ValueError
            If CATCH has no estimate for the particle.
        '''
        x_p, y_p = self.find_center(hologram)
        x_0, y_0 = hologram.corner
        exclude = (hologram.excluded if self.exclude is None
                   else self.exclude)
        with _lock:
            app = self._configured()
            found = app.estimator(hologram.data,
                                  [x_p - x_0],
                                  [y_p - y_0],
                                  app.instrument,
                                  saturation=exclude)
            warning = _warning(app, app.estimator)
        names = ['z_p', 'a_p', 'n_p']
        if len(found) == 0 or not np.isfinite(
                found[names].iloc[0].to_numpy(float)).all():
            raise ValueError('CATCH has no estimate for this particle')
        values = {name: float(found[name].iloc[0]) for name in names}
        result = pd.Series(dict(x_p=x_p, y_p=y_p, **values))
        result.attrs['warning'] = warning
        return result
