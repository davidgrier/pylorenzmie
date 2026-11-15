'''Global search for the starting values of a fit.

A :class:`DEEstimator` searches for the height, radius and refractive index of
a particle with differential evolution
(:func:`scipy.optimize.differential_evolution`), minimizing the squared
difference between the model and a random subset of the pixels of the
hologram.  The search needs no starting values, only bounds, and it is the
default estimator of :class:`~pylorenzmie.analysis.Feature.Feature`.

The position in the plane is not searched: it is taken as the center of the
hologram.  The bounds on the three parameters are wide by default, so a
search can settle on an unphysical solution for a feature that several
different particles could explain; narrower bounds that reflect what is known
about the sample, and a ``seed`` for reproducibility, help.  The search runs
in one process by default; ``settings['workers'] = -1`` distributes it over the
cores of the processor, which pays off for slow calculators and large
holograms.
'''

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import differential_evolution

from pylorenzmie.lib.lmtypes import Bounds, Coordinates, Image, Properties
from pylorenzmie.analysis.BaseEstimator import BaseEstimator
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.analysis.Mask import Mask
from pylorenzmie.theory import LorenzMie


def default_bounds(magnification: float) -> Bounds:
    '''The search ranges of an estimator that has no bounds of its own.

    The height of a particle is measured in pixels, like its position in the
    plane, but the range that a search should cover is set by the sample,
    not by the camera: a fixed range of pixels stops at a different height
    in the sample at each magnification.  The default is therefore 1 to
    200 μm, converted to pixels with the magnification of the instrument.

    Parameters
    ----------
    magnification : float
        Size of a pixel in the sample (μm/pixel).

    Returns
    -------
    bounds : dict
        Mapping of parameter name to ``(min, max)`` search range: ``z_p``
        in pixels, ``a_p`` in μm and ``n_p``.
    '''
    return {
        'z_p': (1. / magnification, 200. / magnification),
        'a_p': (0.25, 10.0),
        'n_p': (1.0, 3.0),
    }


class _DEObjective:
    '''Picklable objective function for differential evolution.

    The sum of the squared residuals, in units of the noise, between the
    model and *data*: the intensity at the coordinates of the model, whether
    those are some of the pixels of a hologram or the points of its azimuthal
    profile.

    Using a top-level callable class rather than a closure allows
    ``differential_evolution(workers=-1)`` to distribute candidate
    evaluations across CPU cores via ``multiprocessing``.
    '''

    def __init__(self,
                 model: LorenzMie,
                 data: Image,
                 variables: list[str],
                 noise: float) -> None:
        self._model = model
        self._data = data
        self._variables = variables
        self._noise = noise

    def __call__(self, values: np.ndarray) -> float:
        self._model.properties = dict(
            zip(self._variables, values, strict=True))
        with np.errstate(over='ignore', invalid='ignore'):
            diff = (self._model.hologram() - self._data) / self._noise
            return float(np.nansum(diff ** 2))


class _DESearch:
    '''The differential-evolution search that the estimators share.

    A subclass is a dataclass with the attributes that are declared here,
    and calls :meth:`_search` with what it compares the model with.
    '''

    model: LorenzMie
    bounds: Bounds | None
    popsize: int
    seed: int | None
    settings: dict

    def _bounds(self) -> Bounds:
        '''The bounds of the search: the default if there are none.'''
        if self.bounds is not None:
            return self.bounds
        return default_bounds(self.model.instrument.magnification)

    def _search(self,
                data: Image,
                coordinates: Coordinates,
                bounds: Bounds) -> None:
        '''Set the model to the best fit of *data* at *coordinates*.

        Parameters
        ----------
        data : numpy.ndarray
            The intensity at *coordinates*.
        coordinates : numpy.ndarray
            Where the model is evaluated.  The coordinates of the model are
            replaced by them during the search, and restored on exit, even
            if an exception occurs.
        bounds : dict
            Search range ``(min, max)`` of each parameter that is varied, by
            the name of the property, in the units of the property (pixels
            for the height).

        Raises
        ------
        ValueError
            If ``settings['workers']`` asks for a pool of processes and the
            calculator cannot be forked (``fork_safe`` is False).
        '''
        if self.settings.get('workers', 1) != 1 and not self.model.fork_safe:
            raise ValueError(
                f'{type(self.model).__name__} holds the memory of a device, '
                'which a pool of worker processes cannot share: '
                "set settings['workers'] = 1")
        variables = list(bounds)
        noise = self.model.instrument.noise
        saved_coordinates = self.model.coordinates
        self.model.coordinates = coordinates
        objective = _DEObjective(self.model, data, variables, noise)
        try:
            with np.errstate(over='ignore', invalid='ignore'):
                result = differential_evolution(
                    objective,
                    list(bounds.values()),
                    popsize=self.popsize,
                    seed=self.seed,  # pyright: ignore[reportCallIssue]
                    **self.settings)
        finally:
            self.model.coordinates = saved_coordinates
        self.model.properties = dict(zip(variables, result.x, strict=True))


@dataclass
class DEEstimator(_DESearch, BaseEstimator):
    '''Estimate initial particle parameters by global search.

    Uses differential evolution (DE) to minimize the sum-squared residual
    between the forward model and a subsampled hologram crop, providing robust
    initial values for :class:`~pylorenzmie.analysis.Optimizer.Optimizer` even
    when the conventional :class:`~pylorenzmie.analysis.Estimator.Estimator`
    fails to converge.

    Inherits from :class:`~pylorenzmie.analysis.BaseEstimator.BaseEstimator`.

    Parameters
    ----------
    model : LorenzMie
        Generative scattering model shared with
        :class:`~pylorenzmie.analysis.Optimizer.Optimizer`. The particle
        parameters on this model are updated in-place.
    bounds : dict, optional
        Mapping of parameter name to ``(min, max)`` search range, in the
        units of the parameter: ``z_p`` in pixels, ``a_p`` in μm.
        Default: ``None``, which searches ``z_p`` from 1 to 200 μm (at the
        magnification of the instrument, see :func:`default_bounds`),
        ``a_p`` from 0.25 to 10 μm and ``n_p`` from 1 to 3.
    fraction : float, optional
        Fraction of pixels used for the DE objective function.
        Default: 0.02.  A fixed random subsample is drawn once per
        call and held constant throughout the search so the objective
        is deterministic.
    exclude : numpy.ndarray of bool, optional
        Boolean mask of pixels to force-exclude from sampling (e.g.
        saturated or dead pixels).  Default: ``None`` (no exclusions).
        Set via :attr:`~pylorenzmie.analysis.Feature.Feature.mask` when using
        :class:`~pylorenzmie.analysis.Feature.Feature`.
    popsize : int, optional
        DE population size multiplier (population = popsize ×
        len(bounds)).  Default: 10.
    seed : int or None, optional
        Random seed passed to :func:`scipy.optimize.differential_evolution`
        for reproducibility.  Default: ``None``.

    Notes
    -----
    ``x_p`` and ``y_p`` are pinned to the pixel-coordinate means before
    the search begins; only the parameters listed in ``bounds`` are
    varied.

    For large particles (``a_p`` ≳ 1 μm) the hologram fringe pattern
    is dense and the coarsely-sampled objective function becomes
    multi-modal.  In that regime increase ``fraction`` (e.g. 0.05)
    and ``popsize`` (e.g. 15) at the cost of longer run time, or supply
    tighter ``bounds`` to reduce the search volume.

    The model coordinates are temporarily overridden during the search
    and restored on exit, even if an exception occurs.

    By default ``settings['workers'] = 1``: the candidates are evaluated
    in one process.  With a compiled calculator
    (:class:`~pylorenzmie.theory.numbaLorenzMie.numbaLorenzMie`) the
    search takes a fraction of a second, and starting a pool of worker
    processes costs about two seconds on macOS, so a pool would make it
    several times slower.  Set ``workers=-1`` to distribute the
    candidates across all CPU cores via ``multiprocessing``, which pays
    off for the NumPy calculator on large crops (about 1.5 times faster
    for a 370-pixel crop on a 10-core Mac; on Linux, where workers are
    forked, the cost of starting them is small and the speedup is closer
    to the number of cores).  Where processes are started by importing
    the main script again (macOS and Windows), a script that uses a pool
    must be protected by ``if __name__ == '__main__':``; otherwise it
    runs inside each worker.  Notebooks and interactive sessions need no
    guard.  A pool cannot be used with ``cupyLorenzMie`` (which holds GPU
    memory and is not picklable).  The companion setting
    ``updating='deferred'`` (also the default) is required for
    ``workers != 1`` and suppresses a SciPy warning.

    Use :class:`~pylorenzmie.analysis.Estimator.Estimator` for a fast
    conventional estimate when the fringe pattern is clean.  Use
    :class:`DEEstimator` when the conventional estimator produces starting
    points too far from the true solution for
    :class:`~pylorenzmie.analysis.Optimizer.Optimizer` to converge. '''

    model: LorenzMie
    bounds: Bounds | None = None
    fraction: float = 0.02
    exclude: NDArray[np.bool_] | None = None
    popsize: int = 10
    seed: int | None = None
    settings: dict = field(default_factory=lambda: {'tol': 0.01,
                                                     'polish': False,
                                                     'updating': 'deferred',
                                                     'workers': 1})

    @BaseEstimator.properties.getter
    def properties(self) -> Properties:
        '''DEEstimator configuration.'''
        return dict(fraction=self.fraction,
                    popsize=self.popsize,
                    bounds=self.bounds,
                    settings=self.settings)

    def estimate(self, hologram: Hologram) -> pd.Series:
        '''Estimate particle parameters by differential evolution.

        Parameters
        ----------
        hologram : Hologram
            Normalized hologram crop to analyze.

        Returns
        -------
        result : pandas.Series
            Estimated particle properties (same keys as
            :attr:`~pylorenzmie.theory.Particle.Particle.properties`).
        '''
        particle = self.model.single()
        particle.x_p = float(hologram.coordinates[0].mean())
        particle.y_p = float(hologram.coordinates[1].mean())

        # seeded with the estimator, so that a seed makes the whole
        # estimate reproducible
        mask = Mask(fraction=self.fraction, seed=self.seed)
        if self.exclude is not None:
            mask.exclude = self.exclude
        de_data, de_coords = mask.apply(hologram)
        self._search(de_data, de_coords, self._bounds())
        return pd.Series(self.model.single().properties)

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        '''Run a demonstration of the class, for use as a script.'''
        from time import perf_counter
        from pylorenzmie.utilities import example_hologram

        model = LorenzMie()
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34

        estimator = cls(model=model, seed=0)

        print(f'{cls.__name__} example')
        start = perf_counter()
        result = estimator.estimate(example_hologram())
        print(f'Time: {perf_counter() - start:.3f} s')
        print(result)


if __name__ == '__main__':  # pragma: no cover
    DEEstimator.example()
