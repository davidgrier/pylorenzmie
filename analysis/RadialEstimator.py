'''Global parameter estimator using the azimuthal radial profile.'''

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import differential_evolution

from pylorenzmie.lib import Azimuthal
from pylorenzmie.lib.lmtypes import Properties, Result
from pylorenzmie.analysis.BaseEstimator import BaseEstimator
from pylorenzmie.analysis.DEEstimator import DEFAULT_BOUNDS
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import LorenzMie


class _RadialObjective:
    '''Picklable objective comparing a 1-D radial model to the azimuthal profile.

    Using a top-level callable class rather than a closure allows
    ``differential_evolution(workers=-1)`` to distribute candidate
    evaluations across CPU cores via ``multiprocessing``.
    '''

    def __init__(self,
                 model: LorenzMie,
                 profile: NDArray[float],
                 variables: list[str],
                 noise: float) -> None:
        self._model = model
        self._profile = profile
        self._variables = variables
        self._noise = noise

    def __call__(self, values: np.ndarray) -> float:
        self._model.properties = dict(zip(self._variables, values))
        with np.errstate(over='ignore', invalid='ignore'):
            diff = (self._model.hologram() - self._profile) / self._noise
            return float(np.nansum(diff ** 2))


@dataclass
class RadialEstimator(BaseEstimator):
    '''Estimate particle parameters by global search on the azimuthal profile.

    Computes the azimuthal average of the hologram once, then uses
    differential evolution (DE) to find the parameter set that minimises
    the residual between a 1-D radial model evaluation and that profile.

    Because a sphere's hologram is rotationally symmetric, the full 2-D
    pixel comparison in :class:`DEEstimator` can be reduced to a 1-D
    radial comparison with no loss of information about *z_p*, *a_p*, or
    *n_p*.  This makes each DE objective evaluation roughly an order of
    magnitude cheaper than :class:`DEEstimator` while retaining the
    robustness of global search and adding *n_p* estimation that
    :class:`Estimator` cannot provide.

    Inherits from :class:`BaseEstimator`.

    Parameters
    ----------
    model : LorenzMie
        Generative scattering model shared with :class:`Optimizer`.
        The particle parameters on this model are updated in-place.
    bounds : dict, optional
        Mapping of parameter name to ``(min, max)`` search range.
        Default: ``z_p`` (10, 600) pixels, ``a_p`` (0.25, 10.0) μm,
        ``n_p`` (1.0, 3.0).
    popsize : int, optional
        DE population size multiplier (population = popsize ×
        len(bounds)).  Default: 10.
    seed : int or None, optional
        Random seed for :func:`scipy.optimize.differential_evolution`.
        Default: ``None``.

    Notes
    -----
    ``x_p`` and ``y_p`` are pinned to the pixel-coordinate means before
    the search begins; only the parameters listed in ``bounds`` are
    varied.

    The model coordinates are temporarily replaced by a 1-D radial spoke
    ``(x_p + r, y_p)`` for ``r = 0, 1, …, n_radii - 1`` and restored on
    exit, even if an exception occurs.

    This estimator assumes the hologram is rotationally symmetric about
    the particle position, which holds for spherical particles.  For
    non-spherical scatterers use :class:`DEEstimator` instead.

    The default ``settings['workers'] = 1`` is appropriate here because
    each objective evaluation is inexpensive (~100 model points vs
    ~1000–2000 for :class:`DEEstimator`).  Set ``workers=-1`` to
    distribute over all cores if you are running very large population
    sizes or tight tolerances.

    Use :class:`Estimator` for a fast conventional estimate when the
    fringe pattern is clean and *n_p* is known.  Use this class when you
    need a robust estimate of *n_p* as well as *z_p* and *a_p*.  Use
    :class:`DEEstimator` for non-spherical particles or when rotational
    symmetry cannot be assumed.
    '''

    model: LorenzMie
    bounds: dict = field(default_factory=lambda: DEFAULT_BOUNDS.copy())
    popsize: int = 10
    seed: int | None = None
    settings: dict = field(default_factory=lambda: {'tol': 0.01,
                                                     'polish': False,
                                                     'updating': 'deferred',
                                                     'workers': 1})

    @BaseEstimator.properties.getter
    def properties(self) -> Properties:
        '''RadialEstimator configuration.'''
        return dict(popsize=self.popsize,
                    bounds=self.bounds,
                    settings=self.settings)

    def estimate(self, hologram: Hologram) -> Result:
        '''Estimate particle parameters from the azimuthal radial profile.

        Parameters
        ----------
        hologram : Hologram
            Normalized hologram crop to analyze.

        Returns
        -------
        result : pandas.Series
            Estimated particle properties (same keys as
            :attr:`~pylorenzmie.theory.Particle.properties`).
        '''
        x_p = float(hologram.coordinates[0].mean())
        y_p = float(hologram.coordinates[1].mean())
        self.model.particle.x_p = x_p
        self.model.particle.y_p = y_p

        cx, cy = hologram.corner
        profile = Azimuthal.avg(hologram.data,
                                center=(x_p - cx, y_p - cy))
        n_radii = len(profile)
        noise = self.model.instrument.noise

        radii = np.arange(n_radii, dtype=float)
        coords = np.vstack([x_p + radii, np.full(n_radii, y_p)])

        de_vars = list(self.bounds.keys())
        de_bounds = list(self.bounds.values())

        saved_coords = self.model.coordinates
        self.model.coordinates = coords
        objective = _RadialObjective(self.model, profile, de_vars, noise)
        try:
            with np.errstate(over='ignore', invalid='ignore'):
                result = differential_evolution(
                    objective, de_bounds,
                    popsize=self.popsize,
                    seed=self.seed,
                    **self.settings,
                )
        finally:
            self.model.coordinates = saved_coords

        self.model.properties = dict(zip(de_vars, result.x))
        return pd.Series(self.model.particle.properties)

    @classmethod
    def example(cls) -> None:  # pragma: no cover
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
    RadialEstimator.example()
