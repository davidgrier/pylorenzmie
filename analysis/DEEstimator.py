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
from pylorenzmie.lib.lmtypes import Image, Properties, Result
from pylorenzmie.analysis.BaseEstimator import BaseEstimator
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.analysis.Mask import Mask
from pylorenzmie.theory import LorenzMie
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import differential_evolution


DEFAULT_BOUNDS: Properties = {
    'z_p': (10., 600.),
    'a_p': (0.25, 10.0),
    'n_p': (1.0, 3.0),
}


class _DEObjective:
    '''Picklable objective function for differential evolution.

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
        self._model.properties = dict(zip(self._variables, values))
        with np.errstate(over='ignore', invalid='ignore'):
            diff = (self._model.hologram() - self._data) / self._noise
            return float(np.nansum(diff ** 2))


@dataclass
class DEEstimator(BaseEstimator):
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
        Mapping of parameter name to ``(min, max)`` search range.
        Default: ``z_p`` (10, 600) pixels, ``a_p`` (0.25, 10.0) μm,
        ``n_p`` (1.0, 3.0).
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
    bounds: dict = field(default_factory=lambda: DEFAULT_BOUNDS.copy())
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

    def estimate(self, hologram: Hologram) -> Result:
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
        self.model.particle.x_p = float(hologram.coordinates[0].mean())
        self.model.particle.y_p = float(hologram.coordinates[1].mean())

        # seeded with the estimator, so that a seed makes the whole
        # estimate reproducible
        mask = Mask(fraction=self.fraction, seed=self.seed)
        if self.exclude is not None:
            mask.exclude = self.exclude
        de_data, de_coords = mask.apply(hologram)
        noise = self.model.instrument.noise

        de_vars = list(self.bounds.keys())
        de_bounds = list(self.bounds.values())

        saved_coords = self.model.coordinates
        self.model.coordinates = de_coords
        objective = _DEObjective(self.model, de_data, de_vars, noise)
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
    DEEstimator.example()
