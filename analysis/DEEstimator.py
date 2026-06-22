from dataclasses import dataclass, field
from pylorenzmie.lib.lmtypes import Image, Properties, Result
from pylorenzmie.analysis.BaseEstimator import BaseEstimator
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.analysis.Mask import Mask
from pylorenzmie.theory import LorenzMie
import numpy as np
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

    Uses differential evolution (DE) to minimize the sum-squared
    residual between the forward model and a subsampled hologram crop,
    providing robust initial values for :class:`Optimizer` even when
    the conventional :class:`Estimator` fails to converge.

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
    fraction : float, optional
        Fraction of pixels used for the DE objective function.
        Default: 0.02.  A fixed random subsample is drawn once per
        call and held constant throughout the search so the objective
        is deterministic.
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

    By default ``settings['workers'] = -1`` distributes candidate
    evaluations across all CPU cores via ``multiprocessing``.  On Linux
    (``fork`` start method) the speedup is near-linear with core count
    at negligible overhead.  On macOS (``spawn`` start method) each
    :meth:`estimate` call pays a process-pool startup cost; on machines
    with many cores this is still a net win (~2× on a 10-core Mac).
    Set ``workers=1`` to disable parallelism, e.g. when using
    ``cupyLorenzMie`` (which holds GPU memory and is not picklable).
    The companion setting ``updating='deferred'`` (also the default) is
    required for ``workers != 1`` and suppresses a SciPy warning.

    Use :class:`Estimator` for a fast conventional estimate when the
    fringe pattern is clean.  Use :class:`DEEstimator` when the
    conventional estimator produces starting points too far from the
    true solution for :class:`Optimizer` to converge.
    '''

    model: LorenzMie
    bounds: dict = field(default_factory=lambda: DEFAULT_BOUNDS.copy())
    fraction: float = 0.02
    popsize: int = 10
    seed: int | None = None
    settings: dict = field(default_factory=lambda: {'tol': 0.01,
                                                     'polish': False,
                                                     'updating': 'deferred',
                                                     'workers': -1})

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
            :attr:`~pylorenzmie.theory.Particle.properties`).
        '''
        self.model.particle.x_p = float(hologram.coordinates[0].mean())
        self.model.particle.y_p = float(hologram.coordinates[1].mean())

        mask = Mask(fraction=self.fraction)
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
