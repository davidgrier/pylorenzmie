from dataclasses import dataclass, field
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.types import Image, Coordinates, Properties
from pylorenzmie.theory import LorenzMie
from pylorenzmie.analysis.Mask import Mask
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


DEFAULT_BOUNDS: Properties = {
    'z_p': (10., 600.),
    'a_p': (0.05, 5.0),
    'n_p': (1.0, 3.0),
}


@dataclass
class DEEstimator(LMObject):
    '''Estimate initial particle parameters by global search.

    Uses differential evolution (DE) to minimize the sum-squared
    residual between the forward model and a subsampled hologram crop,
    providing robust initial values for :class:`Optimizer` even when
    the conventional :class:`Estimator` fails to converge.

    Inherits from :class:`pylorenzmie.lib.LMObject`.

    Parameters
    ----------
    model : LorenzMie
        Generative scattering model shared with :class:`Optimizer`.
        The particle parameters on this model are updated in-place.
    bounds : dict, optional
        Mapping of parameter name to ``(min, max)`` search range.
        Default: ``z_p`` (10, 600) pixels, ``a_p`` (0.05, 5.0) μm,
        ``n_p`` (1.0, 3.0).
    de_fraction : float, optional
        Fraction of pixels used for the DE objective function.
        Default: 0.05.  A fixed random subsample is drawn once per
        call and held constant throughout the search so the objective
        is deterministic.
    popsize : int, optional
        DE population size multiplier (population = popsize ×
        len(bounds)).  Default: 15.
    seed : int or None, optional
        Random seed passed to :func:`scipy.optimize.differential_evolution`
        for reproducibility.  Default: ``None``.

    Notes
    -----
    ``x_p`` and ``y_p`` are pinned to the pixel-coordinate means before
    the search begins; only the parameters listed in ``bounds`` are
    varied.

    The model coordinates are temporarily overridden during the search
    and restored on exit, even if an exception occurs.

    Use :class:`Estimator` for a fast conventional estimate when the
    fringe pattern is clean.  Use :class:`DEEstimator` when the
    conventional estimator produces starting points too far from the
    true solution for :class:`Optimizer` to converge.
    '''

    model: LorenzMie
    bounds: dict = field(default_factory=lambda: DEFAULT_BOUNDS.copy())
    de_fraction: float = 0.05
    popsize: int = 15
    seed: int | None = None
    settings: dict = field(default_factory=lambda: {'tol': 0.01,
                                                     'polish': False})

    @LMObject.properties.getter
    def properties(self) -> Properties:
        '''DEEstimator configuration.'''
        return dict(de_fraction=self.de_fraction,
                    popsize=self.popsize,
                    bounds=self.bounds,
                    settings=self.settings)

    def estimate(self,
                 data: Image,
                 coordinates: Coordinates | None = None) -> pd.Series:
        '''Estimate particle parameters by differential evolution.

        Parameters
        ----------
        data : numpy.ndarray
            Normalized hologram crop, shape ``(height, width)``.
        coordinates : numpy.ndarray
            Pixel coordinates, shape ``(2, npts)``.  Required.

        Returns
        -------
        result : pandas.Series
            Estimated particle properties (same keys as
            :attr:`~pylorenzmie.theory.Particle.properties`).

        Raises
        ------
        ValueError
            If ``coordinates`` is not provided.
        '''
        if coordinates is None:
            raise ValueError(
                'DEEstimator.estimate() requires pixel coordinates')

        # Pin x_p, y_p to the ROI centre; DE searches the remaining params
        self.model.particle.x_p = float(coordinates[0].mean())
        self.model.particle.y_p = float(coordinates[1].mean())

        # Draw a fixed random subsample — held constant for the whole search
        mask = Mask(shape=data.shape, fraction=self.de_fraction)
        m = mask()
        de_data = data[m]
        ndx = np.nonzero(m.ravel())
        de_coords = np.take(coordinates, ndx, axis=1).squeeze()
        noise = self.model.instrument.noise

        de_vars = list(self.bounds.keys())
        bounds_list = [self.bounds[v] for v in de_vars]

        def _objective(values: np.ndarray) -> float:
            self.model.properties = dict(zip(de_vars, values))
            with np.errstate(over='ignore', invalid='ignore'):
                diff = (self.model.hologram() - de_data) / noise
                return float(np.nansum(diff ** 2))

        saved_coords = self.model.coordinates
        self.model.coordinates = de_coords
        try:
            result = differential_evolution(
                _objective,
                bounds_list,
                popsize=self.popsize,
                seed=self.seed,
                **self.settings,
            )
        finally:
            self.model.coordinates = saved_coords

        # Write the best DE solution to the particle
        self.model.properties = dict(zip(de_vars, result.x))
        return pd.Series(self.model.particle.properties)

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        from time import perf_counter
        from pylorenzmie.utilities import example_hologram
        from pylorenzmie.lib import LMObject

        shape = (201, 201)
        model = LorenzMie()
        model.coordinates = LMObject.meshgrid(shape)
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34

        data = example_hologram()
        coordinates = LMObject.meshgrid(data.shape)

        estimator = cls(model=model, seed=0)
        print('DEEstimator example:')
        start = perf_counter()
        result = estimator.estimate(data, coordinates)
        print(f'Time: {perf_counter() - start:.2f} s')
        print(result)


if __name__ == '__main__':  # pragma: no cover
    DEEstimator.example()
