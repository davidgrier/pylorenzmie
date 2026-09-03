from dataclasses import dataclass
from pylorenzmie.lib.lmtypes import Result
from pylorenzmie.analysis.DEEstimator import DEEstimator, _DEObjective
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.analysis.Mask import Mask
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


@dataclass
class PairEstimator(DEEstimator):
    '''Estimate initial parameters of a two-particle Pair by global search.

    Extends :class:`DEEstimator` to a :class:`~pylorenzmie.theory.Pair`
    model: rather than pinning a single ``x_p``/``y_p`` to the crop's
    coordinate mean and searching one set of ``z_p``, ``a_p``, ``n_p``,
    it pins both particles' centers from caller-supplied estimates and
    searches ``z_p1``/``a_p1``/``n_p1`` and ``z_p2``/``a_p2``/``n_p2``
    together using the same :attr:`~DEEstimator.bounds` for each.

    Inherits from :class:`DEEstimator`.

    Notes
    -----
    ``model.particle`` must be a :class:`~pylorenzmie.theory.Pair`.
    The rough centers passed to :meth:`estimate` typically come from
    the ``x_p``/``y_p`` lists that
    :func:`~pylorenzmie.analysis.pair_grouping.group_overlapping`
    stores on a merged two-particle row.
    '''

    def estimate(self,
                 hologram: Hologram,
                 centers: tuple[tuple[float, float],
                                tuple[float, float]] | None = None
                 ) -> Result:
        '''Estimate Pair parameters by differential evolution.

        Parameters
        ----------
        hologram : Hologram
            Normalized hologram crop containing both particles.
        centers : tuple[tuple[float, float], tuple[float, float]], optional
            ``((x_p1, y_p1), (x_p2, y_p2))`` rough centers of the two
            particles, in pixels.  Default: ``None``, which reads the
            centers already set on ``model.particle`` (``x_p1``/``y_p1``
            and ``x_p2``/``y_p2``).

        Returns
        -------
        result : pandas.Series
            Estimated Pair properties (same keys as
            :attr:`~pylorenzmie.theory.Pair.properties`).
        '''
        if centers is None:
            p = self.model.particle
            centers = ((p.x_p1, p.y_p1), (p.x_p2, p.y_p2))
        (x_p1, y_p1), (x_p2, y_p2) = centers
        self.model.particle.x_p1 = float(x_p1)
        self.model.particle.y_p1 = float(y_p1)
        self.model.particle.x_p2 = float(x_p2)
        self.model.particle.y_p2 = float(y_p2)

        mask = Mask(fraction=self.fraction)
        if self.exclude is not None:
            mask.exclude = self.exclude
        de_data, de_coords = mask.apply(hologram)
        noise = self.model.instrument.noise

        de_vars = [f'{key}{i}' for i in (1, 2) for key in self.bounds]
        de_bounds = list(self.bounds.values()) * 2

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
        from pylorenzmie.theory import LorenzMie, Pair

        model = LorenzMie(particle=Pair())
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34
        model.particle.properties = dict(
            x_p1=85., y_p1=100., z_p1=250., a_p1=0.5, n_p1=1.42,
            x_p2=115., y_p2=100., z_p2=250., a_p2=0.6, n_p2=1.40)

        shape = (201, 201)
        model.coordinates = LorenzMie.meshgrid(shape)
        hologram = Hologram(model.hologram().reshape(shape))

        estimator = cls(model=model, seed=0)
        # LorenzMie may resolve to cupyLorenzMie, which holds GPU memory
        # and cannot survive a fork; keep the search single-process.
        estimator.settings['workers'] = 1

        print(f'{cls.__name__} example')
        start = perf_counter()
        result = estimator.estimate(hologram, centers=((85., 100.),
                                                        (115., 100.)))
        print(f'Time: {perf_counter() - start:.3f} s')
        print(result)


if __name__ == '__main__':  # pragma: no cover
    PairEstimator.example()
