import numpy as np
import pandas as pd
from pylorenzmie.analysis import Mask, DEEstimator, PairEstimator, Optimizer
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import LorenzMie, Particle, Pair
from pylorenzmie.lib.lmtypes import Image


# Optimizer.fixed for a Pair: the single-sphere default names 'k_p',
# which a Pair does not have; hold both spheres' absorption instead.
_PAIR_FIXED = ('wavelength magnification numerical_aperture '
               'n_m noise k_p1 k_p2').split()


class Feature(Hologram):
    '''A holographic feature associated with one or two particles.

    A Feature is a Hologram crop augmented with an estimator and
    optimizer for localizing and characterizing its particle(s).  With
    ``centers`` omitted it models a single :class:`~pylorenzmie.theory.Sphere`
    and estimates with :class:`~pylorenzmie.analysis.DEEstimator`.  Given
    two centers it models a :class:`~pylorenzmie.theory.Pair` and
    estimates with :class:`~pylorenzmie.analysis.PairEstimator`.

    Parameters
    ----------
    hologram : Hologram
        Normalized hologram crop with pixel coordinates.
    model : LorenzMie, optional
        Generative scattering model.  Default: ``LorenzMie()``.
    centers : list[tuple[float, float]], optional
        Rough ``(x_p, y_p)`` pixel centers of the particles in the crop.
        ``None`` (default) or one center configures a single-particle
        feature; two centers configure a :class:`~pylorenzmie.theory.Pair`.
    fixed : list[str], optional
        Model properties held constant during fitting.  Default: the
        Optimizer default for a single particle, or ``k_p1``/``k_p2``
        plus the instrument parameters for a Pair.
    '''

    def __init__(self,
                 hologram: Hologram,
                 model: LorenzMie | None = None,
                 centers: list[tuple[float, float]] | None = None,
                 fixed: list[str] | None = None) -> None:
        # Copy Hologram state directly; super().__init__() is skipped to avoid
        # calling __post_init__ (which would rerun meshgrid on hologram.data).
        self.data = hologram.data
        self.corner = hologram.corner
        self._coordinates = hologram._coordinates
        self.model = model or LorenzMie()

        self._centers = (None if centers is None else
                         [(float(x), float(y)) for x, y in centers])
        pair = self._centers is not None and len(self._centers) == 2
        if pair:
            self.model.particle = Pair()
            (x_p1, y_p1), (x_p2, y_p2) = self._centers
            self.model.particle.x_p1, self.model.particle.y_p1 = x_p1, y_p1
            self.model.particle.x_p2, self.model.particle.y_p2 = x_p2, y_p2

        m = Mask()
        m.shape = self.data.shape
        # TODO: detect bad pixels to exclude from Mask
        m.exclude = (
            (self.data == np.max(self.data))
            | np.isnan(self.data)
            | np.isinf(self.data)
        )

        estimator_cls = PairEstimator if pair else DEEstimator
        self.estimator = estimator_cls(model=self.model, exclude=m.exclude)
        # cupy/jax models hold device state that cannot survive a fork;
        # keep the DE search single-process for those backends.
        if any(b in type(self.model).method for b in ('cupy', 'jax')):
            self.estimator.settings['workers'] = 1

        self.optimizer = Optimizer(model=self.model, mask=m)
        if fixed is None and pair:
            fixed = _PAIR_FIXED
        if fixed is not None:
            self.optimizer.fixed = fixed

    @property
    def centers(self) -> list[tuple[float, float]] | None:
        '''Pinned centers for a two-particle feature, else ``None``.'''
        return self._centers

    @property
    def mask(self) -> Mask:
        '''Shared pixel exclusion mask for estimator and optimizer.

        Bad pixels (saturated, dead) set via :attr:`~Mask.exclude` are
        excluded from both the DE estimation and the LM optimization.
        Each component controls its own pixel fraction independently.
        '''
        return self.optimizer.mask

    @mask.setter
    def mask(self, mask: Mask) -> None:
        self.estimator.exclude = mask.exclude
        self.optimizer.mask = mask

    @property
    def particle(self) -> Particle:
        '''Particle associated with the scattering model.'''
        return self.model.particle

    @particle.setter
    def particle(self, particle: Particle) -> None:
        self.model.particle = particle

    @property
    def model(self) -> LorenzMie:
        '''Generative scattering model.'''
        return self._model

    @model.setter
    def model(self, model: LorenzMie) -> None:
        self._model = model

    def estimate(self) -> pd.Series:
        '''Estimate initial particle parameters from the hologram crop.

        For a two-particle feature the :class:`PairEstimator` reads the
        pinned :attr:`centers` off the model's :class:`Pair` particle.
        The estimated values are written onto :attr:`particle` in place,
        ready for :meth:`optimize`.

        Returns
        -------
        properties : pandas.Series
            Estimated particle properties.
        '''
        return self.estimator.estimate(self)

    def optimize(self) -> pd.Series:
        '''Optimize particle parameters to fit the hologram crop.

        Returns
        -------
        result : pandas.Series
            Fitted values, uncertainties, and goodness-of-fit statistics.

        Notes
        -----
        When a mask with ``fraction < 1.0`` is set, a fresh random pixel
        subsample is drawn on every call.
        '''
        return self.optimizer.optimize(self)

    def predicted(self) -> Image:
        '''Hologram predicted by the current model over all pixels.

        Returns
        -------
        predicted : numpy.ndarray
            Predicted intensity, same shape as this Feature.
        '''
        self.model.coordinates = self.flat_coordinates
        return self.model.hologram().reshape(self.shape)

    def residuals(self) -> Image:
        '''Difference between the predicted hologram and the data.

        Returns
        -------
        residuals : numpy.ndarray
            ``predicted() - data``, same shape as this Feature.
        '''
        return self.predicted() - self.data

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        from time import perf_counter
        from pylorenzmie.utilities import example_hologram

        feature = cls(example_hologram())

        instrument = feature.model.instrument
        instrument.wavelength = 0.447
        instrument.magnification = 0.048
        instrument.n_m = 1.34
        instrument.noise = 0.05

        feature.estimator.fraction = 0.01
        feature.optimizer.fraction = 0.25
        feature.optimizer.variables = 'x_p y_p z_p a_p n_p'.split()

        feature.model.coordinates = feature.flat_coordinates
        feature.model.hologram()  # warm up JIT / caches

        start = perf_counter()
        estimate = feature.estimate()
        print(f'Estimated in {perf_counter() - start:.3f} s')
        print(estimate)

        start = perf_counter()
        result = feature.optimize()
        print(f'Optimized in {perf_counter() - start:.3f} s')
        print(feature.optimizer.report())


if __name__ == '__main__':  # pragma: no cover
    Feature.example()
