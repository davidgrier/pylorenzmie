'''A holographic feature: one particle, its model and its fit.

A :class:`Feature` is a :class:`~pylorenzmie.analysis.Hologram.Hologram` of one
particle, usually a crop of a larger image, together with a
:class:`~pylorenzmie.theory.LorenzMie.LorenzMie` model, an estimator that
finds starting values and an optimizer that fits the model to the data.
:meth:`~Feature.estimate` runs the estimator, which sets the properties of the
model's particle, and :meth:`~Feature.optimize` runs the optimizer from there
and returns the fitted values with their uncertainties.
:meth:`~Feature.predicted` and :meth:`~Feature.residuals` give the fitted
hologram and its difference from the data.

The pixels that the fit uses are those that
the :class:`~pylorenzmie.analysis.Mask.Mask` in :attr:`~Feature.mask` selects.
By default the mask leaves out the pixels in
:attr:`~pylorenzmie.analysis.Hologram.Hologram.excluded`: the saturated
pixels that the hologram names, and any NaN or infinite ones.  A feature is
also what :class:`~pylorenzmie.analysis.Frame.Frame` creates for each particle
that it finds.
'''

import logging

import pandas as pd
from pylorenzmie.analysis import (BaseEstimator, DEEstimator, Mask, Optimizer,
                                  PairEstimator)
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import LorenzMie, Particle, Pair
from pylorenzmie.lib.lmtypes import Image


logger = logging.getLogger(__name__)

# Optimizer.fixed for a Pair: the single-sphere default names 'k_p',
# which a Pair does not have; hold both spheres' absorption instead.
_PAIR_FIXED = ['wavelength', 'magnification', 'n_m', 'noise', 'k_p1',
               'k_p2']


class Feature(Hologram):
    '''A holographic feature associated with one or two particles.

    A Feature is a Hologram crop augmented with an estimator and
    optimizer for localizing and characterizing its particle(s).  With
    ``centers`` omitted it models a single :class:`~pylorenzmie.theory.Sphere`
    and estimates with :class:`~pylorenzmie.analysis.DEEstimator`.  Given
    two centers it models a :class:`~pylorenzmie.theory.Pair.Pair` and
    estimates with
    :class:`~pylorenzmie.analysis.PairEstimator.PairEstimator`.

    Parameters
    ----------
    hologram : Hologram
        Normalized hologram crop with pixel coordinates.
    model : LorenzMie, optional
        Generative scattering model.  Default: ``LorenzMie()``.
    centers : list[tuple[float, float]], optional
        Rough ``(x_p, y_p)`` pixel centers of the particles in the crop, as
        a localizer saw them.  ``None`` (default) or one center configures a
        single-particle feature; two centers configure a
        :class:`~pylorenzmie.theory.Pair.Pair`.  One center is given to the
        estimator as its
        :attr:`~pylorenzmie.analysis.BaseEstimator.BaseEstimator.center`,
        where it starts to look, and the particle is put there until it is
        estimated.
    fixed : list[str], optional
        Model properties held constant during fitting.  Default: the
        Optimizer default for a single particle, or ``k_p1``/``k_p2``
        plus the instrument parameters for a Pair.
    estimator : BaseEstimator, optional
        Estimator that finds the starting values.  It is used as given: the
        feature sets its ``exclude`` to the pixels of the mask, so an
        estimator belongs to one feature: sharing it overwrites the mask of
        the others.  The estimator is responsible for its own model.
        Default: ``None``, a
        :class:`~pylorenzmie.analysis.DEEstimator.DEEstimator` (a
        :class:`~pylorenzmie.analysis.PairEstimator.PairEstimator` for two
        centers) that shares the feature's model.
    '''

    def __init__(self,
                 hologram: Hologram,
                 model: LorenzMie | None = None,
                 centers: list[tuple[float, float]] | None = None,
                 fixed: list[str] | None = None,
                 estimator: BaseEstimator | None = None) -> None:
        super().__init__(hologram.data,
                         hologram.corner,
                         hologram.saturated)
        self._coordinates = hologram.coordinates    # the crop's own view
        self.model = model or LorenzMie()

        self._centers = (None if centers is None else
                         [(float(x), float(y)) for x, y in centers])
        pair = self._centers is not None and len(self._centers) == 2
        if self._centers is not None and pair:
            particles = Pair()
            self.model.particle = particles
            (x_p1, y_p1), (x_p2, y_p2) = self._centers
            particles.x_p1, particles.y_p1 = x_p1, y_p1
            particles.x_p2, particles.y_p2 = x_p2, y_p2

        m = Mask()
        m.shape = self.data.shape
        # TODO: detect bad pixels to exclude from Mask
        m.exclude = self.excluded

        if estimator is None:
            estimator_cls = PairEstimator if pair else DEEstimator
            estimator = estimator_cls(model=self.model, exclude=m.exclude)
        else:
            estimator.exclude = m.exclude
        self.estimator = estimator
        if self._centers is not None and len(self._centers) == 1:
            estimator.center = self._centers[0]
            particle = self.model.single()
            particle.x_p, particle.y_p = self._centers[0]

        self.optimizer = Optimizer(model=self.model, mask=m)
        if fixed is None and pair:
            fixed = _PAIR_FIXED
        if fixed is not None:
            self.optimizer.fixed = fixed

    @property
    def centers(self) -> list[tuple[float, float]] | None:
        '''Where the particles were seen, if the feature was told.'''
        return self._centers

    @property
    def mask(self) -> Mask:
        '''Shared pixel exclusion mask for estimator and optimizer.

        Bad pixels (saturated, dead) set via
        :attr:`~pylorenzmie.analysis.Mask.Mask.exclude` are excluded from both
        the DE estimation and the LM optimization. Each component controls its
        own pixel fraction independently. '''
        mask = self.optimizer.mask
        if mask is None:
            raise RuntimeError('the optimizer has no mask')
        return mask

    @mask.setter
    def mask(self, mask: Mask) -> None:
        self.estimator.exclude = mask.exclude
        self.optimizer.mask = mask

    @property
    def particle(self) -> Particle:
        '''Particle associated with the scattering model.'''
        return self.model.single()

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

        For a two-particle feature the
        :class:`~pylorenzmie.analysis.PairEstimator.PairEstimator` reads
        the pinned :attr:`centers` off the model's
        :class:`~pylorenzmie.theory.Pair.Pair` particle.
        The estimator returns its result, and the feature writes the
        properties that the model has onto its particle in place,
        ready for :meth:`optimize`.  An estimator need not touch the
        particle, and one that does (the global estimators) loses nothing:
        the values are the same.

        Returns
        -------
        properties : pandas.Series
            Estimated particle properties.
        '''
        result = self.estimator.estimate(self)
        known = self.model.properties
        self.model.properties = {str(name): value for name, value
                                 in result.items() if name in known}
        ignored = [str(name) for name in result.index if name not in known]
        if ignored:
            logger.debug('Ignoring estimated properties that the model '
                         'does not have: %s', ', '.join(ignored))
        return result

    def optimize(self) -> pd.Series:
        '''Optimize particle parameters to fit the hologram crop.

        Returns
        -------
        result : pandas.Series
            Fitted values, uncertainties, and goodness-of-fit statistics.

        Notes
        -----
        When a mask with ``fraction < 1.0`` is set, a new random pixel
        subsample is drawn on every call, unless the mask has a ``seed``,
        in which case the same pixels are used every time.
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
        '''Run a demonstration of the class, for use as a script.'''
        from time import perf_counter
        from pylorenzmie.utilities import example_hologram

        feature = cls(example_hologram())

        instrument = feature.model.instrument
        instrument.wavelength = 0.447
        instrument.magnification = 0.048
        instrument.n_m = 1.34
        instrument.noise = 0.05

        assert isinstance(feature.estimator, DEEstimator)
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
        feature.optimize()
        print(f'Optimized in {perf_counter() - start:.3f} s')
        print(feature.optimizer.report())


if __name__ == '__main__':  # pragma: no cover
    Feature.example()
