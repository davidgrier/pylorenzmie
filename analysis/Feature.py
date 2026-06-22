import numpy as np
import pandas as pd
from pylorenzmie.analysis import Mask, Estimator, Optimizer
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import LorenzMie, Particle
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.lmtypes import Image, Properties


class Feature(LMObject):
    '''A holographic feature associated with a single particle.

    Bundles a normalized hologram crop, a pixel mask, a generative
    scattering model, an initial-parameter estimator, and an optimizer
    for a single particle.

    Parameters
    ----------
    hologram : Hologram, optional
        Normalized hologram crop with pixel coordinates.
    mask : Mask, optional
        Pixel selection mask.  Default: ``Mask()``.
    model : LorenzMie, optional
        Generative scattering model.  Default: ``LorenzMie()``.
    fixed : list[str], optional
        Model properties held constant during fitting.
        Default: the Optimizer default (``['noise', 'numerical_aperture']``).
    '''

    def __init__(self,
                 hologram: Hologram | None = None,
                 mask: Mask | None = None,
                 model: LorenzMie | None = None,
                 fixed: list[str] | None = None) -> None:
        super().__init__()
        self.model = model or LorenzMie()
        self._mask = mask or Mask()
        self.estimator = Estimator(instrument=self.model.instrument)
        self.optimizer = Optimizer(model=self.model)
        if fixed is not None:
            self.optimizer.fixed = fixed
        self.hologram = hologram

    @property
    def hologram(self) -> Hologram | None:
        '''Normalized hologram crop with pixel coordinates.'''
        return self._hologram

    @hologram.setter
    def hologram(self, hologram: Hologram | None) -> None:
        self._hologram = hologram
        if hologram is not None:
            data = hologram.data
            self._mask.shape = data.shape
            self._mask.exclude = (
                (data == np.max(data))
                | np.isnan(data)
                | np.isinf(data)
            )

    @property
    def mask(self) -> Mask:
        '''Pixel selection mask.'''
        return self._mask

    @mask.setter
    def mask(self, mask: Mask) -> None:
        self._mask = mask

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

    @property
    def fraction(self) -> float:
        '''Fraction of pixels passed to the optimizer.'''
        return self._mask.fraction

    @fraction.setter
    def fraction(self, fraction: float) -> None:
        self._mask.fraction = fraction

    @LMObject.properties.getter
    def properties(self) -> Properties:
        '''Feature pipeline configuration.'''
        return dict(fraction=self._mask.fraction)

    def estimate(self) -> pd.Series:
        '''Estimate initial particle parameters from the hologram crop.

        Returns
        -------
        properties : pandas.Series
            Estimated particle properties.
        '''
        properties = self.estimator.estimate(self.hologram)
        self.particle.properties = properties
        return properties

    def optimize(self) -> pd.Series:
        '''Optimize particle parameters to fit the hologram crop.

        Returns
        -------
        result : pandas.Series
            Fitted values, uncertainties, and goodness-of-fit statistics.
        '''
        self.optimizer.mask = self._mask
        return self.optimizer.optimize(self.hologram)

    def predicted(self) -> Image:
        '''Hologram predicted by the current model over all pixels.

        Returns
        -------
        predicted : numpy.ndarray
            Predicted intensity, same shape as :attr:`hologram`.
        '''
        self.model.coordinates = self.hologram.flat_coordinates
        return self.model.hologram().reshape(self.hologram.shape)

    def residuals(self) -> Image:
        '''Difference between the predicted hologram and the data.

        Returns
        -------
        residuals : numpy.ndarray
            ``predicted() - hologram.data``, same shape as :attr:`hologram`.
        '''
        return self.predicted() - self.hologram.data

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        from time import perf_counter
        from pylorenzmie.utilities import example_hologram

        hologram = example_hologram()
        feature = cls(hologram=hologram)

        instrument = feature.model.instrument
        instrument.wavelength = 0.447
        instrument.magnification = 0.048
        instrument.n_m = 1.34

        h, w = hologram.shape
        particle = feature.particle
        particle.r_p = [w / 2., h / 2., 330.]
        particle.a_p = 1.1
        particle.n_p = 1.4
        feature.mask.fraction = 0.25
        feature.optimizer.variables = 'x_p y_p z_p a_p n_p'.split()
        print(f'Initial estimates:\n{particle}')

        feature.model.coordinates = hologram.flat_coordinates
        feature.model.hologram()  # warm up JIT / caches

        start = perf_counter()
        result = feature.optimize()
        print(f'Refined estimates:\n{particle}')
        print(f'Time to fit: {perf_counter() - start:.3f} s')
        print(feature.optimizer.report())


if __name__ == '__main__':  # pragma: no cover
    Feature.example()
