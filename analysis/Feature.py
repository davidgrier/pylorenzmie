import numpy as np
import pandas as pd
from pylorenzmie.analysis import Mask, Estimator, Optimizer
from pylorenzmie.theory import LorenzMie, Particle
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.types import Image, Coordinates, Properties


class Feature(LMObject):
    '''A holographic feature associated with a single particle.

    Bundles image data, pixel coordinates, a pixel mask, a generative
    scattering model, an initial-parameter estimator, and an optimizer
    for a single particle crop.

    Parameters
    ----------
    data : numpy.ndarray, optional
        Normalized hologram crop.
    coordinates : numpy.ndarray, optional
        Pixel coordinates of shape ``(2, npts)``.
    mask : Mask, optional
        Pixel selection mask.  Default: ``Mask()``.
    model : LorenzMie, optional
        Generative scattering model.  Default: ``LorenzMie()``.
    fixed : list[str], optional
        Model properties held constant during fitting.
        Default: the Optimizer default (``['noise', 'numerical_aperture']``).
    '''

    def __init__(self,
                 data: Image | None = None,
                 coordinates: Coordinates | None = None,
                 mask: Mask | None = None,
                 model: LorenzMie | None = None,
                 fixed: list[str] | None = None) -> None:
        super().__init__()
        self._coordinates = None
        self._data = None
        self.mask = mask or Mask()
        self.model = model or LorenzMie()
        self.data = data
        self.coordinates = coordinates
        self.estimator = Estimator(instrument=self.model.instrument)
        self.optimizer = Optimizer(model=self.model)
        if fixed is not None:
            self.optimizer.fixed = fixed

    @property
    def mask(self) -> Mask:
        '''Pixel selection mask.'''
        return self._mask

    @mask.setter
    def mask(self, mask: Mask) -> None:
        self._mask = mask
        self._mask_data()

    @property
    def data(self) -> Image:
        '''Normalized hologram crop.'''
        return self._data

    @data.setter
    def data(self, data: Image) -> None:
        self._data = data
        self._mask_data()

    def _mask_data(self) -> None:
        if self.data is None:
            return
        self.mask.shape = self.data.shape
        self.mask.exclude = (
            (self.data == np.max(self.data))
            | np.isnan(self.data)
            | np.isinf(self.data)
        )

    @property
    def coordinates(self) -> Coordinates:
        '''Pixel coordinates, shape ``(2, npts)``.'''
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: Coordinates) -> None:
        self._coordinates = coordinates

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
        return self.mask.fraction

    @fraction.setter
    def fraction(self, fraction: float) -> None:
        self.mask.fraction = fraction

    @LMObject.properties.getter
    def properties(self) -> Properties:
        '''Feature pipeline configuration.'''
        return dict(fraction=self.mask.fraction)

    def estimate(self) -> pd.Series:
        '''Estimate initial particle parameters from the hologram crop.

        Sets z_p, a_p, n_p on the particle from the azimuthal profile.
        Sets x_p, y_p to the center of the feature in the coordinate
        system of :attr:`coordinates` (local or frame).

        Returns
        -------
        properties : pandas.Series
            Estimated particle properties.
        '''
        properties = self.estimator.estimate(self.data)
        self.particle.properties = properties
        if self.coordinates is not None:
            self.particle.x_p = float(self.coordinates[0].mean())
            self.particle.y_p = float(self.coordinates[1].mean())
        return properties

    def optimize(self) -> pd.Series:
        '''Optimize particle parameters to fit the hologram crop.

        Returns
        -------
        result : pandas.Series
            Fitted values, uncertainties, and goodness-of-fit statistics.
        '''
        mask = self.mask()
        self.optimizer.data = self.data[mask]
        ndx = np.nonzero(mask.ravel())
        self.model.coordinates = np.take(
            self.coordinates, ndx, axis=1).squeeze()
        return self.optimizer.optimize()

    def hologram(self) -> Image:
        '''Hologram predicted by the current model over all pixels.

        Returns
        -------
        hologram : numpy.ndarray
            Predicted intensity, same shape as :attr:`data`.
        '''
        self.model.coordinates = self.coordinates
        return self.model.hologram().reshape(self.data.shape)

    def residuals(self) -> Image:
        '''Difference between the predicted hologram and the data.

        Returns
        -------
        residuals : numpy.ndarray
            ``hologram() - data``, same shape as :attr:`data`.
        '''
        return self.hologram() - self.data

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        from time import perf_counter
        from pylorenzmie.utilities import example_hologram

        data = example_hologram()

        feature = cls()
        feature.data = data
        feature.coordinates = feature.model.meshgrid(data.shape)
        feature.mask.fraction = 0.25

        instrument = feature.model.instrument
        instrument.wavelength = 0.447
        instrument.magnification = 0.048
        instrument.n_m = 1.34

        particle = feature.model.particle
        particle.r_p = [data.shape[1] / 2., data.shape[0] / 2., 330.]
        particle.a_p = 1.1
        particle.n_p = 1.4
        feature.optimizer.variables = 'x_p y_p z_p a_p n_p'.split()
        print(f'Initial estimates:\n{particle}')

        feature.model.hologram()  # warm up JIT / caches

        start = perf_counter()
        result = feature.optimize()
        print(f'Refined estimates:\n{particle}')
        print(f'Time to fit: {perf_counter() - start:.3f} s')
        print(feature.optimizer.report())


if __name__ == '__main__':  # pragma: no cover
    Feature.example()
