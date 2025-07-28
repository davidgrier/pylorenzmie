#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray
from pylorenzmie.analysis import (Mask, Estimator, Optimizer)
from pylorenzmie.theory import (LorenzMie, Particle)
import pandas as pd


class Feature(object):
    '''
    Abstraction of a feature in an in-line hologram

    ...

    Properties
    ----------
    data : numpy.ndarray
        [npts] normalized intensity values
    coordinates : numpy.ndarray
        [3, npts] coordinates of pixels in data
    mask : Mask
        Mask for selecting pixels from data for analysis
    model : LorenzMie
        Incorporates information about the Particle and the Instrument
        and for supported models, uses this information to compute a
        hologram at the specified coordinates.

    Methods
    -------
    optimize() : pandas.Series
        Optimize adjustable parameters and return a report containing
        the optimized values and their numerical uncertainties.
        This report also can be retrieved from optimizer.report
        Raw fitting results are available from optimizer.results
        Metadata is available from optimizer.metadata
    hologram() : numpy.ndarray
        Intensity value at each coordinate computed with current model.
    residuals() : numpy.ndarray
        Difference between the current model and the data.

    '''

    def __init__(self,
                 data: LorenzMie.Image | None = None,
                 coordinates: LorenzMie.Coordinates | None = None,
                 mask: Mask | None = None,
                 model: LorenzMie | None = None,
                 fixed: list[str] | None = None) -> None:
        self._coordinates = None
        self._data = None
        self.mask = mask or Mask()
        self.model = model or LorenzMie()
        self.data = data
        self.coordinates = coordinates
        self.estimator = Estimator(instrument=self.model.instrument)
        self.optimizer = Optimizer(model=self.model)
        self.optimizer.fixed = fixed or self.optimizer.fixed

    @property
    def mask(self) -> Mask:
        '''Mask for selecting pixels to analyze'''
        return self._mask

    @mask.setter
    def mask(self, mask: Mask) -> None:
        self._mask = mask
        self._mask_data()

    @property
    def data(self) -> LorenzMie.Image:
        '''Values of the (normalized) data at each pixel'''
        return self._data

    @data.setter
    def data(self, data: LorenzMie.Image) -> None:
        self._data = data
        self._mask_data()

    def _mask_data(self) -> None:
        if self.data is None:
            return
        self.mask.shape = self.data.shape
        saturated = (self.data == np.max(self.data))
        nan = np.isnan(self.data)
        infinite = np.isinf(self.data)
        self.mask.exclude = saturated | nan | infinite

    @property
    def coordinates(self) -> LorenzMie.Coordinates:
        '''Array of pixel coordinates'''
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: LorenzMie.Coordinates):
        self._coordinates = coordinates

    @property
    def particle(self) -> Particle:
        return self.model.particle

    @particle.setter
    def particle(self, particle: Particle) -> None:
        self.model.particle = particle

    @property
    def model(self) -> LorenzMie:
        return self._model

    @model.setter
    def model(self, model: LorenzMie) -> None:
        self._model = model

    def estimate(self) -> Particle.Properties:
        properties = self.estimator.estimate(self)
        self.particle.properties = properties
        return properties

    def optimize(self) -> pd.Series:
        self.optimizer.data = self.data[self.mask()]
        # The following nasty hack is required for cupy because
        # opt.coordinates = self.coordinates[:,self.mask().ravel()]
        # yields garbled results on GPU. Memory organization?
        ndx = np.nonzero(self.mask().ravel())
        coordinates = np.take(self.coordinates, ndx, axis=1).squeeze()
        self.model.coordinates = coordinates
        return self.optimizer.optimize()

    def hologram(self) -> LorenzMie.Image:
        self.model.coordinates = self.coordinates
        return self.model.hologram().reshape(self.data.shape)

    def residuals(self) -> LorenzMie.Image:
        return self.hologram() - self.data


def example() -> None:
    import cv2
    from time import perf_counter
    from pylorenzmie.lib import coordinates
    from pathlib import Path

    feature = Feature()

    # Normalized hologram
    basedir = Path(__file__).parent.parent.resolve()
    filename = str(basedir / 'docs' / 'tutorials' / 'crop.png')
    data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float)
    data /= 100.
    feature.data = data
    feature.coordinates = coordinates(data.shape)
    feature.mask.fraction = 0.25

    # instrument properties
    instrument = feature.model.instrument
    instrument.wavelength = 0.447
    instrument.magnification = 0.048
    instrument.n_m = 1.34

    # Initial estimates for particle properties
    particle = feature.model.particle
    particle.r_p = [data.shape[0]//2, data.shape[1]//2, 330.]
    particle.a_p = 1.1
    particle.n_p = 1.4
    print(f'Initial estimates:\n{particle}')

    feature.optimizer.variables = 'x_p y_p z_p a_p n_p'.split()

    # init dummy hologram for proper speed gauge
    feature.model.hologram()

    # perform fit
    start = perf_counter()
    feature.optimize()
    delta = perf_counter() - start
    print(f'Refined estimates:\n{particle}')
    print(f'Time to fit: {delta:.3f} s')


if __name__ == '__main__':  # pragma: no cover
    example()
