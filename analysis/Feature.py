#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import (Optional, List)
from pylorenzmie.analysis import (Mask, Estimator, Optimizer)
from pylorenzmie.theory import LorenzMie


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
    model : LMHologram
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
                 data: Optional[np.ndarray] = None,
                 coordinates: Optional[np.ndarray] = None,
                 model: Optional[LorenzMie] = None,
                 fixed: Optional[List[str]] = None) -> None:
        self._coordinates = None
        self.mask = Mask()
        self.model = model or LorenzMie()
        self.data = data
        self.coordinates = coordinates
        self.estimator = Estimator()
        self.optimizer = Optimizer(model=self.model)
        self.optimizer.fixed = fixed or self.optimizer.fixed

    @property
    def data(self) -> np.ndarray:
        '''Values of the (normalized) data at each pixel'''
        return self._data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        if data is not None:
            self.mask.shape = data.shape
            saturated = (data == np.max(data))
            nan = np.isnan(data)
            infinite = np.isinf(data)
            self.mask.exclude = (saturated or nan or infinite)
        self._data = data

    @property
    def coordinates(self):
        '''Array of pixel coordinates'''
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates = coordinates

    @property
    def particle(self):
        return self.model.particle

    @particle.setter
    def particle(self, particle):
        self.model.particle = particle

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def estimate(self):
        properties = self.estimator.predict(self)
        self.particle.properties = properties
        return properties

    def optimize(self):
        self.optimizer.data = self.data[self.mask()]
        # The following nasty hack is required for cupy because
        # opt.coordinates = self.coordinates[:,mask]
        # yields garbled results on GPU. Memory organization?
        ndx = np.nonzero(self.mask())
        coordinates = np.take(self.coordinates, ndx, axis=1).squeeze()
        self.model.coordinates = coordinates
        return self.optimizer.optimize()

    def hologram(self):
        self.model.coordinates = self.coordinates
        return self.model.hologram().reshape(self.data.shape)

    def residuals(self):
        return self.hologram() - self.data


def example():
    from pathlib import Path
    import cv2
    from time import perf_counter
    from pylorenzmie.lib import coordinates

    basedir = Path(__file__).parent.parent.resolve()
    filename = str(basedir / 'docs' / 'tutorials' / 'crop.png')

    # Normalized hologram
    data = cv2.imread(filename, cv2.COLOR_GRAY).astype(float)
    data /= 100.

    # Feature
    a = Feature()

    # model properties
    a.model.wavelength = 0.447
    a.model.magnification = 0.048
    a.model.n_m = 1.34

    # pixel selection mask
    a.mask.fraction = 0.25

    a.data = data

    # Pixel coordinates
    a.coordinates = coordinates(data.shape)

    # Initial estimates for particle properties
    p = a.model.particle
    p.r_p = [data.shape[0]//2, data.shape[1]//2, 330.]
    p.a_p = 1.1
    p.n_p = 1.4
    print(f'Initial estimates:\n{p}')

    # init dummy hologram for proper speed gauge
    a.model.hologram()
    start = perf_counter()
    a.optimize()
    delta = perf_counter() - start
    print(f'Refined estimates:\n{p}')
    print(f'Time to fit: {delta:.3f} s')


if __name__ == '__main__':  # pragma: no cover
    example()
