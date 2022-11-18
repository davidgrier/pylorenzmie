#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import (Optional, List)
from pylorenzmie.analysis import (Mask, Estimator, Optimizer)
from pylorenzmie.theory import LorenzMie
from pylorenzmie.utilities import coordinates


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
        self.estimator = Estimator(feature=self)
        self.optimizer = Optimizer(model=self.model)
        self.optimizer.fixed = fixed or self.optimizer.fixed

    @property
    def data(self) -> np.ndarray:
        '''Values of the (normalized) data at each pixel'''
        return self._data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        if data is not None:
            saturated = (data == np.max(data))
            nan = np.isnan(data)
            infinite = np.isinf(data)
            bad = (saturated | nan | infinite).flatten()
            self.mask.exclude = np.nonzero(bad)[0]
        self._data = data

    @property
    def coordinates(self):
        '''Array of pixel coordinates'''
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        self.mask.coordinates = coordinates
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
        mask = self.mask.selected
        opt = self.optimizer
        opt.data = self.data.ravel()[mask]
        # The following nasty hack is required for cupy because
        # opt.coordinates = self.coordinates[:,mask]
        # yields garbled results on GPU. Memory organization?
        ndx = np.nonzero(mask)
        coordinates = np.take(self.coordinates, ndx, axis=1).squeeze()
        self.model.coordinates = coordinates
        return self.optimizer.optimize()

    def hologram(self):
        self.model.coordinates = self.coordinates
        return self.model.hologram().reshape(self.data.shape)

    def residuals(self):
        return self.hologram() - self.data


def example():
    import os
    import cv2
    from time import perf_counter

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    path = (THIS_DIR, '..', 'docs', 'tutorials', 'crop.png')
    TEST_IMAGE = os.path.join(*path)

    # Feature
    a = Feature()

    # model properties
    a.model.wavelength = 0.447
    a.model.magnification = 0.048
    a.model.n_m = 1.34

    # pixel selection mask
    a.mask.distribution='radial'
    a.mask.percentpix=0.1

    # Normalized image data
    data = cv2.imread(TEST_IMAGE)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY).astype(np.float)
    data /= np.mean(data)
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
