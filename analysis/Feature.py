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
        self.estimator = Estimator(instrument=self.model.instrument)
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
            self.mask.exclude = saturated | nan | infinite
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
        # opt.coordinates = self.coordinates[:,self.mask().ravel()]
        # yields garbled results on GPU. Memory organization?
        ndx = np.nonzero(self.mask().ravel())
        coordinates = np.take(self.coordinates, ndx, axis=1).squeeze()
        self.model.coordinates = coordinates
        return self.optimizer.optimize()

    def hologram(self):
        self.model.coordinates = self.coordinates
        return self.model.hologram().reshape(self.data.shape)

    def residuals(self):
        return self.hologram() - self.data


def example():
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
