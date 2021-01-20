#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
from pylorenzmie.fitting import Optimizer
from pylorenzmie.theory import (LMHologram, coordinates)
from .Mask import Mask


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
    model : [LMHologram, ]
        Incorporates information about the Particle and the Instrument
        and for supported models, uses this information to compute a
        hologram at the specified coordinates.
    optimizer : Optimizer
        Computational pipeline for fitting model to data.
        Properties of optimizer control the fitting procedure.

    Methods
    -------
    optimize() : pandas.Series
        Optimize adjustable parameters and return a report containing
        the optimized values and their numerical uncertainties.
        This report also can be retrieved from optimizer.report
        Raw fitting results are available from optimizer.results
    hologram() : numpy.ndarray
        Intensity value at each coordinate computed with current model.
    residuals() : numpy.ndarray
        Difference between the current model and the data,
        normalized by the noise estimate.

    '''

    def __init__(self,
                 data=None,
                 coordinates=None,
                 optimizer=None,
                 **kwargs):

        self.mask = Mask(**kwargs)
        self.optimizer = optimizer or Optimizer(**kwargs)
        self.data = data
        self.coordinates = coordinates
        
    @property
    def data(self):
        '''Values of the (normalized) data at each pixel'''
        return self._data

    @data.setter
    def data(self, data):
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
    def model(self):
        if self._optimizer is None:
            return None
        else:
            return self._optimizer.model

    @model.setter
    def model(self, model):
        self._optimizer.model = model
            
    @property
    def optimizer(self):
        '''Optimizer to refine holographic model parameters'''
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def optimize(self):
        mask = self.mask.selected
        opt = self.optimizer
        opt.data = self.data.ravel()[mask]
        # The following nasty hack is required for cupy because
        # opt.coordinates = self.coordinates[:,mask]
        # yields garbled results on GPU. Memory organization?
        ndx = np.nonzero(mask)
        opt.coordinates = np.take(self.coordinates, ndx, axis=1).squeeze()
        return self.optimizer.optimize()

    def hologram(self):
        self.optimizer.model.coordinates = self.coordinates
        return self.model.hologram().reshape(self.data.shape)

    def residuals(self):
        return self.hologram() - self.data

if __name__ == '__main__': # pragma: no cover
    import os
    import cv2
    from time import time

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_IMAGE = os.path.join(THIS_DIR, '../docs/tutorials/crop.png')

    # Feature with instrumental properties and mask properties
    a = Feature(wavelength=0.447, magnification=0.048, n_m=1.34,
                distribution='radial', percentpix=0.1)

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
    print('Initial estimates:\n{}'.format(p))

    # init dummy hologram for proper speed gauge
    b = a.model.hologram()
    start = time()
    result = a.optimize()
    delta = time() - start
    print('Refined estimates:\n{}'.format(p))
    print('Time to fit: {:.3f} s'.format(time() - start))
