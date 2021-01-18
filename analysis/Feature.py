#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
from pylorenzmie.fitting import Optimizer
from pylorenzmie.theory import (LMHologram, coordinates)
from . import Mask


class Feature(object):
    '''
    Abstraction of a feature in an in-line hologram

    ...

    Attributes
    ----------
    data : numpy.ndarray
        [npts] normalized intensity values
    model : [LMHologram, ]
        Incorporates information about the Particle and the Instrument
        and for supported models, uses this information to compute a
        hologram at the specified coordinates.
    optimizer : Optimizer
        Optimization equipment for fitting holographic models to data.
        Supported models decide whether or not to initialize
        optimizers based on config file, which is obtained with
        Optimizer.dump. See pylorenzmie/analysis/LMHologram.json for example.

    Methods
    -------
    residuals() : numpy.ndarray
        Difference between the current model and the data,
        normalized by the noise estimate.
    optimize() : FitResult
        Optimize the Model to fit the data. A FitResult is
        returned and can be printed for a comprehensive report,
        which is also reflected in updates to the properties of
        the Model.
    serialize() : dict
        Serialize select attributes and properties of Feature to a dict.
    deserialize(info) : None
        Restore select attributes and properties to a Feature from a dict.

    '''

    def __init__(self,
                 optimizer=None,
                 data=None,
                 coordinates=None,
                 **kwargs):

        self.optimizer = optimizer or Optimizer(**kwargs)
        self.mask = Mask(**kwargs)
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
        opt.data = self.data.flatten()[mask]
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
    from pylorenzmie.theory import coordinates
    #from pylorenzmie.theory.cuholo import cucoordinates as coordinates
    import cv2
    import matplotlib.pyplot as plt
    from time import time

    a = Feature(model=LMHologram())

    # Read example image
    img = cv2.imread('../tutorials/crop.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / np.mean(img)
    shape = img.shape
    a.data = img

    # Instrument configuration
    a.model.coordinates = coordinates(shape, dtype=np.float32)
    ins = a.model.instrument
    ins.wavelength = 0.447
    ins.magnification = 0.048
    ins.n_m = 1.34

    # Initial estimates for particle properties
    p = a.model.particle
    p.r_p = [shape[0]//2, shape[1]//2, 330.]
    p.a_p = 1.1
    p.n_p = 1.4
    # add errors to parameters
    p.r_p += np.random.normal(0., 1, 3)
    p.z_p += np.random.normal(0., 30, 1)
    p.a_p += np.random.normal(0., 0.1, 1)
    p.n_p += np.random.normal(0., 0.04, 1)
    print("Initial guess:\n{}".format(p))
    a.model.double_precision = False
    # init dummy hologram for proper speed gauge
    a.model.hologram()
    a.optimizer.mask.settings['distribution'] = 'fast'
    a.optimizer.mask.settings['percentpix'] = .1
    # a.amoeba_settings.options['maxevals'] = 1
    # ... and now fit
    start = time()
    a.model.coordinates = coordinates(shape, dtype=np.float32)
    result = a.optimize(method='lm', verbose=False)
    print("Time to fit: {:03f}".format(time() - start))
    print(result)

    # classify
    a.label = {'material': 'silica'}

    # test serialization
    out = a.serialize()
    f = Feature()
    f.deserialize(out)
    f.model.double_precision = False
    f.optimizer.mask.settings['distribution'] = 'fast'
    f.optimizer.mask.settings['percentpix'] = .1
    start = time()
    f.model.coordinates = coordinates(shape, dtype=np.float32)
    f.optimize(method='lm')
    print("Time to fit (after deserialize): {:03f}".format(time() - start))

    # plot residuals
    resid = f.residuals().reshape(shape)
    hol = f.model.hologram().reshape(shape)
    data = f.data.reshape(shape)
    plt.imshow(np.hstack([hol, data, resid+1]), cmap='gray')
    plt.show()

    # plot mask
    plt.imshow(data, cmap='gray')

    f.optimizer.mask.draw_mask()
