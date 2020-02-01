#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
from pylorenzmie.fitting import Optimizer
from pylorenzmie.theory import LMHologram


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
        IMPORTANT: Supported models decide whether or not to initialize
                   optimizers based on config files! See
                   pylorenzmie/fitting/.LMHologram and
                   pylorenzmie/theory/.LMHologram
                   for examples.


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

    def __init__(self, model=None, data=None, info=None):
        # Set fields
        self._optimizer = None
        self._model = None
        # Run setters
        self.data = data
        if model is not None:
            self.model = model
        # Deserialize if needed
        self.deserialize(info)

    #
    # Fields for user to set data and model's initial guesses
    #
    @property
    def data(self):
        '''Values of the (normalized) data at each pixel'''
        if type(self._data) is np.ndarray:
            data = self._data.reshape(self._shape)
        else:
            data = self._data
        return data

    @data.setter
    def data(self, data):
        if type(data) is np.ndarray:
            self._shape = data.shape
            data = data.flatten()
            if type(self.optimizer) is Optimizer:
                self.optimizer.data = data
        self._data = data

    @property
    def model(self):
        '''Model for hologram formation'''
        return self._model

    @model.setter
    def model(self, model):
        if model is not None:
            try:
                path = '/'.join(__file__.split('/')[:-1])
                fn = '.'+str(type(model)).split('.')[-1][:-2]
                with open(os.path.join(path, fn), 'r') as f:
                    d = json.load(f)
                optimize = d['optimize']
            except Exception:
                optimize = False
            if optimize:
                if self._optimizer is None:
                    self.optimizer = Optimizer(model)
                else:
                    self.optimizer.model = model
                if self.data is not None:
                    self.optimizer.data = self._data
        self._model = model

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def residuals(self):
        '''Returns difference bewteen data and current model

        Returns
        -------
        residuals : numpy.ndarray
            Difference between model and data at each pixel
        '''
        return self.model.hologram().reshape(self._shape) - self.data

    def optimize(self, **kwargs):
        '''
        Fit Model to data. See pylorenzmie.fitting.Optimizer
        for documentation
        '''
        return self.optimizer.optimize(**kwargs)

    def serialize(self, filename=None, exclude=[]):
        '''
        Serialization: Save state of Feature in dict

        Arguments
        ---------
        filename: str
            If provided, write data to file. filename should
            end in .json
        exclude : list of keys
            A list of keys to exclude from serialization.
            If no variables are excluded, then by default,
            data, coordinates, noise, and all instrument +
            particle properties) are serialized.
        Returns
        -------
        dict: serialized data

        NOTE: For a shallow serialization (i.e. for graphing/plotting),
              use exclude = ['data', 'coordinates']
        '''
        info = {}
        # Data
        if self.data is not None:
            if 'data' not in exclude:
                info['data'] = self.data.tolist()
            else:
                info['data'] = None
        # Model type
        if self.model is not None:
            model = str(type(self.model)).split('.')[-1][:-2]
            info['model'] = model
        # Coordinates
        if self.model.coordinates is None:
            shape = None
            corner = None
        else:
            coor = self.model.coordinates
            shape = (int(coor[0][-1] - coor[0][0])+1,
                     int(coor[1][-1] - coor[1][0])+1)
            corner = (int(coor[0][0]), int(coor[1][0]))
            info['coordinates'] = (shape, corner)
        # Add reduced chi-squared
        if self.optimizer is not None:
            if self.optimizer.result is not None:
                redchi = self.optimizer.result.redchi
                info['redchi'] = redchi
        else:
            redchi = None
        # Exclude things, if provided
        keys = self.model.properties.keys()
        for ex in exclude:
            if ex in keys:
                keys.pop(ex)
            elif ex in info.keys():
                info.pop(ex)
            else:
                print(ex + " not found in Feature's keylist")
        # Combine dictionaries + finish serialization
        out = self.model.properties
        out.update(info)
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(out, f)
        return out

    def deserialize(self, info):
        '''
        Restore serialized state of Feature from dict

        Arguments
        ---------
        info: dict | str
            Restore keyword/value pairs from dict.
            Alternatively restore dict from named file.
        '''
        if info is None:
            return
        if isinstance(info, str):
            with open(info, 'rb') as f:
                info = json.load(f)
        if 'model' in info.keys():
            if info['model'] == 'LMHologram':
                self.model = LMHologram()
                self.model.properties = {k: info[k] for k in
                                         self.model.properties.keys()}
        if 'coordinates' in info.keys():
            if hasattr(self.model, 'coordinates'):
                args = info['coordinates']
                self.model.coordinates = coordinates(*args)
        if 'data' in info.keys():
            data = np.array(info['data'])
            if 'shape' in info.keys():
                data = data.reshape(info['shape'])
            self.data = data


if __name__ == '__main__':
    from pylorenzmie.theory import coordinates
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
    a.model.coordinates = coordinates(shape)
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
    #a.model.using_cuda = False
    #a.model.double_precision = False
    # init dummy hologram for proper speed gauge
    a.model.hologram()
    a.optimizer.mask.settings['distribution'] = 'donut'
    a.optimizer.mask.settings['percentpix'] = .1
    # a.amoeba_settings.options['maxevals'] = 1
    # ... and now fit
    start = time()
    result = a.optimize(method='amoeba-lm', nfits=1)
    print("Time to fit: {:03f}".format(time() - start))
    print(result)

    # plot residuals
    resid = a.residuals().reshape(shape)
    hol = a.model.hologram().reshape(shape)
    data = a.data.reshape(shape)
    plt.imshow(np.hstack([hol, data, resid+1]), cmap='gray')
    plt.show()

    # plot mask
    plt.imshow(data, cmap='gray')
    a.optimizer.mask.draw_mask()

    # test serialization
    out = a.serialize()
    f = Feature()
    f.deserialize(out)
    f.optimize()
