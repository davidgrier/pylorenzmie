#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pickle
import os
import numpy as np
from pylorenzmie.fitting import Optimizer
from pylorenzmie.theory import LMHologram, coordinates


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
                   pylorenzmie/analysis/.LMHologram
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

    def __init__(self, model=None, data=None, info=None, label=None):
        # Set fields
        self._optimizer = None
        self._model = None
        self._label = None
        # Run setters
        self.data = data
        if model is not None:
            self.model = model
        self.label = label
        # Deserialize if needed
        self.deserialize(info)

    #
    # Fields for user to set data, model, optimizer, and classification label
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
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.data = data
        self._data = data

    @property
    def model(self):
        '''Model for hologram formation'''
        return self._model

    @model.setter
    def model(self, model):
        if model is not None:
            path = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(path, '../fitting')
            fn = model.__class__.__name__+'.pickle'
            config = os.path.join(path, fn)
            try:
                optimizer = pickle.load(open(config, 'rb'))
                optimizer.model = model
                if self.data is not None:
                    optimizer.data = self._data
                self.optimizer = optimizer
            except FileNotFoundError:
                self.optimizer = None
        self._model = model

    @property
    def optimizer(self):
        '''Optimizer to refine holographic model parameters'''
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def label(self):
        '''Classifies what a Feature may be'''
        return self._label

    @label.setter
    def label(self, label):
        self._label = str(label)

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
                info['data'] = self._data.tolist()
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
        # Classification label
        if self.label is not None:
            info['label'] = self.label
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
        if 'label' in info.keys():
            self.label = info['label']


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
    a.model.coordinates = a.model.coordinates.astype(np.float32)
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
    a.label = 'silica'

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
