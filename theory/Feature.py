#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import numpy as np
from pylorenzmie.fitting import Optimizer


class Feature(object):
    '''
    Abstraction of a feature in an in-line hologram

    ...

    Attributes
    ----------
    data : numpy.ndarray
        [npts] normalized intensity values
    coordinates : numpy.ndarray
        [npts, 3] array of pixel coordinates
        Note: This property is shared with the underlying Model
    model : LMHologram
        Incorporates information about the Particle and the Instrument
        and uses this information to compute a hologram at the
        specified coordinates.  Keywords for the Model can be
        provided at initialization.


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
                 model=None,
                 data=None,
                 info=None,
                 **kwargs):
        if model is not None:
            self.optimizer = Optimizer(model)
            self._model = model
        else:
            self.model, self.optimizer = (None, None)
        # Set fields
        self._shape = None
        self.data = data
        self.coordinates = self.model.coordinates
        # Initialize Feature properties
        self.params = tuple(self.model.properties.keys())
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
            self.optimizer.data = data
        self._data = data

    @property
    def model(self):
        '''Model for hologram formation'''
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.optimizer.model = model

    #
    # Methods to show residuals and optimize
    #
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

    #
    # Methods for saving data
    #
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
              use exclude = ['data', 'shape', 'corner', 'noise', 'redchi']
        '''
        coor = self.model.coordinates
        if self.data is None:
            data = None
            shape = None
            corner = None
        else:
            data = self.data.tolist()
            shape = (int(coor[0][-1] - coor[0][0])+1,
                     int(coor[1][-1] - coor[1][0])+1)
            corner = (int(coor[0][0]), int(coor[1][0]))
        # Dict for variables not in properties
        info = {'data': data,
                'shape': shape,
                'corner': corner}
        # Add reduced chi-squared
        if self.optimizer.result is not None:
            redchi = self.optimizer.result.redchi
        else:
            redchi = None
        info.update({'redchi': redchi})
        # Exclude things, if provided
        keys = self.params
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
        self.model.properties = {k: info[k] for k in
                                 self.model.properties.keys()}
        if 'shape' in info.keys():
            if 'corner' in info.keys():
                corner = info['corner']
            else:
                corner = (0, 0)
            self.model.coordinates = coordinates(info['shape'],
                                                 corner=corner)
            self.mask.coordinates = self.model.coordinates
        if 'data' in info.keys():
            self.data = np.array(info['data'])


if __name__ == '__main__':
    from pylorenzmie.theory import coordinates
    from pylorenzmie.theory import LMHologram
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
