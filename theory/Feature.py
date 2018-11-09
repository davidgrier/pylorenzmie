#!/usr/bin/env python
# -*- coding: utf-8 -*-

from LMHologram import LMHologram as Model
from lmfit import Parameters, Minimizer


class Feature(object):

    '''Abstraction for a feature in an in-line hologram'''

    def __init__(self,
                 data=None,
                 noise=0.05,
                 **kwargs):
        self.model = Model(**kwargs)
        self.data = data
        self.noise = noise
        self._keys = ('x_p', 'y_p', 'z_p', 'a_p', 'n_p')

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def residuals(self):
        '''Returns difference bewteen data and current model'''
        return (self.model.hologram() - self.data) / self.noise

    def _loss(self, params):
        particle = self.model.particle
        for key in self._keys:
            setattr(particle, key, params[key].value)
        return self.residuals()

    def optimize(self):
        params = Parameters()
        particle = self.model.particle
        for key in self._keys:
            params.add(key, getattr(particle, key))
        optimizer = Minimizer(self._loss, params)
        return optimizer.minimize(ftol=1e-5, xtol=1e-5)


if __name__ == '__main__':
    from Instrument import coordinates
    import numpy as np
    import matplotlib.pyplot as plt

    a = Feature()
    # Use model to generate synthetic data
    shape = [201, 201]
    a.model.coordinates = coordinates(shape)
    p = a.model.particle
    p.r_p = [100, 100, 100]
    p.a_p = 0.75
    p.n_p = 1.45
    h = a.model.hologram()
    h += np.random.normal(0., 0.05, h.size)
    a.data = h
    # add errors to parameters
    p.r_p += np.random.normal(0., 1, 3)
    p.a_p += np.random.normal(0., 0.01, 1)
    p.n_p += np.random.normal(0., 0.01, 1)
    # ... and now fit
    print(a.optimize().message)
    # plot residuals
    # plt.imshow(a.residuals().reshape(shape), cmap='gray')
    # plt.show()
