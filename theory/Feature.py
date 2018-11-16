#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylorenzmie.theory.LMHologram import LMHologram as Model
from lmfit import Parameters, Minimizer


class Feature(object):

    '''
    Abstraction of a feature in an in-line hologram

    ...

    Attributes
    ----------
    data : numpy.ndarray
        [npts] normalized intensity values
    noise : float
        Estimate for the additive noise value at each data pixel
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
    optimize() : lmfit.MinimzerResult
        Optimize the Model to fit the data.  Results are
        returned in a comprehensive report and are reflected
        in updates to the properties of the Model.
    '''

    def __init__(self,
                 data=None,
                 noise=0.05,
                 **kwargs):
        self.model = Model(**kwargs)
        self.data = data
        self.noise = noise
        self.coordinates = self.model.coordinates
        self._keys = ('x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'k_p')
        self._minimizer = Minimizer(self._loss, None)
        self._minimizer.nan_policy = 'omit'

    @property
    def data(self):
        '''Values of the (normalized) hologram at each pixel'''
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def model(self):
        '''Model for hologram formation'''
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def residuals(self):
        '''Returns difference bewteen data and current model

        Returns
        -------
        residuals : numpy.ndarray
            Difference between model and data at each pixel
        '''
        return (self.model.hologram() - self.data) / self.noise

    def _loss(self, params):
        '''Updates particle properties and returns residuals'''
        particle = self.model.particle
        for key in self._keys:
            setattr(particle, key, params[key].value)
        return self.residuals()

    def optimize(self):
        '''Fit Model to data

        Returns
        -------
        results : lmfit.MinimzerResult
            Comprehensive report on the outcome of fitting the
            Model to the provided data.  The format is described
            in the documentation for lmfit.
        '''
        params = Parameters()
        particle = self.model.particle
        for key in self._keys:
            params.add(key, getattr(particle, key))
        self._minimizer.params = params
        return self._minimizer.minimize()


if __name__ == '__main__':
    from Instrument import coordinates
    import numpy as np
    from lmfit import report_fit
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
    result = a.optimize()
    report_fit(result)
    # plot residuals
    plt.imshow(a.residuals().reshape(shape), cmap='gray')
    plt.show()
