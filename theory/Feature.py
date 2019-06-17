#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/group/endtoend/OOe2e/')
try:
    from pylorenzmie.theory.CudaLMHologram import CudaLMHologram as Model
    print("CudaLMHologram loaded.")
except Exception:
    from pylorenzmie.theory.LMHologram import LMHologram as Model
from lmfit import Parameters, Minimizer
import pickle


'''
callback function for timing out of least-squares fit

if the result is not found after stopiter # of iterations, the fit stops,
and the parameters have the value from the last iteration.
'''
def timeoutwrapper(stopiter=1000):
    def timeoutCB(params, iter, resid, *args, **kws):
        if iter >= stopiter:
            return True
        else:
            return False
    return timeoutCB


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
                 info=None,
                 **kwargs):
        self.model = Model(**kwargs)
        self.data = data
        self.fixed = ['k_p']
        self.noise = noise
        self.coordinates = self.model.coordinates
        self._keys = ('x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'k_p')
        self._minimizer = Minimizer(self._loss, None)
        self._minimizer.nan_policy = 'omit'
        self.deserialize(info)

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

    @property
    def fixed(self):
        '''Parameters to fix during optimization'''
        return self._fixed

    @fixed.setter
    def fixed(self, fixed):
        self._fixed = fixed

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

    def optimize(self,
                 diag = [1.e-4, 1.e-4, 1.e-3, 1.e-4, 1.e-5, 1.e-7],
                 ftol = 1.e-3,
                 xtol = 1.e-6,
                 epsfcn = 1.e-5,
                 stopiter = None):
        '''Fit Model to data

        Returns
        -------
        results : lmfit.MinimzerResult
            Comprehensive report on the outcome of fitting the
            Model to the provided data.  The format is described
            in the documentation for lmfit.
        '''
        if stopiter is not None:
            self._minimizer.iter_cb = timeoutwrapper(stopiter=stopiter)
        params = Parameters()
        particle = self.model.particle
        for key in self._keys:
            params.add(key, getattr(particle, key))
        for key in self.fixed:
            params[key].vary = False
        self._minimizer.params = params
        return self._minimizer.minimize(diag=diag, ftol=ftol, xtol=xtol, epsfcn=epsfcn)

    def serialize(self, filename=None):
        '''Save state of Feature

        Arguments
        ---------
        filename: str
            If provided, write data to file

        Returns
        -------
        dict: serialized data
        '''
        info = {'data': self.data,
                'coordinates': self.coordinates,
                'noise': self.noise}
        if filename is not None:
            with open(filename, 'wb') as f:
                pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
        return info

    def deserialize(self, info):
        '''Restore serialized state of Feature

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
                info = pickle.load(f)
        for key in info:
            if hasattr(self, key):
                setattr(self, key, info[key])

if __name__ == '__main__':
    from Instrument import coordinates
    import numpy as np
    from lmfit import report_fit
    import matplotlib.pyplot as plt
    from time import time

    a = Feature()
    # Use model to generate synthetic data
    shape = [201, 201]
    a.model.coordinates = coordinates(shape)
    ins = a.model.instrument
    ins.wavelength=0.447
    ins.magnification=0.048
    ins.n_m=1.34
    p = a.model.particle
    p.r_p = [100, 100, 252]
    p.a_p = 1.3
    p.n_p = 1.44
    h = a.model.hologram()
    h += np.random.normal(0., 0.05, h.size)
    a.data = h
    #plt.imshow(a.data.reshape(shape), cmap='gray')
    #plt.show()

    # add errors to parameters
    p.r_p += np.random.normal(0., 1, 3)
    p.z_p += np.random.normal(0., 5, 1)
    p.a_p += np.random.normal(0., 0.3, 1)
    p.n_p += np.random.normal(0., 0.1, 1)
    print(p)
    # ... and now fit
    start = time()
    result = a.optimize()
    print("Time to fit: {:03f}".format(time() - start))
    report_fit(result)
    # plot residuals
    plt.imshow(a.residuals().reshape(shape), cmap='gray')
    plt.show()
