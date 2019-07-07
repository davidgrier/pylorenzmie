#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pickle
import numpy as np
from lmfit import Parameters, Minimizer
from pylorenzmie.fitting.minimizers import nelder_mead
from pylorenzmie.theory.LMHologram import LMHologram as Model
try:
    import cupy as cp
    cp.cuda.Device()
except(ImportError, cp.cuda.runtime.CUDARuntimeError):
    cp = None

sys.path.append('/home/group/endtoend/OOe2e/')


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
    parameterBounds : dict of tuples
        Allows user to select range over which fitting parameters
        may vary. Set each entry as a tuple of format (min, max)
    parameterVary : dict of booleans
        Allows user to select whether or not to vary parameter
        during fitting. True means the parameter will vary.
    

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
        self.noise = noise
        self.coordinates = self.model.coordinates
        # Create minimizer and set settings
        self._minimizer = Minimizer(self._loss, None)
        self._minimizer.nan_policy = 'omit'
        # Initialize options for fitting
        self._properties = list(self.model.particle.properties.keys())
        self._properties.extend(list(self.model.instrument.properties.keys()))
        self._properties = tuple(self._properties)
        N = range(len(self._properties))
        self.parameterBounds = dict(zip(self._properties,
                                        [(-np.inf, np.inf) for i in N]))
        self.parameterVary = dict(zip(self._properties,
                                      [True for i in N]))
        self.parameterTol = dict(zip(self._properties,
                                     [0. for i in N]))
        # Default settings
        self.parameterVary['k_p'] = False
        self.parameterVary['n_m'] = False
        self.parameterVary['wavelength'] = False
        self.parameterVary['magnification'] = False
        self.parameterTol['x_p'] = .1
        self.parameterTol['y_p'] = .1
        self.parameterTol['z_p'] = .1
        self.parameterTol['a_p'] = .005
        self.parameterTol['n_p'] = .002
        # Deserialize if needed
        self.deserialize(info)

    @property
    def data(self):
        '''Values of the (normalized) hologram at each pixel'''
        return self._data

    @data.setter
    def data(self, data):
        # Find indices where data is saturated or nan/inf
        if data is not None:
            self.saturated = np.where(data == np.max(data))[0]
            self.nan = np.append(np.where(np.isnan(data))[0], np.where(np.isinf(data))[0])
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
        return self.model.hologram() - self.data

    def optimize(self,
                 diag=[1.e-4, 1.e-4, 1.e-3, 1.e-4, 1.e-5, 1.e-7],
                 ftol=1.e-3, xtol=1.e-6, epsfcn=1.e-5, maxfev=2e3,
                 method='lm',
                 **kwargs):
        '''Fit Model to data
        Arguments
        ________
        see arguments for scipy.optimize.leastsq()
        method  : str
            Optimization method. Use 'lm' for Levenberg-Marquardt,
            'nelder' for Nelder-Mead, and 'custom' to pass custom
            kwargs into lmfit's Minimizer.minimize.
        
        Returns
        -------
        results : lmfit.MinimzerResult
            Comprehensive report on the outcome of fitting the
            Model to the provided data.  The format is described
            in the documentation for lmfit.
        '''
        params = Parameters()
        particle, instrument = self.model.particle, self.model.instrument
        for key in self._properties:
            if hasattr(particle, key):
                params.add(key, getattr(particle, key))
            else:
                params.add(key, getattr(instrument, key))
            params[key].vary = self.parameterVary[key]
            params[key].min = self.parameterBounds[key][0]
            params[key].max = self.parameterBounds[key][1]
        self._minimizer.params = params
        if method == 'lm':
            maxfev = int(maxfev)
            result = self._minimizer.minimize(diag=diag, ftol=ftol,
                                              xtol=xtol, epsfcn=epsfcn,
                                              maxfev=maxfev)
        elif method == 'nelder':
            if self.model.using_gpu:
                self._data = cp.asarray(self._data)
            result = nelder_mead(self._chisq, params,
                                 ndata=self.data.size,
                                 xtol=self.parameterTol,
                                 ftol=ftol,
                                 **kwargs)
            if self.model.using_gpu:
                self._data = cp.asnumpy(self._data)
        elif method == 'custom':
            result = self._minimizer.minimize(**kwargs)
        else:
            raise ValueError("method keyword must either be \'lm\', \'nelder\', or \'custom\'")
        return result

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

    def _loss(self, params, return_gpu=False):
        '''Updates particle properties and returns residuals'''
        particle, instrument = self.model.particle, self.model.instrument
        for key in particle.properties.keys():
            setattr(particle, key, params[key].value)
        for key in instrument.properties.keys():
            setattr(instrument, key, params[key].value)
        residuals = self._residuals(return_gpu)
        #don't fit on saturated or nan/infinite pixels
        residuals[self.saturated] = 0.
        residuals[self.nan] = 0.
        return residuals

    def _residuals(self, return_gpu):
        return (self.model.hologram(return_gpu) - self._data) / self.noise

    def _chisq(self, params):
        r = self._loss(params, self.model.using_gpu)
        chisq = r.dot(r)
        if self.model.using_gpu:
            chisq = chisq.get()
        return chisq


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
    ins.wavelength = 0.447
    ins.magnification = 0.048
    ins.n_m = 1.34
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
    p.a_p += np.random.normal(0., 0.05, 1)
    p.n_p += np.random.normal(0., 0.05, 1)
    print(p)
    # ... and now fit
    a.parameterBounds["x_p"] = (p.x_p-20., p.x_p+20)
    a.parameterBounds["y_p"] = (p.y_p-20., p.y_p+20)
    a.parameterBounds["z_p"] = (20., 1000.)
    a.parameterBounds["a_p"] = (.2, 4.)
    a.parameterBounds["n_p"] = (1., 3.)
    start = time()
    result = a.optimize(method='nelder')
    print("Time to fit: {:03f}".format(time() - start))
    report_fit(result)
    # plot residuals
    plt.imshow(a.residuals().reshape(shape), cmap='gray')
    plt.show()
