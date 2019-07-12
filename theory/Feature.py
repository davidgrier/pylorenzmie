#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pickle
import logging
import numpy as np
from lmfit import Parameters, Minimizer
from pylorenzmie.theory.Instrument import coordinates
from pylorenzmie.theory.LMHologram import LMHologram as Model
#from pylorenzmie.fitting.minimizers import amoebas
from pylorenzmie.fitting.minimizers import amoeba
try:
    import cupy as cp
    cp.cuda.Device()
except ImportError:
    cp = None
except cp.cuda.runtime.CUDARuntimeError:
    cp = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

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
        self.properties = list(self.model.particle.properties.keys())
        self.properties.extend(list(self.model.instrument.properties.keys()))
        self.properties = tuple(self.properties)
        N = range(len(self.properties))
        self.parameterVary = dict(zip(self.properties,
                                      [True for i in N]))
        self.amoebaAbsoluteBounds = dict(zip(self.properties,
                                             [(-np.inf, np.inf) for i in N]))
        self.amoebaBounds = dict(zip(self.properties,
                                     [(-np.inf, np.inf) for i in N]))  # centered around initial guess
        self.amoebaTol = dict(zip(self.properties,
                                  [0. for i in N]))
        # Default settings
        self.parameterVary['k_p'] = False
        self.parameterVary['n_m'] = False
        self.parameterVary['wavelength'] = False
        self.parameterVary['magnification'] = False
        self.amoebaTol['x_p'] = 2.
        self.amoebaTol['y_p'] = 2.
        self.amoebaTol['z_p'] = 15.
        self.amoebaTol['a_p'] = .4
        self.amoebaTol['n_p'] = .025
        self.amoebaAbsoluteBounds['x_p'] = (-20, 20)
        self.amoebaAbsoluteBounds['y_p'] = (-20, 20)
        self.amoebaAbsoluteBounds['z_p'] = (20., 1000.)
        self.amoebaAbsoluteBounds['a_p'] = (.2, 5.)
        self.amoebaAbsoluteBounds['n_p'] = (1.32, 3.)
        self.amoebaAbsoluteBounds['k_p'] = (0.0, 5.0)
        self.amoebaAbsoluteBounds['n_m'] = (1., 2.)
        self.amoebaAbsoluteBounds['wavelength'] = (.200, 1.100)
        self.amoebaAbsoluteBounds['magnification'] = (.001, .140)
        self.amoebaBounds['x_p'] = (-10., 10.)
        self.amoebaBounds['y_p'] = (-10., 10.)
        self.amoebaBounds['z_p'] = (-100., 100.)
        self.amoebaBounds['a_p'] = (-.25, .25)
        self.amoebaBounds['n_p'] = (-.2, .2)
        self.amoebaBounds['k_p'] = (-.1, .1)
        self.amoebaBounds['n_m'] = (-.1, .1)
        self.amoebaBounds['wavelength'] = (-.3, .3)
        self.amoebaBounds['magnification'] = (-.01, .01)
        # Set default kwargs to pass to levenberg and nelder
        xscale = [1.e4, 1.e4, 1.e3, 1.e4, 1.e5, 1.e7, 1.e2, 1.e2, 1.e2]
        self.x_scale = dict(zip(self.properties,
                                xscale))
        self.lm_kwargs = {'method': 'lm',
                          'x_scale': self.x_scale,
                          'xtol': 1.e-6, 'ftol': 1.e-3,
                          'gtol': 1e-6,
                          'max_nfev': int(2e3),
                          'diff_step': 1e-5,
                          'verbose': 0}
        # simplex_scale = -np.array([4., 4., 95., 0.48, 0.19,
        #                           .2, .1, .1, .05])
        simplex_scale = np.array([10., 10., 100., 0.25, 0.2,
                                  .1, .1, .3, .01])
        self.simplex_scale = dict(zip(self.properties, simplex_scale))
        self.amoeba_kwargs = {'initial_simplex': None,
                              'simplex_scale': self.simplex_scale,
                              # 'namoebas': 1,
                              'ftol': 1e-2,
                              'xtol': self.amoebaTol,
                              'maxevals': int(1e3)}
        # Deserialize if needed
        self.deserialize(info)
        # Initialize a dummy hologram to start CuPy
        self.model.coordinates = coordinates((5, 5))
        self.model.hologram()
        self.model.coordinates = self.coordinates

    @property
    def data(self):
        '''Values of the (normalized) hologram at each pixel'''
        return self._data

    @data.setter
    def data(self, data):
        if type(data) is np.ndarray:
            avg = np.mean(data)
            if not np.isclose(avg, 1., rtol=0, atol=.05):
                msg = "Mean of data ({:.02f}) is not near 1. Fit may not converge."
                logger.warning(msg.format(avg))
            # Find indices where data is saturated or nan/inf
            self.saturated = np.where(data == np.max(data))[0]
            self.nan = np.append(np.where(np.isnan(data))[0],
                                 np.where(np.isinf(data))[0])
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

    def optimize(self, method='lm', **kwargs):
        '''Fit Model to data
        Arguments
        ________
        method  : str
            Optimization method. Use 'lm' for Levenberg-Marquardt,
            'amoeba-lm' for Nelder-Mead/Levenberg-Marquardt hybrid,
            and 'custom' to pass custom kwargs into lmfit's 
            Minimizer.minimize.

        For Levenberg-Marquardt fitting, see arguments for
        scipy.optimize.least_squares()
        For Nelder-Mead fitting, see arguments for amoebas in
        pylorenzmie/fitting/minimizers.py

        Returns
        -------
        results : lmfit.MinimzerResult
            Comprehensive report on the outcome of fitting the
            Model to the provided data.  The format is described
            in the documentation for lmfit.
        '''
        if self.saturated.size > 1:
            msg = "Discluding {} saturated pixels from optimization."
            logger.warning(msg.format(self.saturated.size))
        params = Parameters()
        particle, instrument = self.model.particle, self.model.instrument
        x_scale = []
        for key in self.properties:
            if hasattr(particle, key):
                params.add(key, getattr(particle, key))
            else:
                params.add(key, getattr(instrument, key))
            params[key].vary = self.parameterVary[key]
            if self.parameterVary[key]:
                x_scale.append(self.x_scale[key])
        self.lm_kwargs['x_scale'] = x_scale
        self._minimizer.params = params
        if method == 'lm':
            result = self._minimizer.least_squares(**self.lm_kwargs)
        elif method == 'amoeba-lm':
            result = self._amoebaLM(params, **self.amoeba_kwargs)
        elif method == 'custom':
            result = self._minimizer.minimize(**kwargs)
        else:
            raise ValueError(
                "method keyword must either be \'lm\', \'amoeba-lm\', or \'custom\'")
        self.lm_kwargs['x_scale'] = self.x_scale
        if not result.success:
            msg = "Fit did not converge. Max number of function evaluations exceeded"
            logging.warning(msg)
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
        # don't fit on saturated or nan/infinite pixels
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

    def _amoebaLM(self, params, **kwargs):
        bounds = {}
        for param in params:
            if param in self.model.particle.properties:
                attr = getattr(self.model.particle, param)
            elif param in self.model.instrument.properties:
                attr = getattr(self.model.instrument, param)
            if param == 'x_p' or param == 'y_p':
                # params[param].min = attr + \
                mini = attr + \
                    max(self.amoebaBounds[param][0],
                        self.amoebaAbsoluteBounds[param][0])
                # params[param].max = attr + \
                maxi = attr + \
                    min(self.amoebaBounds[param][1],
                        self.amoebaAbsoluteBounds[param][1])
            else:
                # params[param].min = max(
                mini = max(
                    attr+self.amoebaBounds[param][0],
                    self.amoebaAbsoluteBounds[param][0])
                # params[param].max = min(
                maxi = min(
                    attr+self.amoebaBounds[param][1],
                    self.amoebaAbsoluteBounds[param][1])
            bounds[param] = mini, maxi
        if self.model.using_gpu:
            self._data = cp.asarray(self._data)
        #resultNM = amoebas(self._chisq, params, **kwargs)
        resultNM = amoeba(self._chisq, params, bounds, **kwargs)
        resultNM.ndata = self.data.size
        resultNM.redchi = resultNM.chisqr / (resultNM.ndata-resultNM.nvarys)
        if self.model.using_gpu:
            self._data = cp.asnumpy(self._data)
        for param in params:
            resultNM.params[param].max = np.inf
            resultNM.params[param].min = -np.inf
        self._minimizer.params = resultNM.params
        result = self._minimizer.least_squares(**self.lm_kwargs)
        result.method = 'Nelder-Mead/least_squares hybrid'
        result.nfev = '{}+{}'.format(resultNM.nfev, result.nfev)
        #result.amoeba_chi = resultNM.chis / result.nfree
        return result


if __name__ == '__main__':
    from Instrument import coordinates
    import cv2
    from lmfit import report_fit
    import matplotlib.pyplot as plt
    from time import time

    a = Feature()
    # Read example image
    img = cv2.imread('../tutorials/image0400.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / np.mean(img)
    shape = img.shape
    img = np.array([item for sublist in img for item in sublist])
    plt.show()
    a.data = img
    # Set instrument and particle initial guesses
    a.model.coordinates = coordinates(shape)
    ins = a.model.instrument
    ins.wavelength = 0.447
    ins.magnification = 0.048
    ins.n_m = 1.34
    p = a.model.particle
    p.r_p = [shape[0]//2, shape[1]//2, 375.]
    p.a_p = 1.11
    p.n_p = 1.4
    # add errors to parameters
    p.r_p += np.random.normal(0., 1, 3)
    p.z_p += np.random.normal(0., 20, 1)
    p.a_p += np.random.normal(0., 0.05, 1)
    p.n_p += np.random.normal(0., 0.05, 1)
    print("Initial guess:\n{}".format(p))
    # set settings
    start = time()
    # ... and now fit
    result = a.optimize(method='amoeba-lm')
    print("Time to fit: {:03f}".format(time() - start))
    #print("Reduced chi values from Amoeba fits {}".format(result.amoeba_chi))
    report_fit(result)
    # plot residuals
    resid = a.residuals().reshape(shape)
    hol = a.model.hologram().reshape(shape)
    data = a.data.reshape(shape)
    plt.imshow(np.hstack([hol, data, resid+1]), cmap='gray')
    plt.show()
