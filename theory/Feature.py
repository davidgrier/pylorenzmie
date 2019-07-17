#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pickle
import logging
import numpy as np
from lmfit import Parameters, report_fit
from lmfit.minimizer import MinimizerResult
from scipy.optimize import least_squares
from pylorenzmie.theory.Instrument import coordinates
from pylorenzmie.theory.LMHologram import LMHologram as Model
from pylorenzmie.fitting.minimizers import amoeba
try:
    import cupy as cp
except ImportError:
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
                 model=None,
                 data=None,
                 noise=0.05,
                 info=None,
                 **kwargs):
        # If using numba or CUDA accelerated model, good idea
        # to pass in model as keyword if instantiating many Features.
        self.model = Model(**kwargs) if model is None else model
        # Set fields
        self.data = data
        self.noise = noise
        self.coordinates = self.model.coordinates
        # Initialize Feature properties
        self.properties = list(self.model.particle.properties.keys())
        self.properties.extend(list(self.model.instrument.properties.keys()))
        self.properties = tuple(self.properties)
        # Set default options for fitting
        self.params = self._initParams()
        # Deserialize if needed
        self.deserialize(info)
        # Initialize a dummy hologram to start cupy and jit compile
        self.model.coordinates = coordinates((5, 5))
        self.model.hologram()
        self.model.coordinates = self.coordinates

    #
    # Fields for user to set data and model's initial guesses
    #
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
        return self.model.hologram() - self.data

    def optimize(self, method='lm', **kwargs):
        '''Fit Model to data
        Arguments
        ________
        method  : str
            Optimization method. Use 'lm' for scipy least_squares or
            'amoeba->lm' for Nelder-Mead/Levenberg-Marquardt hybrid

        For Levenberg-Marquardt fitting, see arguments for
        scipy.optimize.least_squares()
        For Nelder-Mead fitting, see arguments for amoeba in
        pylorenzmie/fitting/minimizers.py

        Returns
        -------
        result : lmfit.MinimzerResult
            Comprehensive report on the outcome of fitting the
            Model to the provided data.  The format is described
            in the documentation for lmfit.
        '''
        x0 = self._prepareFit(method)
        nmresult, lmresult = [None] * 2
        if method == 'lm':
            lmresult = least_squares(self._loss, x0, **self.lm_kwargs)
            self.lmResult = self._generateResult('lm', x0, lmresult)
            result = self.lmResult
        elif method == 'amoeba->lm':
            nmresult = amoeba(self._chisq, x0, **self.amoeba_kwargs)
            if self.model.using_gpu:
                self.data = cp.asnumpy(self._data)
            lmresult = least_squares(self._loss, nmresult.x,
                                     **self.lm_kwargs)
            self.lmResult = self._generateResult('lm', x0, lmresult,
                                                 init_values=nmresult.x)
            self.amoebaResult = self._generateResult('amoeba', x0, nmresult)
            result = self.lmResult
        else:
            raise ValueError(
                "method keyword must either be \'lm\', \'amoeba->lm\'")
        return result

    def printReport(self):
        '''
        Use lmfit's report_fit to print a nice report 
        of the currently stored fits.
        '''
        if type(self.amoebaResult) is MinimizerResult:
            report_fit(self.amoebaResult)
        if type(self.lmResult) is MinimizerResult:
            report_fit(self.lmResult)

    #
    # Methods for saving data
    #
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

    # TODO: method to save fit results

    #
    # Loss function, residuals, and chisqr for under the hood
    #
    def _loss(self, x, return_gpu=False):
        '''Updates properties and returns residuals'''
        particle, instrument = self.model.particle, self.model.instrument
        idx = 0
        for key in self.properties:
            param = self.params[key]
            if param.vary:
                if hasattr(particle, key):
                    setattr(particle, key, x[idx])
                else:
                    setattr(instrument, key, x[idx])
                setattr(param, 'value', x[idx])
                idx += 1
        residuals = self._residuals(return_gpu)
        # don't fit on saturated or nan/infinite pixels
        residuals[self.saturated] = 0.
        residuals[self.nan] = 0.
        return residuals

    def _residuals(self, return_gpu):
        return (self.model.hologram(return_gpu) - self._data) / self.noise

    def _chisq(self, x):
        r = self._loss(x, self.model.using_gpu)
        chisq = r.dot(r)
        if self.model.using_gpu:
            chisq = chisq.get()
        return chisq

    #
    # Fitting preparation and cleanup
    #
    def _initParams(self):
        '''
        Initialize default settings for levenberg-marquardt and
        nelder-mead optimization
        '''
        # Default parameters to vary, in the following order:
        # x_p, y_p, z_p [pixels], a_p [um], n_p,
        # k_p, n_m, wavelength [um], magnification [um/pixel]
        vary = [True] * 5
        vary.extend([False] * 4)
        # ... levenberg-marquardt variable scale factor
        x_scale = [1.e4, 1.e4, 1.e3, 1.e4, 1.e5, 1.e7, 1.e2, 1.e2, 1.e2]
        # ... bounds around intial guess for bounded nelder-mead
        simplex_bounds = [(-100, 100), (-100, 100), (-200, 200),
                          (-1., 1.), (-1., 1.), (-1., 1.),
                          (-1., 1.), (-1., 1.), (-.5, .5)]
        # ... scale of initial simplex
        simplex_scale = np.array([4., 4., 95., 0.48, 0.19, .2, .1, .1, .05])
        # ... tolerance for nelder-mead termination
        simplex_tol = [2., 2., 5., .005, .005, .001, .01, .01, .01]
        params = Parameters()
        for idx, prop in enumerate(self.properties):
            params.add(prop)
            params[prop].vary = vary[idx]
            params[prop].bounds = (simplex_bounds[idx][0],
                                   simplex_bounds[idx][1])
            # Custom vector properties for each lmfit Parameter
            params[prop].x_scale = x_scale[idx]
            params[prop].simplex_tol = simplex_tol[idx]
            params[prop].simplex_scale = simplex_scale[idx]
        # Keyword dictionaries for levenberg-marquardt and nelder-mead
        self.lm_kwargs = {'method': 'lm', 'xtol': 1.e-6, 'ftol': 1.e-3,
                          'gtol': 1e-6, 'max_nfev': int(2e3),
                          'diff_step': 1e-5, 'verbose': 0}
        self.amoeba_kwargs = {'initial_simplex': None,
                              'ftol': 1000., 'maxevals': int(1e3)}
        # MinimizerResults to be filled after fitting
        self.lmResult = None
        self.amoebaResult = None
        return params

    def _prepareFit(self, method):
        # Warnings
        if self.saturated.size > 1:
            msg = "Discluding {} saturated pixels from optimization."
            logger.warning(msg.format(self.saturated.size))
        # Unwrap vector keywords and initial guess
        x_scale, simplex_scale, simplex_tol, x0 = [np.empty(0)] * 4
        for key in self.properties:
            param = self.params[key]
            if hasattr(self.model.particle, key):
                val = getattr(self.model.particle, key)
            else:
                val = getattr(self.model.instrument, key)
            param.value = val
            param.init_value = val
            if param.vary:
                x_scale = np.append(x_scale, param.x_scale)
                simplex_scale = np.append(simplex_scale,
                                          param.simplex_scale)
                simplex_tol = np.append(simplex_tol,
                                        param.simplex_tol)
                x0 = np.append(x0, val)  # initial guess
        self.lm_kwargs['x_scale'] = x_scale
        self.amoeba_kwargs['simplex_scale'] = simplex_scale
        self.amoeba_kwargs['xtol'] = simplex_tol
        # Method specific actions
        if method == 'amoeba->lm':
            if self.model.using_gpu:
                self._data = cp.asarray(self._data)
        return x0

    def _generateResult(self, method, x0, sciresult, init_values=None):
        result = MinimizerResult(params=self.params,
                                 method=method,
                                 errorbars=True,
                                 nvarys=x0.size,
                                 ndata=self.data.size,
                                 nfev=sciresult.nfev,
                                 success=sciresult.success,
                                 message=sciresult.message)
        # Handle case where initial guess is amoeba result
        if type(init_values) in (np.ndarray, list):
            idx = 0
            for prop in self.properties:
                if result.params[prop].vary:
                    result.params[prop].init_value = init_values[idx]
                    idx += 1
        # Calculate statistics
        result.nfree = result.ndata - result.nvarys
        if type(sciresult.fun) == np.ndarray:
            result.residual = sciresult.fun
            result.chisqr = (result.residual).dot(result.residual)
            result.njev = sciresult.njev
        else:
            result.chisqr = sciresult.fun
        result.redchi = result.chisqr / max(1, result.nfree)
        neg2_log_likel = result.ndata + np.log(result.chisqr) \
            * result.nvarys
        result.aic = neg2_log_likel + 2 * result.nvarys
        result.bic = neg2_log_likel + np.log(result.ndata) \
            * result.nvarys
        return result


if __name__ == '__main__':
    import cv2
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
    result = a.optimize(method='amoeba->lm')
    print("Time to fit: {:03f}".format(time() - start))
    a.printReport()
    # plot residuals
    resid = a.residuals().reshape(shape)
    hol = a.model.hologram().reshape(shape)
    data = a.data.reshape(shape)
    plt.imshow(np.hstack([hol, data, resid+1]), cmap='gray')
    plt.show()
