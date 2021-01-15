#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json

from scipy.optimize import (least_squares, minimize)
# from . import amoeba

from scipy.linalg import svd
import pandas as pd

from .Mask import Mask

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Optimizer(object):
    '''
    Fit generative light-scattering model to data

    ...

    Properties
    ----------
    model : LMHologram
        Incorporates information about the Particle and the Instrument
        and uses this information to compute a hologram at the
        specified coordinates.  Keywords for the Model can be
        provided at initialization    
    data : numpy.ndarray
        [npts] normalized intensity values
    noise : float
        Estimate for the additive noise value at each data pixel
    method : str
        Optimization method.
        'lm': scipy.least_squares
        'amoeba' : Nelder-Mead optimization from pylorenzmie.fitting
        'amoeba-lm': Nelder-Mead/Levenberg-Marquardt hybrid

    Methods
    -------
    optimize() : pandas.Series
        Optimize the Model to fit the data.
    '''

    def __init__(self, model, data=None, noise=0.05, method=None, config=None):
        self.model = model
        self.mask = Mask(model.coordinates)
        self._default_settings()
            
        self.data = data
        self.noise = noise
        self.method = method or 'lm'
        self.result = None

    @property
    def data(self):
        '''Values of the (normalized) data at each pixel'''
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            self._data = None
            self._subset_data = None
            return
        saturated = data == np.max(data)
        nan = np.isnan(data)
        infinite = np.isinf(data)
        self.mask.exclude = np.nonzero(saturated | nan | infinite)[0]
        self._data = data
    
    @property
    def model(self):
        '''Model for hologram formation'''
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    #
    # Public methods
    #
    def optimize(self, robust=False):
        '''
        Fit Model to data

        Keywords
        --------- 
        robust : bool
            If True, attempt to minimize the absolute error.

        For Levenberg-Marquardt fitting, see arguments for
        scipy.optimize.least_squares()
        For Nelder-Mead fitting, see arguments for amoeba either in
        pylorenzmie/fitting/minimizers.py or
        pylorenzmie/fitting/cython/cminimizers.pyx.

        Returns
        -------
        result : pandas.Series
            Values, uncertainties and statistics from fit
        '''
        # Get array of pixels to sample
        self.mask.coordinates = self.model.coordinates
        self.model.coordinates = self.mask.masked_coords()
        self._subset_data = self._data[self.mask.index]
        self.npix = len(self.mask.index)

        # Perform fit
        p0 = self._initial_estimates()
        if 'amoeba' in self.method:
            objective = self._absolute if robust else self._chisq
            result = minimize(objective, p0, **self.nm_settings)
            converged = result.success
        if self.method == 'amoeba-lm':
            if converged:
                p0 = result.x
        if 'lm' in self.method:
            result = least_squares(self._residuals, p0, **self.lm_settings)

        self.result = result
        
        # Restore original coordinates
        self.model.coordinates = self.mask.coordinates
            
        #if not result.success or (result.redchi > 100.):
        #    logger.info('Optimization did not succeed')
        #    avg = self._subset_data.mean()
        #    if not np.isclose(avg, 1., rtol=0, atol=0.1):
        #        msg = 'Mean of data ({:.02f}) should be near 1.'
        #        logger.info(msg.format(avg))
        #    nbad = len(self.mask.exclude)
        #    if self.mask.distribution == 'fast':
        #        nbad = len(self.mask.exclude)
        #        msg = 'Fit included {} potentially bad pixels.'
        #        logger.info(msg.format(nbad))

        return self.report()

    def report(self):
        if self.result is None:
            return None
        a = self.variables
        b = ['d'+c for c in a]
        keys = list(sum(zip(a, b), ()))
        keys.extend(['success', 'npix', 'redchi'])

        values = self.result.x
        redchi, uncertainties = self._statistics()
        values = list(sum(zip(values, uncertainties), ()))
        values.extend([self.result.success, self.npix, redchi])

        return pd.Series(dict(zip(keys, values)))

    @property
    def properties(self):
        p = dict()
        p['lm'] = self.lm_settings
        p['nm'] = self.nm_settings
        p['fixed'] = self.fixed
        p['variables'] = self.variables
        return p

    @properties.setter
    def properties(self, p):
        self.lm_settings = p['lm']
        self.nm_settings = p['nm']
        self.fixed = p['fixed']
        self.variables = p['variables']
        
    def dumps(self, **kwargs):
        return json.dumps(self.properties, **kwargs)

    def loads(self, serial):
        self.properties = json.loads(serial)
        
    #
    # Private methods
    #
    def _default_settings(self):
        # least_squares
        settings = {'method': 'lm',    # (a)
                    'ftol': 1e-3,      # default: 1e-8
                    'xtol': 1e-6,      # default: 1e-8
                    'gtol': 1e-6,      # default: 1e-8
                    'loss': 'linear',  # (b)
                    'max_nfev': 2000,  # max function evaluations
                    'diff_step': 1e-5, # default: machine epsilon
                    'x_scale': 'jac'}  # (c)
        self.lm_settings = settings
        # NOTES:
        # (a) trf:     Trust Region Reflective
        #     dogbox:  
        #     lm:      Levenberg-Marquardt
        # (b) linear:  default: standard least squares
        #     soft_l1: robust least squares
        #     huber:   robust least squares
        #     cauchy:  strong least squares
        #     arctan:  limits maximum loss
        # (c) jac:     dynamic rescaling
        #     x_scale: specify scale for each adjustable variable

        # amoeba
        options = {'maxfev': 2000,
                   'xatol': 1e-2,
                   'fatol': 1e-2,
                   'adaptive': True}
        settings = {'method': 'Nelder-Mead',
                    'options': options}
        self.nm_settings = settings

        properties = self.model.properties
        self.fixed = ['k_p', 'n_m', 'alpha', 'wavelength', 'magnification']
        self.variables = [p for p in properties if p not in self.fixed]

    def _initial_estimates(self):
        p0 = []
        for p in self.variables:
            p0.append(self.model.properties[p])
        return np.array(p0)
    
    def _residuals(self, values):
        '''Updates properties and returns residuals'''
        self.model.properties = dict(zip(self.variables, values))
        return (self.model.hologram() - self._subset_data) / self.noise

    def _chisq(self, x):
        delta = self._residuals(x)
        chisq = delta.dot(delta)
        return chisq

    def _absolute(self, x):
        delta = self._residuals(x)
        return np.absolute(delta).sum()

    def _statistics(self):
        '''return standard uncertainties in fit parameters'''
        res = self.result
        ndeg = self.npix - res.x.size       # number of degrees of freedom
        if self.method == 'amoeba':
            redchi = res.fun / ndeg
            uncertainty = res.x * 0.        # no uncertainty estimate
        else:
            redchi = 2.*res.cost / ndeg # reduced chi-squared
            # covariance matrix
            # Moore-Penrose inverse discarding zero singular values.
            _, s, VT = svd(res.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
            s = s[s > threshold]
            VT = VT[:s.size]
            pcov = np.dot(VT.T / s**2, VT)
            uncertainty = np.sqrt(redchi * np.diag(pcov))
        return redchi, uncertainty
