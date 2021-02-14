#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json

from scipy.optimize import (least_squares, minimize)

from scipy.linalg import svd
import pandas as pd

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Optimizer(object):
    '''
    Fit generative light-scattering model to data

    ...

    Properties
    ----------
    noise : float
        Estimate for the additive noise value at each data pixel
    method : str
        Optimization method.
        'lm': scipy.least_squares
        'amoeba' : Nelder-Mead optimization from pylorenzmie.fitting
        'amoeba-lm': Nelder-Mead/Levenberg-Marquardt hybrid
    fixed : list
        List of properties of the model that should not vary during fitting.
        Default: ['k_p', 'n_m', 'alpha', 'wavelength', 'magnification']
    variables : list
        List of properties of the model that will be optimized.
        Default: All model.properties that are not listed in fixed
    properties : dict
        Dictionary of settings for the optimizer
    result : scipy.optimize.OptimizeResult
        Set by optimize()
    report : pandas.Series
        Optimized values of the variables, together with numerical
        uncertainties

    Methods
    -------
    optimize() : pandas.Series
        Parameters that optimize model to fit the data.
    '''

    def __init__(self,
                 model=None,
                 data=None,
                 noise=0.05,
                 method=None,
                 **kwargs):
        self.model = model
        self.data = data
        self.noise = noise
        self.method = method or 'lm'
        self._result = None
        self._default_settings()

    @property
    def result(self):
        return self._result

    @property
    def report(self):
        '''Parse result into pandas.Series'''
        if self.result is None:
            return None
        a = self.variables
        b = ['d'+c for c in a]
        keys = list(sum(zip(a, b), ()))
        keys.extend(['success', 'npix', 'redchi'])

        values = self.result.x
        npix = self.data.size
        redchi, uncertainties = self._statistics()
        values = list(sum(zip(values, uncertainties), ()))
        values.extend([self.result.success, npix, redchi])
        return pd.Series(dict(zip(keys, values)))

    @property
    def metadata(self):
        metadata = {key: self.model.properties[key] for key in self.fixed}
        lm = 'lm' in self.method
        metadata['settings'] = self.lm_settings if lm else self.nm_settings
        return pd.Series(metadata)

    @property
    def properties(self):
        properties = dict(method=self.method,
                          lm_settings=self.lm_settings,
                          nm_settings=self.nm_settings,
                          noise=self.noise,
                          fixed=self.fixed,
                          variables=self.variables)
        properties.update(self.model.properties)
        return properties

    @properties.setter
    def properties(self, properties):
        self.model.properties = properties
        for property, value in properties.items():
            if hasattr(self, property):
                setattr(self, property, value)
        
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

        self._result = result
        
        #if not result.success or (result.redchi > 100.):
        #    logger.info('Optimization did not succeed')
        #    avg = self._subset_data.mean()
        #    if not np.isclose(avg, 1., rtol=0, atol=0.1):
        #        msg = 'Mean of data ({:.02f}) should be near 1.'
        #        logger.info(msg.format(avg))
        #        msg = 'Fit included {} potentially bad pixels.'
        #        logger.info(msg.format(nbad))

        return self.report
  
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
        p0 = [self.model.properties[p] for p in self.variables]
        return np.array(p0)
    
    def _residuals(self, values):
        '''Updates properties and returns residuals'''
        self.model.properties = dict(zip(self.variables, values))
        return (self.model.hologram() - self.data) / self.noise

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
        ndeg = self.data.size - res.x.size  # number of degrees of freedom
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
