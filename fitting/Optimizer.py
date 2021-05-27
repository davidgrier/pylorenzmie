# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import least_squares
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
    settings : dict
        Dictionary of settings for the optimization method
    fixed : list
        List of properties of the model that should not vary during fitting.
        Default: ['k_p', 'n_m', 'alpha', 'wavelength', 'magnification']
    variables : list
        List of properties of the model that will be optimized.
        Default: All model.properties that are not fixed
    properties : dict
        Dictionary of settings for the optimizer as a whole
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
                 settings=None,
                 fixed=None,
                 **kwargs):
        self.model = model
        self.data = data
        self.noise = noise
        self.settings = settings
        defaults = ['k_p', 'n_m', 'alpha', 'wavelength', 'magnification']
        self.fixed = fixed or defaults      
        self._result = None

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, settings):
        '''Dictionary of settings for scipy.optimize.least_squares
        
        NOTES:
        (a) method:
            trf:     Trust Region Reflective
            dogbox:  
            lm:      Levenberg-Marquardt
                     NOTE: only supports linear loss
        (b) loss
            linear:  default: standard least squares
            soft_l1: robust least squares
            huber:   robust least squares
            cauchy:  strong least squares
            arctan:  limits maximum loss
        (c) x_scale
            jac:     dynamic rescaling
            x_scale: specify scale for each adjustable variable
        '''
        if settings is None:
            settings = {'method': 'lm',    # (a)
                        'ftol': 1e-3,      # default: 1e-8
                        'xtol': 1e-6,      # default: 1e-8
                        'gtol': 1e-6,      # default: 1e-8
                        'loss': 'linear',  # (b)
                        'max_nfev': 2000,  # max function evaluations
                        'diff_step': 1e-5, # default: machine epsilon
                        'x_scale': 'jac'}  # (c)
        self._settings = settings

    @property
    def fixed(self):
        '''list of fixed properties'''
        return self._fixed

    @fixed.setter
    def fixed(self, fixed):
        self._fixed = fixed
        properties = self.model.properties
        self._variables = [p for p in properties if p not in self.fixed]

    @property
    def variables(self):
        return self._variables
        
    @property
    def result(self):
        return self._result

    @property
    def report(self):
        '''Parse result into pandas.Series'''
        if self.result is None:
            return None
        a = self.variables
        b = ['d' + c for c in a]
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
        metadata['settings'] = self.settings
        return pd.Series(metadata)

    @property
    def properties(self):
        properties = dict(settings=self.settings,
                          noise=self.noise,
                          fixed=self.fixed)
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
    def optimize(self):
        '''
        Fit Model to data

        Keywords
        --------- 
        See arguments for scipy.optimize.least_squares()

        Returns
        -------
        result : pandas.Series
            Values, uncertainties and statistics from fit
        '''
        p0 = self._initial_estimates()
        self._result = least_squares(self._residuals, p0, **self.settings)
        return self.report
    
    #
    # Private methods
    #
    def _initial_estimates(self):
        p0 = [self.model.properties[p] for p in self.variables]
        return np.array(p0)
    
    def _residuals(self, values):
        '''Updates properties and returns residuals'''
        self.model.properties = dict(zip(self.variables, values))
        return (self.model.hologram() - self.data) / self.noise

    def _statistics(self):
        '''return standard uncertainties in fit parameters'''
        res = self.result
        ndeg = self.data.size - res.x.size  # number of degrees of freedom
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
