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
    model : LMHologram
        Computational model for calculating holograms
    data : numpy.ndarray
        Target for optimization with model
    noise : float
        Estimate for the additive noise value at each data pixel
    robust : bool
        If True, use robust optimization (absolute deviations)
        otherwise use least-squares optimization
        Default: False (least-squares)
    fixed : list of str
        Names of properties of the model that should not vary during fitting.
        Default: ['k_p', 'n_m', 'alpha', 'wavelength', 'magnification']
    variables : list of str
        Names of properties of the model that will be optimized.
        Default: All model.properties that are not fixed
    settings : dict
        Dictionary of settings for the optimization method
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
                 robust=False,
                 fixed=None,
                 settings=None,                                 
                 **kwargs):
        self.model = model
        self.data = data
        self.settings = settings
        self.robust = robust
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
    def robust(self):
        return self.settings['loss'] in ['soft_l1', 'huber', 'cauchy']

    @robust.setter
    def robust(self, robust):
        if robust:
            self.settings['method'] = 'dogbox'
            self.settings['loss'] = 'cauchy'
        else:
            self.settings['method'] = 'lm'
            self.settings['loss'] = 'linear'

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
        noise = self.model.instrument.noise
        return (self.model.hologram() - self.data) / noise

    def _statistics(self):
        '''return reduced chi-squared and standard uncertainties

        Uncertainties are the square roots of the diagonal
        elements of the covariance matrix. The covariance matrix
        is obtained from the Jacobian of the fit by singular
        value decomposition, using the Moore-Penrose inverse
        after discarding small singular values.
        '''
        res = self.result
        ndeg = self.data.size - res.x.size  # number of degrees of freedom
        redchi = 2.*res.cost / ndeg         # reduced chi-squared

        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        uncertainty = np.sqrt(redchi * np.diag(pcov))
        
        return redchi, uncertainty
