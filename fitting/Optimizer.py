#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json

from scipy.optimize import least_squares
from . import amoeba

from .Settings import FitSettings, FitResult
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
    data : numpy.ndarray
        [npts] normalized intensity values
    noise : float
        Estimate for the additive noise value at each data pixel
    model : LMHologram
        Incorporates information about the Particle and the Instrument
        and uses this information to compute a hologram at the
        specified coordinates.  Keywords for the Model can be
        provided at initialization.
    vary : dict of booleans
        Allows user to select whether or not to vary parameter
        during fitting. True means the parameter will vary.
        Setting FitSettings.parameters.vary manually will not
        work.
    nm_settings : FitSettings
        Settings for nelder-mead optimization. Refer to minimizers.py
        or cminimizers.pyx -> amoeba and Settings.py -> FitSettings
        for documentation.
    lm_settings : FitSettings
        Settings for Levenberg-Marquardt optimization. Refer to
        scipy.optimize.least_squares and Settings.py -> FitSettings
        for documentation.
    mask : Mask
        Controls sampling scheme for random subset fitting.
        Refer to pylorenzmie/fitting/Mask.py for documentation.

    Methods
    -------
    optimize() : FitResult
        Optimize the Model to fit the data. A FitResult is
        returned and can be printed for a comprehensive report,
        which is also reflected in updates to the properties of
        the Model.
    '''

    def __init__(self, model, data=None, noise=0.05, config=None):
        self.params = tuple(model.properties.keys())
        self.model = model
        self.mask = Mask(model.coordinates)
        self.nm_settings = FitSettings(self.params)
        self.lm_settings = FitSettings(self.params)
        if type(config) == str:
            self.load(config)
        else:
            self._default_settings()
            
        self.data = data
        self.noise = noise
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
    def optimize(self, method='amoeba', robust=False):
        '''
        Fit Model to data

        Keywords
        ---------
        method : str
            Optimization method.
            'lm': scipy.least_squares
            'amoeba' : Nelder-Mead optimization from pylorenzmie.fitting
            'amoeba-lm': Nelder-Mead/Levenberg-Marquardt hybrid
        robust : bool
            If True, attempt to minimize the absolute error.

        For Levenberg-Marquardt fitting, see arguments for
        scipy.optimize.least_squares()
        For Nelder-Mead fitting, see arguments for amoeba either in
        pylorenzmie/fitting/minimizers.py or
        pylorenzmie/fitting/cython/cminimizers.pyx.

        Returns
        -------
        result : FitResult
            Stores useful information about the fit. It also has this
            nice quirk where if it's printed, it gives a nice-looking
            fit report. For further description, see documentation for
            FitResult in pylorenzmie.fitting.Settings.py.
        '''
        # Get array of pixels to sample
        self.mask.coordinates = self.model.coordinates
        self.model.coordinates = self.mask.masked_coords()
        self._subset_data = self._data[self.mask.index]

        # Perform fit
        p0 = self._initial_estimates()
        result, options = self._optimize(method, p0, robust=robust)

        # Restore original coordinates
        self.model.coordinates = self.mask.coordinates
        
        # Store last result
        if method == 'amoeba':
            settings = self.nm_settings
        else:
            settings = self.lm_settings
        npix = self.model.coordinates.shape[1]
        result = FitResult(method, result, settings, self.model, npix)

        if not result.success or (result.redchi > 100.):
            logger.info('Optimization did not succeed')
            avg = self._subset_data.mean()
            if not np.isclose(avg, 1., rtol=0, atol=0.1):
                msg = 'Mean of data ({:.02f}) should be near 1.'
                logger.info(msg.format(avg))
            nbad = len(self.mask.exclude)
            if self.mask.distribution == 'fast':
                nbad = len(self.mask.exclude)
                msg = 'Fit included {} potentially bad pixels.'
                logger.info(msg.format(nbad))

        self.result = result
        return result

    def dump(self, filename=None):
        '''
        Saves current fit settings for Optimizer.
        '''
        settings = dict()
        settings['lm'] = self.lm_settings.settings
        settings['nm'] = self.nm_settings.settings
        settings['vary'] = self.vary
        fn = filename or 'Optimizer.json'
        with open(fn, 'w') as f:
            json.dump(settings, f)

    def load(self, filename=None):
        '''
        Configure Optimizer settings from Optimizer.dump
        output.
        '''
        fn = filename or 'Optimizer.json'
        with open(fn, 'rb') as f:
            settings = json.load(f)
        self.lm_settings.settings = settings['lm']
        self.nm_settings.settings = settings['nm']
        self.vary = settings['vary']

    #
    # Private methods
    #
    def _default_settings(self):
        x_scale = [10000.0, 10000.0, 1000.0, 10000.0, 100000.0,
                   10000000.0, 100.0, 100.0, 100.0, 1]
        settings = {'method': 'lm', 'xtol': 1e-06, 'ftol': 0.001,
                    'gtol': 1e-06, 'max_nfev': 2000, 'diff_step': 1e-05,
                    'verbose': 0, 'x_scale': x_scale}
        self.lm_settings.settings = settings

        simplex_scale = [4.0, 4.0, 5.0, 0.01, 0.01,
                         0.2, 0.1, 0.1, 0.05, 0.05]
        xtol = [0.1, 0.1, 0.01, 0.001, 0.001,
                0.001, 0.01, 0.01, 0.01, 0.01]
        xmin = [-np.inf, -np.inf, 0.0, 0.05, 1.0, 0.0, 1.0, 0.1, 0.0, 0.0]
        xmax = [np.inf, np.inf, 2000.0, 4.0, 3.0, 3.0, 3.0, 2.0, 1.0, 5.0]
        settings = {'ftol': 0.001, 'maxevals': 800,
                    'simplex_scale': simplex_scale,
                    'xtol': xtol, 'xmin': xmin, 'xmax': xmax}
        self.nm_settings.settings = settings

        self.vary = {p: True for p in self.params}
        for p in ['k_p', 'n_m', 'alpha', 'wavelength', 'magnification']:
            self.vary[p] = False

    def _initial_estimates(self):
        p0 = []
        for p in self.params:
            value = self.model.properties[p]
            self.lm_settings.parameters[p].initial = value
            self.nm_settings.parameters[p].initial = value
            if self.vary[p]:
                p0.append(value)
        return np.array(p0)
    
    def _optimize(self, method, p0, robust=False):
        '''Perform optimization'''
        options = {}
        vary = self.vary
        nmkwargs = self.nm_settings.getkwargs(vary)
        lmkwargs = self.lm_settings.getkwargs(vary)
        converged = False
        if 'amoeba' in method:
            objective = self._absolute if robust else self._chisq
            result = amoeba(objective, p0, **nmkwargs)
            converged = result.success
        if method == 'amoeba-lm':
            options['nmresult'] = result
            if converged:
                p0 = result.x
        if 'lm' in method:
            result = least_squares(self._residuals, p0, **lmkwargs)
        return result, options

    def _residuals(self, values):
        '''Updates properties and returns residuals'''
        variables = [p for p in self.params if self.vary[p]]
        self.model.properties = dict(zip(variables, values))
        return (self.model.hologram() - self._subset_data) / self.noise

    def _chisq(self, x):
        delta = self._residuals(x)
        chisq = delta.dot(delta)
        return chisq

    def _absolute(self, x):
        delta = self._residuals(x)
        return np.absolute(delta).sum()

   
