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
        # Initialize properties
        self.params = tuple(model.properties.keys())
        # Set model and fitting equipment
        self.model = model
        self.mask = Mask(model.coordinates)
        self.nm_settings = FitSettings(self.params)
        self.lm_settings = FitSettings(self.params)
        if type(config) == str:
            self.load(config)
        else:
            params = self.params
            self.vary = dict(zip(params, len(params)*[True]))
        # Set fields
        self._shape = None
        self.data = data
        self.noise = noise
        self.result = None

    @property
    def data(self):
        '''Values of the (normalized) data at each pixel'''
        return self._data.reshape(self._shape)

    @data.setter
    def data(self, data):
        if type(data) is np.ndarray:
            print(type(data))
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
    def optimize(self, method='amoeba', robust=False, verbose=False):
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
        verbose : bool
            If True, print verbose warning messages

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
        npix = self.model.coordinates.shape[1]
        # Prepare
        x0 = self._prepare(method, verbose=verbose)
        # Check mean of data
        if verbose:
            avg = self._subset_data.mean()
            if not np.isclose(avg, 1., rtol=0, atol=.1):
                msg = ('Mean of data ({:.02f}) is not near 1. '
                       'Fit may not converge.')
                logger.warning(msg.format(avg))
        # Fit
        result, options = self._optimize(method, x0,
                                         robust=robust, verbose=verbose)
        # Post-fit cleanup
        result, settings = self._cleanup(method, result, options=options)
        # Reassign original coordinates
        self.model.coordinates = self.mask.coordinates
        # Store last result
        result = FitResult(method, result, settings, self.model, npix)
        self.result = result

        return result

    def dump(self, fn=None):
        '''
        Saves current fit settings for Optimizer.
        '''
        settings = {}
        settings['lm'] = self.lm_settings.settings
        settings['nm'] = self.nm_settings.settings
        settings['vary'] = self.vary
        if fn is None:
            return settings
        else:
            with open(fn, 'w') as f:
                json.dump(settings, f)

    def load(self, fn=None):
        '''
        Configure Optimizer settings from Optimizer.dump
        output.
        '''
        if type(fn) is str:
            with open(fn, 'rb') as f:
                settings = json.load(f)
            self.lm_settings.settings = settings['lm']
            self.nm_settings.settings = settings['nm']
            self.vary = settings['vary']

    #
    # Under the hood optimization helper functions
    #
    def _optimize(self, method, x0, robust=False, verbose=False):
        options = {}
        vary = self.vary
        nmkwargs = self.nm_settings.getkwargs(vary)
        lmkwargs = self.lm_settings.getkwargs(vary)
        converged = False
        if 'amoeba' in method:
            objective = self._absolute if robust else self._chisq
            result = amoeba(objective, x0, **nmkwargs)
            converged = result.success
        if method == 'lm-amoeba':
            options['nmresult'] = result
            if converged:
                x0 = nmresult.x
        if 'lm' in method:
            result = least_squares(self._residuals, x0, **lmkwargs)
        return result, options

    #
    # Under the hood objective function and its helpers
    #
    def _objective(self, reduce=False, square=True):
        holo = self.model.hologram()
        data = self._subset_data
        noise = self.noise
        obj = (holo - data) / noise
        if reduce:
            if square:
                obj = obj.dot(obj)
            else:
                obj = np.absolute(obj).sum()
        return obj

    def _residuals(self, x):
        '''Updates properties and returns residuals'''
        vary = [p for p in self.params if self.vary[p]]
        self.model.properties = dict(zip(vary, x))
        return (self.model.hologram() - self._subset_data) / self.noise

    def _chisq(self, x):
        delta = self._residuals(x)
        chisq = delta.dot(delta)
        return chisq

    def _absolute(self, x):
        delta = self._residuals(x)
        return np.absolute(delta).sum()

    def _prepare(self, method, verbose=False):
        # Warnings
        if (self.mask.distribution == 'fast'):
            nbad = len(self.mask.exclude)
            if nbad > 10:
                msg = 'Including {} saturated pixels'
                logger.warning(msg.format(nbad))
        # Get initial guess for fit
        x0 = []
        for p in self.params:
            val = self.model.properties[p]
            self.lm_settings.parameters[p].initial = val
            self.nm_settings.parameters[p].initial = val
            if self.vary[p]:
                x0.append(val)
        x0 = np.array(x0)
        self._subset_data = self._data[self.mask.index]
        return x0

    def _cleanup(self, method, result, options=None):
        if method == 'amoeba-lm':
            result.nfev += options['nmresult'].nfev
            settings = self.lm_settings
        elif method == 'amoeba':
            settings = self.nm_settings
        else:
            settings = self.lm_settings
        return result, settings

    
