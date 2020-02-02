#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import logging
import numpy as np
from scipy.optimize import least_squares
from pylorenzmie.fitting import amoeba

from .Settings import FitSettings, FitResult
from .Mask import Mask
from .GlobalSampler import GlobalSampler

try:
    import cupy as cp
    from pylorenzmie.fitting import cukernels as cuk
except Exception:
    cp = None
try:
    from pylorenzmie.fitting import fastkernels as fk
except Exception:
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Optimizer(object):
    '''
    Optimization equipment for fitting a holographic model
    to data.

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
    sampler : GlobalSampler
        Controls sampling scheme of new initial conditions for
        globalized optimization.


    Methods
    -------
    optimize() : FitResult
        Optimize the Model to fit the data. A FitResult is
        returned and can be printed for a comprehensive report,
        which is also reflected in updates to the properties of
        the Model.
    '''

    def __init__(self, model, data=None, noise=0.05):
        # Initialize properties
        self.params = tuple(model.properties.keys())
        # Set model
        self.model = model
        # Set fields
        self._shape = None
        self.data = data
        self.noise = noise
        self.result = None

    #
    # Fields for user to set data and model's initial guesses
    #
    @property
    def data(self):
        '''Values of the (normalized) data at each pixel'''
        if type(self._data) is np.ndarray:
            data = self._data.reshape(self._shape)
        else:
            data = self._data
        return data

    @data.setter
    def data(self, data):
        if type(data) is np.ndarray:
            # Find indices where data is saturated or nan/inf
            self.saturated = np.where(data == np.max(data))[0]
            self.nan = np.append(np.where(np.isnan(data))[0],
                                 np.where(np.isinf(data))[0])
            exclude = np.append(self.saturated, self.nan)
            self.mask.exclude = exclude
        self._data = data

    @property
    def model(self):
        '''Model for hologram formation'''
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        # Random subset sampling
        self.mask = Mask(model.coordinates)
        # Try config
        try:
            abs = os.path.abspath(__file__)
            path = '/'.join(abs.split('/')[:-1])
            fn = '.'+str(type(model)).split('.')[-1][:-2]
            with open(os.path.join(path, fn), 'r') as f:
                config = json.load(f)
            self._init_settings(config)
            # Globalized optimization sampling
            self.sampler = GlobalSampler(self, config['global'])
        except Exception:
            self.lm_settings = None
            self.nm_settings = None
            self.sampler = None

    #
    # Public methods
    #
    def optimize(self, method='amoeba', square=True, nfits=1):
        '''
        Fit Model to data

        Keywords
        ---------
        method : str
            Optimization method.
            'lm': scipy.least_squares
            'amoeba' : Nelder-Mead optimization from pylorenzmie.fitting
            'amoeba-lm': Nelder-Mead/Levenberg-Marquardt hybrid
        square : bool
            If True, 'amoeba' fitting method will minimize chi-squared.
            If False, 'amoeba' fitting method will minimize the sum of
            absolute values of the residuals. This keyword has no effect
            on 'amoeba-lm' or 'lm' methods.

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
        self.mask.initialize_sample()
        self.model.coordinates = self.mask.masked_coords()
        npix = self.model.coordinates.shape[1]
        # Prepare
        x0 = self._prepare(method)
        # Check mean of data
        avg = self._subset_data.mean()
        avg = avg.get() if self.model.using_cuda else avg
        if not np.isclose(avg, 1., rtol=0, atol=.05):
            msg = ('Mean of data ({:.02f}) is not near 1. '
                   'Fit may not converge.')
            logger.warning(msg.format(avg))
        # Fit
        if nfits > 1 and self.sampler is not None:
            result, options = self._globalize(
                method, nfits, x0, square)
        elif nfits == 1:
            result, options = self._optimize(method, x0, square)
        # Post-fit cleanup
        result, settings = self._cleanup(method, square, result, nfits,
                                         options=options)
        # Reassign original coordinates
        self.model.coordinates = self.mask.coordinates
        # Store last result
        result = FitResult(method, result, settings, self.model, npix)
        self.result = result
        return result

    #
    # Under the hood optimization helper functions
    #
    def _optimize(self, method, x0, square):
        options = {}
        if method == 'lm':
            result = least_squares(
                self._residuals, x0,
                **self.lm_settings.getkwargs(self.vary))
        elif method == 'amoeba':
            if square:
                objective = self._chisqr
            else:
                objective = self._absolute
            result = amoeba(
                objective, x0, **self.nm_settings.getkwargs(self.vary))
        elif method == 'amoeba-lm':
            nmresult = amoeba(
                self._chisqr, x0,
                **self.nm_settings.getkwargs(self.vary))
            if not nmresult.success:
                msg = 'Nelder-Mead: {}. Falling back to least squares.'
                logger.warning(msg.format(nmresult.message))
                x1 = x0
            else:
                x1 = nmresult.x
            result = least_squares(
                self._residuals, x1,
                **self.lm_settings.getkwargs(self.vary))
            options['nmresult'] = nmresult
        else:
            raise ValueError(
                "Method keyword must either be lm, amoeba, or amoeba-lm")
        return result, options

    def _globalize(self, method, nfits, x0, square):
        # Initialize for fitting iteration
        self.sampler.x0 = x0
        x1 = x0
        best_eval, best_result = (np.inf, None)
        for i in range(nfits):
            result, options = self._optimize(method, x1, square)
            # Determine if this result is better than previous
            eval = result.fun
            if type(result.fun) is np.ndarray:
                eval = (result.fun).dot(result.fun)
            if eval < best_eval:
                best_eval = eval
                best_result = (result, options)
            if i < nfits - 1:
                # Find new starting point and update distributions
                self.sampler.xfit = result.x
                x1 = self.sampler.sample()
        return best_result

    #
    # Under the hood objective function and its helpers
    #
    def _objective(self, reduce=False, square=True):
        holo = self.model.hologram(self.model.using_cuda)
        if self.model.using_cuda:
            (cuchisqr, curesiduals, cuabsolute) = (
                cuk.cuchisqr, cuk.curesiduals, cuk.cuabsolute)  \
                if self.model.double_precision else (
                cuk.cuchisqrf, cuk.curesidualsf, cuk.cuabsolutef)
            if reduce:
                if square:
                    obj = cuchisqr(holo, self._subset_data, self.noise)
                else:
                    obj = cuabsolute(holo, self._subset_data, self.noise)
            else:
                obj = curesiduals(holo, self._subset_data, self.noise)
            obj = obj.get()
        elif self.model.using_numba:
            if reduce:
                if square:
                    obj = fk.fastchisqr(
                        holo, self._subset_data, self.noise)
                else:
                    obj = fk.fastabsolute(holo, self._subset_data, self.noise)
            else:
                obj = fk.fastresiduals(holo, self._subset_data, self.noise)
        else:
            obj = (holo - self._subset_data) / self.noise
            if reduce:
                if square:
                    obj = obj.dot(obj)
                else:
                    obj = np.absolute(obj).sum()
        return obj

    def _residuals(self, x, reduce=False, square=True):
        '''Updates properties and returns residuals'''
        self._update_model(x)
        objective = self._objective(reduce=reduce, square=square)
        return objective

    def _chisqr(self, x):
        return self._residuals(x, reduce=True)

    def _absolute(self, x):
        return self._residuals(x, reduce=True, square=False)

    #
    # Fitting preparation and cleanup
    #
    def _init_settings(self, config):
        '''
        Initialize default settings for levenberg-marquardt and
        nelder-mead optimization
        '''
        cfg_lm = config['lm']
        cfg_nm = config['nm']
        # Default parameters to vary during fitting
        vary = config['vary']
        # ... levenberg-marquardt variable scale factor
        x_scale = cfg_lm['x_scale']
        # ... scale of initial simplex
        simplex_scale = cfg_nm['simplex_scale']
        # ... bounds around intial guess for bounded nelder-mead
        simplex_bounds = cfg_nm['simplex_bounds']
        # ... tolerance for nelder-mead termination
        simplex_tol = cfg_nm['simplex_tol']
        # Non-vector arguments to fitting
        lm_options = cfg_lm['options']
        nm_options = cfg_nm['options']
        self.nm_settings = FitSettings(self.params,
                                       options=nm_options)
        self.lm_settings = FitSettings(self.params,
                                       options=lm_options)
        self.vary = dict(zip(self.params, vary))
        for idx, p in enumerate(self.params):
            amparam = self.nm_settings.parameters[p]
            lmparam = self.lm_settings.parameters[p]
            amparam.options['simplex_scale'] = simplex_scale[idx]
            amparam.options['xtol'] = simplex_tol[idx]
            amparam.options['xmax'] = simplex_bounds[idx][1]
            amparam.options['xmin'] = simplex_bounds[idx][0]
            lmparam.options['x_scale'] = x_scale[idx]

    def _prepare(self, method):
        # Warnings
        if self.saturated.size > 10:
            msg = "Excluding {} saturated pixels from optimization."
            logger.warning(msg.format(self.saturated.size))
        # Get initial guess for fit
        x0 = []
        for p in self.params:
            val = self.model.properties[p]
            self.lm_settings.parameters[p].initial = val
            self.nm_settings.parameters[p].initial = val
            if self.vary[p]:
                x0.append(val)
        x0 = np.array(x0)
        self._subset_data = self._data[self.mask.sampled_index]
        if self.model.using_cuda:
            dtype = float if self.model.double_precision else np.float32
            self._subset_data = cp.asarray(self._subset_data,
                                           dtype=dtype)
        return x0

    def _cleanup(self, method, square, result, nfits, options=None):
        if nfits > 1:
            self._update_model(result.x)
        if method == 'amoeba-lm':
            result.nfev += options['nmresult'].nfev
            settings = self.lm_settings
        elif method == 'amoeba':
            if not square:
                result.fun = float(self._objective(reduce=True))
            settings = self.nm_settings
        else:
            settings = self.lm_settings
        if self.model.using_cuda:
            self._subset_data = cp.asnumpy(self._subset_data)
        return result, settings

    def _update_model(self, x):
        vary = []
        for p in self.params:
            if self.vary[p]:
                vary.append(p)
        self.model.properties = dict(zip(vary, x))
