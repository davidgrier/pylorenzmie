#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pickle
import logging
import numpy as np
from scipy.optimize import least_squares
from pylorenzmie.theory.Instrument import coordinates
from pylorenzmie.theory.LMHologram import LMHologram as Model
from pylorenzmie.fitting.minimizers import amoeba
from pylorenzmie.fitting.Settings import FitSettings, FitResult
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
    vary : dict of booleans
        Allows user to select whether or not to vary parameter
        during fitting. True means the parameter will vary.
    amoebaSettings : FitSettings
        Settings for nelder-mead optimization. Refer to minimizers.py
        -> amoeba and Settings.py -> FitSettings for documentation
    lmSettings : FitSettings
        Settings for levenberg-marquardt optimization. Refer to
        scipy.optimize.least_squares and Settings.py -> FitSettings
        for documentation


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
        self.params = self._init_params()
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

    def optimize(self, method='amoeba'):
        '''Fit Model to data
        Arguments
        ________
        method  : str
            Optimization method. Use 'lm' for scipy least_squares or
            'amoeba-lm' for Nelder-Mead/Levenberg-Marquardt hybrid

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
        x0 = self._prepare(method)
        vary = self.vary
        if method == 'lm':
            result = least_squares(self._loss, x0,
                                   **self.lmSettings.getkwargs(vary))
        elif method == 'amoeba':
            result = amoeba(self._chisq, x0,
                            **self.amoebaSettings.getkwargs(vary))
        elif method == 'amoeba-lm':
            nmresult = amoeba(self._chisq, x0,
                              **self.amoebaSettings.getkwargs(vary))
            self._cleanup('amoeba')
            if not nmresult.success:
                logger.warning('Nelder-Mead '+nmresult.message)
                x1 = x0
            else:
                x1 = nmresult.x
            result = least_squares(self._loss, x1,
                                   **self.lmSettings.getkwargs(vary))
            result.nfev += nmresult.nfev
        else:
            raise ValueError(
                "method keyword must either be lm, amoeba, or amoeba-lm")
        self._cleanup(method)
        return FitResult(method, result,
                         self.lmSettings, self.model, self.data.size)

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
            if self.vary[key]:
                if hasattr(particle, key):
                    setattr(particle, key, x[idx])
                else:
                    setattr(instrument, key, x[idx])
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
    def _init_params(self):
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
        simplex_bounds = [(-np.inf, np.inf), (-np.inf, np.inf),
                          (0., 2000.), (.2, 4.), (1., 3.),
                          (0., 3.), (1., 3.), (.100, 2.00), (0., 1.)]
        # ... scale of initial simplex
        simplex_scale = np.array([4., 4., 5., 0.01, 0.01, .2, .1, .1, .05])
        # ... tolerance for nelder-mead termination
        simplex_tol = [1., 1., .05, .005, .005, .001, .01, .01, .01]
        # Default options for amoeba and lm not parameter dependent
        lm_options = {'method': 'lm', 'xtol': 1.e-6, 'ftol': 1.e-3,
                      'gtol': 1e-6, 'max_nfev': int(2e3),
                      'diff_step': 1e-5, 'verbose': 0}
        amoeba_options = {'initial_simplex': None,
                          'ftol': 1000., 'maxevals': int(1e3)}
        # Initialize settings for fitting
        self.amoebaSettings = FitSettings(self.properties,
                                          options=amoeba_options)
        self.lmSettings = FitSettings(self.properties,
                                      options=lm_options)
        self.vary = dict(zip(self.properties, vary))
        for idx, prop in enumerate(self.properties):
            amparam = self.amoebaSettings.parameters[prop]
            lmparam = self.lmSettings.parameters[prop]
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
        for prop in self.properties:
            if hasattr(self.model.particle, prop):
                val = getattr(self.model.particle, prop)
            else:
                val = getattr(self.model.instrument, prop)
            self.lmSettings.parameters[prop].initial = val
            self.amoebaSettings.parameters[prop].initial = val
            if self.vary[prop]:
                x0.append(val)
        x0 = np.array(x0)
        # Method specific actions
        if method == 'amoeba-lm' or method == 'amoeba':
            if self.model.using_gpu:
                self._data = cp.asarray(self._data)
        return x0

    def _cleanup(self, method):
        if method == 'amoeba':
            if self.model.using_gpu:
                self._data = cp.asnumpy(self._data)


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
    p.r_p = [shape[0]//2, shape[1]//2, 370.]
    p.a_p = 1.1
    p.n_p = 1.4
    # add errors to parameters
    p.r_p += np.random.normal(0., 1, 3)
    p.z_p += np.random.normal(0., 10, 1)
    p.a_p += np.random.normal(0., 0.05, 1)
    p.n_p += np.random.normal(0., 0.03, 1)
    print("Initial guess:\n{}".format(p))
    # set settings
    start = time()
    # ... and now fit
    result = a.optimize(method='amoeba')
    print("Time to fit: {:03f}".format(time() - start))
    print(result)
    # plot residuals
    resid = a.residuals().reshape(shape)
    hol = a.model.hologram().reshape(shape)
    data = a.data.reshape(shape)
    plt.imshow(np.hstack([hol, data, resid+1]), cmap='gray')
    plt.show()
