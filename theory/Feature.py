#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import numpy as np
from scipy.optimize import least_squares
from pylorenzmie.theory.Instrument import coordinates
from pylorenzmie.theory.LMHologram import LMHologram as Model
from pylorenzmie.fitting.Settings import FitSettings, FitResult
from pylorenzmie.fitting.Mask import Mask
from pylorenzmie.fitting import amoeba

try:
    import cupy as cp
    from pylorenzmie.theory.cukernels \
        import curesiduals, cuchisqr, curesidualsf, cuchisqrf
except Exception:
    cp = None
try:
    from pylorenzmie.theory.fastkernels import fastresiduals, fastchisqr
except Exception:
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


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
    amoeba_settings : FitSettings
        Settings for nelder-mead optimization. Refer to minimizers.py
        -> amoeba and Settings.py -> FitSettings for documentation
    lm_settings : FitSettings
        Settings for Levenberg-Marquardt optimization. Refer to
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
    serialize() : dict
        Serialize select attributes and properties of Feature to a dict.
    deserialize(info) : None
        Restore select attributes and properties to a Feature from a dict.

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
        self.mask = Mask(self.model.coordinates)

    #
    # Fields for user to set data and model's initial guesses
    #
    @property
    def data(self):
        '''Values of the (normalized) hologram at each pixel'''
        return self._data

    @property
    def subset_data(self):
        return self._subset_data

    @data.setter
    def data(self, data):
        if type(data) is np.ndarray:
            avg = np.mean(data)
            if not np.isclose(avg, 1., rtol=0, atol=.05):
                msg = ('Mean of data ({:.02f}) is not near 1. '
                       'Fit may not converge.')
                logger.warning(msg.format(avg))
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
        ---------
        method : str
            Optimization method.
            'lm': scipy.least_squares
            'amoeba-lm': Nelder-Mead/Levenberg-Marquardt hybrid

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
        # Get array of pixels to sample
        self.mask.coordinates = self.model.coordinates
        self.mask.initialize_sample()
        self.model.coordinates = self.mask.masked_coords()

        npix = self.model.coordinates.shape[1]
        x0 = self._prepare(method)
        options = {}
        if method == 'lm':
            result = least_squares(
                self._residuals, x0,
                **self.lm_settings.getkwargs(self.vary))
        elif method == 'amoeba':
            result = amoeba(
                self._chisqr, x0,
                **self.amoeba_settings.getkwargs(self.vary))
        elif method == 'amoeba-lm':
            nmresult = amoeba(
                self._chisqr, x0,
                **self.amoeba_settings.getkwargs(self.vary))
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

        # reassign original coordinates before returning the fit
        self.model.coordinates = self.mask.coordinates

        result = self._cleanup(method, result, options=options)

        fit_result = FitResult(method, result,
                               self.lm_settings, self.model, npix)

        return fit_result

    #
    # Methods for saving data
    #
    def serialize(self, filename=None, exclude=[]):
        '''
        Serialization: Save state of Feature in dict

        Arguments
        ---------
        filename: str
            If provided, write data to file. filename should
            end in .json
        exclude : list of keys
            A list of keys to exclude from serialization.
            If no variables are excluded, then by default,
            data, coordinates, noise, and all instrument +
            particle properties) are serialized.
        Returns
        -------
        dict: serialized data

        NOTE: For a shallow serialization (i.e. for graphing/plotting),
              use exclude = ['data', 'coordinates', 'noise']
        '''
        data = self.data.tolist() if self.data is not None \
            else self.data
        coor = self.coordinates.tolist() if self.coordinates \
            is not None else self.coordinates
        info = {'data': data,  # dict for variables not in properties
                'coordinates': coor,
                'noise': self.noise}

        keys = self.properties  # Keys for variables in properties

        for ex in exclude:  # Exclude things, if provided
            if ex in keys:
                keys.pop(ex)
            elif ex in info.keys():
                info.pop(ex)
            else:
                print(ex + " not found in Feature's keylist")

        vals = []  # Next, get values for variables in properties
        for key in keys:
            if hasattr(self.model.particle, key):
                vals.append(getattr(self.model.particle, key))
            else:
                vals.append(getattr(self.model.instrument, key))

        out = dict(zip(self.properties, vals))
        out.update(info)  # Combine dictionaries + finish serialization
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(out, f)
        return out

    def deserialize(self, info):
        '''
        Restore serialized state of Feature from dict

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
                info = json.load(f)
        for key in info:
            if hasattr(self, key):
                setattr(self, key, info[key])
            elif hasattr(self.model.particle, key):
                setattr(self.model.particle, key, info[key])
            else:
                setattr(self.model.instrument, key, info[key])

    #
    # Objective functions for under the hood
    #
    def _residuals(self, x, reduce=False):
        '''Updates properties and returns residuals'''
        particle = self.model.particle
        instrument = self.model.instrument
        idx = 0
        for key in self.properties:
            if self.vary[key]:
                if hasattr(particle, key):
                    setattr(particle, key, x[idx])
                else:
                    setattr(instrument, key, x[idx])
                idx += 1
        objective = self._objective(reduce=reduce)
        return objective

    def _chisqr(self, x):
        return self._residuals(x, reduce=True)

    def _objective(self, reduce=False):
        holo = self.model.hologram(self.model.using_cuda)
        if self.model.using_cuda:
            (cuchi, curesid) = (cuchisqr, curesiduals)  \
                if self.model.double_precision else (cuchisqrf, curesidualsf)
            if reduce:
                obj = cuchi(holo, self._subset_data, self.noise)
            else:
                obj = curesid(holo, self._subset_data, self.noise)
            obj = obj.get()
        elif self.model.using_numba:
            if reduce:
                obj = fastchisqr(holo, self._subset_data, self.noise)
            else:
                obj = fastresiduals(holo, self._subset_data, self.noise)
        else:
            obj = (holo - self._subset_data) / self.noise
            if reduce:
                obj = obj.dot(obj)
        return obj

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
        simplex_tol = [.1, .1, .01, .0005, .0005, .001, .01, .01, .01]
        # Default options for amoeba and lm not parameter dependent
        lm_options = {'method': 'lm',
                      'xtol': 1.e-6,
                      'ftol': 1.e-3,
                      'gtol': 1e-6,
                      'max_nfev': 2000,
                      'diff_step': 1e-5,
                      'verbose': 0}
        amoeba_options = {'ftol': 1.e-3, 'maxevals': 1000}
        # Initialize settings for fitting
        self.amoeba_settings = FitSettings(self.properties,
                                           options=amoeba_options)
        self.lm_settings = FitSettings(self.properties,
                                       options=lm_options)
        self.vary = dict(zip(self.properties, vary))
        for idx, prop in enumerate(self.properties):
            amparam = self.amoeba_settings.parameters[prop]
            lmparam = self.lm_settings.parameters[prop]
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
            self.lm_settings.parameters[prop].initial = val
            self.amoeba_settings.parameters[prop].initial = val
            if self.vary[prop]:
                x0.append(val)
        x0 = np.array(x0)
        self._subset_data = self._data[self.mask.sampled_index]
        if self.model.using_cuda:
            dtype = float if self.model.double_precision else np.float32
            self._subset_data = cp.asarray(self._subset_data,
                                           dtype=dtype)
        return x0

    def _cleanup(self, method, result, options=None):
        if self.model.using_cuda:
            self._subset_data = cp.asnumpy(self._subset_data)
        if method == 'amoeba-lm':
            result.nfev += options['nmresult'].nfev
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
    a.data = img

    # Instrument configuration
    a.model.coordinates = coordinates(shape)
    ins = a.model.instrument
    ins.wavelength = 0.447
    ins.magnification = 0.048
    ins.n_m = 1.34

    # Initial estimates for particle properties
    p = a.model.particle
    p.r_p = [shape[0]//2, shape[1]//2, 370.]
    p.a_p = 1.1
    p.n_p = 1.4
    # add errors to parameters
    #p.r_p += np.random.normal(0., 1, 3)
    #p.z_p += np.random.normal(0., 10, 1)
    #p.a_p += np.random.normal(0., 0.05, 1)
    #p.n_p += np.random.normal(0., 0.03, 1)
    print("Initial guess:\n{}".format(p))
    # a.model.using_cuda = False
    a.model.double_precision = True
    # init dummy hologram for proper speed gauge
    a.model.hologram()
    a.mask.settings['distribution'] = 'uniform'
    a.mask.settings['percentpix'] = 1.
    # a.amoeba_settings.options['maxevals'] = 1
    # ... and now fit
    start = time()
    result = a.optimize(method='amoeba')
    print("Time to fit: {:03f}".format(time() - start))
    print(result)

    # plot residuals
    resid = a.residuals().reshape(shape)
    hol = a.model.hologram().reshape(shape)
    data = a.data.reshape(shape)
    plt.imshow(np.hstack([hol, data, resid+1]), cmap='gray')
    plt.show()

    # plot mask
    plt.imshow(data, cmap='gray')
    a.mask.draw_mask()
