#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import numpy as np
from scipy.optimize import least_squares
from pylorenzmie.theory import coordinates
from pylorenzmie.theory import LMHologram as Model
from pylorenzmie.fitting import FitSettings, FitResult
from pylorenzmie.fitting import Mask, GlobalSampler, amoeba

try:
    import cupy as cp
    import cukernels as cuk
except Exception:
    cp = None
try:
    from fastkernels import fastresiduals, fastchisqr, fastabsolute
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
        Setting FitSettings.parameters.vary manually will not
        work.
    amoeba_settings : FitSettings
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
    residuals() : numpy.ndarray
        Difference between the current model and the data,
        normalized by the noise estimate.
    optimize() : FitResult
        Optimize the Model to fit the data. A FitResult is
        returned and can be printed for a comprehensive report,
        which is also reflected in updates to the properties of
        the Model.
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
        self.model = Model(**kwargs) if model is None else model
        # Set fields
        self.data = data
        self.noise = noise
        self.coordinates = self.model.coordinates
        # Initialize Feature properties
        self.params = tuple(self.model.properties.keys())
        # Set default options for fitting
        self._init_params()
        # Deserialize if needed
        self.deserialize(info)
        # Random subset sampling
        self.mask = Mask(self.model.coordinates)
        # Globalized optimization sampling
        self.sampler = GlobalSampler(self)

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
            data = data.flatten()
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

    @property
    def redchi(self):
        r = self.resdiuals()
        return r.dot(r) / self.data.size

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
        # Fit
        if nfits > 1:
            result, options = self._globalize(
                method, nfits, x0, square)
        elif nfits == 1:
            result, options = self._optimize(method, x0, square)
        else:
            raise ValueError("nfits must be greater than or equal to 1.")
        # Post-fit cleanup
        result, settings = self._cleanup(method, square, result, nfits,
                                         options=options)
        # Reassign original coordinates
        self.model.coordinates = self.mask.coordinates

        return FitResult(method, result, settings, self.model, npix)

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
              use exclude = ['data', 'shape', 'corner', 'noise', 'redchi']
        '''
        coor = self.model.coordinates
        if self.data is None:
            data = None
            shape = None
            corner = None
            redchi = None
        else:
            data = self.data.tolist()
            shape = (int(coor[0][-1] - coor[0][0])+1,
                     int(coor[1][-1] - coor[1][0])+1)
            corner = (coor[0][0], coor[1][0])
            redchi = self.redchi
        info = {'data': data,  # dict for variables not in properties
                'shape': shape,
                'corner': corner,
                'noise': self.noise,
                'redchi': redchi}
        keys = self.params
        for ex in exclude:  # Exclude things, if provided
            if ex in keys:
                keys.pop(ex)
            elif ex in info.keys():
                info.pop(ex)
            else:
                print(ex + " not found in Feature's keylist")

        out = self.model.properties
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
        self.model.properties = {k: info[k] for k in
                                 self.model.properties.keys()}
        if 'data' in info.keys():
            self.data = np.array(info['data'])
        if 'shape' in info.keys():
            if 'corner' in info.keys():
                corner = info['corner']
            else:
                corner = (0, 0)
            self.model.coordinates = coordinates(info['shape'],
                                                 corner=corner)
        if 'noise' in info.keys():
            self.noise = info['noise']

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
                objective, x0, **self.amoeba_settings.getkwargs(self.vary))
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
                    obj = fastchisqr(
                        holo, self._subset_data, self.noise)
                else:
                    obj = fastabsolute(holo, self._subset_data, self.noise)
            else:
                obj = fastresiduals(holo, self._subset_data, self.noise)
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
    def _init_params(self):
        '''
        Initialize default settings for levenberg-marquardt and
        nelder-mead optimization
        '''
        # Default parameters to vary, in the following order:
        # x_p, y_p, z_p [pixels], a_p [um], n_p,
        # k_p, n_m, wavelength [um], magnification [um/pixel]
        vary = [True] * 5
        vary.extend([False] * 5)
        # ... levenberg-marquardt variable scale factor
        x_scale = [1.e4, 1.e4, 1.e3, 1.e4, 1.e5, 1.e7, 1.e2, 1.e2, 1.e2, 1]
        # ... bounds around intial guess for bounded nelder-mead
        simplex_bounds = [(-np.inf, np.inf), (-np.inf, np.inf),
                          (0., 2000.), (.05, 4.), (1., 3.),
                          (0., 3.), (1., 3.), (.100, 2.00), (0., 1.),
                          (0., 5.)]
        # ... scale of initial simplex
        simplex_scale = np.array(
            [4., 4., 5., 0.01, 0.01, .2, .1, .1, .05, .05])
        # ... tolerance for nelder-mead termination
        simplex_tol = [.1, .1, .01, .001, .001, .001, .01, .01, .01, .01]
        # Default options for amoeba and lm not parameter dependent
        lm_options = {'method': 'lm',
                      'xtol': 1.e-6,
                      'ftol': 1.e-3,
                      'gtol': 1e-6,
                      'max_nfev': 2000,
                      'diff_step': 1e-5,
                      'verbose': 0}
        amoeba_options = {'ftol': 1.e-3, 'maxevals': 800}
        # Initialize settings for fitting
        self.amoeba_settings = FitSettings(self.params,
                                           options=amoeba_options)
        self.lm_settings = FitSettings(self.params,
                                       options=lm_options)
        self.vary = dict(zip(self.params, vary))
        for idx, p in enumerate(self.params):
            amparam = self.amoeba_settings.parameters[p]
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
            self.amoeba_settings.parameters[p].initial = val
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
            settings = self.amoeba_settings
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


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    from time import time

    a = Feature()

    # Read example image
    img = cv2.imread('../tutorials/crop.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / np.mean(img)
    shape = img.shape
    a.data = img

    # Instrument configuration
    a.model.coordinates = coordinates(shape)
    ins = a.model.instrument
    ins.wavelength = 0.447
    ins.magnification = 0.048
    ins.n_m = 1.34

    # Initial estimates for particle properties
    p = a.model.particle
    p.r_p = [shape[0]//2, shape[1]//2, 330.]
    p.a_p = 1.1
    p.n_p = 1.4
    # add errors to parameters
    p.r_p += np.random.normal(0., 1, 3)
    p.z_p += np.random.normal(0., 30, 1)
    p.a_p += np.random.normal(0., 0.1, 1)
    p.n_p += np.random.normal(0., 0.04, 1)
    print("Initial guess:\n{}".format(p))
    # a.model.using_cuda = False
    # a.model.double_precision = False
    # init dummy hologram for proper speed gauge
    a.model.hologram()
    a.mask.settings['distribution'] = 'donut'
    a.mask.settings['percentpix'] = .1
    # a.amoeba_settings.options['maxevals'] = 1
    # ... and now fit
    start = time()
    result = a.optimize(method='amoeba-lm', square=True, nfits=1)
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
