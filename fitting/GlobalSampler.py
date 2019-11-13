#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pylorenzmie.fitting.Settings import FitSettings
from pylorenzmie.fitting.Mask import normalize, gaussian


class GlobalSampler(object):
    '''
    TODO
    '''

    def __init__(self, feature):
        self.feature = feature
        self.params = ("z_p", "a_p", "n_p")

        self._init_settings()

        self._param_space = None
        self._param_range = None
        self._xfit = None
        self._x0 = None

        self._independent = True
        self._npts = 100

    @property
    def distribution(self):
        return self._distribution

    @property
    def param_space(self):
        return self._param_space

    @property
    def param_range(self):
        return self._param_range

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        vary = self.feature.vary
        parameters = []
        for param in params:
            if param in vary.keys():
                if vary[param]:
                    parameters.append(param)
                else:
                    raise ValueError("{} is not set to vary".
                                     format(param))
            else:
                raise ValueError("{} is not a valid parameter".
                                 format(param))
        self._params = parameters
        self._idx_map = {}
        for idx, prop in enumerate(self.feature.properties):
            if prop in self._params:
                self._idx_map[prop] = idx

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0):
        self._x0 = x0
        self._reset()

    @property
    def xfit(self):
        return self._xfit

    @xfit.setter
    def xfit(self, xfit):
        self._xfit = xfit

    @property
    def npts(self):
        return self._npts

    @npts.setter
    def npts(self, npts):
        self._npts = npts
        self._reset()

    @property
    def nparams(self):
        return len(self._params)

    def sample(self):
        s = self.x0
        if self.sampling_settings.options['independent']:
            for param in self.params:
                idx = self._idx_map[param]
                space = self.param_space[param]
                d = self.distribution[param]
                s[idx] = np.random.choice(space, p=d)
        else:
            raise ValueError("Dependent sampling not implemented yet.")
        self._update(s)
        print(s)
        return s

    def _update(self, sample):
        if self.sampling_settings.options['distribution'] == 'wells':
            unpacked = self.well_settings.getkwargs(self.params)
            well_std = unpacked['std']
            if self.sampling_settings.options['independent']:
                for j, param in enumerate(self.params):
                    d = self.distribution[param]
                    space = self.param_space[param]
                    min, max = self.param_range[param]
                    i = self._idx_map[param]
                    std = well_std[j]
                    s = sample[i]
                    start_well = normalize(1 - gaussian(space, s, std))
                    if type(self.xfit) is np.ndarray:
                        xi = self.xfit[i]
                        if (xi > max+3*std) or (xi < min-3*std):
                            fit_well = 0.
                        else:
                            fit_well = normalize(
                                1 - gaussian(space, xi, std))
                    else:
                        fit_well = 0.
                    d += fit_well + start_well
                    self._distribution[param] = normalize(d)
            else:
                raise ValueError("Dependent sampling not implemented yet.")
        else:
            raise ValueError("dist must be set to \'wells\'.")

    def _reset(self):
        settings = self.sampling_settings.getkwargs(self._params)
        sample_range = settings['sample_range']
        x0 = self.x0
        npts = self.npts
        if self.sampling_settings.options['independent']:
            self._distribution = {}
            self._param_space = {}
            self._param_range = {}
            for j, prop in enumerate(self.params):
                i = self._idx_map[prop]
                r = sample_range[j]
                o = x0[i]
                self._distribution[prop] = np.zeros(npts)
                self._param_range[prop] = (o-r, o+r)
                self._param_space[prop] = np.linspace(o-r, o+r, npts)
        else:
            raise ValueError("Dependent sampling not implemented yet.")
        self._update(x0)

    def _init_settings(self):
        # Gaussian well standard deviation
        well_std = [None, None, 5, .03, .02, None, None, None, None, None]
        # Sampling range for globalized optimization based on Estimator
        sample_range = [None, None, 30, .2, .1, None, None, None, None, None]
        sample_options = {"independent": True, "distribution": "wells"}
        self.well_settings = FitSettings(self.feature.properties)
        self.sampling_settings = FitSettings(self.feature.properties,
                                             options=sample_options)
        for idx, prop in enumerate(self.feature.properties):
            well_param = self.well_settings.parameters[prop]
            param = self.sampling_settings.parameters[prop]
            well_param.options['std'] = well_std[idx]
            param.options['sample_range'] = sample_range[idx]
