#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .Settings import FitSettings


def normalize(distribution):
    total = np.sum(distribution)
    normed = distribution / total
    return normed


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class GlobalSampler(object):

    def __init__(self, optimizer, config):

        self.optimizer = optimizer

        self._init_settings(config)

        self._param_space = None
        self._param_range = None
        self._xfit = None
        self._x0 = None

        self._independent = True
        self._npts = 500

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
        vary = self.optimizer.vary
        parameters = []
        for param in params:
            if param in vary.keys():
                if vary[param]:
                    parameters.append(param)
            else:
                raise ValueError("{} is not a valid parameter".
                                 format(param))
        self._params = parameters
        self._idx_map = {}
        idx = 0
        for prop in self.optimizer.params:
            if prop in parameters:
                self._idx_map[prop] = idx
            if self.optimizer.vary[prop]:
                idx += 1

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
                dmax = d.max()
                s[idx] = np.random.choice(space, p=normalize(dmax - d))
        else:
            raise ValueError("Dependent sampling not implemented yet.")
        self._update(s)
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
                    start = gaussian(space, s, std)
                    if type(self.xfit) is np.ndarray:
                        xi = self.xfit[i]
                        if (xi > max+3*std) or (xi < min-3*std):
                            fit = 0.
                        else:
                            fit = gaussian(space, xi, std)
                    else:
                        fit = 0.
                    d += fit + start
                    self._distribution[param] = d
            else:
                raise ValueError("Dependent sampling not implemented yet.")
        else:
            raise ValueError("dist must be set to \'wells\'.")

    def _reset(self):
        self.params = self._params
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

    def _init_settings(self, config):
        self.params = config['params']
        # Gaussian well standard deviation
        well_std = config['well_std']
        # ...sampling range for globalized optimization
        sample_range = config['sample_range']
        # ... options for distribution
        sample_options = config['options']
        self.well_settings = FitSettings(self.optimizer.params)
        self.sampling_settings = FitSettings(self.optimizer.params,
                                             options=sample_options)
        for idx, p in enumerate(self.optimizer.params):
            well_param = self.well_settings.parameters[p]
            param = self.sampling_settings.parameters[p]
            well_param.options['std'] = well_std[idx]
            param.options['sample_range'] = sample_range[idx]
