# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class FitSettings(object):
    '''
    Stores information about an algorithm's general and 
    parameter specific options during fitting.

    ...

    Attributes
    ----------
    options : dict
        Dictionary that can be passed to a fitting algorithm.
        Consists of vector and scalar keywords.
    parameters: dict of ParameterSettings
        See ParameterSettings documentation to see attributes.
    '''

    def __init__(self, keys, options=None):
        self._keys = keys
        if type(options) is not dict:
            self.options = dict()
        else:
            self.options = options
        self.parameters = {}
        for idx, key in enumerate(keys):
            self.parameters[key] = ParameterSettings()

    def getkwargs(self, vary):
        '''
        Returns keyword dictionary for fitting algorithm

        Arguments
        ---------
        vary : dict of bools
            Dictionary that determines whether or not parameter
            will vary during fitting
        '''
        options = self.options
        temp = []
        for key in self._keys:
            param = self.parameters[key]
            param.vary = vary[key]
            temp.append(list(param.options.keys()))
        names = temp[0]  # TODO: error checking
        noptions = len(names)
        lists = [list() for i in range(noptions)]
        for idx, l in enumerate(lists):
            name = names[idx]
            for key in self._keys:
                param = self.parameters[key]
                if param.vary:
                    l.append(param.options[name])
            options[name] = np.array(l)
        return options


class ParameterSettings(object):
    '''
    Stores information about each parameter's initial value, 
    final value, whether or not it will vary during fitting,
    and algorithm-specific options.

    ...

    Attributes
    ----------
    options: dict
    vary: boolean
    '''

    def __init__(self):
        self._options = dict()
        self._vary = bool()
        self._initial = float()

    @property
    def vary(self):
        '''Vary parameter during fitting'''
        return self._vary

    @vary.setter
    def vary(self, vary):
        self._vary = vary

    @property
    def initial(self):
        '''Initial value of parameter for fit'''
        return self._initial

    @initial.setter
    def initial(self, initial):
        self._initial = initial

    @property
    def options(self):
        '''Fitting algorithm options for each parameter'''
        return self._options

    @options.setter
    def options(self, options):
        self._options = options


class FitResult(object):

    def __init__(self, method, scipy_result, settings, model, ndata):
        self._method = method
        self.result = scipy_result
        self.settings = settings
        self.model = model
        self._ndata = ndata
        self._redchi = None
        self._set_properties()

    @property
    def method(self):
        '''Name of fitting method'''
        return self._method

    @property
    def redchi(self):
        '''Reduced chi-squared value of fits'''
        return self._redchi

    @property
    def nfev(self):
        return self.result.nfev

    @property
    def initial(self):
        '''Dictionary telling initial values of parameters'''
        return self._initial

    @property
    def final(self):
        '''Dictionary telling final values of parameters'''
        return self._final

    @property
    def vary(self):
        '''Dictionary telling which parameters varied during fitting'''
        return self._vary

    @property
    def success(self):
        '''Boolean indicating fit success'''
        return self.result.success

    @property
    def message(self):
        '''Message regarding fit success'''
        return self.result.message

    def __str__(self):
        i, f, v = self.initial, self.final, self.vary
        pstr = ''
        pstr += 'FIT REPORT\n'
        pstr += '---------------------------------------------\n'
        props = ['method', 'success', 'message', 'redchi', 'nfev']
        for prop in props:
            pstr += prop + ': {}\n'.format(getattr(self, prop))
        for p in self.settings.parameters:
            pstr += p+': {:.05f} (init: {:.05f})'.format(f[p], i[p])
            if not v[p]:
                pstr += ' (fixed)'
            pstr += '\n'
        return pstr

    def _set_properties(self):
        params = self.settings.parameters
        self._initial = {}
        self._vary = {}
        self._final = {}
        for param in params:
            self._initial[param] = params[param].initial
            self._vary[param] = params[param].vary
            if hasattr(self.model.particle, param):
                val = getattr(self.model.particle, param)
            else:
                val = getattr(self.model.instrument, param)
            self._final[param] = val
        self._calculate_statistics()

    def _calculate_statistics(self):
        nfree = self._ndata - self.result.x.size
        if type(self.result.fun) is np.ndarray:
            redchi = (self.result.fun).dot(self.result.fun) / nfree
        else:
            redchi = self.result.fun / nfree
        self._redchi = redchi

    def save(self):
        pass


if __name__ == '__main__':
    settings = FitSettings(('x', 'y'))
    settings.parameters['x'].vary = True
    settings.parameters['y'].vary = True
    settings.parameters['x'].initial = 5.
    settings.parameters['y'].initial = 10.
    settings.parameters['x'].options['xtol'] = 2.
    settings.parameters['y'].options['xtol'] = 1.
    settings.options['ftol'] = 1.
    print(settings.keywords)
