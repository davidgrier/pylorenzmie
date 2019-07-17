#!/usr/bin/env python
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

    @property
    def vary(self):
        '''Vary parameter during fitting'''
        return self._vary

    @vary.setter
    def vary(self, vary):
        self._vary = vary

    @property
    def options(self):
        '''Fitting algorithm options for each parameter'''
        return self._options

    @options.setter
    def options(self, options):
        self._options = options


class FitResult(object):

    def __init__(self, scipyresult):
        self.scipyresult = scipyresult


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
