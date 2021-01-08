# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class FitSettings(object):
    '''
    Stores information about an algorithm's general and
    parameter-specific options during fitting.

    ...

    Attributes
    ----------
    options : dict
        Dictionary that can be passed to a fitting algorithm,
        consisting of vector and scalar keywords
    parameters: dict of ParameterSettings
        See ParameterSettings documentation to see attributes.

    Methods
    -------
    getkwargs(vary)
        Returns a dictionary of arguments and keywords to be
        passed to a fitting algorithm. Before calling, set
        all non-vector keywords in FitSettings.options
        and all vector keywords in each ParameterSettings.options
        stored in FitSettings.parameters. However, setting
        FitSettings.parameters.vary will not have an effect on the
        output; only the argument to this function, vary, will.
    '''

    def __init__(self, keys, options=None):
        '''
        Arguments
        ---------
        keys : list or tuple
            Names of parameters in model

        Keywords
        --------
        options : dict
            Non-vector args or kwargs options for fitting algorithm.
        '''
        self._keys = keys
        self.options = dict(options or dict())
        self.parameters = {key: ParameterSettings() for key in keys}

    @property
    def settings(self):
        vary = {key: True for key in self._keys}
        return self.getkwargs(vary, ndarray=False)

    @settings.setter
    def settings(self, settings):
        for key, value in settings.items():
            if isinstance(value, list):
                if len(value) == len(self._keys):
                    for idx, val in enumerate(value):
                        p = self._keys[idx]
                        param = self.parameters[p]
                        param.options[key] = val
            else:
                self.options[key] = value

    def getkwargs(self, vary, ndarray=True):
        '''
        Returns keyword dictionary for fitting algorithm

        Arguments
        ---------
        vary : dict of bools or tuple
            Dictionary or tuple that determines whether or not
            parameter will vary during fitting. If a tuple, all 
            elements in tuple are assumed to vary. Setting
            FitSettings.parameters.vary individually before
            calling this method will no have no effect.

        Returns
        -------
        options : dict
            Both vector and non-vector args and kwargs for
            fitting algorithm
        '''
        options = self.options
        temp = []
        for key in self._keys:
            param = self.parameters[key]
            if type(vary) is dict:
                param.vary = vary[key]
            else:
                param.vary = (key in vary)
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
            if ndarray:
                l = np.array(l)
            options[name] = l
        return options


class ParameterSettings(object):
    '''
    Stores information about each parameter's initial value, 
    final value, whether or not it will vary during fitting,
    and algorithm-specific options.

    ...

    Attributes
    ----------
    options : dict
        Settings for fitting algorithm relating to this specific
        parameter. For example, an argument requiring a vector,
        where each entry is a value specific to a parameter,
        would store its value as a scalar here.
    initial
        Storage for parameter initial value during fitting.
    vary : boolean
        Determines whether or not parameter will vary during
        fitting
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
    '''
    Convenient object for storing results of a fit.
    Provides nice print statement report. Future plans
    include serialization and options to calculate
    statistics beyond reduced chi-squared.

    ...

    Attributes
    ----------
    method : str
        Name of fitting method
    redchi : float
        Reduced chi squared value from fit
    nfev : int
        Number of function evaluations before convergence
    initial : dict
        Initial values of parameters during fitting
    final : dict
        Final values of parameters during fitting
    vary : dict of bools
        Whether or not a parameter varied during fitting
    success : bool
        Whether or not fit converged
    message : str
        Message about reason for convergence or failure
    Methods
    -------

    '''

    def __init__(self, method, scipy_result, settings, model, ndata):
        '''
        Arguments
        ---------
        method : str
            Fitting method
        scipy_result : scipy.optimize.OptimizeResult
            See scipy's documentation. The idea of this object
            is to build on scipy's object and provide a friendlier
            report.
        settings : FitSettings
            See FitSettings object for documentation. Used to
            gather information about the fit.
        model
            Model that was optimized during fitting. For example,
            an LMHologram object.
        ndata : int
            Number of data points during fitting. Used for reduced
            chi-squared calculation.
        '''
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
        '''Number of function evaluations before convergence'''
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
            elif hasattr(self.model, param):
                val = getattr(self.model, param)
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
