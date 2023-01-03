# -*- coding: utf-8 -*-

from pylorenzmie.lib import LMObject
from pylorenzmie.theory import LorenzMie
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import svd
import pandas as pd
from typing import (Optional, List, Dict)
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Optimizer(LMObject):
    '''
    Fit generative light-scattering model to data

    ...

    Properties
    ----------
    model : LorenzMie
        Generative model for calculating holograms.
    data : numpy.ndarray
        Target for optimization with model.
    robust : bool
        If True, use robust optimization (absolute deviations)
        otherwise use least-squares optimization
        Default: False (least-squares)
    fixed : list of str
        Names of properties of the model that should remain constant
        during fitting
    variables : list of str
        Names of properties of the model that will be optimized.
        Default: All model.properties that are not fixed
    settings : dict
        Dictionary of settings for the optimization method
    properties : dict
        Dictionary of settings for the optimizer as a whole
    result : pandas.Series
        Optimized values of the variables, together with numerical
        uncertainties.

    Methods
    -------
    optimize() : pandas.Series
        Optimizes model parameters to fit the model to the data.
        Returns result.
    '''

    def __init__(self,
                 model: Optional[LorenzMie] = None,
                 data: Optional[np.ndarray] = None,
                 robust: bool = False,
                 fixed: List[str] = [],
                 settings: Optional[Dict] = None,
                 **kwargs) -> None:
        self.model = model or LorenzMie(**kwargs)
        self.data = data
        self.settings = settings
        self.robust = robust
        self.fixed = fixed
        self._result = None

    #
    # Properties that control the fitting process
    #
    @property
    def settings(self) -> dict:
        return self._settings

    @settings.setter
    def settings(self, settings: dict) -> None:
        '''Dictionary of settings for scipy.optimize.least_squares

        NOTES:
        (a) method:
            trf:     Trust Region Reflective
            dogbox:
            lm:      Levenberg-Marquardt
                     NOTE: only supports linear loss
        (b) loss
            linear:  default: standard least squares
            soft_l1: robust least squares
            huber:   robust least squares
            cauchy:  strong least squares
            arctan:  limits maximum loss
        (c) x_scale
            jac:     dynamic rescaling
            x_scale: specify scale for each adjustable variable
        '''
        if settings is None:
            settings = {'method': 'lm',     # (a)
                        'ftol': 1e-4,       # default: 1e-8
                        'xtol': 1e-6,       # default: 1e-8
                        'gtol': 1e-6,       # default: 1e-8
                        'loss': 'linear',   # (b)
                        'max_nfev': 2000,   # max function evaluations
                        'diff_step': 1e-5,  # default: machine epsilon
                        'x_scale': 'jac'}   # (c)
        self._settings = settings

    @property
    def robust(self) -> bool:
        return self.settings['loss'] != 'linear'

    @robust.setter
    def robust(self, robust: bool) -> None:
        '''Convenience property for selecting robust fitting'''
        if robust:
            self.settings['method'] = 'trf'
            self.settings['loss'] = 'cauchy'
        else:
            self.settings['method'] = 'lm'
            self.settings['loss'] = 'linear'

    @property
    def fixed(self) -> List[str]:
        '''list of fixed properties'''
        return self._fixed

    @fixed.setter
    def fixed(self, fixed: List[str]) -> None:
        self._fixed = fixed
        self._variables = [p for p in self.model.properties
                           if p not in fixed]

    @property
    def variables(self) -> List[str]:
        '''list of variable properties'''
        return self._variables

    #
    # Properties resulting from optimization
    #
    @property
    def result(self) -> pd.Series:
        '''Optimized properties formatted as a pandas.Series'''
        if self._result is None:
            return None
        a = self.variables
        b = ['d' + c for c in a]
        keys = list(sum(zip(a, b), ()))
        keys.extend(['success', 'npix', 'redchi'])

        values = self._result.x
        npix = self.data.size
        redchi, uncertainties = self._statistics()
        values = list(sum(zip(values, uncertainties), ()))
        values.extend([self._result.success, npix, redchi])
        return pd.Series(dict(zip(keys, values)))

    @property
    def metadata(self):
        '''Fixed properties and fit settings'''
        metadata = {key: self.model.properties[key] for key in self.fixed
                    if key in self.model.properties}
        metadata.update(self.settings)
        return pd.Series(metadata)

    @property
    def properties(self):
        '''Properties of the Optimizer object'''
        properties = dict(settings=self.settings,
                          fixed=self.fixed)
        properties.update(self.model.properties)
        return properties

    @properties.setter
    def properties(self, properties):
        self.model.properties = properties
        for property, value in properties.items():
            if hasattr(self, property):
                setattr(self, property, value)

    #
    # Public methods
    #
    def optimize(self) -> pd.Series:
        '''
        Fit Model to data

        Returns
        -------
        result : pandas.Series
            Values, uncertainties and statistics from fit
        '''
        p0 = np.array([self.model.properties[p] for p in self.variables])
        self._result = least_squares(self._residuals, p0, **self.settings)
        return self.result

    #
    # Private methods
    #
    def _residuals(self, values):
        '''Updates properties and returns residuals'''
        self.model.properties = dict(zip(self.variables, values))
        noise = self.model.instrument.noise
        return (self.model.hologram() - self.data) / noise

    def _statistics(self):
        '''return reduced chi-squared and standard uncertainties

        Uncertainties are the square roots of the diagonal
        elements of the covariance matrix. The covariance matrix
        is obtained from the Jacobian of the fit by singular
        value decomposition, using the Moore-Penrose inverse
        after discarding small singular values.
        '''
        res = self._result
        ndeg = self.data.size - res.x.size  # number of degrees of freedom
        redchi = 2.*res.cost / ndeg         # reduced chi-squared

        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        uncertainty = np.sqrt(redchi * np.diag(pcov))

        return redchi, uncertainty


def test_case():
    from pylorenzmie.utilities import coordinates

    shape = (201, 201)
    model = LorenzMie()
    model.coordinates = coordinates(shape)
    model.particle.a_p = 0.75
    model.particle.n_p = 1.42
    model.particle.r_p = [100., 100., 225.]
    data = model.hologram()
    data += model.instrument.noise * np.random.normal(size=shape).flatten()

    fixed = 'wavelength magnification numerical_aperture n_m k_p'.split()
    a = Optimizer(model=model, fixed=fixed)
    settings = a.settings
    settings['method'] = 'trf'
    settings['loss'] = 'cauchy'
    settings['ftol'] = 1e-3
    settings['xtol'] = None
    settings['gtol'] = None
    # settings['verbose'] = 2
    a.data = data
    result = a.optimize()
    print(result)


if __name__ == '__main__':
    test_case()
