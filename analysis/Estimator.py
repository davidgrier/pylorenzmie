from dataclasses import dataclass
from pylorenzmie.lib import LMObject
from typing import (Optional, Dict)
import pandas as pd
import numpy as np
from pylorenzmie.lib import aziavg
from scipy.signal import (argrelmin, argrelmax)
from scipy.stats import siegelslopes
from scipy.special import jn_zeros


@dataclass
class Estimator(LMObject):
    '''Estimate parameters of a holographic feature

    Properties
    ----------
    z_p : float
        Axial particle position [pixels]
    a_p : float
        Particle radius [um]
    n_p : float
        Particle refractive index

    Methods
    -------
    predict(feature) : pandasSeries
        Returns estimated properties
    '''
    x_p: Optional[float] = None
    y_p: Optional[float] = None
    z_p: Optional[float] = None
    a_p: Optional[float] = None
    n_p: float = 1.5

    def __post_init__(self) -> None:
        self.predict = self.estimate

    @LMObject.properties.fget
    def properties(self) -> Dict[str, float]:
        props = 'x_p y_p z_p a_p n_p'.split()
        return {p: getattr(self, p) for p in props}

    def _initialize(self, feature):
        '''Prepare for estimation

        self.k: wavenumber in the medium [radian/pixels]
        self.noise: noise estimate from instrument
        self.profile: aximuthal average of data
        '''
        if (feature is None) | (feature.data is None):
            return
        instrument = feature.model.instrument
        self.k = instrument.wavenumber()
        self.noise = instrument.noise
        self.magnification = instrument.magnification
        center = np.array(feature.data.shape) // 2
        self.profile = aziavg(feature.data, center) - 1.
        self.coordinates = feature.coordinates

    def _estimate_z(self) -> None:
        '''Estimate axial position of particle

        Particle is assumed to be at the center of curvature
        of spherical waves interfering with a plane wave.
        '''
        if self.z_p is not None:
            return
        b = self.profile
        nmax = argrelmax(b, order=5)
        nmin = argrelmin(b, order=5)
        xj = np.concatenate([np.sqrt(b[nmax])-1., 1.-np.sqrt(b[nmin])])
        rj = np.concatenate(nmax + nmin)
        good = xj > 0.
        xj = xj[good]
        rj = rj[good]
        res = siegelslopes(1./xj, rj**2)
        self.z_p = np.sqrt(res.intercept/res.slope)

    def _estimate_xy(self) -> None:
        '''Estimate in-plane position of particle'''
        if (self.x_p is None) or (self.y_p is None):
            self.x_p, self.y_p = np.mean(self.coordinates, axis=1)

    def _estimate_a(self) -> None:
        '''Estimate radius of particle

        Model interference pattern as spherical wave
        eminating from z_p and interfering with a plane wave.
        '''
        if self.a_p is not None:
            return
        minima = argrelmin(self.profile)
        alpha_n = np.sqrt((self.z_p/minima)**2 + 1.)
        a_p = np.median(jn_zeros(1, len(alpha_n)) * alpha_n) / self.k
        return 2. * self.magnification * a_p

    def estimate(self, feature=None) -> pd.Series:
        ''' Estimate particle properties '''
        if feature is None:
            return pd.Series()
        self._initialize(feature)
        self._estimate_center()
        self._estimate_z()
        self._estimate_a()
        return pd.Series(self.properties)
