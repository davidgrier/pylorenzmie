from dataclasses import dataclass
from pylorenzmie.lib import LMObject
from typing import Optional
import pandas as pd
import numpy as np
from pylorenzmie.utilities import aziavg
from scipy.signal import (savgol_filter, argrelmin)
from scipy.stats import sigmaclip
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

    @LMObject.properties.fget
    def properties(self):
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

    def _estimate_z(self):
        '''Estimate axial position of particle

        Particle is assumed to be at the center of curvature
        of spherical waves interfering with a plane wave.
        '''
        if self.z_p is not None:
            return
        a = self.profile
        rho = np.arange(len(a)) + 0.5
        lap = savgol_filter(a, 11, 3, 2) + savgol_filter(a, 11, 3, 1)/rho

        good = np.abs(a) > self.noise
        qsq = -lap[good] / a[good] / self.k**2
        rho = rho[good]

        good = (abs(qsq) > 0.01) & (abs(qsq) < 1)
        rho = rho[good]
        qsq = qsq[good]

        zsq = rho**2 * (1./qsq - 1.)

        self.z_p = np.sqrt(np.mean(sigmaclip(zsq).clipped))

    def _estimate_center(self):
        if (self.x_p is None) | (self.y_p is None):
            self.x_p, self.y_p = np.mean(self.coordinates, axis=1)

    def _estimate_a(self):
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

    def predict(self, feature=None):
        if feature is None:
            return None
        self._initialize(feature)
        self._estimate_center()
        self._estimate_z()
        self._estimate_a()
        return pd.Series(self.properties)
