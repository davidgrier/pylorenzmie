import numpy as np
from pylorenzmie.utilities import aziavg
from scipy.signal import (savgol_filter, argrelmin)
from scipy.stats import sigmaclip
from scipy.special import jn_zeros


class Estimator(object):
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
    predict(feature) :
        Returns a dictionary of estimated properties
    '''
    def __init__(self,
                 z_p=None,
                 a_p=None,
                 n_p=None,
                 **kwargs):
        self.z_p = z_p
        self.a_p = a_p
        self.n_p = n_p or 1.5

    def _initialize(self, feature):
        '''Prepare for estimation

        self.k: wavenumber in the medium [pixels^{-1}]
        self.noise: noise estimate from instrument
        self.profile: aximuthal average of data
        '''
        ins = feature.model.instrument
        self.k = (2.*np.pi * ins.n_m / ins.wavelength) * ins.magnification
        self.noise = ins.noise  
        center = np.array(feature.data.shape) // 2
        self.profile = aziavg(feature.data, center) - 1.
        self._initialized = True
        
    def estimate_z(self, feature):
        '''Estimate axial position of particle

        Particle is assumed to be at the center of curvature
        of spherical waves interfering with a plane wave.
        '''
        if not self._initialized:
            self._initialize(feature)
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
    
        return np.sqrt(np.mean(sigmaclip(zsq).clipped))

    def estimate_a(self, feature):
        if not self._initialized:
            self._initialize(feature)
        ins = feature.model.instrument
        minima = argrelmin(self.profile)
        alpha_n = np.sqrt((self.z_p/minima)**2 + 1.)
        a_p = np.median(jn_zeros(1, len(alpha_n)) * alpha_n) / self.k
        return 2.*ins.magnification*a_p
    
    def predict(self, feature):
        self._initialized = False
        if self.z_p is None:
            self.z_p = self.estimate_z(feature)
        if self.a_p is None:
            self.a_p = self.estimate_a(feature)
        if self.n_p is None:
            self.n_p = 1.5
        return dict(z_p=self.z_p,
                    a_p=self.a_p,
                    n_p=self.n_p)
