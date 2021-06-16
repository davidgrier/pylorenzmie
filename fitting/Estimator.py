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
                 a_p=1.,
                 n_p=1.5,
                 z_p=100.,
                 feature=None,
                 **kwargs):
        self.a_p = float(a_p)
        self.n_p = float(n_p)
        self.z_p = float(z_p)
        self._initialize(feature)

    def _initialize(self, feature):
        '''Prepare for estimation

        self.k: wavenumber in the medium [pixels^{-1}]
        self.noise: noise estimate from instrument
        self.profile: aximuthal average of data
        '''
        self._initialized = False
        self.feature = feature
        if feature is None:
            return
        if feature.data is None:
            return
        ins = self.feature.model.instrument
        self.k = (2.*np.pi * ins.n_m / ins.wavelength) * ins.magnification
        self.noise = ins.noise
        center = np.array(self.feature.data.shape) // 2
        self.profile = aziavg(self.feature.data, center) - 1.
        self._initialized = True

    def _estimate_z(self):
        '''Estimate axial position of particle

        Particle is assumed to be at the center of curvature
        of spherical waves interfering with a plane wave.
        '''
        if not self._initialized:
            return self.z_p
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

    def _estimate_a(self, z_p):
        '''Estimate radius of particle

        Model interference pattern as spherical wave 
        eminating from z_p and interfering with a plane wave.
        '''
        if not self._initialized:
            return self.a_p
        instrument = self.feature.model.instrument
        minima = argrelmin(self.profile)
        alpha_n = np.sqrt((z_p/minima)**2 + 1.)
        a_p = np.median(jn_zeros(1, len(alpha_n)) * alpha_n) / self.k
        return 2. * instrument.magnification * a_p
    
    def predict(self, feature=None):
        if feature is not None:
            self._initialize(feature)
        z_p = self._estimate_z()
        center = np.mean(self.feature.coordinates, axis=1)
        r_p = [center[0], center[1], z_p]
        a_p = self._estimate_a(z_p)
        n_p = self.n_p
        return dict(r_p=r_p, a_p=a_p, n_p=n_p)
