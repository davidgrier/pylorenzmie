from dataclasses import dataclass
from pylorenzmie.lib import (LMObject, Azimuthal)
from pylorenzmie.theory import Instrument
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy.signal import (argrelmin, argrelmax)
from scipy.special import jn_zeros


Feature = NDArray[float]
Features = Feature | list[Feature]
Prediction = pd.Series
Predictions = Prediction | list[Prediction]


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
    estimate(feature, center): pandas.Series

        Arguments
        ---------
        feature : numpy.ndarray

        center : list-like

    predict :
        synonum for estimate for backward compatibility
    '''
    instrument: Instrument
    z_p: float | None = None
    a_p: float | None = None
    n_p: float = 1.5

    def __post_init__(self) -> None:
        self.predict = self.estimate

    @property
    def properties(self) -> dict[str, float]:
        return dict(z_p=self.z_p,
                    a_p=self.a_p,
                    n_p=self.n_p)

    def _initialize(self, feature: Feature) -> None:
        '''Prepare for estimation

        self.k: wavenumber in the medium [radian/pixels]
        self.noise: noise estimate from instrument
        self.profile: aximuthal average of data
        '''
        self.k = self.instrument.wavenumber()
        self.noise = self.instrument.noise
        self.magnification = self.instrument.magnification
        # NOTE: Allow to pass in profile without aziavg
        self.profile = Azimuthal.avg(feature)

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
        rn = np.sort(np.concatenate(nmax + nmin))
        zn = self.k / (2.*np.pi) * (rn[1:]**2 - rn[:-1]**2)
        self.z_p = np.median(zn)

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
        self.a_p = 2. * self.magnification * a_p

    def estimate(self, feature: Features) -> Predictions:
        '''Estimate particle properties'''
        if isinstance(feature, list):
            return [self.estimate(this) for this in feature]
        self._initialize(feature)
        self._estimate_z()
        self._estimate_a()
        return pd.Series(self.properties)


def example() -> None:
    import cv2

    estimator = Estimator(Instrument())
    basedir = estimator.directory.parent
    filename = str(basedir / 'docs' / 'tutorials' / 'crop.png')
    hologram = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float)
    hologram /= 100.
    print(estimator.estimate(hologram))


if __name__ == '__main__':
    example()
