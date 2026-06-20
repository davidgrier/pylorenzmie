from dataclasses import dataclass
from pathlib import Path
from pylorenzmie.lib import LMObject, Azimuthal
from pylorenzmie.lib.types import Image, Images, Properties, Results
from pylorenzmie.theory import Instrument
import pandas as pd
import numpy as np
from scipy.signal import argrelmin, argrelmax
from scipy.special import jn_zeros


@dataclass
class Estimator(LMObject):
    '''Estimate initial parameters of a holographic feature.

    Uses the azimuthal average of a cropped hologram to estimate the
    axial position and radius of the scattering particle, providing
    initial values for :class:`Optimizer`.

    Inherits from :class:`pylorenzmie.lib.LMObject`.

    Parameters
    ----------
    instrument : Instrument
        Optical parameters of the microscope.
    n_p : float, optional
        Particle refractive index. Default: 1.5. Cannot be estimated
        from the azimuthal profile; serves as an initial value for
        the optimizer.
    k_p : float, optional
        Imaginary part of the refractive index. Default: 0.

    Notes
    -----
    Call :meth:`estimate` to populate all particle parameters.
    Results are available via :attr:`properties` afterwards.

    :meth:`predict` is a backward-compatibility alias for
    :meth:`estimate`.

    This class provides a lightweight conventional-algorithm baseline.
    For higher-quality initial estimates — including ``n_p`` — use the
    CATCH deep neural network model [1]_ [2]_.

    References
    ----------
    .. [1] L. E. Altman and D. G. Grier,
       "CATCH: Characterizing and Tracking Colloids Holographically
       Using Deep Neural Networks,"
       J. Phys. Chem. B **124**, 1602 (2020).
    .. [2] L. E. Altman and D. G. Grier,
       "Machine learning enables precise holographic characterization
       of colloidal materials in real time,"
       Soft Matter **19**, 3002 (2023).
    '''

    instrument: Instrument
    n_p: float = 1.5
    k_p: float = 0.

    def __post_init__(self) -> None:
        self.predict = self.estimate
        self.x_p: float | None = None
        self.y_p: float | None = None
        self.z_p: float | None = None
        self.a_p: float | None = None

    @LMObject.properties.getter
    def properties(self) -> Properties:
        '''Estimated particle properties.'''
        return dict(x_p=self.x_p, y_p=self.y_p,
                    z_p=self.z_p, a_p=self.a_p,
                    n_p=self.n_p, k_p=self.k_p)

    def _initialize(self, data: Image) -> None:
        self._k = self.instrument.wavenumber()
        self._magnification = self.instrument.magnification
        self._profile = Azimuthal.avg(data)

    def _estimate_z(self) -> None:
        '''Estimate axial position from fringe spacing.

        Uses the paraxial approximation: consecutive extrema at radii
        r_n satisfy Δ(r²) = 2π z_p / k.
        '''
        b = self._profile
        nmax = argrelmax(b, order=5)
        nmin = argrelmin(b, order=5)
        rn = np.sort(np.concatenate(nmax + nmin))
        zn = self._k / (2. * np.pi) * (rn[1:]**2 - rn[:-1]**2)
        self.z_p = float(np.median(zn))

    def _estimate_a(self) -> None:
        '''Estimate radius from positions of scattering minima.

        Uses the Fraunhofer approximation: minima of the azimuthal
        profile occur where the size parameter equals zeros of J_1.
        '''
        minima = argrelmin(self._profile)[0].astype(float)
        if len(minima) == 0:
            self.logger.warning(
                'No intensity minima found; a_p could not be estimated')
            return
        alpha_n = np.sqrt(np.square(self.z_p / minima) + 1.)
        a_p = np.median(jn_zeros(1, len(alpha_n)) * alpha_n) / self._k
        self.a_p = float(2. * self._magnification * a_p)

    def estimate(self, feature: Images) -> Results:
        '''Estimate particle properties from a holographic crop.

        Parameters
        ----------
        feature : numpy.ndarray or list of numpy.ndarray
            Normalized hologram crop(s).

        Returns
        -------
        result : pandas.Series or list of pandas.Series
            Estimated particle properties.
        '''
        if isinstance(feature, list):
            return [self.estimate(f) for f in feature]
        self._initialize(feature)
        h, w = feature.shape
        self.x_p = float(w / 2.)
        self.y_p = float(h / 2.)
        self._estimate_z()
        self._estimate_a()
        return pd.Series(self.properties)

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        import cv2

        estimator = cls(Instrument())
        basedir = Path(__file__).parent.parent
        filename = str(basedir / 'docs' / 'tutorials' / 'crop.png')
        hologram = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float)
        hologram /= 100.
        print(estimator.estimate(hologram))


if __name__ == '__main__':  # pragma: no cover
    Estimator.example()
