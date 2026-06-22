from dataclasses import dataclass
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.lmtypes import Properties
import numpy as np


@dataclass
class Instrument(LMObject):
    '''In-line holographic microscope for Lorenz-Mie microscopy.

    Encapsulates the optical parameters of the instrument.  All five
    attributes (``n_m``, ``wavelength``, ``magnification``,
    ``numerical_aperture``, ``noise``) are exposed via ``properties``
    and are therefore available to :class:`Optimizer` during fitting.

    Attributes
    ----------
    wavelength : float
        Vacuum wavelength of the illuminating light, in μm.
        Default: 0.447.
    magnification : float
        Effective pixel size (object-space), in μm/pixel.
        Default: 0.048.
    numerical_aperture : float
        Numerical aperture of the objective lens. Default: 1.45.
    noise : float
        Camera noise as a fraction of the mean intensity. Default: 0.05.
    n_m : float
        Refractive index of the medium. Default: 1.340.
    '''

    wavelength: float = 0.447
    magnification: float = 0.048
    numerical_aperture: float = 1.45
    noise: float = 0.05
    n_m: float = 1.340

    @LMObject.properties.getter
    def properties(self) -> Properties:
        return {'n_m': self.n_m,
                'wavelength': self.wavelength,
                'magnification': self.magnification,
                'numerical_aperture': self.numerical_aperture,
                'noise': self.noise}

    def wavenumber(self,
                   in_medium: bool = True,
                   scaled: bool = True) -> float:
        '''Wave number of the illuminating light.

        Parameters
        ----------
        in_medium : bool
            If True (default), return the wave number in the medium.
            If False, return the wave number in vacuum.
        scaled : bool
            If True (default), return in rad/pixel.
            If False, return in rad/μm.

        Returns
        -------
        k : float
            Wave number.
        '''
        k = 2. * np.pi / self.wavelength  # wave number in vacuum
        if in_medium:
            k *= self.n_m                 # ... in medium
        if scaled:
            k *= self.magnification       # ... in image units
        return k


if __name__ == '__main__':  # pragma: no cover
    Instrument.example()
