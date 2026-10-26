'''Optical parameters of the holographic microscope.

An :class:`Instrument` holds what the Lorenz-Mie model needs to know about
the microscope: the vacuum wavelength of the illumination in micrometers, the
size of a pixel in the sample in micrometers per pixel (``magnification``),
the refractive index of the medium, and the noise of the camera as a fraction
of the background, which sets the weight of a pixel in a fit.  The defaults
describe a 447 nm laser, 0.048 um per pixel and water.

The parameters are exposed through ``properties``, so an
:class:`~pylorenzmie.analysis.Optimizer.Optimizer` can fit them, although it
holds them fixed by default.  :meth:`~Instrument.wavenumber` gives the wave
number in the units that the calculators use.
'''

from dataclasses import dataclass
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.lmtypes import Properties
import numpy as np


@dataclass
class Instrument(LMObject):
    '''In-line holographic microscope for Lorenz-Mie microscopy.

    Encapsulates the optical parameters of the instrument.  All four attributes
    (``n_m``, ``wavelength``, ``magnification``, ``noise``) are exposed via
    ``properties`` and are therefore available to
    :class:`~pylorenzmie.analysis.Optimizer.Optimizer` during fitting.

    Attributes
    ----------
    wavelength : float
        Vacuum wavelength of the illuminating light, in μm.
        Default: 0.447.
    magnification : float
        Effective pixel size (object-space), in μm/pixel.
        Default: 0.048.
    noise : float
        Camera noise as a fraction of the mean intensity. Default: 0.05.
    n_m : float
        Refractive index of the medium. Default: 1.340.
    '''

    wavelength: float = 0.447
    magnification: float = 0.048
    noise: float = 0.05
    n_m: float = 1.340

    @LMObject.properties.getter
    def properties(self) -> Properties:
        return {'n_m': self.n_m,
                'wavelength': self.wavelength,
                'magnification': self.magnification,
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
