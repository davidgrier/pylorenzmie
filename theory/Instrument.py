# /usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from pylorenzmie.lib import LMObject
import numpy as np
import logging
from typing import Union

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@dataclass
class Instrument(LMObject):
    '''
    Abstraction of an in-line holographic microscope

    The instrument forms an image of the light scattering
    pattern for Lorenz-Mie microscopy

    ...

    Properties
    ----------
    wavelength : float
        Vacuum wavelength of light [um]
    magnification : float
        Effective size of pixels [um/pixel]
    n_m : float
        Refractive index of medium
    background : float or numpy.ndarray
        Background image
    noise : float
        Estimated noise as a percentage of the mean value
    dark_count : float
        Dark count of camera
    properties : dict
        Adjustable properties of the instrument model

    Methods
    -------
    wavenumber(in_medium=True, magnified=True) : float
        Wavenumber of light
    '''

    wavelength: float = 0.532
    magnification: float = 0.135
    n_m: float = 1.335
    background: Union[float, np.ndarray] = field(repr=False, default=1.)
    noise: float = field(repr=False, default=0.05)
    darkcount: float = field(repr=False, default=0.)

    @LMObject.properties.getter
    def properties(self) -> dict:
        return {'n_m': self.n_m,
                'wavelength': self.wavelength,
                'magnification': self.magnification}

    def wavenumber(self,
                   in_medium: bool = True,
                   magnified: bool = True) -> float:
        '''Return the wave number of light

        Parameters
        ----------
        in_medium : bool
            If set (default) return the wave number in the medium
            Otherwise, return the wave number in vacuum
        magnified : bool
            If set (default) return the scaled value [radian/pixel]
            Otherwise, return SI value [radian/um]

        Returns
        -------
        k : float
            Wave number
        '''
        k = 2. * np.pi / self.wavelength  # wave number in vacuum
        if in_medium:
            k *= self.n_m                 # ... in medium
        if magnified:
            k *= self.magnification       # ... in image units
        return k


if __name__ == '__main__':  # pragma: no cover
    a = Instrument()
    print(a.wavelength, a.magnification)
