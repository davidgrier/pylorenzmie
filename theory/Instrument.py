# /usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pylorenzmie.lib import LMObject
import numpy as np


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
        System magnification [um/pixel]
    numerical_aperture : float
        Numerical aperture (NA) of objective lens
    noise : float
        Estimated camera noise as a percentage of the mean intensity
    dark_count : float
        Dark count of camera
    n_m : float
        Refractive index of medium
    properties : dict
        Adjustable properties of the instrument model

    Methods
    -------
    wavenumber(in_medium=True, magnified=True) : float
        Wavenumber of light
    '''

    wavelength: float = 0.447
    magnification: float = 0.048
    numerical_aperture: float = 1.45
    noise: float = 0.05
    darkcount: float = 0.
    n_m: float = 1.340

    @LMObject.properties.getter
    def properties(self) -> LMObject.Properties:
        return {'n_m': self.n_m,
                'wavelength': self.wavelength,
                'magnification': self.magnification,
                'numerical_aperture': self.numerical_aperture}

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
    Instrument.example()
