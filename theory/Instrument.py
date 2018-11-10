# /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def coordinates(shape):
    '''Return coordinate system for Lorenz-Mie microscopy images'''
    (ny, nx) = shape
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    zv = np.zeros_like(xv)
    return np.stack((xv, yv, zv))


class Instrument(object):

    '''
    Abstraction of an in-line holographic microscope

    The instrument forms an image of the light scattering
    pattern for Lorenz-Mie microscopy

    ...

    Attributes
    ----------
    wavelength : float
        Vacuum wavelength of light [um]
    magnification : float
        Effective size of pixels [um/pixel]
    n_m : complex
        Refractive index of medium

    Methods
    -------
    wavenumber(in_medium=True, magnified=True) : float
        Wavenumber of light
    '''

    def __init__(self,
                 wavelength=0.532,
                 magnification=0.135,
                 n_m=1.335):
        self.wavelength = wavelength
        self.magnification = magnification
        self.n_m = n_m

    def __str__(self):
        str = '{}(wavelength={}, magnification={}, n_m={})'
        return str.format(self.__class__.__name__,
                          self.wavelength,
                          self.magnification,
                          self.n_m)

    @property
    def wavelength(self):
        '''Wavelength of light in vacuum [um]'''
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        self._wavelength = float(wavelength)

    @property
    def magnification(self):
        '''Magnification of microscope [um/pixel]'''
        return self._magnification

    @magnification.setter
    def magnification(self, magnification):
        self._magnification = float(magnification)

    @property
    def n_m(self):
        '''Complex refractive index of medium'''
        return self._n_m

    @n_m.setter
    def n_m(self, n_m):
        self._n_m = complex(n_m)

    def wavenumber(self, in_medium=True, magnified=True):
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
            k *= np.real(self.n_m)    # wave number in medium
        if magnified:
            k *= self.magnification
        return k


if __name__ == '__main__':
    a = Instrument()
    print(a.wavelength, a.magnification)
