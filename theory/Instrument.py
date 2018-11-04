# /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Instrument(object):
    '''Abstraction of an in-line holographic microscope for
    Lorenz-Mie microscopy'''

    def __init__(self,
                 wavelength=0.532,
                 magnification=0.135,
                 n_m=1.335):
        self.wavelength = wavelength
        self.magnification = magnification
        self.n_m = n_m

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
        '''Wave number of light in the medium [um^-1]'''
        k = 2.*np.pi/self.wavelength  # wave number in vacuum
        if in_medium:
            k *= np.real(self.n_m)    # wave number in medium
        if magnified:
            k *= self.magnification
        return k


if __name__ == '__main__':
    a = Instrument()
    print(a.wavelength, a.magnification)
