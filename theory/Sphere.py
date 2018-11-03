#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Particle import Particle
from sphere_coefficients import sphere_coefficients
import numpy as np


class Sphere(Particle):
    '''Abstraction of a spherical particle for Lorenz-Mie micrsocopy'''

    def __init__(self,
                 a_p=1.,   # radius of sphere [um]
                 n_p=1.5,  # refractive index of sphere
                 **kwargs):
        super(Sphere, self).__init__(**kwargs)
        self.a_p = a_p
        self.n_p = n_p

    @property
    def a_p(self):
        '''Radius of sphere [um]'''
        return self._a_p

    @a_p.setter
    def a_p(self, a_p):
        self._a_p = float(a_p)

    @property
    def n_p(self):
        '''Complex refractive index of sphere'''
        return self._n_p

    @n_p.setter
    def n_p(self, n_p):
        self._n_p = np.complex(n_p)

    def ab(self, n_m, wavelength, resolution=None):
        '''Lorenz-Mie ab coefficients for given wavelength
        and refractive index'''
        self._ab = sphere_coefficients(self.a_p, self.n_p,
                                       n_m, wavelength, resolution)
        return self._ab


if __name__ == '__main__':
    s = Sphere(a_p=0.75, n_p=1.45)
    print(s.r_p)
    print(s.ab(1.335, 0.532))
