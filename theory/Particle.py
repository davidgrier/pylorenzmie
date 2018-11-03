#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Particle(object):
    '''Abstraction of a particle for Lorenz-Mie microscopy'''

    def __init__(self,
                 r_p=[0, 0, 0]):   # 3-dimensional coordinates of center

        self.r_p = r_p

    @property
    def r_p(self):
        '''Three-dimensional coordinates of particle's center'''
        return self._r_p

    @r_p.setter
    def r_p(self, r_p):
        self._r_p = np.asarray(r_p, dtype=float)

    @property
    def x_p(self):
        return self._r_p[0]

    @x_p.setter
    def x_p(self, x_p):
        self._r_p[0] = float(x_p)

    @property
    def y_p(self):
        return self._r_p[1]

    @y_p.setter
    def y_p(self, y_p):
        self._r_p[1] = float(y_p)

    @property
    def z_p(self):
        return self._r_p[2]

    @z_p.setter
    def z_p(self, z_p):
        self._r_p[2] = float(z_p)

    def ab(self, n_m=1., wavelength=0., resolution=0):
        return np.asarray([1, 1], dtype=np.complex)


if __name__ == '__main__':
    p = Particle()
    print(p.r_p)
    p.x_p = 100.
    print(p.r_p)
    print(p.ab())
