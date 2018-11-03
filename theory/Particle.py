#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Particle(object):
    '''Abstraction of a particle for Lorenz-Mie microscopy'''

    def __init__(self,
                 rp=None,  # 3-dimensional coordinates of center
                 ab=None):  # Lorenz-Mie ab coefficients
        if rp is None:
            self.rp = [0, 0, 0]
        else:
            self.rp = rp

        if ab is None:
            self.ab = [1, 1]

    @property
    def rp(self):
        '''Three-dimensional coordinates of particle's center'''
        return self._rp

    @rp.setter
    def rp(self, rp):
        self._rp = np.asarray(rp, dtype=float)

    @property
    def xp(self):
        return self._rp[0]

    @xp.setter
    def xp(self, xp):
        self._rp[0] = float(xp)

    @property
    def yp(self):
        return self._rp[1]

    @yp.setter
    def yp(self, yp):
        self._rp[1] = float(yp)

    @property
    def zp(self):
        return self._rp[2]

    @zp.setter
    def zp(self, zp):
        self._rp[2] = float(zp)

    @property
    def ab(self):
        '''Lorenz-Mie a and b coefficients'''
        return self._ab

    @ab.setter
    def ab(self, ab):
        self._ab = np.asarray(ab, dtype=np.complex)


if __name__ == '__main__':
    p = Particle()
    print(p.rp)
    p.xp = 100.
    print(p.rp)
    print(p.ab)
