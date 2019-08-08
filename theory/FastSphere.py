#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylorenzmie.theory.Sphere import Sphere, mie_coefficients
from numba import jit


class FastSphere(Sphere):

    '''
    Numba accelerated abstraction of a spherical scatterer for
    Lorenz-Mie micrsocopy
    '''

    def __init__(self, **kwargs):
        super(FastSphere, self).__init__(**kwargs)
        self._mie_coefficients = jit(mie_coefficients, nopython=True)


if __name__ == '__main__':
    from time import time
    s = FastSphere(a_p=0.75, n_p=1.5)
    print(s.a_p, s.n_p)
    print(s.ab(1.339, 0.447).shape)
    start = time()
    s.ab(1.339, .447)
    print(time() - start)
