#!/usr/bin/env python
# -*- coding:utf-8 -*-

from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory import GeneralizedLorenzMie
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None


class LMHologram(LorenzMie):

    '''
    A class that computes in-line holograms of spheres

    ...

    Attributes
    ----------
    alpha : float, optional
        weight of scattered field in superposition

    Methods
    -------
    hologram() : numpy.ndarray
        Computed hologram of sphere
    '''

    def __init__(self,
                 alpha=1.,
                 *args, **kwargs):
        super(LMHologram, self).__init__(*args, **kwargs)
        if cp is not None and 'cuda' in str(GeneralizedLorenzMie):
            self.using_gpu = True
        else:
            self.using_gpu = False
        self.alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = float(alpha)

    def hologram(self, return_gpu=False):
        '''Return hologram of sphere

        Returns
        -------
        hologram : numpy.ndarray
            Computed hologram.
        '''
        xp = cp if self.using_gpu else np
        field = self.alpha * self.field()
        field[0, :] += 1.
        hologram = xp.sum(xp.real(field * xp.conj(field)), axis=0)
        if return_gpu is False and xp == cp:
            hologram = hologram.get()
        return hologram


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Instrument import coordinates
    from time import time

    shape = [201, 251]
    h = LMHologram(coordinates=coordinates(shape))
    h.particle.r_p = [125, 75, 100]
    h.particle.a_p = 0.9
    h.particle.n_p = 1.45
    h.instrument.wavelength = 0.447
    start = time()
    hol = h.hologram()
    print("Time to calculate {}".format(time() - start))
    plt.imshow(hol.reshape(shape), cmap='gray')
    plt.show()
