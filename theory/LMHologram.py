#!/usr/bin/env python
# -*- coding:utf-8 -*-

from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory import GeneralizedLorenzMie
import numpy as np
try:
    import cupy as cp
    cp.cuda.Device()
    if 'Cuda' not in str(GeneralizedLorenzMie):
        raise Exception()
    from pylorenzmie.theory.cukernels import cuhologram
except Exception:
    cp = None
try:
    import numba as nb
    from pylorenzmie.theory.fastkernels import fasthologram
except Exception:
    nb = None


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
        if self.using_cuda:
            field = self.field()
            hologram = self.holo
            alpha = self.alpha
            alpha = cp.float32(alpha)
            Ex, Ey, Ez = field
            cuhologram((self.blockspergrid,), (self.threadsperblock,),
                       (Ex, Ey, Ez, alpha, hologram.size, hologram))
            if return_gpu is False:
                hologram = hologram.get()
        elif self.using_numba:
            field = self.field()
            hologram = self.holo
            fasthologram(field, self.alpha, hologram.size, hologram)
        else:
            field = self.alpha * self.field()
            field[0, :] += 1.
            hologram = np.sum(np.real(field * np.conj(field)), axis=0)
        return hologram


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Instrument import coordinates
    from time import time

    shape = [201, 201]
    h = LMHologram(coordinates=coordinates(shape))
    h.particle.r_p = [125, 75, 100]
    h.particle.a_p = 0.9
    h.particle.n_p = 1.45
    h.instrument.wavelength = 0.447
    h.hologram()
    start = time()
    hol = h.hologram()
    print("Time to calculate {}".format(time() - start))
    plt.imshow(hol.reshape(shape), cmap='gray')
    plt.show()
