#!/usr/bin/env python
# -*- coding:utf-8 -*-

from . import LorenzMie
import numpy as np

class LMHologram(LorenzMie):

    '''
    Compute in-line holograms of spheres

    ...

    Properties
    ----------
    alpha : float, optional
        weight of scattered field in superposition

    Methods
    -------
    hologram() : numpy.ndarray
        Computed hologram of sphere
    '''

    def __init__(self, *args, alpha=1., **kwargs):
        super(LMHologram, self).__init__(*args, **kwargs)
        self.alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = float(alpha)

    @LorenzMie.properties.getter
    def properties(self):
        p = LorenzMie.properties.fget(self)
        p['alpha'] = self.alpha
        return p

    def hologram(self):
        '''Return hologram of sphere

        Returns
        -------
        hologram : numpy.ndarray
            Computed hologram.
        '''
        try:
            field = self.alpha * self.field()
        except TypeError:
            return None
        field[0, :] += 1.
        hologram = np.sum(np.real(field * np.conj(field)), axis=0)
        return hologram


if __name__ == '__main__': # pragma: no cover
    import matplotlib.pyplot as plt
    from pylorenzmie.utilities import coordinates
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
