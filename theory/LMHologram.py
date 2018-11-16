#!/usr/bin/env python
# -*- coding:utf-8 -*-

from pylorenzmie.theory.LorenzMie import LorenzMie
import numpy as np


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

    def hologram(self):
        '''Return hologram of sphere

        Returns
        -------
        hologram : numpy.ndarray
            Computed hologram.
        '''
        field = self.alpha * self.field()
        field[0, :] += 1.
        return np.sum(np.real(field * np.conj(field)), axis=0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Instrument import coordinates

    shape = [201, 251]
    h = LMHologram(coordinates=coordinates(shape))
    h.particle.r_p = [125, 75, 100]
    h.particle.a_p = 0.9
    h.particle.n_p = 1.45
    h.instrument.wavelength = 0.447
    plt.imshow(h.hologram().reshape(shape), cmap='gray')
    plt.show()
