#!/usr/bin/env python
# -*- coding:utf-8 -*-

from LorenzMie import LorenzMie
import numpy as np


class LMHologram(LorenzMie):

    '''
    A class that computes in-line holograms of spheres

    ...

    Attributes
    ----------
    shape : list
        [height, width] specification of hologram shape
    alpha : float, optional
        weight of scattered field in superposition

    Methods
    -------
    hologram() : numpy.ndarray
        Computed hologram of sphere
    '''

    def __init__(self,
                 shape=[201, 201],
                 alpha=1.,
                 **kwargs):
        super(LMHologram, self).__init__(**kwargs)
        self.shape = shape
        self.alpha = alpha

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        (ny, nx) = shape
        x = np.arange(0, nx)
        y = np.arange(0, ny)
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten()
        yv = yv.flatten()
        zv = np.zeros_like(xv)
        self.coordinates = np.stack((xv, yv, zv))
        self._shape = shape

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
        res = np.sum(np.real(field * np.conj(field)), axis=0)
        return res.reshape(self.shape)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    h = LMHologram(shape=[201, 251])
    h.particle.r_p = [125, 75, 100]
    h.instrument.wavelength = 0.447
    plt.imshow(h.hologram(), cmap='gray')
    plt.show()
