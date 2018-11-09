#!/usr/bin/env python
# -*- coding:utf-8 -*-

from CudaLorenzMie import CudaLorenzMie
import numpy as np


class CudaLMHologram(CudaLorenzMie):

    '''
    A class that computes in-line holograms of spheres with CUDA acceleration

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
        super(CudaLMHologram, self).__init__(*args, **kwargs)
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
        gpufield = self.alpha * self.field(return_gpu=True)
        gpufield[0, :] += 1.
        gpufield = gpufield * gpufield.conj()
        return np.sum(gpufield.real.get(), axis=0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Instrument import coordinates

    shape = [201, 251]
    h = CudaLMHologram(coordinates=coordinates(shape))
    h.particle.r_p = [125, 75, 100]
    h.instrument.wavelength = 0.447
    plt.imshow(h.hologram().reshape(shape), cmap='gray')
    plt.show()
