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
except Exception:
    cp = None

if cp is not None:
    compute = cp.RawKernel(r'''
#include <cuComplex.h>

extern "C" __global__
void hologram(cuFloatComplex *Ex,
              cuFloatComplex *Ey,
              cuFloatComplex *Ez,
              float alpha,
              int n,
              float *hologram) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < n;          idx += blockDim.x * gridDim.x) {
        cuFloatComplex ex = Ex[idx];
        cuFloatComplex ey = Ey[idx];
        cuFloatComplex ez = Ez[idx];

        ex = cuCaddf(ex, make_cuFloatComplex(1., 0.));

        ex = cuCmulf(ex, make_cuFloatComplex(alpha, 0.));
        ey = cuCmulf(ey, make_cuFloatComplex(alpha, 0.));
        ez = cuCmulf(ez, make_cuFloatComplex(alpha, 0.));

        cuFloatComplex ix = cuCmulf(ex, cuConjf(ex));
        cuFloatComplex iy = cuCmulf(ey, cuConjf(ey));
        cuFloatComplex iz = cuCmulf(ez, cuConjf(ez));

        hologram[idx] = cuCrealf(cuCaddf(ix, cuCaddf(iy, iz)));
    }
}
''', 'hologram')


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
        if cp is not None:
            self._using_gpu = True
        else:
            self._using_gpu = False

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = float(alpha)

    @property
    def using_gpu(self):
        return self._using_gpu

    def hologram(self, return_gpu=False):
        '''Return hologram of sphere

        Returns
        -------
        hologram : numpy.ndarray
            Computed hologram.
        '''
        if self._using_gpu:
            hologram = self.holo
            field = self.field()
            alpha = self.alpha
            alpha = cp.float32(alpha)
            Ex, Ey, Ez = field
            compute((self.blockspergrid,), (self.threadsperblock,),
                    (Ex, Ey, Ez, alpha, hologram.size, hologram))
            if return_gpu is False:
                hologram = hologram.get()
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
