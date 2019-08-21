#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cupy as cp
from pylorenzmie.theory.cukernels import cufield
from pylorenzmie.theory.GeneralizedLorenzMie import GeneralizedLorenzMie


cp.cuda.Device()

'''
This object uses generalized Lorenz-Mie theory to compute the
in-line hologram of a particle with specified Lorenz-Mie scattering
coefficients.  The hologram is calculated at specified
three-dimensional coordinates under the assumption that the
incident illumination is a plane wave linearly polarized along x.
q
REFERENCES:
1. Adapted from Chapter 4 in
   C. F. Bohren and D. R. Huffman,
   Absorption and Scattering of Light by Small Particles,
   (New York, Wiley, 1983).

2. W. J. Wiscombe, "Improved Mie scattering algorithms,"
   Appl. Opt. 19, 1505-1509 (1980).

3. W. J. Lentz, "Generating Bessel function in Mie scattering
   calculations using continued fractions," Appl. Opt. 15,
   668-671 (1976).

4. S. H. Lee, Y. Roichman, G. R. Yi, S. H. Kim, S. M. Yang,
   A. van Blaaderen, P. van Oostrum and D. G. Grier,
   "Characterizing and tracking single colloidal particles with
   video holographic microscopy," Opt. Express 15, 18275-18282
   (2007).

5. F. C. Cheong, B. Sun, R. Dreyfus, J. Amato-Grill, K. Xiao,
   L. Dixon and D. G. Grier,
   "Flow visualization and flow cytometry with holographic video
   microscopy," Opt. Express 17, 13071-13079 (2009).

HISTORY
This code was adapted from the IDL implementation of
generalizedlorenzmie__define.pro
which was written by David G. Grier.
This version is

Copyright (c) 2018 David G. Grier
'''


class CudaGeneralizedLorenzMie(GeneralizedLorenzMie):

    '''
    A class that computes scattered light fields with CUDA 
    acceleration

    ...

    Attributes
    ----------
    particle : Particle
        Object representing the particle scattering light
    instrument : Instrument
        Object resprenting the light-scattering instrument
    coordinates : numpy.ndarray
        [3, npts] array of x, y and z coordinates where field
        is calculated

    Methods
    -------
    field(cartesian=True, bohren=True)
        Returns the complex-valued field at each of the coordinates.
    '''

    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        coordinates : numpy.ndarray
           [3, npts] array of x, y and z coordinates where field
           is calculated
        particle : Particle
           Object representing the particle scattering light
        instrument : Instrument
           Object resprenting the light-scattering instrument
        n_m : complex, optional
           Refractive index of medium
        magnification : float, optional
           Magnification of microscope [um/pixel]
        wavelength : float, optional
           Vacuum wavelength of light [um]
        '''
        super(CudaGeneralizedLorenzMie, self).__init__(**kwargs)

    def _allocate(self, shape):
        '''Allocates ndarrays for calculation'''
        self.this = cp.empty(shape, dtype=np.complex64)
        self.result = cp.empty(shape, dtype=np.complex64)
        self.device_coordinates = cp.asarray(self.coordinates
                                             .astype(np.float32))
        self.holo = cp.empty(shape[1], dtype=np.float32)
        self.threadsperblock = 32
        self.blockspergrid = (shape[1] + (self.threadsperblock - 1))\
            // self.threadsperblock

    def field(self, cartesian=True, bohren=True):
        '''Return field scattered by particles in the system'''
        if (self.coordinates is None or self.particle is None):
            return None

        if not hasattr(self, 'this'):
            self._allocate(self.coordinates.shape)
        elif self.this.shape != self.coordinates.shape:
            self._allocate(self.coordinates.shape)

        self.result.fill(0.+0.j)

        k = np.float32(self.instrument.wavenumber())

        for p in np.atleast_1d(self.particle):
            ab = p.ab(self.instrument.n_m,
                      self.instrument.wavelength)
            ar = ab[:, 0].real.astype(np.float32)
            ai = ab[:, 0].imag.astype(np.float32)
            br = ab[:, 1].real.astype(np.float32)
            bi = ab[:, 1].imag.astype(np.float32)
            ar, ai, br, bi = cp.asarray([ar, ai, br, bi])
            coordsx, coordsy, coordsz = self.device_coordinates
            x_p, y_p, z_p = p.r_p.astype(np.float32)
            phase = np.complex64(np.exp(-1.j * k * z_p))
            cufield((self.blockspergrid,), (self.threadsperblock,),
                    (coordsx, coordsy, coordsz,
                     x_p, y_p, z_p, k, phase,
                     ar, ai, br, bi,
                     ab.shape[0], coordsx.shape[0],
                     cartesian, bohren,
                     *self.this))
            self.result += self.this
        return self.result


if __name__ == '__main__':
    from pylorenzmie.theory.FastSphere import FastSphere
    from pylorenzmie.theory.Instrument import Instrument
    import matplotlib.pyplot as plt
    from time import time
    # Create coordinate grid for image
    x = np.arange(0, 201)
    y = np.arange(0, 201)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    zv = np.zeros_like(xv)
    coordinates = np.stack((xv, yv, zv))
    # Place a sphere in the field of view, above the focal plane
    particle = FastSphere()
    particle.r_p = [150, 150, 200]
    particle.a_p = 0.5
    particle.n_p = 1.45
    particle2 = FastSphere()
    particles = [particle, particle2]
    particles.reverse()
    # Form image with default instrument
    instrument = Instrument()
    instrument.magnification = 0.135
    instrument.wavelength = 0.447
    instrument.n_m = 1.335
    k = instrument.wavenumber()
    # Use Generalized Lorenz-Mie theory to compute field
    kernel = CudaGeneralizedLorenzMie(coordinates=coordinates,
                                      particle=particles,
                                      instrument=instrument)
    kernel.field()
    start = time()
    field = kernel.field()
    print("Time to calculate field: {}".format(time() - start))
    # Compute hologram from field and show it
    field[0, :] += 1.
    hologram = cp.sum(cp.real(field * cp.conj(field)), axis=0)
    hologram = hologram.get()
    plt.imshow(hologram.reshape(201, 201), cmap='gray')
    plt.show()
