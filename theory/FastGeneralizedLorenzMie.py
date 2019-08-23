#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pylorenzmie.theory.GeneralizedLorenzMie import GeneralizedLorenzMie
from pylorenzmie.theory.fastkernels import fastfield

'''
This object uses generalized Lorenz-Mie theory to compute the
in-line hologram of a particle with specified Lorenz-Mie scattering
coefficients.  The hologram is calculated at specified
three-dimensional coordinates under the assumption that the
incident illumination is a plane wave linearly polarized along x.

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


class FastGeneralizedLorenzMie(GeneralizedLorenzMie):

    '''
    A class that computes scattered light fields with numba
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
        super(FastGeneralizedLorenzMie, self).__init__(**kwargs)
        self._using_numba = True

    def _allocate(self, shape):
        '''Allocates ndarrays for calculation'''
        self.this = np.empty(shape, dtype=np.complex128)
        self.result = np.empty(shape, dtype=np.complex128)
        self.holo = np.empty(shape[1], dtype=np.float64)

    def field(self, cartesian=True, bohren=True):
        '''Return field scattered by particles in the system'''
        if (self.coordinates is None or self.particle is None):
            return None

        if not hasattr(self, 'this'):
            self._allocate(self.coordinates.shape)
        elif self.this.shape != self.coordinates.shape:
            self._allocate(self.coordinates.shape)

        self.result.fill(0.+0.j)

        k = self.instrument.wavenumber()
        for p in np.atleast_1d(self.particle):
            ab = p.ab(self.instrument.n_m,
                      self.instrument.wavelength)
            phase = np.exp(-1.j * k * p.z_p)
            fastfield(self.coordinates, p.r_p, k, phase,
                      ab, self.this, cartesian, bohren)
            self.result += self.this
        return self.result


if __name__ == '__main__':
    from pylorenzmie.theory.FastSphere import FastSphere
    from pylorenzmie.theory.Instrument import Instrument
    import matplotlib.pyplot as plt
    # from time import time
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
    # Form image with default instrument
    instrument = Instrument()
    instrument.magnification = 0.135
    instrument.wavelength = 0.447
    instrument.n_m = 1.335
    k = instrument.wavenumber()
    # Use Generalized Lorenz-Mie theory to compute field
    kernel = FastGeneralizedLorenzMie(coordinates=coordinates,
                                      particle=particle,
                                      instrument=instrument)
    kernel.field()
    start = time()
    field = kernel.field()
    # Compute hologram from field and show it
    field[0, :] += 1.
    hologram = np.sum(np.real(field * np.conj(field)), axis=0)
    print("Time to calculate: {}".format(time() - start))
    plt.imshow(hologram.reshape(201, 201), cmap='gray')
    plt.show()
