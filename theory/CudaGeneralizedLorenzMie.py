#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylorenzmie.theory.GeneralizedLorenzMie import GeneralizedLorenzMie
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import numpy as np

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


class CudaGeneralizedLorenzMie(GeneralizedLorenzMie):

    '''
    A class that computes scattered light fields with CUDA acceleration

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

    def __init__(self, *args, **kwargs):
        super(CudaGeneralizedLorenzMie, self).__init__(*args, **kwargs)

    def _allocate(self, shape):
        '''Allocates GPUArrays for calculation'''
        self.krv = gpuarray.empty(shape, dtype=np.float32)
        self.mo1n = gpuarray.empty(shape, dtype=np.complex64)
        self.ne1n = gpuarray.empty(shape, dtype=np.complex64)
        self.es = gpuarray.empty(shape, dtype=np.complex64)
        self.ec = gpuarray.empty(shape, dtype=np.complex64)
        self.result = gpuarray.empty(shape, dtype=np.complex64)

    def compute(self, ab, krv, cartesian=True, bohren=True):
        '''Returns the field scattered by the particle at each coordinate

        Parameters
        ----------
        ab : numpy.ndarray
            [2, norders] Mie scattering coefficients
        krv : numpy.ndarray
            Reduced vector displacements of particle from image coordinates
        cartesian : bool
            If set, return field projected onto Cartesian coordinates.
            Otherwise, return polar projection.
        bohren : bool
            If set, use sign convention from Bohren and Huffman.
            Otherwise, use opposite sign convention.

        Returns
        -------
        field : numpy.ndarray
            [3, npts] array of complex vector values of the
            scattered field at each coordinate.
        '''

        norders = ab.shape[0]  # number of partial waves in sum

        # GEOMETRY
        # 1. particle displacement [pixel]
        # Note: The sign convention used here is appropriate
        # for illumination propagating in the -z direction.
        # This means that a particle forming an image in the
        # focal plane (z = 0) is located at positive z.
        # Accounting for this by flipping the axial coordinate
        # is equivalent to using a mirrored (left-handed)
        # coordinate system.
        self.krv.set(krv.astype(np.float32))
        kx = self.krv[0, :]
        ky = self.krv[1, :]
        kz = self.krv[2, :]
        # rather than setting kz = -kz, we propagate the - sign below

        # 2. geometric factors
        krho = kx * kx
        krho += ky * ky
        kr = krho + kz * kz
        kr = cumath.sqrt(kr)
        krho = cumath.sqrt(krho)

        cosphi = kx / krho
        sinphi = ky / krho
        costheta = -kz / kr  # z convention
        sintheta = krho / kr
        sinkr = cumath.sin(kr)
        coskr = cumath.cos(kr)

        # SPECIAL FUNCTIONS
        # starting points for recursive function evaluation ...
        # 1. Riccati-Bessel radial functions, page 478.
        # Particles above the focal plane create diverging waves
        # described by Eq. (4.13) for $h_n^{(1)}(kr)$. These have z > 0.
        # Those below the focal plane appear to be converging from the
        # perspective of the camera. They are descrinbed by Eq. (4.14)
        # for $h_n^{(2)}(kr)$, and have z < 0. We can select the
        # appropriate case by applying the correct sign of the imaginary
        # part of the starting functions...
        factor = 1.j * kz / abs(kz)
        if bohren:           # z convention
            factor *= -1.
        xi_nm2 = coskr + factor * sinkr  # \xi_{-1}(kr)
        xi_nm1 = sinkr - factor * coskr  # \xi_0(kr)

        # 2. Angular functions (4.47), page 95
        pi_nm1 = 0.                      # \pi_0(\cos\theta)
        pi_n = 1.                        # \pi_1(\cos\theta)

        # 3. Vector spherical harmonics: [r,theta,phi]
        self.mo1n.fill(np.complex64(0.j))

        # 4. Scattered field
        self.es.fill(np.complex64(0.j))

        # COMPUTE field by summing partial waves
        for n in range(1, norders):
            # upward recurrences ...
            # 4. Legendre factor (4.47)
            # Method described by Wiscombe (1980)
            swisc = pi_n * costheta
            twisc = swisc - pi_nm1
            tau_n = pi_nm1 - n * twisc  # -\tau_n(\cos\theta)

            # ... Riccati-Bessel function, page 478
            xi_n = (2. * n - 1.) * (xi_nm1 / kr) - xi_nm2  # \xi_n(kr)

            # ... Deirmendjian's derivative
            dn = (n * xi_n) / kr - xi_nm1

            # vector spherical harmonics (4.50)
            # mo1n[0, :] = 0.j           # no radial component
            self.mo1n[1, :] = pi_n * xi_n     # ... divided by cosphi/kr
            self.mo1n[2, :] = tau_n * xi_n    # ... divided by sinphi/kr

            # ... divided by cosphi sintheta/kr^2
            self.ne1n[0, :] = n * (n + 1.) * pi_n * xi_n
            self.ne1n[1, :] = tau_n * dn      # ... divided by cosphi/kr
            self.ne1n[2, :] = pi_n * dn       # ... divided by sinphi/kr

            # prefactor, page 93
            en = 1.j**n * (2. * n + 1.) / n / (n + 1.)

            # the scattered field in spherical coordinates (4.45)
            self.es += np.complex64(1.j * en * ab[n, 0]) * self.ne1n
            self.es -= np.complex64(en * ab[n, 1]) * self.mo1n

            # upward recurrences ...
            # ... angular functions (4.47)
            # Method described by Wiscombe (1980)
            pi_nm1 = pi_n
            pi_n = swisc + ((n + 1.) / n) * twisc

            # ... Riccati-Bessel function
            xi_nm2 = xi_nm1
            xi_nm1 = xi_n
        # n: multipole sum

        # geometric factors were divided out of the vector
        # spherical harmonics for accuracy and efficiency ...
        # ... put them back at the end.
        radialfactor = 1. / kr
        self.es[0, :] *= cosphi * sintheta * radialfactor**2
        self.es[1, :] *= cosphi * radialfactor
        self.es[2, :] *= sinphi * radialfactor

        # By default, the scattered wave is returned in spherical
        # coordinates.  Project components onto Cartesian coordinates.
        # Assumes that the incident wave propagates along z and
        # is linearly polarized along x
        if cartesian:
            self.ec[0, :] = self.es[0, :] * sintheta * cosphi
            self.ec[0, :] += self.es[1, :] * costheta * cosphi
            self.ec[0, :] -= self.es[2, :] * sinphi

            self.ec[1, :] = self.es[0, :] * sintheta * sinphi
            self.ec[1, :] += self.es[1, :] * costheta * sinphi
            self.ec[1, :] += self.es[2, :] * cosphi

            self.ec[2, :] = self.es[0, :] * costheta - self.es[1, :] * sintheta
            return self.ec
        else:
            return self.es

    def field(self, cartesian=True, bohren=True, return_gpu=False):
        '''Return field scattered by particles in the system'''
        if (self.coordinates is None or self.particle is None):
            return None

        k = self.instrument.wavenumber()
        self.result.fill(np.complex64(0.j))
        for p in np.atleast_1d(self.particle):
            krv = k * (self.coordinates - p.r_p[:, None])
            ab = p.ab(self.instrument.n_m,
                      self.instrument.wavelength)
            this = self.compute(ab, krv,
                                cartesian=cartesian, bohren=bohren)
            this *= np.complex64(np.exp(-1j * k * p.z_p))
            self.result += this
        if return_gpu:
            return self.result
        else:
            return self.result.get()


if __name__ == '__main__':
    from Instrument import Instrument
    from Sphere import Sphere
    import matplotlib.pyplot as plt

    # Create coordinate grid for image
    x = np.arange(0, 201)
    y = np.arange(0, 201)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    zv = np.zeros_like(xv)
    coordinates = np.stack((xv, yv, zv))
    # Place a sphere in the field of view, above the focal plane
    particle = Sphere()
    particle.r_p = [125, 75, 100]
    particle.a_p = 0.5
    particle.n_p = 1.45
    # Form image with default instrument
    instrument = Instrument()
    instrument.magnification = 0.135
    instrument.wavelength = 0.447
    instrument.n_m = 1.335
    k = instrument.wavenumber()
    # Use Generalized Lorenz-Mie theory to compute field
    kernel = CudaGeneralizedLorenzMie(coordinates, particle, instrument)
    field = kernel.field()
    # Compute hologram from field and show it
    field *= np.complex64(np.exp(-1.j * k * particle.z_p))
    field[0, :] += 1.
    hologram = np.sum(np.real(field * np.conj(field)), axis=0)
    plt.imshow(hologram.reshape(201, 201), cmap='gray')
    plt.show()
