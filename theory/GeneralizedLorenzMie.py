#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from Particle import Particle
from Instrument import Instrument

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


class GeneralizedLorenzMie(object):
    def __init__(self,
                 coordinates=None,
                 particle=None,
                 instrument=None):
        self.coordinates = coordinates
        self.particle = particle
        self.instrument = instrument

    @property
    def coordinates(self):
        '''Three-dimensional coordinates at which field is calculated'''
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates = coordinates

    @property
    def particle(self):
        return self._particle

    @particle.setter
    def particle(self, particle):
        if isinstance(particle, Particle):
            self._particle = particle

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        if isinstance(instrument, Instrument):
            self._instrument = instrument

    def field(self, cartesian=True, bohren=True, strength=False):
        '''Compute scattered field'''
        if (self.coordinates is None or
            self.particle is None or
                self.instrument is None):
            return None

        # scattering coefficients
        ab = self.particle.ab(self.instrument.n_m, self.instrument.wavelength)
        nc = ab.shape[0]  # number of partial waves in sum

        # wavenumber in medium [pixel^-1]
        k = self.instrument.wavenumber()

        # GEOMETRY
        # 1. particle displacement [pixel]
        # Note: The sign convention used here is appropriate
        # for illumination propagating in the -z direction.
        # This means that a particle forming an image in the
        # focal plane (z = 0) is located at positive z.
        # Accounting for this by flipping the axial coordinate
        # is equivalent to using a mirrored (left-handed)
        # coordinate system.
        dr = self.coordinates - self.particle.r_p
        x = dr[:, 0]
        y = dr[:, 1]
        z = -dr[:, 2]
        npts = len(x)

        # 2. geometric factors
        phi = np.arctan2(y, x)
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)

        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(rho, z)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        kr = k * np.sqrt(rho**2 + z**2)
        sinkr = np.sin(kr)
        coskr = np.cos(kr)

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
        if bohren:
            factor = 1.j * np.sign(z)
        else:
            factor = -1.j * np.sign(z)
        xi_nm2 = coskr + factor * sinkr  # \xi_{-1}(kr)
        xi_nm1 = sinkr - factor * coskr  # \xi_0(kr)

        # 2. Angular functions (4.47), page 95
        pi_nm1 = 0.                      # \pi_0(\cos\theta)
        pi_n = 1.                        # \pi_1(\cos\theta)

        # 3. Vector spherical harmonics: [r,theta,phi]
        mo1n = np.empty([3, npts], complex)
        mo1n[0, :] = 0.j                 # no radial component
        ne1n = np.empty([3, npts], complex)

        # storage for scattered field
        es = np.zeros([3, npts], complex)

        # COMPUTE field by summing partial waves
        for n in range(1, nc):
            # upward recurrences ...
            # 4. Legendre factor (4.47)
            # Method described by Wiscombe (1980)
            swisc = pi_n * costheta
            twisc = swisc - pi_nm1
            tau_n = pi_nm1 - n * twisc  # -\tau_n(\cos\theta)

            # ... Riccati-Bessel function, page 478
            xi_n = (2.*n - 1.) * (xi_nm1 / kr) - xi_nm2  # \xi_n(kr)

            # ... Deirmendjian's derivative
            dn = (n * xi_n) / kr - xi_nm1

            # vector spherical harmonics (4.50)
            # mo1n[0, :] = 0.j           # no radial component
            mo1n[1, :] = pi_n * xi_n     # ... divided by cosphi/kr
            mo1n[2, :] = tau_n * xi_n    # ... divided by sinphi/kr

            # ... divided by cosphi sintheta/kr^2
            ne1n[0, :] = n*(n + 1.) * pi_n * xi_n
            ne1n[1, :] = tau_n * dn      # ... divided by cosphi/kr
            ne1n[2, :] = pi_n * dn       # ... divided by sinphi/kr

            # prefactor, page 93
            en = 1.j**n * (2.*n + 1.) / n / (n + 1.)

            # the scattered field in spherical coordinates (4.45)
            es += (1.j * en * ab[n, 0]) * ne1n
            es -= (en * ab[n, 1]) * mo1n

            # upward recurrences ...
            # ... angular functions (4.47)
            # Method described by Wiscombe (1980)
            pi_nm1 = pi_n
            pi_n = swisc + ((n + 1.)/n) * twisc

            # ... Riccati-Bessel function
            xi_nm2 = xi_nm1
            xi_nm1 = xi_n
        # n: multipole sum

        # geometric factors were divided out of the vector
        # spherical harmonics for accuracy and efficiency ...
        # ... put them back at the end.
        if strength:
            # Compute the electric field strength factor
            # by removing radial dependence.
            radialfactor = np.exp(-1.j*kr) / k
        else:
            radialfactor = 1. / kr
        es[0, :] *= cosphi * sintheta * radialfactor / kr
        es[1, :] *= cosphi * radialfactor
        es[2, :] *= sinphi * radialfactor

        # By default, the scattered wave is returned in spherical
        # coordinates.  Project components onto Cartesian coordinates.
        # Assumes that the incident wave propagates along z and
        # is linearly polarized along x
        if cartesian:
            ec = np.empty_like(es)

            ec[0, :] = es[0, :] * sintheta * cosphi
            ec[0, :] += es[1, :] * costheta * cosphi
            ec[0, :] -= es[2, :] * sinphi

            ec[1, :] = es[0, :] * sintheta * sinphi
            ec[1, :] += es[1, :] * costheta * sinphi
            ec[1, :] += es[2, :] * cosphi

            ec[2, :] = es[0, :] * costheta - es[1, :] * sintheta
            return ec
        else:
            return es


if __name__ == '__main__':
    from Sphere import Sphere
    import matplotlib.pyplot as plt

    # Create coordinate grid for image
    x = np.arange(0, 201)
    y = np.arange(0, 201)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    zv = np.zeros_like(xv)
    coordinates = np.stack((xv, yv, zv)).T
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
    kernel = GeneralizedLorenzMie(coordinates, particle, instrument)
    field = kernel.field()
    # Compute hologram from field and show it
    field *= np.exp(-1.j*k*particle.z_p)
    field[0, :] += 1.
    hologram = np.sum(np.real(field*np.conj(field)), axis=0)
    plt.imshow(hologram.reshape(201, 201), cmap='gray')
    plt.show()
