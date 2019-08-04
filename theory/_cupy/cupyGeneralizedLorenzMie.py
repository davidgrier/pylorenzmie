#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pylorenzmie.theory.Particle import Particle
from pylorenzmie.theory.Instrument import Instrument
import json
import cupy as cp

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

safe_division = cp.ElementwiseKernel(
    "float32 x, float32 y, float32 a",
    "float32 z",
    "if (abs(y) > 1e-6) { z = x/y; } else { z = a; };",
    name="safe_division")


class cupyGeneralizedLorenzMie(object):

    '''
    A class that computes scattered light fields

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

    def __init__(self,
                 coordinates=None,
                 particle=None,
                 instrument=None,
                 n_m=None,
                 magnification=None,
                 wavelength=None):
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
        self.coordinates = coordinates
        self.particle = particle
        if instrument is None:
            self.instrument = Instrument()
        else:
            self.instrument = instrument
        if n_m is not None:
            self.instrument.n_m = n_m
        if magnification is not None:
            self.instrument.magnification = magnification
        if wavelength is not None:
            self.instrument.wavelength = wavelength

    @property
    def coordinates(self):
        '''Three-dimensional coordinates at which field is calculated'''
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        try:
            shape = coordinates.shape
        except AttributeError:
            self._coordinates = None
            return
        if coordinates.ndim == 1:
            self._coordinates = np.zeros((3, shape[0]))
            self._coordinates[0, :] = coordinates
        elif shape[0] == 2:
            self._coordinates = np.zeros((3, shape[1]))
            self._coordinates[[0, 1], :] = coordinates
        else:
            self._coordinates = coordinates
        self._allocate(self._coordinates.shape)

    @property
    def particle(self):
        '''Particle responsible for light scattering'''
        return self._particle

    @particle.setter
    def particle(self, particle):
        try:
            if isinstance(particle[0], Particle):
                self._particle = particle
        except TypeError:
            if isinstance(particle, Particle):
                self._particle = particle

    @property
    def instrument(self):
        '''Imaging instrument'''
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        if isinstance(instrument, Instrument):
            self._instrument = instrument

    def dumps(self, **kwargs):
        '''Returns JSON string of adjustable properties

        Parameters
        ----------
        Accepts all keywords of json.dumps()

        Returns
        -------
        str : string
            JSON-encoded string of properties
        '''
        return json.dumps(self.properties, **kwargs)
        s = {'particle': self.particle.dumps(**kwargs),
             'instrument': self.instrument.dumps(**kwargs)}
        return json.dumps(s, **kwargs)

    def loads(self, str):
        '''Loads JSON string of adjustable properties

        Parameters
        ----------
        str : string
            JSON-encoded string of properties
        '''
        s = json.loads(str)
        self.particle.loads(s['particle'])
        self.instrument.loads(s['instrument'])

    def _allocate(self, shape):
        '''Allocates ndarrays for calculation'''
        cmplx, flt = (np.complex64, np.float32)
        self.sinphi = cp.empty(shape[1], dtype=flt)
        self.cosphi = cp.empty(shape[1], dtype=flt)
        self.sintheta = cp.empty(shape[1], dtype=flt)
        self.costheta = cp.empty(shape[1], dtype=flt)
        self.krv = cp.empty(shape, dtype=flt)
        self.mo1n = cp.empty(shape, dtype=cmplx)
        self.ne1n = cp.empty(shape, dtype=cmplx)
        self.es = cp.empty(shape, dtype=cmplx)
        self.ec = cp.empty(shape, dtype=cmplx)
        self.result = cp.empty(shape, dtype=cmplx)

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
        shape = krv.shape
        kx = krv[0, :]
        ky = krv[1, :]
        kz = -krv[2, :]

        # 2. geometric factors
        krho = cp.sqrt(kx**2 + ky**2)
        kr = cp.sqrt(krho**2 + kz**2)

        self.cosphi[...] = safe_division(kx, krho, 1.)
        self.sinphi[...] = safe_division(ky, krho, 0.)
        self.costheta[...] = safe_division(kz, kr, 1.)  # z convention
        self.sintheta[...] = safe_division(krho, kr, 0.)
        sinkr = cp.sin(kr)
        coskr = cp.cos(kr)

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
            factor = 1.j * cp.sign(kz)
        else:
            factor = -1.j * cp.sign(kz)

        xi_nm2 = coskr + factor * sinkr  # \xi_{-1}(kr)
        xi_nm1 = sinkr - factor * coskr  # \xi_0(kr)

        # 2. Angular functions (4.47), page 95
        pi_nm1 = 0.                      # \pi_0(\cos\theta)
        pi_n = 1.                        # \pi_1(\cos\theta)

        # 3. Vector spherical harmonics: [r,theta,phi]
        self.mo1n[0, :] = 0.j                 # no radial component

        # storage for scattered field
        self.es.fill(0.j)

        # COMPUTE field by summing partial waves
        for n in range(1, norders):
            # upward recurrences ...
            # 4. Legendre factor (4.47)
            # Method described by Wiscombe (1980)

            swisc = pi_n * self.costheta
            twisc = swisc - pi_nm1
            tau_n = pi_nm1 - n * twisc  # -\tau_n(\cos\theta)

            # ... Riccati-Bessel function, page 478
            xi_n = (2. * n - 1.) * (xi_nm1 / kr) - xi_nm2  # \xi_n(kr)

            # ... Deirmendjian's derivative
            dn = (n * xi_n) / kr - xi_nm1

            # vector spherical harmonics (4.50)
            self.mo1n[1, :] = pi_n * xi_n     # ... divided by cosphi/kr
            self.mo1n[2, :] = tau_n * xi_n    # ... divided by sinphi/kr

            # ... divided by cosphi sintheta/kr^2
            self.ne1n[0, :] = n * (n + 1.) * pi_n * xi_n
            self.ne1n[1, :] = tau_n * dn      # ... divided by cosphi/kr
            self.ne1n[2, :] = pi_n * dn       # ... divided by sinphi/kr

            # prefactor, page 93
            en = 1.j**n * (2. * n + 1.) / n / (n + 1.)

            # the scattered field in spherical coordinates (4.45)
            self.es += (1.j * en * ab[n, 0]) * self.ne1n
            self.es -= (en * ab[n, 1]) * self.mo1n

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
        self.es[0, :] *= self.cosphi * self.sintheta * radialfactor**2
        self.es[1, :] *= self.cosphi * radialfactor
        self.es[2, :] *= self.sinphi * radialfactor

        # By default, the scattered wave is returned in spherical
        # coordinates.  Project components onto Cartesian coordinates.
        # Assumes that the incident wave propagates along z and
        # is linearly polarized along x

        if cartesian:
            self.ec[0, :] = self.es[0, :] * self.sintheta * self.cosphi
            self.ec[0, :] += self.es[1, :] * self.costheta * self.cosphi
            self.ec[0, :] -= self.es[2, :] * self.sinphi

            self.ec[1, :] = self.es[0, :] * self.sintheta * self.sinphi
            self.ec[1, :] += self.es[1, :] * self.costheta * self.sinphi
            self.ec[1, :] += self.es[2, :] * self.cosphi
            self.ec[2, :] = (self.es[0, :] * self.costheta -
                             self.es[1, :] * self.sintheta)
            return self.ec
        else:
            return self.es

    def field(self, cartesian=True, bohren=True):
        '''Return field scattered by particles in the system'''
        if (self.coordinates is None or self.particle is None):
            return None

        k = self.instrument.wavenumber()
        '''
        try:               # one particle in field of view
            krv = k * (self.coordinates - self.particle.r_p[:, None])
            ab = self.particle.ab(self.instrument.n_m,
                                  self.instrument.wavelength)
            field = self.compute(ab, krv,
                                 cartesian=cartesian, bohren=bohren)
            field *= np.exp(-1j * k * self.particle.z_p)
        except AttributeError:  # list of particles
        '''
        self.result.fill(0.j)
        for p in np.atleast_1d(self.particle):
            self.krv[...] = cp.asarray(k * (self.coordinates -
                                            p.r_p[:, None]))
            ab = p.ab(self.instrument.n_m,
                      self.instrument.wavelength)
            this = self.compute(ab, self.krv,
                                cartesian=cartesian, bohren=bohren)
            this *= cp.exp(-1j * k * p.z_p)
            try:
                self.result += this
            except NameError:
                self.result = this
        return self.result


if __name__ == '__main__':
    from Sphere import Sphere
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
    particle = Sphere()
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
    kernel = GeneralizedLorenzMie(coordinates, particle, instrument)
    kernel.field()
    start = time()
    field = kernel.field()
    print("Time to calculate: {}".format(time() - start))
    field = field.get()
    # Compute hologram from field and show it
    field *= np.exp(-1.j * k * particle.z_p)
    field[0, :] += 1.
    hologram = np.sum(np.real(field * np.conj(field)), axis=0)
    plt.imshow(hologram.reshape(201, 201), cmap='gray')
    plt.show()
