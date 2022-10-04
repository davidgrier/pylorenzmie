#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pylorenzmie.lib import LMObject
from pylorenzmie.theory import (Particle, Sphere, Instrument)
from typing import Union, Optional, Any

from pylorenzmie.utilities import configuration as config

if config.has_numba():
    from numba import njit
else:  # pragma: no cover
    from pylorenzmie.utilities.numba import njit

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

'''
This object uses generalized Lorenz-Mie theory to compute the
electric field scattered by a particle with specified Lorenz-Mie
scattering coefficients. The field is calculated at specified
three-dimensional coordinates under the assumption that the
incident illumination is a plane wave that is linearly polarized
along x and is propagating along -z.

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

np.seterr(all='raise')


class LorenzMie(LMObject):
    '''
    Compute scattered light field with Generalized Lorenz-Mie theory

    ...

    Properties
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

    method = 'numpy'

    def __init__(self,
                 coordinates: np.ndarray = None,
                 particle: Particle = None,
                 instrument: Instrument = None,
                 **kwargs: Optional[Any]) -> None:
        '''
        Keywords
        ----------
        coordinates : numpy.ndarray
           [3, npts] array of x, y and z coordinates where field
           is calculated
        particle : Particle
           Object representing the particle. Default: Sphere()
        instrument : Instrument
           Object resprenting the light-scattering instrument
        '''
        self.coordinates = coordinates
        self.particle = particle or Sphere(**kwargs)
        self.instrument = instrument or Instrument(**kwargs)

    def __str__(self) -> str:
        fmt = '<{}(particle=part, instrument=inst)>\n\t'
        fmt += 'part = {}\n\tinst = {}'
        return fmt.format(self.__class__.__name__,
                          str(self.particle),
                          str(self.instrument))

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def coordinates(self) -> np.ndarray:
        '''Three-dimensional coordinates at which field is calculated

        Expected shape is (3, npts)
        '''
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: np.ndarray) -> None:
        if coordinates is None:
            self._coordinates = None
            return
        c = np.array(coordinates)
        if c.ndim == 1:            # only x specified
            c = np.vstack((c, np.zeros((2, c.size))))
        elif c.shape[0] == 2:      # only (x, y) specified
            z = np.zeros_like(c[0])
            c = np.vstack((c, z))
        elif c.shape[0] != 3:      # pragma: no cover
            raise ValueError(
                'coordinates should have shape ({1|2|3}, npts).')
        self._coordinates = c
        self.allocate()

    @property
    def particle(self) -> Particle:
        '''Particle responsible for light scattering'''
        return self._particle

    @particle.setter
    def particle(self, particle: Union[Particle, list]) -> None:
        logger.debug('Setting particle')
        p = np.atleast_1d(particle)
        if isinstance(p[0], Particle):
            self._particle = particle

    @property
    def instrument(self) -> Instrument:
        '''Imaging instrument'''
        return self._instrument

    @instrument.setter
    def instrument(self, instrument: Instrument) -> None:
        logger.debug('Setting instrument')
        if isinstance(instrument, Instrument):
            self._instrument = instrument

    @property
    def properties(self) -> dict:
        p = dict()
        p.update(self.particle.properties)
        p.update(self.instrument.properties)
        return p

    @properties.setter
    def properties(self, properties: dict) -> None:
        # Set properties of components
        self.particle.properties = properties
        self.instrument.properties = properties
        # Set own properties: useful for subclassing
        for property, value in properties.items():  # pragma: no cover
            if hasattr(self, property):
                setattr(self, property, value)

    def field(self,
              cartesian: bool = True,
              bohren: bool = True) -> np.ndarray:
        '''Return field scattered by particles in the system'''
        if (self.coordinates is None or self.particle is None):
            return None
        logger.debug('Computing field')
        k = self.instrument.wavenumber()
        n_m = self.instrument.n_m
        wavelength = self.instrument.wavelength
        self.result.fill(0.+0.j)
        for p in np.atleast_1d(self.particle):
            logger.debug(p)
            r_p = p.r_p + p.r_0
            dr = self.coordinates - r_p[:, None]
            self.kdr[...] = np.asarray(k * dr)
            ab = p.ab(n_m, wavelength)
            this = self.compute(ab, self.kdr, *self.buffers,
                                cartesian=cartesian, bohren=bohren)
            this *= np.exp(-1j * k * r_p[2])
            this *= self.correction(dr)
            self.result += this
        return self.result

    def correction(self, dr: np.ndarray):
        '''Correction should be overriden by subclass'''
        return 1.

    def allocate(self) -> None:
        '''Allocate ndarrays for calculation'''
        shape = self.coordinates.shape
        self.kdr = np.empty(shape, dtype=float)
        self.buffers = [np.empty(shape, dtype=complex) for _ in range(4)]
        self.result = np.empty(shape, dtype=complex)

    @staticmethod
    @njit()  # unittest does not cover jitted methods
    def compute(ab: np.ndarray,
                kdr: np.ndarray,
                mo1n: np.ndarray,
                ne1n: np.ndarray,
                es: np.ndarray,
                ec: np.ndarray,
                cartesian: bool = True,
                bohren: bool = True) -> np.ndarray:  # pragma: no cover
        '''Returns the field scattered by the particle at each coordinate

        Arguments
        ----------
        ab : numpy.ndarray
            [2, norders] Mie scattering coefficients
        kdr : numpy.ndarray
            [3, npts] Coordinates at which field is evaluated
            relative to the center of the scatterer. Coordinates
            are assumed to be multiplied by the wavenumber of
            light in the medium, and so are dimensionless.

        Keywords
        --------
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
        kx = kdr[0, :]
        ky = kdr[1, :]
        kz = -kdr[2, :]
        shape = kx.shape

        # 2. geometric factors
        krho = np.hypot(kx, ky)
        kr = np.hypot(krho, kz)

        phi = np.arctan2(ky, kx)
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        theta = np.arctan2(krho, kz)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
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
        factor = 1.j * np.sign(kz) if bohren else -1.j * np.sign(kz)

        xi_nm2 = coskr + factor * sinkr  # \xi_{-1}(kr)
        xi_nm1 = sinkr - factor * coskr  # \xi_0(kr)

        # 2. Angular functions (4.47), page 95
        # \pi_0(\cos\theta)
        pi_nm1 = np.zeros(shape)
        # \pi_1(\cos\theta)
        pi_n = np.ones(shape)

        # 3. Vector spherical harmonics: [r,theta,phi]
        mo1n[0, :] = 0.j                 # no radial component

        # storage for scattered field
        es.fill(0.j)

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
            mo1n[1, :] = pi_n * xi_n     # ... divided by cosphi/kr
            mo1n[2, :] = tau_n * xi_n    # ... divided by sinphi/kr

            # ... divided by cosphi sintheta/kr^2
            ne1n[0, :] = n * (n + 1.) * pi_n * xi_n
            ne1n[1, :] = tau_n * dn      # ... divided by cosphi/kr
            ne1n[2, :] = pi_n * dn       # ... divided by sinphi/kr

            # prefactor, page 93
            en = 1.j**n * (2. * n + 1.) / n / (n + 1.)

            # the scattered field in spherical coordinates (4.45)
            es += (1.j * en * ab[n, 0]) * ne1n
            es -= (en * ab[n, 1]) * mo1n

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
        es[0, :] *= cosphi * sintheta * radialfactor**2
        es[1, :] *= cosphi * radialfactor
        es[2, :] *= sinphi * radialfactor

        # By default, the scattered wave is returned in spherical
        # coordinates.  Project components onto Cartesian coordinates.
        # Assumes that the incident wave propagates along z and
        # is linearly polarized along x

        if cartesian:
            ec[0, :] = es[0, :] * sintheta * cosphi
            ec[0, :] += es[1, :] * costheta * cosphi
            ec[0, :] -= es[2, :] * sinphi

            ec[1, :] = es[0, :] * sintheta * sinphi
            ec[1, :] += es[1, :] * costheta * sinphi
            ec[1, :] += es[2, :] * cosphi
            ec[2, :] = (es[0, :] * costheta -
                        es[1, :] * sintheta)
            return ec
        else:
            return es


if __name__ == '__main__':  # pragma: no cover
    import matplotlib.pyplot as plt
    from pylorenzmie.utilities import coordinates
    from time import perf_counter

    # Create coordinate grid for image
    coords = coordinates((201, 201))
    # Place two spheres in the field of view, above the focal plane
    pa = Sphere()
    pa.r_p = [150, 150, 200]
    pa.a_p = 0.5
    pa.n_p = 1.45
    pb = Sphere()
    pb.r_p = [100, 10, 75]
    pb.a_p = 0.75
    pb.n_p = 1.45
    particle = [pa, pb]
    # Form image with default instrument
    instrument = Instrument()
    instrument.magnification = 0.048
    instrument.wavelength = 0.447
    instrument.n_m = 1.340
    k = instrument.wavenumber()
    # Use Generalized Lorenz-Mie theory to compute field
    kernel = LorenzMie(coords, particle, instrument)
    kernel.field()
    start = perf_counter()
    field = kernel.field()
    print(f'Time to calculate: {perf_counter()-start} s')
    # Compute hologram from field and show it
    field[0, :] += 1.
    hologram = np.sum(np.real(field * np.conj(field)), axis=0)
    plt.imshow(hologram.reshape(201, 201), cmap='gray')
    plt.show()
