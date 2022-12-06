from pylorenzmie.lib import LMObject
from pylorenzmie.theory import (Particle, Sphere, Instrument)
from typing import (List, Dict, Optional, Union)
import numpy as np


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
    coordinates : numpy.ndarray
        [3, npts] array of x, y and z coordinates where field
        is calculated
    particle : Particle | List[Particle]
        Object representing the particle scattering light
    instrument : Instrument
        Object resprenting the light-scattering instrument

    Methods
    -------
    field(cartesian=True, bohren=True)
        Returns the complex-valued field at each of the coordinates.
    '''

    method: str = 'numpy'

    def __init__(self,
                 coordinates: np.ndarray = None,
                 particle: Optional[Union[Particle, List[Particle]]] = None,
                 instrument: Optional[Instrument] = None) -> None:
        super().__init__()
        self.coordinates = coordinates
        self.particle = particle or Sphere()
        self.instrument = instrument or Instrument()

    def __repr__(self) -> str:
        part = f'particle={self.particle!r}'
        inst = f'instrument={self.instrument!r}'
        args = [part, inst]
        return '{}({})'.format(type(self).__name__, ', '.join(args))

    @property
    def properties(self) -> Dict:
        return {**self.particle.properties,
                **self.instrument.properties}

    @properties.setter
    def properties(self, properties: Dict) -> None:
        for name, value in properties.items():
            if hasattr(self.particle, name):
                setattr(self.particle, name, value)
            elif hasattr(self.instrument, name):
                setattr(self.instrument, name, value)
            elif hasattr(self, name):
                setattr(self, name, value)

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords: np.ndarray) -> None:
        '''Ensure coordinates have shape (3, npts)'''
        logger.debug('Setting coordinates')
        c = np.atleast_2d(0. if coords is None else coords)
        ndim, npts = c.shape
        if ndim > 3:
            raise ValueError(f'Incompatible shape: {coords.shape=}')
        self._coordinates = np.vstack([c, np.zeros((3-ndim, npts))])
        self.allocate()

    @property
    def _device_coordinates(self) -> np.ndarray:
        return self._coordinates

    @staticmethod
    def to_field(phase):
        return np.exp(-1j * phase)

    def scattered_field(self, particle, cartesian, bohren):
        '''Return field scattered by one particle'''
        k = self.instrument.wavenumber()
        n_m = self.instrument.n_m
        wavelength = self.instrument.wavelength
        r_p = particle.r_p + particle.r_0
        dr = self.coordinates - r_p[:, None]
        self.kdr[...] = np.asarray(k * dr)
        ab = particle.ab(n_m, wavelength)
        psi = self.compute(ab, self.kdr, self.buffers,
                           cartesian=cartesian, bohren=bohren)
        psi *= self.to_field(k* r_p[2])
        return psi

    def field(self,
              cartesian: bool = True,
              bohren: bool = True) -> np.ndarray:
        '''Return field scattered by particles in the system

        Arguments
        ---------
        cartesian: bool (default True)
            True: field is projected onto Cartesian axes (x, y, z) with
               z being the propagation direction and
               x being the axis of the illumination polarization
            False: field is returned in polar coordinates (r, theta, phi)
        bohren: bool (default True)
            True: z sign convention from Bohren and Huffman

        Returns
        -------
        field : numpy.ndarray
            (3, npts) complex value of the scattered field
        '''
        if (self.coordinates is None or self.particle is None):
            return None
        logger.debug('Computing field')
        self.result.fill(0.+0.j)
        for p in np.atleast_1d(self.particle):
            logger.debug(p)
            self.result += self.scattered_field(p, cartesian, bohren)
        return self.result

    def _device_field(self, **kwargs):
        return self.field(**kwargs)

    def hologram(self) -> np.ndarray:
        '''Return hologram of particle

        Returns
        -------
        hologram : numpy.ndarray
            Computed hologram.
        '''
        psi = self.field()
        psi[0, :] += 1.
        hologram = np.sum(np.real(psi * np.conj(psi)), axis=0)
        return hologram

    def allocate(self) -> None:
        '''Allocate ndarrays for calculation'''
        shape = self.coordinates.shape
        self.kdr = np.empty(shape, dtype=float)
        buffers = [np.empty(shape, dtype=complex) for _ in range(4)]
        self.buffers = np.array(buffers)
        self.result = np.empty(shape, dtype=complex)

    @staticmethod
    def compute(ab: np.ndarray,
                kdr: np.ndarray,
                buffers: List[np.ndarray],
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
        mo1n, ne1n, es, ec = buffers

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


def example(cls=LorenzMie, **kwargs):
    import matplotlib.pyplot as plt
    from pylorenzmie.utilities import coordinates
    from pylorenzmie.theory import (Sphere, Instrument)
    from time import perf_counter

    # Create coordinate grid for image
    shape = (201, 201)
    coords = coordinates(shape)
    # Place two spheres in the field of view, above the focal plane
    pa = Sphere()
    pa.r_p = [150, 150, 200]
    pa.a_p = 0.5
    pa.n_p = 1.45
    pb = Sphere()
    pb.r_p = [100, 10, 250]
    pb.a_p = 1.
    pb.n_p = 1.45
    particle = [pa, pb]
    # Form image with default instrument
    instrument = Instrument()
    instrument.magnification = 0.048
    instrument.wavelength = 0.447
    instrument.n_m = 1.340
    # Use generalized Lorenz-Mie theory to compute field
    kernel = cls(coords, particle, instrument, **kwargs)
    kernel.field()
    start = perf_counter()
    hologram = kernel.hologram()
    print(f'Time to calculate: {perf_counter()-start} s')
    # Compute hologram from field and show it
    plt.imshow(hologram.reshape(shape), cmap='gray')
    plt.show()


if __name__ == '__main__':
    example()
