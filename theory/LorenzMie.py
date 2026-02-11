from pylorenzmie.lib import LMObject
from pylorenzmie.theory import (Particle, Sphere, Instrument)
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

Copyright (c) 2018-2026 David G. Grier
'''

np.seterr(all='raise')


class LorenzMie(LMObject):
    '''
    Compute scattered light field with Generalized Lorenz-Mie theory

    ...

    Inherits
    --------
    pylorenzmie.lib.LMObject

    Properties
    ----------
    coordinates: LMObject.Coordinates
        [3, npts] array of x, y and z coordinates where field
        is calculated
    particle : Particle | list[Particle]
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
                 coordinates: LMObject.Coordinates | None = None,
                 particle: Particle | list[Particle] | None = None,
                 instrument: Instrument | None = None) -> None:
        super().__init__()
        self.coordinates = coordinates
        self.particle = particle or Sphere()
        self.instrument = instrument or Instrument()

    def __repr__(self) -> str:
        classname = self.__class__.__qualname__
        r = f'{classname}(instrument, particle)'
        inst = f'instrument={self.instrument!r}'.replace(',', ',\n\t')
        part = f'particle={self.particle!r}'.replace(',', ',\n\t')
        return '\n    '.join([r, inst, part])

    @property
    def properties(self) -> LMObject.Properties:
        return {**self.particle.properties,
                **self.instrument.properties}

    @properties.setter
    def properties(self, properties: LMObject.Properties) -> None:
        for name, value in properties.items():
            if hasattr(self.particle, name):
                setattr(self.particle, name, value)
            elif hasattr(self.instrument, name):
                setattr(self.instrument, name, value)
            elif hasattr(self, name):
                setattr(self, name, value)

    @property
    def coordinates(self) -> LMObject.Coordinates:
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: LMObject.Coordinates) -> None:
        '''Ensure coordinates have shape (3, npts)'''
        logger.debug('Setting coordinates')
        if coordinates is not None:
            c = np.atleast_2d(coordinates)
        else:
            c = self.meshgrid((201, 201))
        ndim, npts = c.shape
        if ndim > 3:
            raise ValueError(f'Incompatible shape: {coords.shape = }')
        self._coordinates = np.vstack([c, np.zeros((3-ndim, npts))])
        self._allocate()

    def hologram(self, **kwargs) -> LMObject.Image:
        '''Returns hologram of particle

        Returns
        -------
        hologram : LMObject.Image
            Computed hologram.
        '''
        psi = self.field(**kwargs)  # scattered field
        psi[0, :] += 1.0            # incident field
        hologram = np.sum(np.real(psi * np.conj(psi)), axis=0)
        return hologram

    def field(self,
              cartesian: bool = True,
              bohren: bool = True) -> LMObject.Field:
        '''Returns field scattered by particle

        Returns
        -------
        field : LMObject.Field
            complex-valued electric field

        Keywords
        --------
        cartesian: bool
            If True (default), project field onto Cartesian coordinates.
            Otherwise, field is returned in spherical polar coordinates
        bohren: bool
            If True (default), define +z along the beam's propagation direction.
            Otherwise, flip the orientation of z.
        '''
        logger.debug('Computing field')
        self.__field.fill(0.+0.j)
        for p in self.particle:
            logger.debug(p)
            self.__field += self._partial(p, cartesian, bohren)
        return self.__field

    def _field(self, *args, **kwargs) -> LMObject.Field:
        '''Returns field in device format

        Required for API consistency with subclasses.
        '''
        return self.field(*args, **kwargs)

    def _partial(self,
                 particle: Particle,
                 cartesian: bool,
                 bohren: bool) -> LMObject.Field:
        '''Returns field scattered by one particle'''
        k = self.instrument.wavenumber()
        r_p = particle.r_p + particle.r_0
        dr = self.coordinates - r_p[:, None]
        self.kdr = k * dr
        ab = particle.ab(self.instrument.n_m, self.instrument.wavelength)
        field = self.compute(ab, cartesian=cartesian, bohren=bohren)
        field *= np.exp(-1j * k * r_p[2])
        return field

    def _allocate(self) -> None:
        '''Allocate ndarrays for calculation'''
        logger.debug('Allocating buffers')
        shape = self.coordinates.shape
        buffers = [np.empty(shape, dtype=complex) for _ in range(4)]
        self.buffers = np.array(buffers)
        self.__field = np.empty(shape, dtype=complex)

    def compute(self,
                ab: LMObject.Coefficients,
                cartesian: bool = True,
                bohren: bool = True) -> LMObject.Field:
        '''Returns the field scattered by the particle at each coordinate

        Arguments
        ----------
        ab : numpy.ndarray
            [2, norders] Mie scattering coefficients

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
        Mo1n, Ne1n, Es, Ec = self.buffers

        # GEOMETRY
        # 1. particle displacement [pixel]
        # Note: The sign convention used here is appropriate
        # for illumination propagating in the -z direction.
        # This means that a particle forming an image in the
        # focal plane (z = 0) is located at positive z.
        # Accounting for this by flipping the axial coordinate
        # is equivalent to using a mirrored (left-handed)
        # coordinate system.
        kx = self.kdr[0, :]
        ky = self.kdr[1, :]
        kz = -self.kdr[2, :]
        shape = kx.shape

        # 2. geometric factors
        kρ = np.hypot(kx, ky)
        kr = np.hypot(kρ, kz)
        sinkr = np.sin(kr)
        coskr = np.cos(kr)

        φ = np.arctan2(ky, kx)
        cosφ = np.cos(φ)
        sinφ = np.sin(φ)
        θ = np.arctan2(kρ, kz)
        cosθ = np.cos(θ)
        sinθ = np.sin(θ)

        # SPECIAL FUNCTIONS
        # starting points for recursive function evaluation ...
        # 1. Riccati-Bessel radial functions, page 478.
        # Particles above the focal plane create diverging waves
        # described by Eq. (4.13) for $h_n^{(1)}(kr)$. These have z > 0.
        # Those below the focal plane appear to be converging from the
        # perspective of the camera. They are described by Eq. (4.14)
        # for $h_n^{(2)}(kr)$, and have z < 0. We can select the
        # appropriate case by applying the correct sign of the imaginary
        # part of the starting functions...
        factor = 1.j * np.sign(kz) if bohren else -1.j * np.sign(kz)
        ξ_nm2 = coskr + factor * sinkr  # \xi_{-1}(kr)
        ξ_nm1 = sinkr - factor * coskr  # \xi_0(kr)

        # 2. Angular functions (4.47), page 95
        # \pi_0(\cos\theta)
        π_nm1 = np.zeros(shape)
        # \pi_1(\cos\theta)
        π_n = np.ones(shape)

        # 3. Vector spherical harmonics: (r, θ, φ)
        Mo1n[0].fill(0.j)            # no radial component

        # Allocate memory for Wiscombe functions
        swisc = np.empty(shape)
        twisc = np.empty(shape)

        # storage for scattered field
        Es.fill(0.j)

        # COMPUTE field by summing partial waves
        for n in range(1, norders):
            # upward recurrences ...
            # 4. Legendre factor (4.47)
            # Method described by Wiscombe (1980)

            # swisc = π_n * cosθ
            # twisc = swisc - π_nm1
            np.multiply(π_n, cosθ, out=swisc)
            np.subtract(swisc, π_nm1, out=twisc)
            τ_n = π_nm1 - n * twisc  # -\tau_n(\cos\theta)

            # ... Riccati-Bessel function, page 478
            ξ_n = (2.*n - 1.) * (ξ_nm1 / kr) - ξ_nm2  # \xi_n(kr)

            # ... Deirmendjian's derivative
            Dn = n * (ξ_n / kr) - ξ_nm1

            # vector spherical harmonics (4.50)
            Mo1n[1] = π_n * ξ_n      # ... divided by cosφ/kr
            Mo1n[2] = τ_n * ξ_n      # ... divided by sinφ/kr

            # ... divided by cosφ sinθ/kr^2
            Ne1n[0] = (n*n + n) * π_n * ξ_n
            Ne1n[1] = τ_n * Dn       # ... divided by cosφ/kr
            Ne1n[2] = π_n * Dn       # ... divided by sinφ/kr

            # prefactor, page 93
            En = 1.j**n * (2.*n + 1.) / (n*n + n)

            # the scattered field in spherical coordinates (4.45)
            Es += (1.j * En * ab[n, 0]) * Ne1n
            Es -= (En * ab[n, 1]) * Mo1n

            # upward recurrences ...
            # ... angular functions (4.47)
            # Method described by Wiscombe (1980)
            π_nm1 = π_n
            π_n = swisc + (1. + 1./n) * twisc

            # ... Riccati-Bessel function
            ξ_nm2 = ξ_nm1
            ξ_nm1 = ξ_n
            # n: multipole sum

        # geometric factors were divided out of the vector
        # spherical harmonics for accuracy and efficiency ...
        # ... put them back at the end.
        Es[0] *= cosφ * sinθ / kr
        Es[1] *= cosφ
        Es[2] *= sinφ
        Es /= kr

        # By default, the scattered wave is returned in spherical
        # coordinates.  Project components onto Cartesian coordinates.
        # Assumes that the incident wave propagates along z and
        # is linearly polarized along x
        if cartesian:
            Ec[0] = Es[0] * sinθ * cosφ
            Ec[0] += Es[1] * cosθ * cosφ
            Ec[0] -= Es[2] * sinφ

            Ec[1] = Es[0] * sinθ * sinφ
            Ec[1] += Es[1] * cosθ * sinφ
            Ec[1] += Es[2] * cosφ
            Ec[2] = Es[0] * cosθ - Es[1] * sinθ
            return Ec
        else:
            return Es

    @classmethod
    def example(cls, **kwargs) -> None:  # pragma: no cover
        import matplotlib.pyplot as plt
        from pylorenzmie.theory import (Sphere, Instrument)
        from time import perf_counter

        shape = (201, 201)
        c = cls.meshgrid(shape)
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
        instrument.numerical_aperture = 1.45
        instrument.wavelength = 0.447
        instrument.n_m = 1.340
        # Use generalized Lorenz-Mie theory to compute field
        model = cls(coordinates=c, instrument=instrument, **kwargs)
        start = perf_counter()
        model.particle = particle
        hologram = model.hologram()
        print(f'Time to calculate: {perf_counter()-start:.1e} s')
        # Compute hologram from field and show it
        plt.imshow(hologram.reshape(shape), cmap='gray')
        plt.show()


if __name__ == '__main__':  # pragma: no cover
    LorenzMie.example()
