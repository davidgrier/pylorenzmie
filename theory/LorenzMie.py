from pylorenzmie.lib import LMObject
from pylorenzmie.lib.types import (Coordinates, Coefficients, Field,
                                   Image, Properties)
from pylorenzmie.theory import (Particle, Sphere, Instrument)
import numpy as np


class LorenzMie(LMObject):
    '''Scattered light field computed with Generalized Lorenz-Mie theory.

    Computes the electric field scattered by one or more particles,
    then forms a synthetic hologram by interfering the scattered field
    with the incident plane wave.  The incident illumination is assumed
    to be a plane wave linearly polarized along x and propagating
    along -z.

    The class attribute ``method`` is used by :class:`Optimizer` to
    select a compatible calculator.

    Attributes
    ----------
    coordinates : numpy.ndarray, shape (3, npts)
        Pixel coordinates at which the field is evaluated.
        Defaults to a 201×201 grid if not supplied.
    particle : Particle or list of Particle
        Scattering particle(s).  Defaults to a :class:`Sphere`.
    instrument : Instrument
        Optical parameters of the microscope.

    References
    ----------
    .. [1] C. F. Bohren and D. R. Huffman, *Absorption and Scattering
       of Light by Small Particles* (Wiley, 1983), Chapter 4.
    .. [2] W. J. Wiscombe, "Improved Mie scattering algorithms,"
       *Appl. Opt.* **19**, 1505–1509 (1980).
    .. [3] W. J. Lentz, "Generating Bessel functions in Mie scattering
       calculations using continued fractions,"
       *Appl. Opt.* **15**, 668–671 (1976).
    .. [4] S.-H. Lee, Y. Roichman, G.-R. Yi, S.-H. Kim, S.-M. Yang,
       A. van Blaaderen, P. van Oostrum and D. G. Grier,
       "Characterizing and tracking single colloidal particles with
       video holographic microscopy,"
       *Opt. Express* **15**, 18275–18282 (2007).
    .. [5] F. C. Cheong, B. Sun, R. Dreyfus, J. Amato-Grill, K. Xiao,
       L. Dixon and D. G. Grier,
       "Flow visualization and flow cytometry with holographic video
       microscopy," *Opt. Express* **17**, 13071–13079 (2009).
    '''

    method: str = 'numpy'

    def __init__(self,
                 coordinates: Coordinates | None = None,
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
    def properties(self) -> Properties:
        return {**self.particle.properties,
                **self.instrument.properties}

    @properties.setter
    def properties(self, properties: Properties) -> None:
        for name, value in properties.items():
            if hasattr(self.particle, name):
                setattr(self.particle, name, value)
            elif hasattr(self.instrument, name):
                setattr(self.instrument, name, value)
            elif hasattr(self, name):
                setattr(self, name, value)

    @property
    def coordinates(self) -> Coordinates:
        '''Pixel coordinates at which the field is evaluated.

        Shape is always ``(3, npts)``.  If assigned a ``(2, npts)``
        array, the z-coordinates are set to zero.  Assigning ``None``
        creates a default 201×201 grid.
        '''
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: Coordinates | None) -> None:
        self.logger.debug('Setting coordinates')
        if coordinates is None:
            c = self.meshgrid((201, 201))
        else:
            c = np.atleast_2d(coordinates)
        ndim, npts = c.shape
        if ndim > 3:
            raise ValueError(f'Incompatible shape: {coordinates.shape=}')
        self._coordinates = np.vstack([c, np.zeros((3-ndim, npts))])
        self._allocate()

    def _allocate(self) -> None:
        '''Allocate working buffers.  Subclasses should override this.'''
        self.logger.debug('Allocating buffers')
        shape = self.coordinates.shape
        self.buffers = [np.empty(shape, dtype=complex) for _ in range(4)]
        self._field = np.empty(shape, dtype=complex)

    def hologram(self, **kwargs) -> Image:
        '''Hologram of the particle at the current coordinates.

        Parameters
        ----------
        cartesian : bool
            If True (default), project the field onto Cartesian
            coordinates. If False, return the field in spherical
            polar coordinates.
        bohren : bool
            If True (default), use +z along the propagation direction.
            If False, flip the orientation of z.

        Returns
        -------
        hologram : numpy.ndarray, shape (npts,)
            Computed hologram intensity at each coordinate.
        '''
        field = self.field(**kwargs)
        field[0, :] += 1.0
        return np.sum(field.real**2 + field.imag**2, axis=0)

    def field(self, **kwargs) -> Field:
        '''Electric field scattered by the particle(s).

        Parameters
        ----------
        cartesian : bool
            If True (default), project the field onto Cartesian
            coordinates. If False, return the field in spherical
            polar coordinates.
        bohren : bool
            If True (default), use +z along the propagation direction.
            If False, flip the orientation of z.

        Returns
        -------
        field : numpy.ndarray, shape (3, npts), dtype complex
            Complex electric field scattered by the particle.
        '''
        k = self.instrument.wavenumber()
        n_m = self.instrument.n_m
        wavelength = self.instrument.wavelength
        self._field.fill(0.+0.j)
        for particle in self.particle:
            r_p = particle.r_p + particle.r_0
            kdr = k * (self.coordinates - r_p[:, None])
            ab = particle.ab(n_m, wavelength)
            self._field += (self.lorenzmie(ab, kdr, **kwargs) *
                            np.exp(-1.j * k * r_p[2]))
        return self._field

    def lorenzmie(self,
                  ab: Coefficients,
                  kdr: Coordinates,
                  cartesian: bool = True,
                  bohren: bool = True) -> Field:
        '''Scattered field for given Mie coefficients and geometry.

        Parameters
        ----------
        ab : numpy.ndarray, shape (n_orders, 2)
            Mie scattering coefficients.
        kdr : numpy.ndarray, shape (3, npts)
            Wave-number-scaled displacement from particle to each
            coordinate point.
        cartesian : bool
            If True, return field projected onto Cartesian coordinates.
            Default: True.
        bohren : bool
            If True, use sign convention from Bohren and Huffman.
            Default: True.

        Returns
        -------
        field : numpy.ndarray, shape (3, npts), dtype complex
            Complex scattered field at each coordinate.
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
        kx = kdr[0, :]
        ky = kdr[1, :]
        kz = -kdr[2, :]
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

        factor = 1.

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
            Ne1n[0] = (n*n + n) * Mo1n[1]   # do not recompute π_n*ξ_n
            Ne1n[1] = τ_n * Dn       # ... divided by cosφ/kr
            Ne1n[2] = π_n * Dn       # ... divided by sinφ/kr

            # prefactor, page 93
            factor *= 1.j
            En = factor * (2.*n + 1.) / (n*n + n)

            # the scattered field in spherical coordinates (4.45)
            # Mo1n[0] == 0 always, so Es[0] has no Mo1n contribution
            an = 1.j * En * ab[n, 0]
            bn = En * ab[n, 1]
            Es[0] += an * Ne1n[0]
            Es[1] += an * Ne1n[1] - bn * Mo1n[1]
            Es[2] += an * Ne1n[2] - bn * Mo1n[2]

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
    def example(cls,
                shape: tuple[int, int] = (201, 201),
                show: bool = True,
                **kwargs) -> None:  # pragma: no cover
        import matplotlib.pyplot as plt
        from pylorenzmie.theory import (Sphere, Instrument)
        from time import perf_counter

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
        start = perf_counter()
        hologram = model.hologram()
        print(f'Second pass: {perf_counter()-start:.1e} s')
        # Optionally show hologram
        if show:
            plt.figure(num=f'{cls.__name__} example')
            plt.imshow(hologram.reshape(shape), cmap='gray')
            plt.show()


if __name__ == '__main__':  # pragma: no cover
    LorenzMie.example()
