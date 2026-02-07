#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pylorenzmie.theory import Particle
import numpy as np


@dataclass
class Sphere(Particle):

    '''
    Abstraction of a spherical scatterer for Lorenz-Mie micrsocopy

    ...

    Inherits
    --------
    pylorenzmie.theory.Particle

    Properties
    ----------
    a_p: float
        radius of particle [um]
    n_p: float | complex
        refractive index of particle
    k_p: float
        absorption coefficient of particle

    Methods
    -------
    ab(n_m, wavelength): Particle.Coefficients
        returns the Mie scattering coefficients for the sphere

        Arguments
        ---------
        n_m: float | complex
            refractive index of medium
        wavelength: float
            vacuum wavelength of light [micrometers]

    References
    ----------
    1. Adapted from Chapter 8 in
       C. F. Bohren and D. R. Huffman,
       Absorption and Scattering of Light by Small Particles,
       (New York, Wiley, 1983).
    2. W. Yang,
       Improved recursive algorithm for light scattering
       by a multilayered sphere,
       Applied Optics 42, 1710--1720 (2003).
    3. O. Pena, U. Pal,
       Scattering of electromagnetic radiation by a multilayered sphere,
       Computer Physics Communications 180, 2348--2354 (2009).
       NB: Equation numbering follows this reference.
    4. W. J. Wiscombe,
       Improved Mie scattering algorithms,
       Applied Optics 19, 1505-1509 (1980).
    5. A. A. R. Neves and D. Pisignano,
       Effect of finite terms on the truncation error of Mie series,
       Optics Letters 37, 2481-2420 (2012).
    '''

    a_p: float = 1.
    n_p: float = 1.5
    k_p: float = 0.

    @property
    def d_p(self) -> float:
        '''Diameter of sphere [um]'''
        return 2.*self.a_p

    @d_p.setter
    def d_p(self, d_p: float) -> None:
        self.a_p = d_p/2.

    @Particle.properties.getter
    def properties(self) -> Particle.Properties:
        return {**super().properties,
                'a_p': self.a_p,
                'n_p': self.n_p,
                'k_p': self.k_p}

    def ab(self,
           n_m: float | complex,
           wavelength: float) -> Particle.Coefficients:
        '''Returns the Mie scattering coefficients

        Arguments
        ---------
        n_m: float | complex
            Refractive index of medium
        wavelength: float
            Vacuum wavelength of light [micrometers]

        Returns
        -------
        ab : Particle.Coefficients
            Mie AB scattering coefficients
        '''
        return Sphere.mie_coefficients(self.a_p,
                                       self.n_p,
                                       self.k_p,
                                       n_m, wavelength)

    @staticmethod
    def wiscombe_yang(x: float, m: float | complex) -> int:
        '''Return the number of terms to keep in partial wave expansion

        Equation numbers refer to Wiscombe (1980) and Yang (2003).

        ...

        Arguments
        ---------
        x : complex
            size parameters for sphere
        m : float | complex
            relative refractive index of sphere

        Returns
        -------
        ns : int
            Number of terms to retain in the partial-wave expansion
        '''

        # Wiscombe (1980)
        xl = np.abs(x)
        if xl <= 8.:
            ns = np.floor(xl + 4. * np.cbrt(xl) + 1.)
        elif xl <= 4200.:
            ns = np.floor(xl + 4.05 * np.cbrt(xl) + 2.)
        else:
            ns = np.floor(xl + 4. * np.cbrt(xl) + 2.)

        # Yang (2003) Eq. (30)
        xm = np.abs(x * m)
        xm_1 = np.abs(np.roll(x, -1) * m)
        nstop = np.max([ns, np.max(xm), np.max(xm_1)])
        return int(nstop)

    @staticmethod
    def nieves_pisignano(x: float,
                         precision: float | complex = 6.) -> int:
        nstop = x + 0.76 * np.cbrt(precision*precision*x) - 4.1
        return int(nstop)

    @staticmethod
    def mie_coefficients(a_p: float,
                         n_p: float,
                         k_p: float,
                         n_m: complex,
                         wavelength: float) -> Particle.Coefficients:
        '''Returns the Mie scattering coefficients for a sphere

        This works for an isotropic homogeneous sphere illuminated by
        a coherent plane wave linearly polarized along x
        and propagating along z.
        ...

        Arguments
        ---------
        a_p : float
            radius of the sphere [um]
        n_p : float
            refractive index of the sphere
        k_p : float
            absorption coefficient of sphere
        n_m : float | complex
            refractive index of medium
        wavelength : float
            wavelength of light [um]

        Returns
        -------
        ab : Particle.Coefficients
            Mie AB coefficients
        '''

        # size parameters for layers
        k = 2.*np.pi/wavelength    # wave number in vacuum [um^-1]
        k *= np.real(n_m)          # wave number in medium

        x = k * a_p                # size parameter in each layer
        m = (n_p + 1.j*k_p) / n_m  # relative refractive index in each layer

        nmax = Sphere.wiscombe_yang(x, m)

        # storage for results
        ab = np.empty((nmax+1, 2), np.complex128)
        d1_z1 = np.empty(nmax+1, np.complex128)
        d1_z2 = np.empty(nmax+1, np.complex128)
        d3_z1 = np.empty(nmax+1, np.complex128)
        d3_z2 = np.empty(nmax+1, np.complex128)
        psi = np.empty(nmax+1, np.complex128)
        zeta = np.empty(nmax+1, np.complex128)

        # initialization
        d1_z1[nmax] = 0.                                          # (16a)
        d1_z2[nmax] = 0.
        d3_z1[0] = 1.j                                            # (18b)
        d3_z2[0] = 1.j

        # iterate outward from the sphere's core
        z1 = x * m
        for n in range(nmax, 0, -1):
            d1_z1[n-1] = n/z1 - 1./(d1_z1[n] + n/z1)              # (16b)
        ha = d1_z1.copy()                                         # (7a)
        hb = d1_z1.copy()                                         # (8a)

        # iterate into medium (m = 1.)
        z1 = x
        # downward recurrence for D1 (D1[nmax] = 0)
        for n in range(nmax, 0, -1):
            d1_z1[n-1] = n/z1 - (1./(d1_z1[n] + n/z1))            # (16b)

        # upward recurrence for Psi, Zeta, PsiZeta and D3
        psi[0] = np.sin(z1)                                       # (20a)
        zeta[0] = -1.j * np.exp(1.j * z1)                         # (21a)
        psizeta = 0.5 * (1. - np.exp(2.j * z1))                   # (18a)
        for n in range(1, nmax+1):
            psi[n] = psi[n-1] * (n/z1 - d1_z1[n-1])               # (20b)
            zeta[n] = zeta[n-1] * (n/z1 - d3_z1[n-1])             # (21b)
            psizeta *= (n/z1 - d1_z1[n-1]) * (n/z1 - d3_z1[n-1])  # (18c)
            d3_z1[n] = d1_z1[n] + 1.j/psizeta                     # (18d)

        # Scattering coefficients
        n = np.arange(nmax+1)
        fac = ha/m + n/x
        ab[:, 0] = ((fac * psi - np.roll(psi, 1)) /
                    (fac * zeta - np.roll(zeta, 1)))              # (5)
        fac = hb*m + n/x
        ab[:, 1] = ((fac * psi - np.roll(psi, 1)) /
                    (fac * zeta - np.roll(zeta, 1)))              # (6)
        ab[0, :] = 0.j

        return ab

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        from time import perf_counter

        s = cls(a_p=0.75, n_p=1.5)
        print(s)
        ab = s.ab(1.339, 0.447)
        print(f'{ab.shape = }')
        start = perf_counter()
        s.ab(1.339, 0.447)
        print(f'time = {perf_counter() - start:.2e} s')


if __name__ == '__main__':  # pragma: no cover
    Sphere.example()
