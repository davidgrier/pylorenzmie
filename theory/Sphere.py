from dataclasses import dataclass
from pylorenzmie.theory import Particle
from pylorenzmie.lib.types import Coefficients, Properties
import numpy as np


@dataclass
class Sphere(Particle):
    '''Homogeneous spherical scatterer for Lorenz-Mie microscopy.

    Extends :class:`Particle` with the physical parameters needed to
    compute Mie scattering coefficients for an isotropic homogeneous
    sphere illuminated by a coherent plane wave polarized along x and
    propagating along z.

    Attributes
    ----------
    a_p : float
        Radius of the sphere, in μm. Default: 1.
    n_p : float
        Refractive index of the sphere. Default: 1.5.
    k_p : float
        Absorption coefficient of the sphere. Default: 0.

    References
    ----------
    .. [1] C. F. Bohren and D. R. Huffman, *Absorption and Scattering
       of Light by Small Particles* (Wiley, 1983), Chapter 8.
    .. [2] W. Yang, "Improved recursive algorithm for light scattering
       by a multilayered sphere," *Applied Optics* **42**, 1710–1720
       (2003).
    .. [3] O. Pena and U. Pal, "Scattering of electromagnetic radiation
       by a multilayered sphere," *Computer Physics Communications*
       **180**, 2348–2354 (2009).  Equation numbering follows this
       reference.
    .. [4] W. J. Wiscombe, "Improved Mie scattering algorithms,"
       *Applied Optics* **19**, 1505–1509 (1980).
    .. [5] A. A. R. Neves and D. Pisignano, "Effect of finite terms on
       the truncation error of Mie series," *Optics Letters* **37**,
       2481–2483 (2012).
    '''

    a_p: float = 1.
    n_p: float = 1.5
    k_p: float = 0.

    @property
    def d_p(self) -> float:
        '''Diameter of the sphere, in μm.

        Equivalent to ``2 * a_p``.  Setting ``d_p`` updates ``a_p``.

        Returns
        -------
        float
        '''
        return 2. * self.a_p

    @d_p.setter
    def d_p(self, d_p: float) -> None:
        self.a_p = d_p / 2.

    @Particle.properties.getter
    def properties(self) -> Properties:
        return {**super().properties,
                'a_p': self.a_p,
                'n_p': self.n_p,
                'k_p': self.k_p}

    def ab(self,
           n_m: float | complex = 1.42,
           wavelength: float = 0.532) -> Coefficients:
        '''Mie scattering coefficients for the sphere.

        Parameters
        ----------
        n_m : float or complex
            Refractive index of the medium. Default: 1.42.
        wavelength : float
            Vacuum wavelength of the illuminating light, in μm.
            Default: 0.532.

        Returns
        -------
        ab : numpy.ndarray, shape (n_terms, 2), dtype complex
            Mie scattering coefficients.
        '''
        return Sphere.mie_coefficients(self.a_p, self.n_p, self.k_p,
                                       n_m, wavelength)

    @staticmethod
    def wiscombe_yang(x: float, m: float | complex) -> int:
        '''Number of terms to retain in the partial-wave expansion.

        Implements the truncation criterion of Wiscombe (1980) with the
        extension for multilayered spheres from Yang (2003) Eq. (30).

        Parameters
        ----------
        x : float
            Size parameter of the sphere (or outermost layer).
        m : float or complex
            Refractive index of the sphere relative to the medium.

        Returns
        -------
        nstop : int
            Number of terms to retain.
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
    def neves_pisignano(x: float,
                        precision: float = 6.) -> int:
        '''Number of terms to retain, after Neves and Pisignano (2012).

        Alternative termination criterion to :meth:`wiscombe_yang`.

        Parameters
        ----------
        x : float
            Size parameter of the sphere.
        precision : float
            Desired decimal digits of precision. Default: 6.

        Returns
        -------
        nstop : int
            Number of terms to retain.
        '''
        nstop = x + 0.76 * np.cbrt(precision * precision * x) - 4.1
        return int(nstop)

    @staticmethod
    def mie_coefficients(a_p: float,
                         n_p: float,
                         k_p: float,
                         n_m: float | complex,
                         wavelength: float) -> Coefficients:
        '''Mie scattering coefficients for a homogeneous sphere.

        Parameters
        ----------
        a_p : float
            Radius of the sphere, in μm.
        n_p : float
            Refractive index of the sphere.
        k_p : float
            Absorption coefficient of the sphere.
        n_m : float or complex
            Refractive index of the medium.
        wavelength : float
            Vacuum wavelength of the illuminating light, in μm.

        Returns
        -------
        ab : numpy.ndarray, shape (n_terms, 2), dtype complex
            Mie scattering coefficients.
        '''
        # size parameters for layers
        k = 2. * np.pi / wavelength   # wave number in vacuum [um^-1]
        k *= np.real(n_m)             # wave number in medium

        x = k * a_p                   # size parameter
        m = (n_p + 1.j * k_p) / n_m   # relative refractive index

        nmax = Sphere.wiscombe_yang(x, m)

        # storage for results
        ab = np.empty((nmax+1, 2), np.complex128)
        D1 = np.empty(nmax+1, np.complex128)
        D3 = np.empty(nmax+1, np.complex128)
        Ψ = np.empty(nmax+1, np.complex128)
        ζ = np.empty(nmax+1, np.complex128)

        # initialization
        D1[nmax] = 0.                                # Eq. (16a)
        D3[0] = 1.j                                  # Eq. (18b)

        # iterate outward from the sphere's core
        z = x * m
        for n in range(nmax, 0, -1):
            D1[n-1] = n/z - 1./(D1[n] + n/z)         # Eq. (16b)
        Ha = D1.copy()                               # Eq. (7a)
        Hb = D1.copy()                               # Eq. (8a)

        # iterate into medium (m = 1.)
        z = x
        # downward recurrence for D1 (D1[nmax] = 0)
        for n in range(nmax, 0, -1):
            D1[n-1] = n/z - (1./(D1[n] + n/z))       # Eq. (16b)

        # upward recurrence for Ψ, ζ, Ψζ and D3
        Ψ[0] = np.sin(z)                             # Eq. (20a)
        ζ[0] = -1.j * np.exp(1.j * z)                # Eq. (21a)
        Ψζ = 0.5 * (1. - np.exp(2.j * z))            # Eq. (18a)
        for n in range(1, nmax+1):
            Ψ[n] = Ψ[n-1] * (n/z - D1[n-1])          # Eq. (20b)
            ζ[n] = ζ[n-1] * (n/z - D3[n-1])          # Eq. (21b)
            Ψζ *= (n/z - D1[n-1]) * (n/z - D3[n-1])  # Eq. (18c)
            D3[n] = D1[n] + 1.j/Ψζ                   # Eq. (18d)

        # scattering coefficients
        n = np.arange(nmax+1)
        Ψr = np.roll(Ψ, 1)
        ζr = np.roll(ζ, 1)
        fac = Ha/m + n/x                             # Eq. (5)
        ab[:, 0] = (fac * Ψ - Ψr) / (fac * ζ - ζr)
        fac = Hb*m + n/x                             # Eq. (6)
        ab[:, 1] = (fac * Ψ - Ψr) / (fac * ζ - ζr)
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
