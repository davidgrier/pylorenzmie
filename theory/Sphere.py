'''Homogeneous sphere as a scatterer for Lorenz-Mie microscopy.

A :class:`Sphere` has a radius ``a_p`` in micrometers and a complex
refractive index ``n_p + i k_p``.  It computes the Mie scattering coefficients
:math:`a_n` and :math:`b_n` of a homogeneous sphere in a medium of given
refractive index at a given vacuum wavelength, following Bohren and Huffman,
with the number of partial waves chosen by the criterion of Wiscombe, in the
form that Yang extended to layered spheres, or by the criterion of Neves and
Pisignano.  The calculators in ``pylorenzmie.theory`` combine the
coefficients with the geometry of the pixels to obtain the scattered field.

Lengths are in micrometers and positions in pixels, as everywhere in the
package.

See Also
--------
~pylorenzmie.theory.Particle.Particle : base class, with the position.
~pylorenzmie.theory.LorenzMie.LorenzMie : calculator of the scattered field.
'''

from dataclasses import dataclass

import numpy as np

from pylorenzmie.theory import Particle
from pylorenzmie.lib.lmtypes import Coefficients, Properties
from pylorenzmie.theory._mie import mie_ab

class ModelDomainError(ValueError):
    '''The parameters are outside the range that the model computes.

    A :class:`ValueError`, so that code that refuses such parameters in
    general still does.  :class:`~pylorenzmie.analysis.Optimizer.Optimizer`
    catches this one, and no other, to reject a step of a fit that leaves
    the range.
    '''


MAX_RADIUS = 50.
'''Largest radius of a sphere that the model computes, in μm.'''

MAX_ORDERS = 5000
'''Largest number of partial waves that the model computes.

The number of partial waves grows with the radius and with the index.  A
sphere of 50 μm and an index of 3 needs 2,100 of them, and the same in air at
300 nm needs 3,100.  A fit that wanders off to an absurd radius or index asks
for millions or billions, and the computer for tens of gigabytes, before it
notices.
'''


@dataclass
class Sphere(Particle):
    '''Homogeneous spherical scatterer for Lorenz-Mie microscopy.

    Extends :class:`~pylorenzmie.theory.Particle.Particle` with the physical
    parameters needed to compute Mie scattering coefficients for an isotropic
    homogeneous sphere illuminated by a coherent plane wave polarized along x
    and propagating along z.

    Attributes
    ----------
    a_p : float
        Radius of the sphere, in μm. Default: 1.  The coefficients are
        computed for ``abs(a_p) <= MAX_RADIUS`` (50 μm) only, and for
        no more than ``MAX_ORDERS`` (5000) partial waves.
    n_p : float
        Refractive index of the sphere. Default: 1.5.
    k_p : float
        Absorption coefficient of the sphere. Default: 0.

    References
    ----------
    1. C. F. Bohren and D. R. Huffman, *Absorption and Scattering
       of Light by Small Particles* (Wiley, 1983), Chapter 8.
    2. W. Yang, "Improved recursive algorithm for light scattering
       by a multilayered sphere," *Applied Optics* **42**, 1710–1720
       (2003).
    3. O. Pena and U. Pal, "Scattering of electromagnetic radiation
       by a multilayered sphere," *Computer Physics Communications*
       **180**, 2348–2354 (2009).  Equation numbering follows this
       reference.
    4. W. J. Wiscombe, "Improved Mie scattering algorithms,"
       *Applied Optics* **19**, 1505–1509 (1980).
    5. A. A. R. Neves and D. Pisignano, "Effect of finite terms on
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

        Raises
        ------
        ModelDomainError
            If the radius is larger than ``MAX_RADIUS``, or the sum needs
            more than ``MAX_ORDERS`` partial waves.  It is a
            :class:`ValueError`.
        '''
        return Sphere.mie_coefficients(self.a_p,
                                       self.n_p,
                                       self.k_p,
                                       n_m,
                                       wavelength)

    @staticmethod
    def check_radius(a_p: float) -> None:
        '''Refuse a radius that the model does not compute.

        Parameters
        ----------
        a_p : float
            Radius of the sphere, in μm.

        Raises
        ------
        ModelDomainError
            If ``abs(a_p)`` is larger than ``MAX_RADIUS``, or is not a
            number.  It is a :class:`ValueError`.
        '''
        if not abs(a_p) <= MAX_RADIUS:
            raise ModelDomainError(f'the radius of a sphere is at most '
                             f'{MAX_RADIUS:g} um, not a_p = {a_p:g} um')

    @staticmethod
    def check_orders(nmax: int, a_p: float, n_p: float) -> None:
        '''Refuse a sphere whose Mie sum is longer than the model computes.

        Parameters
        ----------
        nmax : int
            Number of partial waves that the sphere needs
            (:meth:`wiscombe_yang`).
        a_p : float
            Radius of the sphere, in μm.
        n_p : float
            Refractive index of the sphere.

        Raises
        ------
        ModelDomainError
            If ``nmax`` is larger than ``MAX_ORDERS``.  It is a
            :class:`ValueError`.
        '''
        if nmax > MAX_ORDERS:
            raise ModelDomainError(f'a sphere of radius {a_p:g} um and index '
                             f'{n_p:g} needs {nmax:,} partial waves, '
                             f'at most {MAX_ORDERS:,} are computed')

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

        Raises
        ------
        ModelDomainError
            If the radius is larger than ``MAX_RADIUS``, or the sum needs
            more than ``MAX_ORDERS`` partial waves.  It is a
            :class:`ValueError`.
        '''
        Sphere.check_radius(a_p)

        # size parameters for layers
        k = 2. * np.pi / wavelength   # wave number in vacuum [um^-1]
        k *= np.real(n_m)             # wave number in medium

        x = k * a_p                   # size parameter
        m = (n_p + 1.j * k_p) / n_m   # relative refractive index

        nmax = Sphere.wiscombe_yang(x, m)
        Sphere.check_orders(nmax, a_p, n_p)

        return mie_ab(x, m, nmax)

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        '''Run a demonstration of the class, for use as a script.'''
        from time import perf_counter

        s = cls(a_p=0.75, n_p=1.5)
        print(s)
        ab = s.ab(1.339, 0.447)
        print(f'{ab.shape = }')
        start = perf_counter()
        s.ab(1.339, 0.447)
        print(f'time = {perf_counter() - start:.2e} s')


def mie_coefficients(x: float,
                     m: float | complex,
                     nmax: int | None = None) -> Coefficients:
    '''Mie scattering coefficients of a homogeneous sphere.

    The coefficients that :meth:`Sphere.ab` returns, for a size parameter
    and a relative index that the caller has worked out.  This is the
    function for code that needs the coefficients of many spheres, or that
    checks another implementation of the series against this one.

    Parameters
    ----------
    x : float
        Size parameter of the sphere in the medium, ``2 pi n_m a_p /
        wavelength``.
    m : float or complex
        Refractive index of the sphere relative to the medium,
        ``(n_p + 1j k_p) / n_m``.
    nmax : int, optional
        Number of partial waves to keep.  Default: ``None``, the number
        that :meth:`Sphere.wiscombe_yang` gives for ``x`` and ``m``.

    Returns
    -------
    ab : numpy.ndarray, shape (nmax + 1, 2), dtype complex
        The coefficients ``a_n`` and ``b_n``, with ``ab[0] = 0``.

    Raises
    ------
    ModelDomainError
        If ``x`` is not positive and finite, ``m`` is not finite, or the
        sum needs more than ``MAX_ORDERS`` partial waves.  It is a
        :class:`ValueError`.
    '''
    m = complex(m)
    if not (np.isfinite(x) and x > 0.):
        raise ModelDomainError(f'the size parameter must be positive and '
                               f'finite, not x = {x:g}')
    if not np.isfinite(m):
        raise ModelDomainError(f'the relative index must be finite, '
                               f'not m = {m:g}')
    if nmax is None:
        nmax = Sphere.wiscombe_yang(x, m)
    if nmax > MAX_ORDERS:
        raise ModelDomainError(f'a size parameter x = {x:g} needs {nmax:,} '
                               f'partial waves, at most {MAX_ORDERS:,} '
                               'are computed')
    return mie_ab(float(x), m, int(nmax))


if __name__ == '__main__':  # pragma: no cover
    Sphere.example()
