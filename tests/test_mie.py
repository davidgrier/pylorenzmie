import unittest

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from pylorenzmie.theory import _mie
from pylorenzmie.theory.Sphere import Sphere


def bohren_huffman(x: float, m: complex, nmax: int) -> np.ndarray:
    '''The coefficients from the spherical Bessel functions (BH, Ch. 4).'''
    n = np.arange(1, nmax + 1)

    def psi(z):
        return z * spherical_jn(n, z), spherical_jn(n, z) + z * spherical_jn(
            n, z, derivative=True)

    def xi(z):
        h = spherical_jn(n, z) + 1.j * spherical_yn(n, z)
        dh = (spherical_jn(n, z, derivative=True)
              + 1.j * spherical_yn(n, z, derivative=True))
        return z * h, h + z * dh

    ψx, dψx = psi(x)
    ψm, dψm = psi(m * x)
    ξx, dξx = xi(x)
    ab = np.zeros((nmax + 1, 2), np.complex128)
    ab[1:, 0] = ((m*ψm*dψx - ψx*dψm) / (m*ψm*dξx - ξx*dψm))
    ab[1:, 1] = ((ψm*dψx - m*ψx*dψm) / (ψm*dξx - m*ξx*dψm))
    return ab


CASES = tuple((a_p, n_p)
              for a_p in (0.1, 0.5, 1.5, 3., 10., 30.)
              for n_p in (1.0, 1.45, 1.59, 2., 3.))


class TestMieKernel(unittest.TestCase):
    '''The recurrences agree with the Bessel functions of the theory.'''

    n_m = 1.34
    wavelength = 0.447

    def case(self, a_p, n_p):
        x = 2. * np.pi * self.n_m / self.wavelength * a_p
        m = n_p / self.n_m
        return x, m + 0j, Sphere.wiscombe_yang(x, m)

    def test_agree_with_the_bessel_functions(self) -> None:
        '''The coefficients are right for large spheres of a high index.

        The recurrence for the logarithmic derivative starts above the
        highest order, and used to start at it: the first coefficients
        were off by 1e-4 for a 1.5 um sphere of polystyrene, by 0.7% for
        3 um, and by 100% for 3 um of an index of 2.  The Bessel functions
        lose their precision at the highest orders, so only the first ten
        are compared.
        '''
        for a_p, n_p in CASES:
            with self.subTest(a_p=a_p, n_p=n_p):
                ab = Sphere.mie_coefficients(a_p,
                                             n_p,
                                             0.,
                                             self.n_m,
                                             self.wavelength)
                x, m, nmax = self.case(a_p, n_p)
                k = min(nmax, 10)
                np.testing.assert_allclose(ab[1:k + 1],
                                           bohren_huffman(x, m, k)[1:k + 1],
                                           rtol=1e-9)

    def test_absorbing_sphere(self) -> None:
        x, m, nmax = self.case(3., 1.59)
        m += 0.05j / self.n_m
        ab = _mie.mie_ab(x, m, nmax)[1:11]
        np.testing.assert_allclose(ab,
                                   bohren_huffman(x, m, 10)[1:11],
                                   rtol=1e-9)

    def test_the_start_of_the_recurrence_is_beyond_the_last_order(self):
        for nmax in (5, 40, 400, 4000):
            self.assertGreaterEqual(_mie.start_margin(nmax), 15)
        self.assertGreater(_mie.start_margin(4000), _mie.start_margin(40))

    def test_shape_and_zeroth_order(self) -> None:
        ab = _mie.mie_ab(5., 1.1 + 0j, 12)
        self.assertEqual(ab.shape, (13, 2))
        self.assertEqual(ab.dtype, np.complex128)
        np.testing.assert_array_equal(ab[0], 0.)

    def test_the_compiled_kernel_is_the_python_one(self) -> None:
        '''With Numba the kernel is compiled; its source is the same.'''
        if not hasattr(_mie.mie_ab, 'py_func'):
            self.skipTest('Numba is not installed')
        for a_p, n_p in CASES:
            with self.subTest(a_p=a_p, n_p=n_p):
                args = self.case(a_p, n_p)
                np.testing.assert_allclose(_mie.mie_ab(*args),
                                           _mie.mie_ab.py_func(*args),
                                           rtol=1e-10,
                                           atol=1e-14)

    def test_the_sphere_uses_the_kernel(self) -> None:
        x, m, nmax = self.case(0.5, 1.45)
        ab = Sphere.mie_coefficients(0.5, 1.45, 0., self.n_m, self.wavelength)
        np.testing.assert_array_equal(ab, _mie.mie_ab(x, m, nmax))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
