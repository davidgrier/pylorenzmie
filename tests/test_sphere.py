import unittest
import numpy as np
from pylorenzmie.theory.Sphere import MAX_ORDERS, MAX_RADIUS, Sphere

# TODO: add spot-check tests for specific ab coefficient values against
# a published reference (e.g. Bohren & Huffman Appendix A) once a
# suitable reference dataset is identified.


class TestSphere(unittest.TestCase):

    def setUp(self):
        self.sphere = Sphere(a_p=1., n_p=1.4, k_p=0.)
        self.n_m = 1.34
        self.wavelength = 0.447

    # --- attributes ---

    def test_default_values(self):
        s = Sphere()
        self.assertEqual(s.a_p, 1.)
        self.assertEqual(s.n_p, 1.5)
        self.assertEqual(s.k_p, 0.)

    def test_d_p_getter(self):
        self.assertAlmostEqual(self.sphere.d_p, 2. * self.sphere.a_p)

    def test_d_p_setter(self):
        self.sphere.d_p = 3.
        self.assertAlmostEqual(self.sphere.a_p, 1.5)

    def test_repr(self):
        self.assertIsInstance(repr(self.sphere), str)

    # --- properties protocol ---

    def test_properties_keys(self):
        props = self.sphere.properties
        for key in ('x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'k_p'):
            self.assertIn(key, props)

    def test_properties_roundtrip(self):
        props = self.sphere.properties
        props['a_p'] += 0.25
        self.sphere.properties = props
        self.assertEqual(self.sphere.a_p, props['a_p'])

    # --- ab coefficients ---

    def test_ab_shape(self):
        ab = self.sphere.ab(self.n_m, self.wavelength)
        self.assertEqual(ab.ndim, 2)
        self.assertEqual(ab.shape[1], 2)

    def test_ab_dtype(self):
        ab = self.sphere.ab(self.n_m, self.wavelength)
        self.assertEqual(ab.dtype, np.complex128)

    def test_ab_zeroth_order(self):
        ab = self.sphere.ab(self.n_m, self.wavelength)
        np.testing.assert_array_equal(ab[0, :], 0j)

    # --- wiscombe_yang ---

    def test_wiscombe_yang_small(self):
        self.assertEqual(Sphere.wiscombe_yang(1., 10.+0.j), 10)

    def test_wiscombe_yang_mid(self):
        self.assertEqual(Sphere.wiscombe_yang(10., 10.+0.j), 100)

    def test_wiscombe_yang_large(self):
        self.assertEqual(Sphere.wiscombe_yang(4210., 10.+0.j), 42100)

    # --- neves_pisignano ---

    def test_neves_pisignano_type(self):
        self.assertIsInstance(Sphere.neves_pisignano(10.), int)

    def test_neves_pisignano_value(self):
        # x=10, precision=6: 10 + 0.76*cbrt(360) - 4.1 ≈ 11.31 → 11
        self.assertEqual(Sphere.neves_pisignano(10.), 11)

    # --- mie_coefficients ---

    def test_mie_coefficients_shape(self):
        ab = Sphere.mie_coefficients(self.sphere.a_p,
                                     self.sphere.n_p,
                                     self.sphere.k_p,
                                     self.n_m,
                                     self.wavelength)
        self.assertEqual(ab.ndim, 2)
        self.assertEqual(ab.shape[1], 2)

    def test_mie_coefficients_dtype(self):
        ab = Sphere.mie_coefficients(self.sphere.a_p,
                                     self.sphere.n_p,
                                     self.sphere.k_p,
                                     self.n_m,
                                     self.wavelength)
        self.assertEqual(ab.dtype, np.complex128)

    def test_mie_coefficients_zeroth_order(self):
        ab = Sphere.mie_coefficients(self.sphere.a_p,
                                     self.sphere.n_p,
                                     self.sphere.k_p,
                                     self.n_m,
                                     self.wavelength)
        np.testing.assert_array_equal(ab[0, :], 0j)

    # --- the largest radius ---

    def test_the_largest_radius_is_computed(self):
        self.sphere.a_p = MAX_RADIUS
        ab = self.sphere.ab(self.n_m, self.wavelength)
        self.assertEqual(ab.shape[0] - 1,
                         Sphere.wiscombe_yang(
                             2. * np.pi * self.n_m * MAX_RADIUS /
                             self.wavelength,
                             self.sphere.n_p / self.n_m))

    def test_a_radius_that_is_too_large_is_refused(self):
        for a_p in (MAX_RADIUS + 0.01, -2. * MAX_RADIUS, 850., np.nan):
            with self.subTest(a_p=a_p):
                self.sphere.a_p = a_p
                with self.assertRaisesRegex(ValueError, 'radius'):
                    self.sphere.ab(self.n_m, self.wavelength)

    def test_a_sphere_that_needs_too_many_orders_is_refused(self):
        # A radius of 1 um with an index of 1e5 needs 1.4 million orders.
        # (A radius of 850 um with an index of 1.6e5 needed 1.9 billion,
        # 60 GB, and ended the process; the radius is refused first.)
        for a_p, n_p in ((1., 1e5), (50., 1e4)):
            with self.subTest(a_p=a_p, n_p=n_p):
                self.sphere.a_p, self.sphere.n_p = a_p, n_p
                with self.assertRaisesRegex(ValueError, 'partial waves'):
                    self.sphere.ab(self.n_m, self.wavelength)

    def test_the_largest_sphere_that_is_physical_is_computed(self):
        # 50 um, an index of 3, in air at 300 nm: 3141 orders
        sphere = Sphere(a_p=MAX_RADIUS, n_p=3., k_p=0.)
        ab = sphere.ab(1., 0.3)
        self.assertLess(ab.shape[0] - 1, MAX_ORDERS)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
