import unittest

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from pylorenzmie.analysis import Optimizer
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.lib import LMObject
from pylorenzmie.theory import LayeredSphere, LorenzMie, Sphere
from pylorenzmie.theory.Sphere import MAX_RADIUS, ModelDomainError

N_M = 1.34
WAVELENGTH = 0.447


def riccati(n, z):
    '''Riccati-Bessel functions psi, psi', chi, chi' (Bohren and Huffman).'''
    j = spherical_jn(n, z)
    jp = spherical_jn(n, z, derivative=True)
    y = spherical_yn(n, z)
    yp = spherical_yn(n, z, derivative=True)
    return z * j, j + z * jp, -z * y, -(y + z * yp)


def bohren_huffman(x, y, m1, m2, nmax):
    '''Coefficients of a coated sphere, Bohren and Huffman Eq. (8.1), with
    Bessel functions of complex argument: an independent evaluation.'''
    n = np.arange(1, nmax + 1)
    psiy, dpsiy, chiy, dchiy = riccati(n, y + 0j)
    xiy, dxiy = psiy - 1j * chiy, dpsiy - 1j * dchiy
    ps2y, dps2y, ch2y, dch2y = riccati(n, m2 * y + 0j)
    ps1x, dps1x, _, _ = riccati(n, m1 * x + 0j)
    ps2x, dps2x, ch2x, dch2x = riccati(n, m2 * x + 0j)
    A = ((m2 * ps2x * dps1x - m1 * dps2x * ps1x)
         / (m2 * ch2x * dps1x - m1 * dch2x * ps1x))
    B = ((m2 * ps1x * dps2x - m1 * ps2x * dps1x)
         / (m2 * dch2x * ps1x - m1 * dps1x * ch2x))
    a = ((psiy * (dps2y - A * dch2y) - m2 * dpsiy * (ps2y - A * ch2y))
         / (xiy * (dps2y - A * dch2y) - m2 * dxiy * (ps2y - A * ch2y)))
    b = ((m2 * psiy * (dps2y - B * dch2y) - dpsiy * (ps2y - B * ch2y))
         / (m2 * xiy * (dps2y - B * dch2y) - dxiy * (ps2y - B * ch2y)))
    ab = np.zeros((nmax + 1, 2), complex)
    ab[1:, 0], ab[1:, 1] = a, b
    return ab


def reference(sphere, n_m=N_M, wavelength=WAVELENGTH):
    k = 2. * np.pi * n_m / wavelength
    x, y = k * sphere.a_p, k * sphere.outer_radius
    m1 = (sphere.n_p + 1j * sphere.k_p) / n_m
    m2 = (sphere.n_s + 1j * sphere.k_s) / n_m
    return bohren_huffman(x, y, m1, m2, Sphere.wiscombe_yang(
        y, max(abs(m1) * x / y, abs(m2))))


CASES = [
    # a_p, n_p, k_p, t_s, n_s, k_s
    (0.5, 1.59, 0., 0.010, 1.45, 0.),        # a protein layer on polystyrene
    (0.5, 1.59, 0., 0.100, 1.45, 0.),
    (1.0, 1.59, 0., 0.005, 1.45, 0.),        # very thin
    (1.5, 1.33, 0., 0.200, 1.59, 0.),        # a shell of a high index
    (1.0, 1.00, 0., 0.050, 1.50, 0.),        # a hollow core
    (1.0, 1.59, 0.05, 0.100, 1.45, 0.02),    # absorbing core and shell
    (3.0, 1.45, 0., 0.300, 1.59, 0.),        # a larger sphere
    (2.0, 1.90, 0., 0.030, 1.40, 0.),
]


class TestLayeredSphere(unittest.TestCase):

    def test_is_a_sphere_with_a_shell(self):
        sphere = LayeredSphere(a_p=0.5, t_s=0.02)
        self.assertIsInstance(sphere, Sphere)
        self.assertEqual(sphere.outer_radius, 0.52)

    def test_properties_have_the_shell(self):
        sphere = LayeredSphere(
            a_p=0.5,
            n_p=1.59,
            t_s=0.02,
            n_s=1.45,
            k_s=0.01,
            x_p=10.,
            y_p=20.,
            z_p=300.,
        )
        properties = sphere.properties
        for name in ('x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'k_p',
                     't_s', 'n_s', 'k_s'):
            self.assertIn(name, properties)
        other = LayeredSphere()
        other.from_json(sphere.to_json())
        self.assertEqual(other, sphere)
        other = LayeredSphere()
        other.from_pandas(sphere.to_pandas())
        self.assertEqual(other, sphere)

    def test_coefficients_agree_with_an_independent_evaluation(self):
        for a_p, n_p, k_p, t_s, n_s, k_s in CASES:
            with self.subTest(a_p=a_p, n_p=n_p, t_s=t_s, n_s=n_s):
                sphere = LayeredSphere(
                    a_p=a_p,
                    n_p=n_p,
                    k_p=k_p,
                    t_s=t_s,
                    n_s=n_s,
                    k_s=k_s,
                )
                ab = sphere.ab(N_M, WAVELENGTH)
                expected = reference(sphere)
                self.assertEqual(ab.shape, expected.shape)
                np.testing.assert_allclose(ab,
                                           expected,
                                           rtol=1e-8,
                                           atol=1e-11)

    def test_a_shell_of_the_index_of_the_core_is_a_larger_sphere(self):
        for t_s in (0.005, 0.1):
            with self.subTest(t_s=t_s):
                layered = LayeredSphere(a_p=0.7, n_p=1.5, t_s=t_s, n_s=1.5)
                sphere = Sphere(a_p=0.7 + t_s, n_p=1.5)
                np.testing.assert_allclose(layered.ab(N_M, WAVELENGTH),
                                           sphere.ab(N_M, WAVELENGTH),
                                           rtol=1e-9,
                                           atol=1e-13)

    def test_a_shell_of_no_thickness_is_the_core(self):
        layered = LayeredSphere(a_p=0.7, n_p=1.59, t_s=0., n_s=1.3)
        sphere = Sphere(a_p=0.7, n_p=1.59)
        np.testing.assert_array_equal(layered.ab(N_M, WAVELENGTH),
                                      sphere.ab(N_M, WAVELENGTH))

    def test_a_very_thin_shell_is_nearly_the_core(self):
        layered = LayeredSphere(a_p=0.7, n_p=1.59, t_s=1e-6, n_s=1.3)
        sphere = Sphere(a_p=0.7, n_p=1.59)
        np.testing.assert_allclose(layered.ab(N_M, WAVELENGTH),
                                   sphere.ab(N_M, WAVELENGTH),
                                   rtol=1e-4,
                                   atol=1e-8)

    def test_a_shell_changes_the_coefficients(self):
        core = Sphere(a_p=0.5, n_p=1.59).ab(N_M, WAVELENGTH)
        shell = LayeredSphere(a_p=0.5, n_p=1.59, t_s=0.05, n_s=1.45)
        shell = shell.ab(N_M, WAVELENGTH)
        self.assertGreater(np.max(np.abs(shell[:core.shape[0]]
                                         - core[:shell.shape[0]])), 1e-3)

    def test_parameters_outside_the_model_are_refused(self):
        refused = [dict(a_p=0.), dict(a_p=-1.), dict(t_s=-0.01),
                   dict(a_p=np.nan), dict(t_s=np.nan),
                   dict(a_p=40., t_s=20.)]
        for parameters in refused:
            with self.subTest(**parameters), \
                    self.assertRaises(ModelDomainError):
                LayeredSphere(**parameters).ab(N_M, WAVELENGTH)

    def test_the_whole_sphere_is_limited_not_the_core(self):
        with self.assertRaises(ModelDomainError):
            LayeredSphere(a_p=MAX_RADIUS - 0.5, t_s=1.).ab(N_M, WAVELENGTH)


class TestLayeredSphereHologram(unittest.TestCase):

    def setUp(self):
        self.shape = (41, 41)
        self.model = LorenzMie(coordinates=LMObject.meshgrid(self.shape))
        self.model.instrument.wavelength = WAVELENGTH
        self.model.instrument.magnification = 0.048
        self.model.instrument.n_m = N_M

    def hologram(self, particle):
        self.model.particle = particle
        return self.model.hologram()

    def position(self):
        return dict(x_p=20., y_p=20., z_p=300.)

    def test_a_shell_of_no_thickness_gives_the_hologram_of_the_core(self):
        layered = LayeredSphere(a_p=0.7, n_p=1.59, t_s=0., **self.position())
        sphere = Sphere(a_p=0.7, n_p=1.59, **self.position())
        np.testing.assert_array_equal(self.hologram(layered),
                                      self.hologram(sphere))

    def test_a_shell_changes_the_hologram(self):
        layered = LayeredSphere(
            a_p=0.7,
            n_p=1.59,
            t_s=0.1,
            n_s=1.45,
            **self.position(),
        )
        sphere = Sphere(a_p=0.7, n_p=1.59, **self.position())
        change = np.abs(self.hologram(layered) - self.hologram(sphere))
        self.assertGreater(change.max(), 1e-3)

    def test_the_hologram_is_that_of_the_reference_coefficients(self):
        layered = LayeredSphere(
            a_p=0.7,
            n_p=1.59,
            t_s=0.05,
            n_s=1.45,
            **self.position(),
        )
        expected = self.hologram(layered)

        class Reference(Sphere):
            def ab(self, n_m=1.42, wavelength=0.532):
                return reference(layered, n_m, wavelength)

        particle = Reference(a_p=0.7, n_p=1.59, **self.position())
        np.testing.assert_allclose(self.hologram(particle),
                                   expected,
                                   rtol=1e-8,
                                   atol=1e-10)


class TestLayeredSphereFit(unittest.TestCase):
    '''Fits of a thick shell, on a small grid so that it is fast.'''

    def setUp(self):
        self.shape = (31, 31)
        self.model = LorenzMie(coordinates=LMObject.meshgrid(self.shape))
        self.model.instrument.wavelength = WAVELENGTH
        self.model.instrument.magnification = 0.048
        self.model.instrument.n_m = N_M
        self.model.instrument.noise = 0.02
        self.truth = dict(
            x_p=15.,
            y_p=15.,
            z_p=200.,
            a_p=0.7,
            n_p=1.40,
            t_s=0.2,
            n_s=1.60,
        )
        self.model.particle = LayeredSphere(**self.truth)
        self.data = self.model.hologram().reshape(self.shape)

    def optimizer(self, **start):
        model = LorenzMie(coordinates=LMObject.meshgrid(self.shape))
        model.instrument.wavelength = WAVELENGTH
        model.instrument.magnification = 0.048
        model.instrument.n_m = N_M
        model.instrument.noise = 0.02
        model.particle = LayeredSphere(**{**self.truth, **start})
        return Optimizer(model=model)

    def test_the_absorption_is_fixed_by_default(self):
        optimizer = self.optimizer()
        for name in ('k_p', 'k_s', 'wavelength', 'noise'):
            self.assertIn(name, optimizer.fixed)
        for name in ('x_p', 'a_p', 'n_p', 't_s', 'n_s'):
            self.assertIn(name, optimizer.variables)

    def test_a_fit_finds_the_shell(self):
        optimizer = self.optimizer(
            a_p=0.75,
            n_p=1.42,
            t_s=0.17,
            n_s=1.55,
            z_p=203.,
            x_p=15.3,
            y_p=14.8,
        )
        result = optimizer.optimize(Hologram(self.data))
        self.assertLess(result['redchi'], 1.5)
        for name in ('a_p', 'n_p', 't_s', 'n_s'):
            with self.subTest(name=name):
                self.assertAlmostEqual(result[name],
                                       self.truth[name],
                                       delta=0.02)

    def test_a_thickness_that_cannot_be_negative_is_bounded(self):
        optimizer = self.optimizer(t_s=0.05)
        optimizer.bounds = {'t_s': (0., 1.)}
        result = optimizer.optimize(Hologram(self.data))
        self.assertGreaterEqual(result['t_s'], 0.)

    def test_a_step_to_a_negative_thickness_is_rejected(self):
        model = self.optimizer().model
        model.particle.t_s = -0.01
        with self.assertRaises(ModelDomainError):
            model.hologram()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
