import pickle
import unittest

import numpy as np

from pylorenzmie.theory import Instrument, Sphere
from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.Aberrated import AberratedLorenzMie


class TestAberratedLorenzMie(unittest.TestCase):

    def setUp(self):
        self.n = 51
        self.xp = self.yp = (self.n - 1) / 2.
        self.zp = 100.
        coordinates = LorenzMie.meshgrid((self.n, self.n))
        self.particle = Sphere(a_p=0.5, n_p=1.5,
                               x_p=self.xp, y_p=self.yp, z_p=self.zp)
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]

    def model(self, **kwargs):
        m = AberratedLorenzMie(coordinates=self.coordinates, **kwargs)
        m.particle = self.particle
        return m

    def mask(self, **kwargs):
        m = self.model(**kwargs)
        return m._aberration(self.particle.r_p)

    def geometry(self):
        '''Return (dx, dy, r, u) of the field points about the particle.'''
        dx = self.x - self.xp
        dy = self.y - self.yp
        rsq = dx*dx + dy*dy
        return dx, dy, np.sqrt(rsq), rsq / (rsq + self.zp**2)

    def test_no_aberration_matches_lorenzmie(self):
        ideal = LorenzMie(coordinates=self.coordinates)
        ideal.particle = self.particle
        aberrated = self.model()
        self.assertTrue(np.allclose(aberrated.hologram(),
                                    ideal.hologram()))

    def test_no_aberration_returns_float(self):
        # GPU code detects 'no aberration' with isinstance(mask, float)
        self.assertIsInstance(self.mask(), float)

    def ray_phase(self, name, c, xc, yc):
        '''Expected phase [rad] for a field-dependent aberration.

        Written from Table I of the paper, with explicit vectors: the
        ray coordinate eta = (r - r_c) / lambda at each field point, and
        r_hat the unit vector from the particle to the field point.
        '''
        inst = Instrument()
        n_m = inst.n_m
        scale = inst.wavelength / inst.magnification       # pixels
        dx, dy, r, u = self.geometry()
        etax = (self.x - xc) / scale
        etay = (self.y - yc) / scale
        rx = np.divide(dx, r, out=np.zeros_like(u), where=r > 0)
        ry = np.divide(dy, r, out=np.zeros_like(u), where=r > 0)
        r_eta = rx * etax + ry * etay                      # r_hat . eta
        eta2 = etax**2 + etay**2
        sin_beta = np.sqrt(u)
        if name == 'coma':
            return 6.*np.pi*n_m**3 * c * sin_beta**3 * r_eta
        if name == 'astigmatism':
            return 4.*np.pi*n_m**2 * c * sin_beta**2 * r_eta**2
        if name == 'distortion':
            return 2.*np.pi*n_m * c * sin_beta * eta2 * r_eta
        raise ValueError(name)

    def test_center_at_particle_is_finite(self):
        # The ray coordinate is the position of the ray at each field
        # point, so an aberration centered on the particle is not zero:
        # eta = rho / lambda.  The guards must prevent NaN where r = 0.
        for name, c in (('coma', 0.5), ('astigmatism', 0.05),
                        ('distortion', 0.01)):
            m = self.mask(**{name: c}, x_c=self.xp, y_c=self.yp)
            self.assertTrue(np.all(np.isfinite(m)), name)
            expected = self.ray_phase(name, c, self.xp, self.yp)
            self.assertGreater(np.abs(expected).max(), 0.1, name)
            self.assertTrue(np.allclose(m, np.exp(1j * expected)), name)

    def test_piston_only_is_pure_phase(self):
        m = self.mask(piston=0.7)
        self.assertTrue(np.allclose(m, np.exp(2j*np.pi*0.7)))

    def test_piston_is_a_property(self):
        m = self.model(piston=0.7)
        self.assertEqual(m.properties['piston'], 0.7)
        m.properties = dict(piston=-0.3)
        self.assertEqual(m.piston, -0.3)

    def test_piston_shifts_the_scattered_phase_only(self):
        # the hologram is |1 + exp(2 pi i piston) E_s|^2, not a phase shift of
        # both fields, which would change nothing
        ideal = LorenzMie(coordinates=self.coordinates)
        ideal.particle = self.particle
        field = ideal.scattered_field(self.particle)
        phi = 0.7
        reference = np.zeros_like(field)
        reference[0] = 1.
        expected = np.sum(np.abs(reference + np.exp(2j*np.pi*phi)*field)**2,
                          axis=0)
        np.testing.assert_allclose(self.model(piston=phi).hologram(),
                                   expected)

    def test_piston_changes_the_hologram(self):
        self.assertGreater(np.abs(self.model(piston=0.7).hologram() -
                                  self.model().hologram()).max(), 1e-3)

    def test_piston_adds_to_the_seidel_phase(self):
        both = self.mask(spherical=1., piston=0.4)
        np.testing.assert_allclose(both, self.mask(spherical=1.) *
                                   np.exp(2j*np.pi*0.4))

    def test_defocus(self):
        n_m = Instrument().n_m
        u = self.geometry()[3]
        m = self.mask(defocus=0.3)
        self.assertTrue(np.allclose(m, np.exp(1j * 4.*np.pi*n_m**2 *
                                              0.3 * u)))

    def test_spherical(self):
        n_m = Instrument().n_m
        u = self.geometry()[3]
        m = self.mask(spherical=0.3)
        self.assertTrue(np.allclose(m, np.exp(1j * 8.*np.pi*n_m**4 *
                                              0.3 * u**2)))

    def test_field_dependent_phases_follow_table_one(self):
        # a center displaced from the particle along -y, as in the paper
        d = 20.
        for name, c in (('coma', 0.5), ('astigmatism', 0.05),
                        ('distortion', 0.01)):
            expected = self.ray_phase(name, c, self.xp, self.yp - d)
            self.assertGreater(np.abs(expected).max(), 0.1, name)
            m = self.mask(**{name: c}, x_c=self.xp, y_c=self.yp - d)
            self.assertTrue(np.allclose(m, np.exp(1j * expected)), name)

    def test_far_center_recovers_the_particle_centered_terms(self):
        # For a center much farther from the particle than the field
        # point, the ray coordinate is nearly the displacement of the
        # particle from the center, and the phases reduce to the leading
        # terms: eta = d / lambda with d along +y gives
        # coma ~ 6 pi n^3 c eta sin^3(beta) sin(theta) and so on.
        n_m = Instrument().n_m
        inst = Instrument()
        scale = inst.wavelength / inst.magnification
        d = 1e6
        dx, dy, r, u = self.geometry()
        sin_theta = np.divide(dy, r, out=np.zeros_like(u), where=r > 0)
        eta = d / scale
        sin_beta = np.sqrt(u)
        # coefficients that give phases of order one
        c = {'coma': 1. / eta, 'astigmatism': 1. / eta**2,
             'distortion': 1. / eta**3}
        expected = {
            'coma': 6.*np.pi*n_m**3 * c['coma'] * eta * sin_beta**3 *
            sin_theta,
            'astigmatism': (4.*np.pi*n_m**2 * c['astigmatism'] * eta**2 *
                            u * sin_theta**2),
            'distortion': (2.*np.pi*n_m * c['distortion'] * eta**3 *
                           sin_beta * sin_theta)}
        for name in c:
            m = self.mask(**{name: c[name]}, x_c=self.xp, y_c=self.yp - d)
            self.assertGreater(np.abs(expected[name]).max(), 0.1, name)
            np.testing.assert_allclose(np.angle(m), expected[name],
                                       rtol=1e-3, atol=1e-4, err_msg=name)

    def test_pickle_class(self):
        cls = pickle.loads(pickle.dumps(AberratedLorenzMie))
        self.assertIs(cls, AberratedLorenzMie)

    def test_rotation_covariance(self):
        # Rotating the aberration center about the particle rotates the
        # mask.  The grid is symmetric about the particle, so a quarter
        # turn maps it onto itself.  The flattened grid has x varying
        # fastest, so reshaped arrays are indexed a[y, x] with y
        # increasing down the rows; moving the displacement from +y to
        # +x is then a counterclockwise quarter turn of the array.
        d = 20.
        for name in ('coma', 'astigmatism', 'distortion'):
            m1 = self.mask(**{name: 0.5},
                           x_c=self.xp, y_c=self.yp - d)
            m2 = self.mask(**{name: 0.5},
                           x_c=self.xp - d, y_c=self.yp)
            a1 = m1.reshape(self.n, self.n)
            a2 = m2.reshape(self.n, self.n)
            self.assertTrue(np.allclose(np.rot90(a1, k=1), a2),
                            msg=name)


class TestAberratedKernelHooks(unittest.TestCase):
    '''The hooks that calculators with an in-kernel phase use.'''

    def setUp(self):
        self.model = AberratedLorenzMie(coordinates=LorenzMie.meshgrid(
            (32, 32)))
        self.model.particle = Sphere(a_p=0.75, n_p=1.45, x_p=16., y_p=16.,
                                     z_p=120.)

    def test_no_coefficients_without_aberrations(self):
        self.assertIsNone(self.model._aberration_coefficients())

    def test_coefficients_in_a_fixed_order(self):
        self.model.defocus = 0.5
        self.model.spherical = 2.
        self.model.coma = 0.1
        self.model.astigmatism = 0.2
        self.model.distortion = 0.3
        self.model.x_c = 7.
        self.model.y_c = 9.
        inst = self.model.instrument
        np.testing.assert_array_equal(
            self.model._aberration_coefficients(),
            [0.5, 2., 0.1, 0.2, 0.3, 7., 9.,
             inst.wavelength / inst.magnification])

    def test_a_calculator_that_applies_the_phase_itself_is_not_multiplied(self):
        class InKernel(type(self.model)):
            _aberrations_in_kernel = True

        plain = LorenzMie(coordinates=self.model.coordinates)
        plain.particle = self.model.particle
        aberrated = InKernel(coordinates=self.model.coordinates,
                             spherical=2.)
        aberrated.particle = self.model.particle
        # the Python mask is skipped: the field is that of the plain model
        np.testing.assert_allclose(
            aberrated.scattered_field(aberrated.particle),
            plain.scattered_field(plain.particle))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
