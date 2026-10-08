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

    def test_center_at_particle_is_finite(self):
        # degenerate aberration direction: the guards must prevent NaN.
        # With no displacement the off-axis terms contribute no phase.
        m = self.mask(coma=1., astigmatism=1., distortion=1.,
                      x_c=self.xp, y_c=self.yp)
        self.assertTrue(np.all(np.isfinite(m)))
        self.assertTrue(np.allclose(m, 1.))

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

    def table_frame(self, d=20.):
        '''Return (n_m, eta, u, sin_theta) for the Table I frame.

        The particle is displaced along +y from the aberration center,
        so cos(phi - psi) = sin(theta) = dy/r.
        '''
        n_m = Instrument().n_m
        dx, dy, r, u = self.geometry()
        sin_theta = np.divide(dy, r, out=np.zeros_like(u), where=r > 0)
        return n_m, d / self.zp, u, sin_theta

    def test_table_frame_coma(self):
        # includes points with sin(theta) < 0, which fix the sign
        d = 20.
        c = 0.7
        n_m, eta, u, sin_theta = self.table_frame(d)
        expected = 6.*np.pi*n_m**3 * c * eta * u**1.5 * sin_theta
        m = self.mask(coma=c, x_c=self.xp, y_c=self.yp - d)
        self.assertTrue(np.allclose(m, np.exp(1j * expected)))

    def test_table_frame_astigmatism(self):
        d = 20.
        a = 0.7
        n_m, eta, u, sin_theta = self.table_frame(d)
        expected = 4.*np.pi*n_m**2 * a * eta**2 * u * sin_theta**2
        m = self.mask(astigmatism=a, x_c=self.xp, y_c=self.yp - d)
        self.assertTrue(np.allclose(m, np.exp(1j * expected)))

    def test_pickle_class(self):
        cls = pickle.loads(pickle.dumps(AberratedLorenzMie))
        self.assertIs(cls, AberratedLorenzMie)

    def test_table_frame_distortion(self):
        d = 20.
        t = 0.7
        n_m, eta, u, sin_theta = self.table_frame(d)
        expected = 2.*np.pi*n_m * t * eta**3 * np.sqrt(u) * sin_theta
        m = self.mask(distortion=t, x_c=self.xp, y_c=self.yp - d)
        self.assertTrue(np.allclose(m, np.exp(1j * expected)))

    def test_orientation_follows_displacement(self):
        # The odd phase is largest for field points along the displacement
        # of the particle from the aberration center and vanishes for
        # points perpendicular to it.  The particle is displaced along +y.
        d = 20.
        dx, dy, r, u = self.geometry()
        n_m = Instrument().n_m
        c = 0.5
        m = self.mask(coma=c, x_c=self.xp, y_c=self.yp - d)
        phase = np.angle(m)
        along = (dx == 0) & (dy == 10.)
        perpendicular = (dy == 0) & (dx == 10.)
        self.assertEqual(along.sum(), 1)
        self.assertEqual(perpendicular.sum(), 1)
        expected = (6.*np.pi*n_m**3 * c * (d / self.zp) *
                    np.sqrt(u[along])**3)
        self.assertAlmostEqual(float(phase[along][0]), float(expected[0]))
        self.assertAlmostEqual(float(phase[perpendicular][0]), 0.)

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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
