import unittest
import numpy as np
from pylorenzmie.theory import Particle


class TestParticle(unittest.TestCase):

    def setUp(self) -> None:
        self.particle = Particle()

    def test_default_values(self) -> None:
        self.assertEqual(self.particle.x_p, 0.)
        self.assertEqual(self.particle.y_p, 0.)
        self.assertEqual(self.particle.z_p, 100.)
        self.assertEqual(self.particle.x_0, 0.)
        self.assertEqual(self.particle.y_0, 0.)
        self.assertEqual(self.particle.z_0, 0.)

    def test_r_p_setter(self) -> None:
        self.particle.r_p = [10., 20., 30.]
        self.assertEqual(self.particle.x_p, 10.)
        self.assertEqual(self.particle.y_p, 20.)
        self.assertEqual(self.particle.z_p, 30.)

    def test_r_p_getter(self) -> None:
        self.particle.x_p = 1.
        self.particle.y_p = 2.
        self.particle.z_p = 3.
        np.testing.assert_array_equal(self.particle.r_p, [1., 2., 3.])

    def test_r_0_setter(self) -> None:
        self.particle.r_0 = [1., 2., 3.]
        self.assertEqual(self.particle.x_0, 1.)
        self.assertEqual(self.particle.y_0, 2.)
        self.assertEqual(self.particle.z_0, 3.)

    def test_r_0_getter(self) -> None:
        self.particle.x_0 = 4.
        self.particle.y_0 = 5.
        self.particle.z_0 = 6.
        np.testing.assert_array_equal(self.particle.r_0, [4., 5., 6.])

    def test_properties_getter(self) -> None:
        self.particle.x_p = 5.
        self.particle.y_p = 6.
        self.particle.z_p = 7.
        self.assertEqual(self.particle.properties,
                         {'x_p': 5., 'y_p': 6., 'z_p': 7.})

    def test_properties_setter(self) -> None:
        self.particle.properties = {'x_p': 9., 'y_p': 8., 'z_p': 7.}
        self.assertEqual(self.particle.x_p, 9.)
        self.assertEqual(self.particle.y_p, 8.)
        self.assertEqual(self.particle.z_p, 7.)

    def test_to_json_roundtrip(self) -> None:
        self.particle.r_p = [1., 2., 3.]
        b = Particle()
        b.from_json(self.particle.to_json())
        self.assertEqual(b.x_p, self.particle.x_p)
        self.assertEqual(b.y_p, self.particle.y_p)
        self.assertEqual(b.z_p, self.particle.z_p)

    def test_to_pandas_roundtrip(self) -> None:
        self.particle.r_p = [1., 2., 3.]
        b = Particle()
        b.from_pandas(self.particle.to_pandas())
        self.assertEqual(b.x_p, self.particle.x_p)
        self.assertEqual(b.y_p, self.particle.y_p)
        self.assertEqual(b.z_p, self.particle.z_p)

    def test_ab_shape(self) -> None:
        ab = self.particle.ab()
        self.assertEqual(ab.shape, (1, 2))
        self.assertEqual(ab.dtype, complex)

    def test_ab_values(self) -> None:
        np.testing.assert_array_equal(self.particle.ab(),
                                      np.ones((1, 2), dtype=complex))

    def test_ab_accepts_parameters(self) -> None:
        ab = self.particle.ab(n_m=1.33 + 0.01j, wavelength=0.447)
        self.assertEqual(ab.shape, (1, 2))

    def test_len(self) -> None:
        self.assertEqual(len(self.particle), 1)

    def test_iter(self) -> None:
        particles = list(self.particle)
        self.assertEqual(len(particles), 1)
        self.assertIs(particles[0], self.particle)

    def test_iter_reentrant(self) -> None:
        self.assertEqual(list(self.particle), list(self.particle))

    def test_getitem(self) -> None:
        self.assertIs(self.particle[0], self.particle)

    def test_getitem_out_of_range(self) -> None:
        with self.assertRaises(IndexError):
            _ = self.particle[1]

    def test_repr(self) -> None:
        self.assertIsInstance(repr(self.particle), str)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
