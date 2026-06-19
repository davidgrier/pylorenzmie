import unittest
import numpy as np
from pylorenzmie.theory import Particle


class TestParticle(unittest.TestCase):

    def setUp(self):
        self.particle = Particle()

    def test_default_values(self):
        self.assertEqual(self.particle.x_p, 0.0)
        self.assertEqual(self.particle.y_p, 0.0)
        self.assertEqual(self.particle.z_p, 100.0)
        np.testing.assert_array_equal(
            self.particle.r_p, np.array([0.0, 0.0, 100.0]))

    def test_r_p_setter_and_getter(self):
        new_position = np.array([10.0, 20.0, 30.0])
        self.particle.r_p = new_position
        np.testing.assert_array_equal(self.particle.r_p, new_position)
        self.assertEqual(self.particle.x_p, 10.0)
        self.assertEqual(self.particle.y_p, 20.0)
        self.assertEqual(self.particle.z_p, 30.0)

    def test_r_0_setter_and_getter(self):
        new_origin = np.array([1.0, 2.0, 3.0])
        self.particle.r_0 = new_origin
        np.testing.assert_array_equal(self.particle.r_0, new_origin)
        self.assertEqual(self.particle.x_0, 1.0)
        self.assertEqual(self.particle.y_0, 2.0)
        self.assertEqual(self.particle.z_0, 3.0)

    def test_properties(self):
        self.particle.x_p = 5.0
        self.particle.y_p = 6.0
        self.particle.z_p = 7.0
        props = self.particle.properties
        expected = {'x_p': 5.0, 'y_p': 6.0, 'z_p': 7.0}
        self.assertEqual(props, expected)

    def test_ab_returns_correct_array(self):
        result = self.particle.ab()
        expected = np.ones((2, 1), dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_ab_accepts_parameters(self):
        # Even with parameters, it should return the same default array
        result = self.particle.ab(n_m=1.33 + 0.01j, wavelength=0.532)
        expected = np.ones((2, 1), dtype=complex)
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
