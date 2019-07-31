import unittest
from pylorenzmie.theory.Sphere import Sphere


class TestSphere(unittest.TestCase):
    def setUp(self):
        self.particle = Sphere()
        self.particle.r_p = (100., 200, 300)

    def test_set_ap(self):
        value = 1.
        self.particle.a_p = value
        self.assertEqual(value, self.particle.a_p)

    def test_set_np(self):
        value = 1.5
        self.particle.n_p = value
        self.assertEqual(value, self.particle.n_p)


if __name__ == '__main__':
    unittest.main()
