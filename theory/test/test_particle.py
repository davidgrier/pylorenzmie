import unittest
from pylorenzmie.theory.Particle import Particle


class TestParticle(unittest.TestCase):
    def setUp(self):
        self.particle = Particle()

    def test_set_rp(self):
        self.particle.r_p = (100., 200, -300)
        self.assertEqual(100., self.particle.x_p)
        self.assertEqual(200., self.particle.y_p)
        self.assertEqual(-300., self.particle.z_p)

    def test_set_xp(self):
        self.particle.x_p = -100.
        self.assertEqual(-100, self.particle.x_p)

    def test_multiply_xp(self):
        self.particle.x_p = 100
        self.particle.x_p *= -1
        self.assertEqual(-100, self.particle.x_p)


if __name__ == '__main__':
    unittest.main()
