import unittest

from theory.Particle import Particle


class TestParticle(unittest.TestCase):

    def setUp(self):
        self.particle = Particle()

    def test_setx(self):
        value = 100.
        self.particle.x_p = value
        self.assertEqual(self.particle.r_p[0], value)

    def test_setr(self):
        value = [100., 200., 300.]
        self.particle.r_p = value
        self.assertEqual(self.particle.x_p, value[0])
        self.assertEqual(self.particle.y_p, value[1])
        self.assertEqual(self.particle.z_p, value[2])

if __name__ == '__main__':
    unittest.main()
