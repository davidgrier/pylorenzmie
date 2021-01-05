import unittest

from theory import Particle


class TestParticle(unittest.TestCase):

    def setUp(self):
        self.particle = Particle()

    def test_setcoordinates(self):
        value = 100.
        self.particle.x_p = value
        self.assertEqual(self.particle.r_p[0], value)
        value += 1.
        self.particle.y_p = value
        self.assertEqual(self.particle.r_p[1], value)
        value += 1.
        self.particle.z_p = value
        self.assertEqual(self.particle.r_p[2], value)

    def test_setposition(self):
        value = [100., 200., 300.]
        self.particle.r_p = value
        self.assertEqual(self.particle.x_p, value[0])
        self.assertEqual(self.particle.y_p, value[1])
        self.assertEqual(self.particle.z_p, value[2])

    def test_properties(self):
        props = self.particle.properties
        value = props['x_p'] + 1.
        props['x_p'] = value
        self.particle.properties = props
        self.assertEqual(self.particle.x_p, value)

    def test_serialize(self):
        ser = self.particle.dumps()
        b = Particle()
        b.loads(ser)
        self.assertEqual(self.particle.x_p, b.x_p)

    def test_ab(self):
        ab = self.particle.ab()
        self.assertEqual(ab[0], 1.)

    def test_repr(self):
        s = repr(self.particle)
        self.assertTrue(isinstance(s, str))

if __name__ == '__main__':
    unittest.main()
