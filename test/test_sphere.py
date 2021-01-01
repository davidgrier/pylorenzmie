import unittest

from theory.Sphere import Sphere


class TestSphere(unittest.TestCase):

    def setUp(self):
        self.particle = Sphere()

    def test_seta(self):
        value = 2.
        self.particle.a_p = value
        self.assertEqual(self.particle.a_p, value)
        
        self.particle.a_p = [value]
        self.assertEqual(self.particle.a_p, value)

        value = [1., 2.]
        self.particle.a_p = value
        self.assertTrue((self.particle.a_p == value).all())

    def test_ab(self):
        ab = self.particle.ab(1.339, 0.447)
        self.assertEqual(ab.size, 64)

    def test_properties(self):
        props = self.particle.properties
        value = props['a_p'] + 0.25
        props['a_p'] = value
        self.particle.properties = props
        self.assertEqual(self.particle.a_p, value)

    def test_repr(self):
        s = repr(self.particle)
        self.assertTrue(isinstance(s, str))
        

if __name__ == '__main__':
    unittest.main()
