import unittest

from theory.Sphere import Sphere


class TestSphere(unittest.TestCase):

    def setUp(self):
        self.particle = Sphere()

    def test_setxa(self):
        value = 2.
        self.particle.a_p = value
        self.assertEqual(self.particle.a_p, value)
        
        self.particle.a_p = [value]
        self.assertEqual(self.particle.a_p, value)

        value = [1., 2.]
        self.particle.a_p = value
        self.assertTrue((self.particle.a_p == value).all())
        

if __name__ == '__main__':
    unittest.main()
