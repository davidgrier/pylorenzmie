import unittest

from theory.LorenzMie import LorenzMie
from theory.Instrument import coordinates
import numpy as np


class TestLorenzMie(unittest.TestCase):

    def setUp(self):
        self.method = LorenzMie()
        self.shape = [256, 256]

    def test_coordinates_None(self):
        self.method.coordinates = None
        self.assertIs(self.method.coordinates, None)

    def test_coordinates_point(self):
        point = np.array([1, 2, 3])
        self.method.coordinates = point
        self.assertTrue(np.allclose(self.method.coordinates[:,0], point))
        
    def test_coordinates_2d(self):
        c = coordinates(self.shape)
        self.method.coordinates = c
        self.assertEqual(self.method.coordinates.shape[0], 3)
        self.assertTrue(np.allclose(self.method.coordinates[0:2,:], c))

    def test_coordinates_3dlist(self):
        c = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
        self.method.coordinates = c
        self.assertTrue(np.allclose(self.method.coordinates, np.array(c)))

    def test_properties(self):
        '''Get properties, change one, and set properties'''
        value = -42
        p = self.method.properties
        p['x_p'] = value
        self.method.properties = p        
        self.assertEqual(self.method.particle.x_p, value)

    def test_serialize(self):
        n_0 = 1.5
        n_1 = 1.4
        self.method.particle.n_p = n_0
        s = self.method.dumps()
        self.method.particle.n_p = n_1
        self.method.loads(s)
        self.assertEqual(self.method.particle.n_p, n_0)

        
if __name__ == '__main__':
    unittest.main()
