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



if __name__ == '__main__':
    unittest.main()
