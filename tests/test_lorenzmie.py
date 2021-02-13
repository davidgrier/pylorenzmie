import unittest

import numpy as np
from theory.LorenzMie import LorenzMie
from utilities import coordinates


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

    def test_field_nocoordinates(self):
        self.method.coordinates = None
        field = self.method.field()
        self.assertEqual(field, None)

    def test_field(self, bohren=False, cartesian=False):
        p = self.method.particle
        p.a_p = 1.
        p.n_p = 1.4
        p.r_p = [64, 64, 100]
        c = coordinates([128, 128])
        self.method.coordinates = c
        field = self.method.field(bohren=bohren, cartesian=cartesian)
        self.assertEqual(field.shape[1], c.shape[1])

    def test_field_bohren(self):
        self.test_field(bohren=True)

    def test_field_cartesian(self):
        self.test_field(cartesian=True)

    def test_field_both(self):
        self.test_field(bohren=True, cartesian=True)
        

        
if __name__ == '__main__':
    unittest.main()
