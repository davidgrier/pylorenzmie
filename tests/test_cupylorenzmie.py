import unittest
import sys

if 'LorenzMie' in sys.modules:
    sys.modules.pop('LorenzMie')

from theory.LorenzMie import LorenzMie as numpyLorenzMie
from theory.cupyLorenzMie import cupyLorenzMie

from utilities import coordinates
import numpy as np


class TestCupyLorenzMie(unittest.TestCase):

    def setUp(self):
        self.method = cupyLorenzMie()
        self.nmethod = numpyLorenzMie()
        if self.method.method != 'cupy':
            self.skipTest('Not using cupy acceleration')

    def test_method(self):
        self.assertEqual(self.method.method, 'cupy')
        self.assertEqual(self.nmethod.method, 'numpy')
            
    def test_doubleprecision(self):
        self.method.double_precision = False
        self.assertFalse(self.method.double_precision)
        self.method.double_precision = True
        self.assertTrue(self.method.double_precision)

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

    def test_compare_methods(self):
        '''check that numpy and cupy pipelines yield consistent results'''
        shape = [128, 128]
        c = coordinates(shape)
        self.method.coordinates = c
        self.nmethod.coordinates = c
        
        p = self.method.particle
        p.a_p = 1.
        p.n_p = 1.4
        p.r_p = [64, 64, 100]
        self.nmethod.particle = p
       
        field = self.method.field()
        nfield = self.nmethod.field()
        inten = np.sum(np.abs(field), axis=0).reshape(shape)
        ninten = np.sum(np.abs(nfield), axis=0).reshape(shape)

        self.assertTrue(np.allclose(inten, ninten))

        
if __name__ == '__main__':
    unittest.main()
