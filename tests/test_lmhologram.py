import unittest

from theory import LMHologram
from utilities import coordinates
import numpy as np


class TestLorenzMie(unittest.TestCase):

    def setUp(self):
        self.method = LMHologram()
        self.shape = [256, 256]

    def test_alpha(self):
        value = 2.
        alpha = self.method.alpha
        self.method.alpha = value
        self.assertEqual(self.method.alpha, value)
        self.method.alpha = alpha

    def test_properties(self):
        value = 0.9
        self.method.alpha = value
        p = self.method.properties
        self.assertEqual(p['alpha'], value)

    def test_hologram_nocoordinates(self):
        self.method.coordinates = None
        field = self.method.hologram()
        self.assertEqual(field, None)

    def test_hologram(self):
        p = self.method.particle
        p.a_p = 1.
        p.n_p = 1.4
        p.r_p = [64, 64, 100]
        c = coordinates([128, 128])
        self.method.coordinates = c
        hologram = self.method.hologram()
        self.assertEqual(hologram.shape[0], c.shape[1])

    def test_hologram_singleprecision(self):
        if self.method.method != 'cupy':
            self.skipTest('not using cupy acceleration')
        else:
            self.method.double_precision = False
            self.test_hologram()

        
if __name__ == '__main__':
    unittest.main()
