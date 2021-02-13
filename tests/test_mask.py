import unittest

from analysis import Mask
from utilities import coordinates
import numpy as np

class TestMask(unittest.TestCase):

    def setUp(self):
        self.shape = [128, 128]
        self.coordinates = coordinates(self.shape)
        self.mask = Mask(self.coordinates)

    def test_percentpix(self, value=0.2):
        a = self.mask.percentpix
        self.mask.percentpix = value
        self.assertEqual(self.mask.percentpix, value)
        self.mask.percentpix = a

    def test_percentpix_unity(self):
        self.test_percentpix(1.)

    def test_distributions(self):
        for d in ['uniform', 'radial', 'donut']:
            self.mask.distribution = d
            self.assertEqual(self.mask.distribution, d)

    def test_distributions_none(self):
        self.mask.distribution = None
        self.assertEqual(self.mask.distribution, 'fast')

    def test_coordinates(self):
        self.mask.coordinates = None
        self.mask.coordinates = self.coordinates
        self.assertTrue(np.array_equal(self.coordinates.shape,
                                       self.mask.coordinates.shape))

    def test_update_nocoordinates(self):
        self.mask.coordinates = None
        self.assertIs(self.mask.selected, None)


if __name__ == '__main__':
    unittest.main()
