import unittest

from pylorenzmie.analysis import Mask
import numpy as np


class TestMask(unittest.TestCase):

    def setUp(self):
        self.shape = [128, 128]
        self.mask = Mask(self.shape)

    def test_fraction(self, value=0.2):
        orig = self.mask.fraction
        self.mask.fraction = value
        self.assertEqual(self.mask.fraction, value)
        fraction = np.sum(self.mask())/self.mask().size
        self.assertAlmostEqual(value, fraction, 2, 'within tolerances')
        self.mask.percentpix = orig

    def test_fraction_unity(self):
        self.test_fraction(1.)

    def test_exclude(self):
        exclude = np.empty(self.shape, dtype=bool)
        self.mask.exclude = exclude
        self.assertEqual(exclude.size, self.mask().size)


if __name__ == '__main__':
    unittest.main()
