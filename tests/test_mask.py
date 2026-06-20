import unittest
import numpy as np

from pylorenzmie.analysis import Mask


class TestMask(unittest.TestCase):

    def setUp(self):
        self.shape = (128, 128)
        self.mask = Mask(shape=self.shape)

    def test_fraction(self, value=0.2):
        self.mask.fraction = value
        self.assertEqual(self.mask.fraction, value)
        fraction = np.sum(self.mask()) / self.mask().size
        self.assertAlmostEqual(value, fraction, 1, 'within tolerances')

    def test_fraction_unity(self):
        self.test_fraction(1.)

    def test_exclude(self):
        exclude = np.zeros(self.shape, dtype=bool)
        self.mask.exclude = exclude
        self.assertEqual(exclude.size, self.mask().size)

    def test_shape_sets_mask(self):
        self.assertEqual(self.mask().shape, self.shape)

    def test_update_regenerates_mask(self):
        m1 = self.mask().copy()
        self.mask.update()
        m2 = self.mask()
        # Two independent draws are unlikely to be identical
        self.assertFalse(np.array_equal(m1, m2))

    def test_exclude_suppresses_pixels(self):
        exclude = np.ones(self.shape, dtype=bool)
        self.mask.fraction = 1.
        self.mask.exclude = exclude
        self.assertEqual(np.sum(self.mask()), 0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
