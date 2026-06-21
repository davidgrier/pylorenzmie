import unittest
import numpy as np

from pylorenzmie.analysis.RadialMask import RadialMask


class TestRadialMask(unittest.TestCase):

    def setUp(self):
        self.shape = (101, 121)

    def test_output_shape(self):
        mask = RadialMask(shape=self.shape, fraction=0.5)
        self.assertEqual(mask().shape, self.shape)

    def test_output_dtype(self):
        mask = RadialMask(shape=self.shape, fraction=0.5)
        self.assertEqual(mask().dtype, bool)

    def test_non_square_shape(self):
        mask = RadialMask(shape=(61, 101), fraction=0.3)
        self.assertEqual(mask().shape, (61, 101))

    def test_center_excluded_at_low_fraction(self):
        '''For fraction < 0.5, the center pixel has p=0 and is never selected.'''
        h, w = self.shape
        mask = RadialMask(shape=self.shape, fraction=0.2)
        selected = 0
        for _ in range(20):
            mask.update()
            selected += int(mask()[h // 2, w // 2])
        self.assertEqual(selected, 0)

    def test_center_selected_at_high_fraction(self):
        '''For fraction >= 0.5, the center pixel has positive probability.'''
        h, w = self.shape
        mask = RadialMask(shape=self.shape, fraction=0.8)
        selected = 0
        for _ in range(20):
            mask.update()
            selected += int(mask()[h // 2, w // 2])
        self.assertGreater(selected, 0)

    def test_update_regenerates(self):
        mask = RadialMask(shape=self.shape, fraction=0.3)
        mask.update()
        m1 = mask().copy()
        mask.update()
        m2 = mask()
        self.assertFalse(np.array_equal(m1, m2))

    def test_inherits_exclude(self):
        h, w = self.shape
        exclude = np.zeros(self.shape, dtype=bool)
        exclude[h // 2, w // 2] = True
        mask = RadialMask(shape=self.shape, fraction=0.9, exclude=exclude)
        self.assertFalse(mask()[h // 2, w // 2])

    def test_any_pixels_selected(self):
        mask = RadialMask(shape=self.shape, fraction=0.3)
        self.assertTrue(mask().any())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
