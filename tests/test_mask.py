import unittest
import numpy as np

from pylorenzmie.analysis import Mask, Hologram


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

    def test_properties(self):
        self.assertIn('fraction', self.mask.properties)
        self.assertEqual(self.mask.properties['fraction'], self.mask.fraction)

    def test_json_roundtrip(self):
        self.mask.fraction = 0.3
        s = self.mask.to_json()
        self.mask.fraction = 0.1
        self.mask.from_json(s)
        self.assertAlmostEqual(self.mask.fraction, 0.3)

    # --- apply ---

    def test_apply_returns_tuple(self):
        '''apply returns a (data, coordinates) tuple.'''
        hologram = Hologram(np.random.rand(*self.shape))
        result = self.mask.apply(hologram)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_apply_data_shape(self):
        '''apply data has shape (nselected,).'''
        hologram = Hologram(np.random.rand(*self.shape))
        data, coords = self.mask.apply(hologram)
        self.assertEqual(data.ndim, 1)
        self.assertEqual(data.shape[0], coords.shape[1])

    def test_apply_coords_shape(self):
        '''apply coordinates have shape (2, nselected).'''
        hologram = Hologram(np.random.rand(*self.shape))
        data, coords = self.mask.apply(hologram)
        self.assertEqual(coords.shape[0], 2)

    def test_apply_sets_shape(self):
        '''apply updates mask shape to match hologram.'''
        new_shape = (64, 80)
        hologram = Hologram(np.random.rand(*new_shape))
        self.mask.apply(hologram)
        self.assertEqual(self.mask.shape, new_shape)

    def test_apply_selects_correct_pixels(self):
        '''apply data matches manual boolean indexing.'''
        hologram = Hologram(np.random.rand(*self.shape))
        data, coords = self.mask.apply(hologram)
        m = self.mask()
        np.testing.assert_array_equal(data, hologram.data[m])
        np.testing.assert_array_equal(coords, hologram.coordinates[:, m])

    def test_apply_respects_exclude(self):
        '''apply does not return pixels marked in exclude.'''
        hologram = Hologram(np.random.rand(*self.shape))
        exclude = np.zeros(self.shape, dtype=bool)
        exclude[0, :] = True  # exclude entire top row (y == 0)
        self.mask.fraction = 1.
        self.mask.exclude = exclude
        _, coords = self.mask.apply(hologram)
        self.assertFalse(np.any(coords[1] == 0.))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
