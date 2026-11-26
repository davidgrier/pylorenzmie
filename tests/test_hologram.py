import unittest
import numpy as np
from pylorenzmie.analysis import Hologram


class TestHologram(unittest.TestCase):

    def setUp(self):
        self.shape = (50, 60)
        self.data = np.random.rand(*self.shape)
        self.hologram = Hologram(self.data)

    # --- construction ---

    def test_shape(self):
        '''shape matches data dimensions.'''
        self.assertEqual(self.hologram.shape, self.shape)

    def test_default_corner(self):
        '''corner defaults to (0., 0.).'''
        self.assertEqual(self.hologram.corner, (0., 0.))

    def test_custom_corner(self):
        '''custom corner shifts coordinate origin.'''
        corner = (10., 20.)
        h = Hologram(self.data, corner=corner)
        self.assertAlmostEqual(h.coordinates[0, 0, 0], 10.)  # x
        self.assertAlmostEqual(h.coordinates[1, 0, 0], 20.)  # y

    # --- coordinates ---

    def test_coordinates_shape(self):
        '''coordinates have shape (2, ny, nx).'''
        ny, nx = self.shape
        self.assertEqual(self.hologram.coordinates.shape, (2, ny, nx))

    def test_coordinates_origin(self):
        '''coordinates start at corner.'''
        self.assertAlmostEqual(self.hologram.coordinates[0, 0, 0], 0.)  # x
        self.assertAlmostEqual(self.hologram.coordinates[1, 0, 0], 0.)  # y

    # --- flat views ---

    def test_flat_data_shape(self):
        '''flat_data has shape (npts,).'''
        ny, nx = self.shape
        self.assertEqual(self.hologram.flat_data.shape, (ny * nx,))

    def test_flat_data_is_view(self):
        '''flat_data shares memory with data (no copy).'''
        self.assertTrue(np.shares_memory(self.hologram.flat_data,
                                         self.hologram.data))

    def test_flat_coordinates_shape(self):
        '''flat_coordinates has shape (2, npts).'''
        ny, nx = self.shape
        self.assertEqual(self.hologram.flat_coordinates.shape, (2, ny * nx))

    def test_flat_coordinates_is_view(self):
        '''flat_coordinates shares memory with coordinates (no copy).'''
        self.assertTrue(np.shares_memory(self.hologram.flat_coordinates,
                                         self.hologram.coordinates))

    # --- __getitem__ ---

    def test_getitem_shape(self):
        '''Crop has the expected shape.'''
        crop = self.hologram[10:30, 15:45]
        self.assertEqual(crop.shape, (20, 30))

    def test_getitem_corner(self):
        '''Crop corner reflects slice origin.'''
        crop = self.hologram[10:30, 15:45]
        self.assertEqual(crop.corner, (15., 10.))

    def test_getitem_coordinates_consistent(self):
        '''Crop coordinates match corresponding parent coordinates.'''
        crop = self.hologram[10:30, 15:45]
        np.testing.assert_array_equal(
            crop.flat_coordinates,
            self.hologram.coordinates[:, 10:30, 15:45].reshape(2, -1))

    def test_getitem_corner_accumulates(self):
        '''Corner accumulates correctly through nested crops.'''
        h = Hologram(self.data, corner=(5., 3.))
        crop = h[10:30, 15:45]
        self.assertEqual(crop.corner, (20., 13.))  # (5+15, 3+10)

    def test_getitem_none_start(self):
        '''Slices with None start (e.g. hologram[:30, :45]) are handled.'''
        crop = self.hologram[:30, :45]
        self.assertEqual(crop.corner, (0., 0.))
        self.assertEqual(crop.shape, (30, 45))

    # --- equality ---

    def test_equality(self):
        '''Two Holograms with the same data and corner are equal.'''
        other = Hologram(self.data.copy())
        self.assertEqual(self.hologram, other)

    def test_inequality_data(self):
        '''Different data → not equal.'''
        other = Hologram(self.data + 1.)
        self.assertNotEqual(self.hologram, other)

    def test_inequality_corner(self):
        '''Different corner → not equal.'''
        other = Hologram(self.data, corner=(1., 0.))
        self.assertNotEqual(self.hologram, other)


    def test_from_coordinates_keeps_the_given_coordinates(self):
        crop = self.hologram[2:6, 3:9]
        hologram = Hologram.from_coordinates(crop.data, crop.coordinates)
        self.assertEqual(hologram.corner, crop.corner)
        self.assertIs(hologram.coordinates, crop.coordinates)
        self.assertEqual(hologram, crop)

    def test_from_coordinates_checks_the_shape(self):
        with self.assertRaises(ValueError):
            Hologram.from_coordinates(self.data, self.hologram.coordinates[0])



class TestExcludedPixels(unittest.TestCase):

    def setUp(self):
        self.data = np.ones((6, 8))
        self.saturated = np.zeros((6, 8), dtype=bool)
        self.saturated[2, 3] = True

    def test_the_brightest_pixel_is_not_excluded_by_itself(self):
        self.data[1, 1] = 3.
        hologram = Hologram(self.data)
        self.assertIsNone(hologram.saturated)
        self.assertFalse(hologram.excluded.any())

    def test_saturated_and_nonfinite_pixels_are_excluded(self):
        self.data[0, 0] = np.nan
        self.data[5, 7] = np.inf
        excluded = Hologram(self.data, saturated=self.saturated).excluded
        self.assertEqual(np.argwhere(excluded).tolist(),
                         [[0, 0], [2, 3], [5, 7]])

    def test_excluding_does_not_change_saturated(self):
        self.data[0, 0] = np.nan
        hologram = Hologram(self.data, saturated=self.saturated)
        hologram.excluded
        self.assertEqual(hologram.saturated.sum(), 1)

    def test_a_crop_has_its_part_of_saturated(self):
        hologram = Hologram(self.data, saturated=self.saturated)
        crop = hologram[1:4, 2:6]
        self.assertEqual(np.argwhere(crop.saturated).tolist(), [[1, 1]])
        self.assertIsNone(Hologram(self.data)[1:4, 2:6].saturated)

    def test_saturated_of_another_shape_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'saturated'):
            Hologram(self.data, saturated=np.zeros((3, 3), dtype=bool))
        hologram = Hologram(self.data)
        hologram.saturated = np.zeros((9, 9), dtype=bool)
        with self.assertRaisesRegex(ValueError, 'saturated'):
            hologram.excluded
        with self.assertRaisesRegex(ValueError, 'saturated'):
            hologram[0:2, 0:2]


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
