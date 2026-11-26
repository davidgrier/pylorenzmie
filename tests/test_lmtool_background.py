'''Tests of the background that lmtool divides its holograms by.

The class needs no Qt, so these tests run without a binding.
'''
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pylorenzmie.lmtool.background import Background

HOLOGRAM = np.linspace(50., 150., 12 * 12).reshape(12, 12)


class TestBackground(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, name, array):
        path = Path(self.directory.name) / name
        Image.fromarray(array.astype(np.uint8)).save(path)
        return str(path)

    def test_the_median_is_the_default(self):
        background = Background()
        self.assertEqual(background.normalizer.method, 'median')
        self.assertEqual(background.dark, 0.)
        self.assertEqual(background.refused, '')
        self.assertEqual(background.describe(), 'background: median')

    def test_the_saturation_level_goes_to_the_normalizer(self):
        background = Background(saturation=140.)
        saturated = background.saturated(HOLOGRAM)
        np.testing.assert_array_equal(saturated, HOLOGRAM >= 140.)
        self.assertEqual(background.describe(),
                         'background: median, saturation 140')
        background.use(100)                 # it stays with a background
        self.assertEqual(background.normalizer.saturation, 140.)
        small = self.write('small.png', np.full((8, 8), 100.))
        background.use(small)               # and when the median is used
        with self.assertLogs(level='ERROR'):
            _, description = background.normalize(HOLOGRAM)
        self.assertIn('saturation 140', description)
        with self.assertRaises(ValueError):
            Background(saturation=0.)

    def test_a_dark_count_is_validated(self):
        with self.assertRaises(ValueError):
            Background(dark=-1.)
        with self.assertRaises(ValueError):
            Background(dark=float('nan'))

    def test_a_number_is_a_background(self):
        background = Background()
        self.assertIsNone(background.use(100))
        self.assertEqual(background.normalizer.reference, 100.)
        self.assertEqual(background.describe(), 'background: 100')
        data, description = background.normalize(HOLOGRAM)
        np.testing.assert_allclose(data, HOLOGRAM / 100.)
        self.assertEqual(description, 'background: 100')

    def test_a_file_is_a_background_and_is_named(self):
        path = self.write('reference.png', np.full((12, 12), 80.))
        background = Background()
        self.assertIsNone(background.use(path))
        self.assertEqual(background.describe(), 'background: reference.png')
        self.assertEqual(background.start_directory(None), path)

    def test_a_background_that_cannot_be_read_keeps_the_previous_one(self):
        background = Background()
        background.use(100)
        error = background.use('/nonexistent/background.png')
        self.assertTrue(error)
        self.assertEqual(background.refused, error)
        self.assertEqual(background.normalizer.reference, 100.)
        self.assertIn(f'({error})', background.describe())
        self.assertIsNone(background.use(50))      # a good one clears it
        self.assertEqual(background.refused, '')
        self.assertEqual(background.describe(), 'background: 50')

    def test_the_dark_count_is_subtracted(self):
        background = Background(dark=20.)
        background.use(100)
        data, description = background.normalize(HOLOGRAM)
        np.testing.assert_allclose(data, (HOLOGRAM - 20.) / 80.)
        self.assertEqual(description, 'background: 100, dark 20')
        background.set_dark(30.)
        self.assertEqual(background.dark, 30.)
        data, _ = background.normalize(HOLOGRAM)
        np.testing.assert_allclose(data, (HOLOGRAM - 30.) / 70.)

    def test_an_invalid_dark_count_leaves_the_old_one(self):
        background = Background(dark=20.)
        with self.assertRaises(ValueError):
            background.set_dark(-5.)
        self.assertEqual(background.dark, 20.)
        self.assertEqual(background.normalizer.dark, 20.)

    def test_a_dark_count_survives_a_new_background(self):
        background = Background(dark=20.)
        background.use(100)
        self.assertEqual(background.normalizer.dark, 20.)

    def test_a_background_of_another_shape_falls_back_to_the_median(self):
        path = self.write('small.png', np.full((5, 5), 80.))
        background = Background()
        background.use(path)
        with self.assertLogs(level='ERROR'):
            data, description = background.normalize(HOLOGRAM)
        np.testing.assert_allclose(data, HOLOGRAM / np.median(HOLOGRAM))
        self.assertIn('using the median', description)
        self.assertTrue(description.startswith('background: median'))
        # the background that was asked for is still the one in use
        self.assertEqual(background.normalizer.source, path)

    def test_a_background_starts_beside_the_hologram(self):
        background = Background()
        self.assertEqual(background.start_directory('/data/run/image.png'),
                         '/data/run')
        self.assertEqual(background.start_directory(None), '')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
