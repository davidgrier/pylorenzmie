import unittest
from pathlib import Path

from PIL import Image
import numpy as np

from pylorenzmie.utilities import Normalizer


def hologram(shape=(64, 64), background=100., seed=0):
    '''A flat background with a bright blob and a little noise.'''
    y, x = np.mgrid[:shape[0], :shape[1]]
    blob = 0.3 * np.exp(-((x - 20.)**2 + (y - 40.)**2) / 50.)
    noise = np.random.default_rng(seed).normal(0., 0.01, shape)
    return background * (1. + blob + noise)


class TestNormalizer(unittest.TestCase):

    def test_methods(self):
        self.assertEqual(set(Normalizer.methods),
                         {'filter', 'median', 'reference'})

    def test_median_divides_by_the_median(self):
        image = hologram()
        out = Normalizer()(image)
        self.assertEqual(out.shape, image.shape)
        self.assertAlmostEqual(float(np.median(out)), 1., places=12)
        np.testing.assert_allclose(out * np.median(image), image)

    def test_scalar_reference(self):
        image = hologram()
        out = Normalizer('reference', reference=80.)(image)
        np.testing.assert_allclose(out, image / 80.)

    def test_image_reference_is_divided_pixel_by_pixel(self):
        image = hologram()
        reference = np.full(image.shape, 100.) * np.linspace(0.9, 1.1, 64)
        out = Normalizer('reference', reference=reference)(image)
        np.testing.assert_allclose(out, image / reference)

    def test_reference_of_the_wrong_shape_is_refused(self):
        normalizer = Normalizer('reference', reference=np.ones((10, 10)))
        with self.assertRaisesRegex(ValueError, 'shape'):
            normalizer(hologram())

    def test_reference_method_without_a_reference_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'needs a reference'):
            Normalizer('reference')(hologram())

    def test_filter_removes_an_illumination_gradient(self):
        gradient = np.linspace(60., 140., 128)[None, :] * np.ones((128, 1))
        out = Normalizer('filter', size=31)(gradient)
        # away from the edges, where the filter is not truncated
        np.testing.assert_allclose(out[20:-20, 20:-20], 1., atol=0.02)

    def test_filter_follows_slow_structure_but_not_a_small_feature(self):
        image = np.full((101, 101), 100.)
        image[48:53, 48:53] = 150.             # smaller than the kernel
        out = Normalizer('filter', size=31)(image)
        self.assertAlmostEqual(float(out[50, 50]), 1.5)
        self.assertAlmostEqual(float(out[10, 10]), 1.)

    def test_pixels_without_a_background_are_nan(self):
        image = np.full((4, 4), 5.)
        reference = np.full((4, 4), 5.)
        reference[1, 2] = 0.
        reference[3, 3] = -1.
        out = Normalizer('reference', reference=reference)(image)
        self.assertTrue(np.isnan(out[1, 2]) and np.isnan(out[3, 3]))
        self.assertEqual(int(np.isnan(out).sum()), 2)
        np.testing.assert_allclose(out[~np.isnan(out)], 1.)

    def test_a_dark_image_has_no_normalization(self):
        self.assertTrue(np.isnan(Normalizer()(np.zeros((4, 4)))).all())

    def test_unknown_method_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'Unknown method'):
            Normalizer('nonsense')
        normalizer = Normalizer()
        with self.assertRaises(ValueError):
            normalizer.method = 'refernce'
        self.assertEqual(normalizer.method, 'median')

    def test_a_scalar_reference_must_be_positive_and_finite(self):
        for value in (0., -3., np.nan, np.inf):
            with self.assertRaises(ValueError, msg=str(value)):
                Normalizer('reference', reference=value)

    def test_the_dark_count_is_subtracted_from_image_and_background(self):
        dark, background = 30., 90.
        light = np.full((8, 8), 60.)            # a pixel of the light alone
        light[3, 4] = 66.                       # a fringe, 10% above
        raw = light + dark
        for normalizer in (Normalizer('reference',
                                      reference=background,
                                      dark=dark),
                           Normalizer('median', dark=dark),
                           Normalizer('filter', size=5, dark=dark)):
            out = normalizer(raw)
            self.assertAlmostEqual(float(out[0, 0]), 1., places=6)
            self.assertAlmostEqual(float(out[3, 4]), 1.1, places=6)
        # dividing by the raw background shows the fringe at 6.7%
        plain = Normalizer('reference', reference=background)(raw)
        self.assertAlmostEqual(float(plain[3, 4]/plain[0, 0]), 96/90)

    def test_the_dark_count_must_be_finite_and_not_negative(self):
        self.assertEqual(Normalizer().dark, 0.)
        for value in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError, msg=str(value)):
                Normalizer(dark=value)
        normalizer = Normalizer(dark=5)
        with self.assertRaises(ValueError):
            normalizer.dark = -5
        self.assertEqual(normalizer.dark, 5.)

    def test_pixels_below_the_dark_count_are_nan(self):
        out = Normalizer('reference',
                         reference=np.full((4, 4), 20.),
                         dark=30.)(np.full((4, 4), 50.))
        self.assertTrue(np.isnan(out).all())

    def test_the_reference_can_be_replaced(self):
        normalizer = Normalizer('reference', reference=80.)
        normalizer.reference = np.full((8, 8), 40.)
        np.testing.assert_allclose(normalizer(np.full((8, 8), 20.)), 0.5)
        normalizer.reference = None
        self.assertIsNone(normalizer.reference)


class TestRead(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / 'background.png'
        self.image = np.full((16, 16), 120, dtype=np.uint8)
        self.image[2, 3] = 60
        Image.fromarray(self.image).save(self.path)

    def test_a_number(self):
        for value in (100, 100., '100', '1e2'):
            normalizer = Normalizer.read(value)
            self.assertEqual(normalizer.method, 'reference')
            self.assertEqual(normalizer.reference, 100.)
            self.assertIsNone(normalizer.source)

    def test_a_file_by_name_or_path(self):
        for background in (str(self.path), self.path):
            normalizer = Normalizer.read(background)
            self.assertEqual(normalizer.method, 'reference')
            np.testing.assert_array_equal(normalizer.reference, self.image)
            self.assertEqual(normalizer.source, str(self.path))

    def test_a_file_normalizes_an_image_of_its_shape(self):
        out = Normalizer.read(self.path)(np.full((16, 16), 120.))
        self.assertAlmostEqual(float(out[0, 0]), 1.)
        self.assertAlmostEqual(float(out[2, 3]), 2.)

    def test_options_reach_the_constructor(self):
        normalizer = Normalizer.read(100, size=11, dark=29)
        self.assertEqual((normalizer.size, normalizer.dark), (11, 29.))

    def test_setting_the_reference_forgets_the_file(self):
        normalizer = Normalizer.read(self.path)
        normalizer.reference = 50.
        self.assertIsNone(normalizer.source)

    def test_a_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            Normalizer.read(self.path.with_name('missing.png'))
        with self.assertRaises(FileNotFoundError):
            Normalizer.read('missing.png')

    def test_a_file_that_is_not_an_image(self):
        text = self.path.with_name('background.txt')
        text.write_text('not an image')
        with self.assertRaisesRegex(ValueError, 'Could not read'):
            Normalizer.read(text)

    def test_a_number_that_is_not_a_background(self):
        for value in ('0', -5, 'nan', float('inf')):
            with self.assertRaises(ValueError, msg=str(value)):
                Normalizer.read(value)

    def test_something_else(self):
        with self.assertRaises(TypeError):
            Normalizer.read(None)



class TestSaturated(unittest.TestCase):

    def image(self, dtype, high):
        image = np.full((4, 4), 100, dtype=dtype)
        image[1, 2] = high
        return image

    def where(self, saturated):
        return np.argwhere(saturated).tolist()

    def test_the_default_is_the_largest_value_of_the_type(self):
        for dtype, high in ((np.uint8, 255), (np.uint16, 65535)):
            with self.subTest(dtype=dtype):
                saturated = Normalizer().saturated(self.image(dtype, high))
                self.assertEqual(self.where(saturated), [[1, 2]])

    def test_a_bright_pixel_below_the_level_is_not_saturated(self):
        saturated = Normalizer().saturated(self.image(np.uint8, 254))
        self.assertFalse(saturated.any())

    def test_a_given_level(self):
        normalizer = Normalizer(saturation=4095)
        for high, expected in ((4094, []), (4095, [[1, 2]])):
            with self.subTest(high=high):
                image = self.image(np.uint16, high)
                self.assertEqual(self.where(normalizer.saturated(image)),
                                 expected)

    def test_a_floating_point_image_needs_a_level(self):
        image = self.image(float, 255)
        self.assertIsNone(Normalizer().saturated(image))
        saturated = Normalizer(saturation=255).saturated(image)
        self.assertEqual(self.where(saturated), [[1, 2]])

    def test_a_saturated_background_counts(self):
        reference = self.image(np.uint8, 255).astype(float)
        normalizer = Normalizer('reference', reference=reference)
        image = np.full((4, 4), 90, dtype=np.uint8)
        image[0, 0] = 255
        self.assertEqual(self.where(normalizer.saturated(image)),
                         [[0, 0], [1, 2]])
        # not when it is not the background that is used
        normalizer.method = 'median'
        self.assertEqual(self.where(normalizer.saturated(image)), [[0, 0]])
        # nor one of another shape
        normalizer = Normalizer('reference', reference=np.full((2, 2), 255.))
        self.assertEqual(self.where(normalizer.saturated(image)), [[0, 0]])

    def test_a_level_must_be_positive_and_finite(self):
        for level in (0., -1., np.nan, np.inf):
            with self.subTest(level=level):
                with self.assertRaises(ValueError):
                    Normalizer(saturation=level)

    def test_read_takes_the_level(self):
        normalizer = Normalizer.read(100, saturation=4095)
        self.assertEqual(normalizer.saturation, 4095.)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
