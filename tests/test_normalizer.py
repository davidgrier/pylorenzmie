import logging
import unittest

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

    def test_reference_of_the_wrong_shape_falls_back_to_the_median(self):
        image = hologram()
        normalizer = Normalizer('reference', reference=np.ones((10, 10)))
        with self.assertLogs('pylorenzmie.utilities.normalizer',
                             level=logging.WARNING):
            out = normalizer(image)
        np.testing.assert_allclose(out, image / np.median(image))

    def test_reference_method_without_a_reference_uses_the_median(self):
        image = hologram()
        out = Normalizer('reference')(image)
        np.testing.assert_allclose(out, image / np.median(image))

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

    def test_nonpositive_background_does_not_divide_by_zero(self):
        out = Normalizer('reference', reference=0.)(np.full((4, 4), 5.))
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out, 5.)

    def test_unknown_method_uses_the_median(self):
        image = hologram()
        np.testing.assert_allclose(Normalizer('nonsense')(image),
                                   Normalizer('median')(image))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
