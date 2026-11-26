'''Tests of reading an image file as gray-scale intensities.'''
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pylorenzmie.utilities import imread


class TestImread(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def path(self, name):
        return Path(self.directory.name) / name

    def test_an_eight_bit_image_is_read_as_it_is(self):
        data = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
        Image.fromarray(data).save(self.path('a.png'))
        image = imread(self.path('a.png'))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image, data)

    def test_a_color_image_is_converted_to_gray(self):
        data = np.full((3, 4, 3), 100, dtype=np.uint8)
        Image.fromarray(data).save(self.path('c.png'))
        image = imread(self.path('c.png'))
        self.assertEqual(image.shape, (3, 4))
        np.testing.assert_array_equal(image, 100)

    def test_a_sixteen_bit_image_keeps_all_its_bits(self):
        data = np.array([[0, 255, 4095, 65535]], dtype=np.uint16)
        for name in ('d.png', 'd.tif'):
            with self.subTest(name=name):
                Image.fromarray(data).save(self.path(name))
                image = imread(self.path(name))
                self.assertEqual(image.dtype, np.uint16)
                np.testing.assert_array_equal(image, data)

    def test_a_big_endian_image_is_read_in_native_order(self):
        data = np.array([[1, 4095]], dtype='>u2')
        Image.fromarray(data).save(self.path('e.tif'))
        image = imread(self.path('e.tif'))
        self.assertEqual(image.dtype, np.uint16)
        np.testing.assert_array_equal(image, data)

    def test_integers_beyond_sixteen_bits_are_kept(self):
        data = np.array([[-5, 70000]], dtype=np.int32)
        Image.fromarray(data).save(self.path('f.tif'))
        image = imread(self.path('f.tif'))
        self.assertEqual(image.dtype, np.int32)
        np.testing.assert_array_equal(image, data)

    def test_a_floating_point_image_is_kept(self):
        data = np.array([[0.25, 1.5, 1000.]], dtype=np.float32)
        Image.fromarray(data).save(self.path('g.tif'))
        image = imread(self.path('g.tif'))
        self.assertEqual(image.dtype, np.float32)
        np.testing.assert_array_equal(image, data)

    def test_a_missing_or_unreadable_file_is_none(self):
        self.assertIsNone(imread(self.path('missing.png')))
        self.path('text.png').write_text('not an image')
        self.assertIsNone(imread(self.path('text.png')))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
