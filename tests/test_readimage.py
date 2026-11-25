'''Tests of reading an image file as 8-bit intensities.'''
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

    def test_a_sixteen_bit_image_keeps_its_high_byte(self):
        data = np.array([[0, 256, 25600, 65535]], dtype=np.uint16)
        Image.fromarray(data).save(self.path('d.png'))
        image = imread(self.path('d.png'))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image, [[0, 1, 100, 255]])

    def test_a_missing_or_unreadable_file_is_none(self):
        self.assertIsNone(imread(self.path('missing.png')))
        self.path('text.png').write_text('not an image')
        self.assertIsNone(imread(self.path('text.png')))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
