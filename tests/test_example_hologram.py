import importlib
import unittest
from pathlib import Path

from pylorenzmie.analysis import Hologram
from pylorenzmie.utilities import example_hologram
# the package namespace holds the function of the same name
example_module = importlib.import_module('pylorenzmie.utilities.example_hologram')


class TestExampleHologram(unittest.TestCase):

    def test_default_is_a_normalized_hologram(self):
        hologram = example_hologram()
        self.assertIsInstance(hologram, Hologram)
        self.assertEqual(hologram.shape, (390, 390))
        self.assertAlmostEqual(hologram.data.mean(), 1., delta=0.05)

    def test_the_images_that_the_examples_use_ship_with_the_package(self):
        # an installed package has no docs/ directory to read them from
        data = Path(example_module.__file__).parent / 'data'
        for name in ('crop.png', 'image0010.png', 'image0400.png'):
            with self.subTest(name):
                self.assertTrue((data / name).exists())
                self.assertIsInstance(example_hologram(name), Hologram)

    def test_unknown_image_raises_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            example_hologram('no_such_image.png')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
