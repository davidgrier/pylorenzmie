import unittest

import numpy as np
from pylorenzmie.theory import Instrument
from pylorenzmie.lib import coordinates


class TestInstrument(unittest.TestCase):

    def setUp(self):
        self.instrument = Instrument()

    def test_repr(self):
        r = repr(self.instrument)
        self.assertIsInstance(r, str)

    def test_coordinates(self, corner=None):
        shape = [128, 128]
        c = coordinates(shape, corner=corner)
        self.assertEqual(c.shape[1], np.prod(shape))

    def test_coordinates_corner(self):
        self.test_coordinates(corner=(10, 20))

    def test_wavelength(self):
        value = 0.447
        self.instrument.wavelength = value
        self.assertEqual(self.instrument.wavelength, value)

    def test_magnification(self):
        value = 0.120
        self.instrument.magnification = value
        self.assertEqual(self.instrument.magnification, value)

    def test_nm(self):
        value = 1.34
        self.instrument.n_m = value
        self.assertEqual(self.instrument.n_m, value)

    def test_str(self):
        s = str(self.instrument)
        self.assertIsInstance(s, str)

    def test_properties(self):
        value = 1.339
        p = self.instrument.properties
        p['n_m'] = value
        self.instrument.properties = p
        self.assertEqual(self.instrument.n_m, value)

    def test_serialize(self):
        n_0 = 1.341
        n_1 = 1.339
        self.instrument.n_m = n_0
        s = self.instrument.to_json()
        self.instrument.n_m = n_1
        self.instrument.from_json(s)
        self.assertEqual(self.instrument.n_m, n_0)

    def test_dark_count(self):
        value = 10.
        self.instrument.dark_count = value
        self.assertEqual(self.instrument.dark_count, value)

    def test_background(self):
        value = 10.
        self.instrument.background = value
        self.assertEqual(self.instrument.background, value)


if __name__ == '__main__':
    unittest.main()
