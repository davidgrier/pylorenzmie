import unittest
import numpy as np
from pylorenzmie.lib import (aziavg, azimedian, azistd, coordinates)


class TestAzi(unittest.TestCase):

    def setUp(self):
        self.data = np.ones((90, 100), dtype=float)

    def test_aziavg(self):
        a = aziavg(self.data, (50, 45))
        rad = int(np.sqrt(50**2 + 45**2)) + 1
        self.assertEqual(len(a), rad)

    def test_azimedian(self):
        a = azimedian(self.data, (50, 45))
        rad = int(np.sqrt(50**2 + 45**2))
        self.assertEqual(len(a), rad)

    def test_azistd(self):
        a, s = azistd(self.data, (50, 45))
        rad = int(np.sqrt(50**2 + 45**2)) + 1
        self.assertEqual(len(a), rad)


class TestCoordinates(unittest.TestCase):

    def setUp(self):
        self.shape = [100, 90]

    def test_flatten(self):
        c = coordinates(self.shape)
        self.assertEqual(c.ndim, 2)

    def test_normal(self):
        c = coordinates(self.shape, flatten=False)
        self.assertEqual(c.ndim, 3)


if __name__ == '__main__':
    unittest.main()
