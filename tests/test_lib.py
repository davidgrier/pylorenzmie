import unittest
import numpy as np
from pylorenzmie.lib import (LMObject, Azimuthal)


class TestAzi(unittest.TestCase):

    def setUp(self):
        self.data = np.ones((90, 100), dtype=float)

    def test_aziavg(self) -> None:
        a = Azimuthal.avg(self.data, (50, 45))
        rad = int(np.hypot(50, 45)) + 1
        self.assertEqual(len(a), rad)

    def test_azimedian(self) -> None:
        a = Azimuthal.med(self.data, (50, 45))
        rad = int(np.hypot(50, 45)) + 1
        self.assertEqual(len(a), rad)

    def test_azistd(self) -> None:
        a, s = Azimuthal.std(self.data, (50, 45))
        rad = int(np.hypot(50, + 45)) + 1
        self.assertEqual(len(a), rad)

    def test_azimad(self) -> None:
        a, m = Azimuthal.mad(self.data, (50, 45))
        rad = int(np.hypot(50, 45)) + 1
        self.assertEqual(len(a), rad)


if __name__ == '__main__':
    unittest.main()
