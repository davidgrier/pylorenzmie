import unittest
import numpy as np
import cv2
from pylorenzmie.lib import (Azimuthal, circletransform)


class TestAzimuthal(unittest.TestCase):

    def setUp(self) -> None:
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


class TestCircleTransform(unittest.TestCase):

    def setUp(self) -> None:
        self.data = cv2.imread('data/crop.png', cv2.IMREAD_GRAYSCALE)

    def test_transform(self) -> None:
        b = circletransform(self.data)
        self.assertEqual(self.data.shape, b.shape)


if __name__ == '__main__':
    unittest.main()
