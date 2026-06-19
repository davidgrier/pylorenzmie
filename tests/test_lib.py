import unittest
import numpy as np
import cv2
from pathlib import Path
from pylorenzmie.lib import (Azimuthal, circletransform)
from pylorenzmie.theory import Sphere, Instrument


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


THIS_DIR = Path(__file__).parent


class TestCircleTransform(unittest.TestCase):

    def setUp(self) -> None:
        self.data = cv2.imread(str(THIS_DIR / 'data' / 'crop.png'),
                               cv2.IMREAD_GRAYSCALE)

    def test_transform(self) -> None:
        b = circletransform(self.data)
        self.assertEqual(self.data.shape, b.shape)


class TestLMObjectEquality(unittest.TestCase):

    def test_equal_same_class(self):
        a = Sphere(a_p=1.0, n_p=1.5)
        b = Sphere(a_p=1.0, n_p=1.5)
        self.assertEqual(a, b)

    def test_not_equal_same_class(self):
        a = Sphere(a_p=1.0, n_p=1.5)
        b = Sphere(a_p=2.0, n_p=1.5)
        self.assertNotEqual(a, b)

    def test_not_equal_different_class(self):
        s = Sphere()
        i = Instrument()
        self.assertNotEqual(s, i)


if __name__ == '__main__':
    unittest.main()
