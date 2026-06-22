import unittest

from pathlib import Path
import cv2

from pylorenzmie.analysis import Feature
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import LorenzMie


THIS_DIR = Path(__file__).parent.parent
TEST_IMAGE = THIS_DIR / 'data' / 'crop.png'


class TestFeature(unittest.TestCase):

    def setUp(self):
        data = cv2.imread(str(TEST_IMAGE), cv2.IMREAD_GRAYSCALE).astype(float)
        data /= 100.
        self.data = data
        model = LorenzMie()
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34
        h, w = data.shape
        model.particle.r_p = [w / 2., h / 2., 330.]
        model.particle.a_p = 1.1
        model.particle.n_p = 1.4
        self.feature = Feature(Hologram(data), model=model)
        self.feature.mask.fraction = 0.1

    def test_residuals(self):
        self.feature.optimize()
        res = self.feature.residuals()
        self.assertEqual(self.data.size, res.size)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
