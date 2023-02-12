import unittest

from pylorenzmie.analysis import Feature
from pylorenzmie.theory import LorenzMie
from pylorenzmie.lib import coordinates
from pathlib import Path
import cv2


THIS_DIR = Path(__file__).parent
TEST_IMAGE = str(THIS_DIR / 'data' / 'crop.png')


class TestFeature(unittest.TestCase):

    def setUp(self):
        data = cv2.imread(TEST_IMAGE, 0).astype(float)
        data /= 100.
        self.data = data
        coords = coordinates(data.shape)
        self.coords = coords
        model = LorenzMie()
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34
        model.particle.r_p = [data.shape[0]//2, data.shape[1]//2, 330]
        model.particle.a_p = 1.1
        model.particle.n_p = 1.4
        self.feature = Feature(data=data,
                               coordinates=coords,
                               model=model)
        self.feature.mask.fraction = 0.1

    def test_data(self):
        self.feature.data = self.data
        self.assertEqual(self.data.size, self.feature.data.size)

    def test_residuals(self):
        self.feature.data = self.data
        self.feature.optimize()
        res = self.feature.residuals()
        self.assertEqual(self.data.size, res.size)

    def test_model(self):
        model = LorenzMie()
        self.feature.model = model
        self.assertIs(self.feature.model, model)

    def test_getsetparticle(self):
        z_p = 200.
        p = self.feature.particle
        p.z_p = z_p
        self.feature.particle = p
        self.assertEqual(self.feature.particle.z_p, z_p)


if __name__ == '__main__':
    unittest.main()
