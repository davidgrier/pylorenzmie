import unittest
import numpy as np
import pandas as pd

from pathlib import Path
import cv2

from pylorenzmie.analysis import Feature
from pylorenzmie.theory import LorenzMie


THIS_DIR = Path(__file__).parent
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
        coords = model.meshgrid(data.shape)
        self.coords = coords
        self.feature = Feature(data=data, coordinates=coords, model=model)
        self.feature.mask.fraction = 0.1

    def test_data(self):
        self.feature.data = self.data
        self.assertEqual(self.data.size, self.feature.data.size)

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

    def test_estimate_returns_series(self):
        result = self.feature.estimate()
        self.assertIsInstance(result, pd.Series)

    def test_estimate_sets_z_p(self):
        self.feature.estimate()
        self.assertGreater(self.feature.particle.z_p, 0)

    def test_estimate_sets_a_p(self):
        self.feature.estimate()
        self.assertGreater(self.feature.particle.a_p, 0)

    def test_estimate_uses_coordinates_for_position(self):
        '''estimate() places x_p/y_p at center of coordinates, not local crop center'''
        self.feature.estimate()
        self.assertAlmostEqual(
            self.feature.particle.x_p,
            float(self.coords[0].mean()), places=5)
        self.assertAlmostEqual(
            self.feature.particle.y_p,
            float(self.coords[1].mean()), places=5)

    def test_hologram_shape(self):
        hologram = self.feature.hologram()
        self.assertEqual(hologram.shape, self.data.shape)

    def test_residuals_shape(self):
        residuals = self.feature.residuals()
        self.assertEqual(residuals.shape, self.data.shape)

    def test_properties(self):
        self.assertIn('fraction', self.feature.properties)
        self.assertEqual(self.feature.properties['fraction'],
                         self.feature.mask.fraction)

    def test_json_roundtrip(self):
        self.feature.fraction = 0.3
        s = self.feature.to_json()
        self.feature.fraction = 0.1
        self.feature.from_json(s)
        self.assertAlmostEqual(self.feature.fraction, 0.3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
