import unittest
import numpy as np
import pandas as pd

from pathlib import Path
import cv2

from pylorenzmie.analysis import Feature
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import LorenzMie


THIS_DIR = Path(__file__).parent
TEST_IMAGE = THIS_DIR / 'data' / 'crop.png'


class TestFeature(unittest.TestCase):

    def setUp(self):
        data = cv2.imread(str(TEST_IMAGE), cv2.IMREAD_GRAYSCALE).astype(float)
        data /= 100.
        model = LorenzMie()
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34
        h, w = data.shape
        model.particle.r_p = [w / 2., h / 2., 330.]
        model.particle.a_p = 1.1
        model.particle.n_p = 1.4
        self.data = data
        self.feature = Feature(Hologram(data), model=model)
        self.feature.mask.fraction = 0.1

    def test_exclude_shared(self):
        '''Estimator and optimizer share the same exclusion array.'''
        self.assertIs(self.feature.estimator.exclude,
                      self.feature.optimizer.mask.exclude)

    def test_is_hologram(self):
        '''Feature is-a Hologram.'''
        self.assertIsInstance(self.feature, Hologram)

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
        '''estimate() returns a pandas.Series.'''
        result = self.feature.estimate()
        self.assertIsInstance(result, pd.Series)

    def test_estimate_sets_z_p(self):
        '''estimate() produces a positive z_p.'''
        self.feature.estimate()
        self.assertGreater(self.feature.particle.z_p, 0)

    def test_estimate_sets_a_p(self):
        '''estimate() produces a positive a_p.'''
        self.feature.estimate()
        self.assertGreater(self.feature.particle.a_p, 0)

    def test_estimate_uses_coordinates_for_position(self):
        '''estimate() places x_p/y_p at center of hologram coordinates.'''
        self.feature.estimate()
        self.assertAlmostEqual(
            self.feature.particle.x_p,
            float(self.feature.coordinates[0].mean()), places=5)
        self.assertAlmostEqual(
            self.feature.particle.y_p,
            float(self.feature.coordinates[1].mean()), places=5)

    def test_predicted_shape(self):
        '''predicted() returns an array with the same shape as the feature.'''
        predicted = self.feature.predicted()
        self.assertEqual(predicted.shape, self.feature.shape)

    def test_residuals_shape(self):
        '''residuals() has the same shape as the feature.'''
        residuals = self.feature.residuals()
        self.assertEqual(residuals.shape, self.feature.shape)

    def test_single_feature_defaults(self):
        '''Without centers, Feature uses a Sphere and a DEEstimator.'''
        from pylorenzmie.analysis import DEEstimator
        from pylorenzmie.theory import Sphere
        feature = Feature(Hologram(self.data))
        self.assertIsInstance(feature.particle, Sphere)
        self.assertIsInstance(feature.estimator, DEEstimator)
        self.assertIsNone(feature.centers)

    def test_pair_centers_configure_pair(self):
        '''Two centers configure a Pair particle and a PairEstimator.'''
        from pylorenzmie.analysis import PairEstimator
        from pylorenzmie.theory import Pair
        centers = [(130., 180.), (260., 210.)]
        feature = Feature(Hologram(self.data), centers=centers)
        self.assertIsInstance(feature.particle, Pair)
        self.assertIsInstance(feature.estimator, PairEstimator)
        self.assertEqual(feature.centers, centers)
        self.assertEqual(
            (feature.particle.x_p1, feature.particle.y_p1), (130., 180.))
        self.assertEqual(
            (feature.particle.x_p2, feature.particle.y_p2), (260., 210.))

    def test_pair_optimizer_frees_geometry_fixes_absorption(self):
        '''Pair optimizer holds k_p1/k_p2 fixed and frees z/a/n for both.'''
        feature = Feature(Hologram(self.data),
                          centers=[(130., 180.), (260., 210.)])
        self.assertIn('k_p1', feature.optimizer.fixed)
        self.assertIn('k_p2', feature.optimizer.fixed)
        for p in ('z_p1', 'a_p1', 'n_p1', 'z_p2', 'a_p2', 'n_p2'):
            self.assertIn(p, feature.optimizer.variables)

    def test_pair_estimate_returns_pair_series(self):
        '''estimate() on a pair feature returns Pair keys, centers pinned.'''
        feature = Feature(Hologram(self.data),
                          centers=[(130., 180.), (260., 210.)])
        feature.estimator.settings['workers'] = 1
        feature.estimator.settings['maxiter'] = 1
        result = feature.estimate()
        self.assertIsInstance(result, pd.Series)
        for key in ('z_p1', 'a_p1', 'n_p1', 'z_p2', 'a_p2', 'n_p2'):
            self.assertIn(key, result)
        self.assertEqual(feature.particle.x_p1, 130.)
        self.assertEqual(feature.particle.x_p2, 260.)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
