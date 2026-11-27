import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pylorenzmie.utilities import imread

from pylorenzmie.analysis import BaseEstimator, DEEstimator, Feature
from pylorenzmie.analysis.Estimator import Estimator
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import LorenzMie


THIS_DIR = Path(__file__).parent
TEST_IMAGE = THIS_DIR / 'data' / 'crop.png'


class _Fixed(BaseEstimator):
    '''An estimator that is not one of the package.'''

    @BaseEstimator.properties.getter
    def properties(self):
        return {}

    def estimate(self, hologram):
        return pd.Series({'a_p': 3.})


class TestFeature(unittest.TestCase):

    def setUp(self):
        data = imread(TEST_IMAGE).astype(float)
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

    def test_only_saturated_and_nonfinite_pixels_are_excluded(self):
        data = np.ones((20, 20))
        data[3, 4] = 2.                     # the brightest, and a good one
        data[5, 5] = np.nan
        saturated = np.zeros(data.shape, dtype=bool)
        saturated[7, 8:10] = True
        feature = Feature(Hologram(data, saturated=saturated))
        self.assertEqual(np.argwhere(feature.mask.exclude).tolist(),
                         [[5, 5], [7, 8], [7, 9]])
        np.testing.assert_array_equal(feature.saturated, saturated)

    def test_is_hologram(self):
        '''Feature is-a Hologram.'''
        self.assertIsInstance(self.feature, Hologram)

    def test_the_default_estimator_is_global_and_shares_the_model(self):
        self.assertIsInstance(self.feature.estimator, DEEstimator)

    def test_an_estimator_can_be_given(self):
        estimator = _Fixed()
        feature = Feature(Hologram(self.data), estimator=estimator)
        self.assertIs(feature.estimator, estimator)
        self.assertTrue(np.array_equal(estimator.exclude,
                                       feature.mask.exclude))

    def test_an_estimator_that_is_shared_takes_the_last_mask(self):
        estimator = _Fixed()
        first = Feature(Hologram(self.data), estimator=estimator)
        second = Feature(Hologram(self.data), estimator=estimator)
        self.assertIs(first.estimator, second.estimator)
        self.assertTrue(np.array_equal(estimator.exclude,
                                       second.mask.exclude))

    def test_a_given_estimator_is_used_to_estimate(self):
        feature = Feature(Hologram(self.data), estimator=_Fixed())
        self.assertEqual(feature.estimate()['a_p'], 3.)

    def test_the_feature_writes_the_estimate_onto_the_particle(self):
        # _Fixed returns a Series and leaves the particle alone
        feature = Feature(Hologram(self.data), estimator=_Fixed())
        feature.particle.a_p = 1.
        result = feature.estimate()
        self.assertEqual(feature.particle.a_p, 3.)
        self.assertEqual(result['a_p'], 3.)

    def test_properties_that_the_particle_lacks_are_ignored(self):
        class Extra(_Fixed):
            def estimate(self, hologram):
                return pd.Series({'a_p': 2., 'snr': 7., 'x_p1': 1.})

        feature = Feature(Hologram(self.data), estimator=Extra())
        with self.assertLogs('pylorenzmie.analysis.Feature',
                             'DEBUG') as logs:
            feature.estimate()
        self.assertEqual(feature.particle.a_p, 2.)
        self.assertNotIn('snr', feature.particle.properties)
        self.assertIn('snr', logs.output[0])
        self.assertIn('x_p1', logs.output[0])

    def test_the_analytic_estimator_sets_the_particle_through_the_feature(self):
        feature = Feature(Hologram(self.data), model=self.feature.model)
        feature.estimator = Estimator(instrument=feature.model.instrument)
        feature.particle.z_p = 1.
        result = feature.estimate()
        self.assertEqual(feature.particle.z_p, result['z_p'])
        self.assertNotEqual(feature.particle.z_p, 1.)

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

    def test_estimate_without_a_center_uses_the_middle(self):
        '''A feature that was not told keeps the middle of its crop.'''
        self.feature.estimate()
        for value, axis in ((self.feature.particle.x_p, 0),
                            (self.feature.particle.y_p, 1)):
            middle = float(self.feature.coordinates[axis].mean())
            self.assertAlmostEqual(value, middle, places=5)

    def test_a_center_goes_to_the_estimator_and_the_particle(self):
        feature = Feature(self.feature, centers=[(190., 200.)])
        self.assertEqual(feature.estimator.center, (190., 200.))
        self.assertEqual((feature.particle.x_p, feature.particle.y_p),
                         (190., 200.))
        given = Feature(self.feature,
                        centers=[(190., 200.)],
                        estimator=DEEstimator(model=feature.model))
        self.assertEqual(given.estimator.center, (190., 200.))

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


class TestFeatureFit(unittest.TestCase):
    '''A fit of the feature in the image of one sphere.'''

    def setUp(self):
        data = imread(TEST_IMAGE).astype(float)
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
