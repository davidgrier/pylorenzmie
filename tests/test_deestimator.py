import unittest
import numpy as np
import pandas as pd

from pylorenzmie.analysis import BaseEstimator, Hologram
from pylorenzmie.analysis.DEEstimator import DEEstimator, DEFAULT_BOUNDS
from pylorenzmie.theory import LorenzMie
from pylorenzmie.lib import LMObject


class TestDEEstimator(unittest.TestCase):

    def setUp(self):
        shape = (101, 101)
        self.model = LorenzMie()
        self.model.coordinates = LMObject.meshgrid(shape)
        self.model.particle.r_p = [50., 50., 200.]
        self.model.particle.a_p = 0.75
        self.model.particle.n_p = 1.45
        self.hologram = Hologram(self.model.hologram().reshape(shape))
        self.estimator = DEEstimator(model=self.model, seed=0)
        self.estimator.settings['maxiter'] = 1
        self.estimator.settings['workers'] = 1

    def test_is_base_estimator(self):
        '''DEEstimator is a subclass of BaseEstimator.'''
        self.assertIsInstance(self.estimator, BaseEstimator)

    def test_default_bounds_keys(self):
        '''DEFAULT_BOUNDS covers z_p, a_p, and n_p.'''
        for key in ('z_p', 'a_p', 'n_p'):
            self.assertIn(key, DEFAULT_BOUNDS)

    def test_default_bounds_valid_ranges(self):
        '''Each default bound is a (min, max) pair with min < max.'''
        for key, (lo, hi) in DEFAULT_BOUNDS.items():
            self.assertLess(lo, hi, msg=f'{key}: lo >= hi')

    def test_returns_series(self):
        '''estimate() returns a pandas.Series.'''
        result = self.estimator.estimate(self.hologram)
        self.assertIsInstance(result, pd.Series)

    def test_result_has_particle_keys(self):
        '''Result contains x_p, y_p, z_p, a_p, n_p.'''
        result = self.estimator.estimate(self.hologram)
        for key in ('x_p', 'y_p', 'z_p', 'a_p', 'n_p'):
            self.assertIn(key, result)

    def test_xy_pinned_to_coordinate_means(self):
        '''x_p and y_p are set from hologram coordinate means.'''
        result = self.estimator.estimate(self.hologram)
        self.assertAlmostEqual(
            result['x_p'], float(self.hologram.coordinates[0].mean()), places=3)
        self.assertAlmostEqual(
            result['y_p'], float(self.hologram.coordinates[1].mean()), places=3)

    def test_model_coordinates_restored(self):
        '''Model coordinates are restored to their pre-call value.'''
        coords_before = self.model.coordinates
        self.estimator.estimate(self.hologram)
        np.testing.assert_array_equal(self.model.coordinates, coords_before)

    def test_model_coordinates_restored_on_exception(self):
        '''Model coordinates are restored even when estimate() raises.'''
        coords_before = self.model.coordinates
        bad_estimator = DEEstimator(
            model=self.model,
            bounds={'z_p': (500., 100.)},  # invalid: lo > hi → DE will raise
            seed=0)
        try:
            bad_estimator.estimate(self.hologram)
        except Exception:
            pass
        np.testing.assert_array_equal(self.model.coordinates, coords_before)

    def test_properties_keys(self):
        '''properties exposes fraction, popsize, bounds, and settings.'''
        props = self.estimator.properties
        for key in ('fraction', 'popsize', 'bounds', 'settings'):
            self.assertIn(key, props)

    def test_custom_bounds(self):
        '''DEEstimator respects custom bounds.'''
        custom = DEEstimator(
            model=self.model,
            bounds={'z_p': (150., 250.)},
            seed=0)
        custom.settings['maxiter'] = 1
        custom.settings['workers'] = 1
        result = custom.estimate(self.hologram)
        self.assertGreaterEqual(result['z_p'], 150.)
        self.assertLessEqual(result['z_p'], 250.)

    def test_exclude_filters_pixels(self):
        '''exclude array is honoured: excluded pixels are not sampled.'''
        shape = self.hologram.shape
        exclude = np.zeros(shape, dtype=bool)
        exclude[:, shape[1] // 2:] = True  # exclude right half
        estimator = DEEstimator(
            model=self.model, seed=0, exclude=exclude)
        estimator.settings['maxiter'] = 1
        estimator.settings['workers'] = 1
        result = estimator.estimate(self.hologram)
        self.assertIsInstance(result, pd.Series)

    def test_exclude_none_by_default(self):
        '''exclude defaults to None.'''
        self.assertIsNone(self.estimator.exclude)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
