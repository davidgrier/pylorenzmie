import unittest
import numpy as np
import pandas as pd

from pylorenzmie.analysis.DEEstimator import DEEstimator, DEFAULT_BOUNDS
from pylorenzmie.theory import LorenzMie
from pylorenzmie.lib import LMObject


class TestDEEstimator(unittest.TestCase):

    def setUp(self):
        shape = (101, 101)
        self.model = LorenzMie()
        self.coordinates = LMObject.meshgrid(shape)
        self.model.coordinates = self.coordinates
        self.model.particle.r_p = [50., 50., 200.]
        self.model.particle.a_p = 0.75
        self.model.particle.n_p = 1.45
        self.data = self.model.hologram().reshape(shape)
        self.estimator = DEEstimator(model=self.model, seed=0)
        self.estimator.settings['maxiter'] = 1
        self.estimator.settings['workers'] = 1

    def test_default_bounds_keys(self):
        '''DEFAULT_BOUNDS covers z_p, a_p, and n_p.'''
        for key in ('z_p', 'a_p', 'n_p'):
            self.assertIn(key, DEFAULT_BOUNDS)

    def test_default_bounds_valid_ranges(self):
        '''Each default bound is a (min, max) pair with min < max.'''
        for key, (lo, hi) in DEFAULT_BOUNDS.items():
            self.assertLess(lo, hi, msg=f'{key}: lo >= hi')

    def test_requires_coordinates(self):
        '''estimate() raises ValueError when coordinates is None.'''
        with self.assertRaises(ValueError):
            self.estimator.estimate(self.data)

    def test_returns_series(self):
        '''estimate() returns a pandas.Series.'''
        result = self.estimator.estimate(self.data, self.coordinates)
        self.assertIsInstance(result, pd.Series)

    def test_result_has_particle_keys(self):
        '''Result contains x_p, y_p, z_p, a_p, n_p.'''
        result = self.estimator.estimate(self.data, self.coordinates)
        for key in ('x_p', 'y_p', 'z_p', 'a_p', 'n_p'):
            self.assertIn(key, result)

    def test_xy_pinned_to_coordinate_means(self):
        '''x_p and y_p are set from coordinate means, not searched.'''
        result = self.estimator.estimate(self.data, self.coordinates)
        self.assertAlmostEqual(
            result['x_p'], float(self.coordinates[0].mean()), places=3)
        self.assertAlmostEqual(
            result['y_p'], float(self.coordinates[1].mean()), places=3)

    def test_model_coordinates_restored(self):
        '''Model coordinates are restored to their pre-call value.'''
        coords_before = self.model.coordinates
        self.estimator.estimate(self.data, self.coordinates)
        np.testing.assert_array_equal(
            self.model.coordinates, coords_before)

    def test_model_coordinates_restored_on_exception(self):
        '''Model coordinates are restored even when estimate() raises.'''
        coords_before = self.model.coordinates
        bad_estimator = DEEstimator(
            model=self.model,
            bounds={'z_p': (500., 100.)},  # invalid: lo > hi → DE will raise
            seed=0)
        try:
            bad_estimator.estimate(self.data, self.coordinates)
        except Exception:
            pass
        np.testing.assert_array_equal(
            self.model.coordinates, coords_before)

    def test_properties_keys(self):
        '''properties exposes de_fraction, popsize, bounds, and settings.'''
        props = self.estimator.properties
        for key in ('de_fraction', 'popsize', 'bounds', 'settings'):
            self.assertIn(key, props)

    def test_custom_bounds(self):
        '''DEEstimator respects custom bounds.'''
        custom = DEEstimator(
            model=self.model,
            bounds={'z_p': (150., 250.)},
            seed=0)
        custom.settings['maxiter'] = 1
        custom.settings['workers'] = 1
        result = custom.estimate(self.data, self.coordinates)
        self.assertGreaterEqual(result['z_p'], 150.)
        self.assertLessEqual(result['z_p'], 250.)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
