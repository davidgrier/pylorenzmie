import unittest
import numpy as np
import pandas as pd

from pylorenzmie.analysis import DEEstimator, Hologram, PairEstimator
from pylorenzmie.theory import LorenzMie, Pair
from pylorenzmie.lib import LMObject


class TestPairEstimator(unittest.TestCase):

    def setUp(self) -> None:
        shape = (101, 101)
        self.model = LorenzMie(particle=Pair())
        self.model.coordinates = LMObject.meshgrid(shape)
        self.model.particle.properties = dict(
            x_p1=35., y_p1=50., z_p1=180., a_p1=0.6, n_p1=1.45,
            x_p2=65., y_p2=50., z_p2=220., a_p2=0.7, n_p2=1.5)
        self.hologram = Hologram(self.model.hologram().reshape(shape))
        self.centers = ((35., 50.), (65., 50.))
        self.estimator = PairEstimator(model=self.model, seed=0)
        self.estimator.settings['maxiter'] = 1
        self.estimator.settings['workers'] = 1

    def test_is_de_estimator(self) -> None:
        '''PairEstimator is a subclass of DEEstimator.'''
        self.assertIsInstance(self.estimator, DEEstimator)

    def test_returns_series(self) -> None:
        '''estimate() returns a pandas.Series.'''
        result = self.estimator.estimate(self.hologram, self.centers)
        self.assertIsInstance(result, pd.Series)

    def test_result_has_pair_keys(self) -> None:
        '''Result contains all twelve Pair properties.'''
        result = self.estimator.estimate(self.hologram, self.centers)
        expected = {'x_p1', 'y_p1', 'z_p1', 'a_p1', 'n_p1', 'k_p1',
                   'x_p2', 'y_p2', 'z_p2', 'a_p2', 'n_p2', 'k_p2'}
        self.assertEqual(set(result.keys()), expected)

    def test_centers_pinned(self) -> None:
        '''x_p1/y_p1 and x_p2/y_p2 are set from the supplied centers.'''
        result = self.estimator.estimate(self.hologram, self.centers)
        self.assertEqual(result['x_p1'], 35.)
        self.assertEqual(result['y_p1'], 50.)
        self.assertEqual(result['x_p2'], 65.)
        self.assertEqual(result['y_p2'], 50.)

    def test_searched_values_within_bounds(self) -> None:
        '''z_p, a_p, n_p for both spheres stay within bounds.'''
        result = self.estimator.estimate(self.hologram, self.centers)
        for key in ('z_p', 'a_p', 'n_p'):
            lo, hi = self.estimator.bounds[key]
            for suffix in ('1', '2'):
                value = result[f'{key}{suffix}']
                self.assertGreaterEqual(value, lo)
                self.assertLessEqual(value, hi)

    def test_model_coordinates_restored(self) -> None:
        '''Model coordinates are restored to their pre-call value.'''
        coords_before = self.model.coordinates
        self.estimator.estimate(self.hologram, self.centers)
        np.testing.assert_array_equal(self.model.coordinates, coords_before)

    def test_custom_bounds_applied_to_both_spheres(self) -> None:
        '''A custom bounds dict constrains both spheres identically.'''
        custom = PairEstimator(
            model=self.model,
            bounds={'z_p': (150., 250.)},
            seed=0)
        custom.settings['maxiter'] = 1
        custom.settings['workers'] = 1
        result = custom.estimate(self.hologram, self.centers)
        self.assertGreaterEqual(result['z_p1'], 150.)
        self.assertLessEqual(result['z_p1'], 250.)
        self.assertGreaterEqual(result['z_p2'], 150.)
        self.assertLessEqual(result['z_p2'], 250.)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
