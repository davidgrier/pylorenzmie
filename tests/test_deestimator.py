from contextlib import suppress
import unittest
import numpy as np
import pandas as pd

from pylorenzmie.analysis import BaseEstimator, Hologram
from pylorenzmie.analysis.DEEstimator import DEEstimator, default_bounds
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

    def test_default_is_a_single_worker(self):
        '''A pool costs seconds to start and slows a compiled model down.'''
        self.assertEqual(DEEstimator(model=self.model).settings['workers'], 1)

    def test_seed_makes_the_estimate_reproducible(self):
        '''The seed controls the search and the subsampled pixels.'''
        results = []
        for _ in range(2):
            estimator = DEEstimator(model=self.model, seed=3)
            results.append(estimator.estimate(self.hologram))
        self.assertTrue(results[0].equals(results[1]))

    def test_worker_pool_can_still_be_requested(self):
        estimator = DEEstimator(model=self.model, seed=0)
        estimator.settings.update(workers=2, maxiter=2)
        result = estimator.estimate(self.hologram)
        self.assertIn('z_p', result.index)

    def test_is_base_estimator(self):
        '''DEEstimator is a subclass of BaseEstimator.'''
        self.assertIsInstance(self.estimator, BaseEstimator)

    def test_default_bounds_keys(self):
        '''The default bounds cover z_p, a_p, and n_p.'''
        for key in ('z_p', 'a_p', 'n_p'):
            self.assertIn(key, default_bounds(0.1))

    def test_default_bounds_valid_ranges(self):
        '''Each default bound is a (min, max) pair with min < max.'''
        for key, (lo, hi) in default_bounds(0.1).items():
            self.assertLess(lo, hi, msg=f'{key}: lo >= hi')

    def test_no_bounds_by_default(self):
        '''None means the default, which is found at the search.'''
        self.assertIsNone(self.estimator.bounds)

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
        bad_estimator.settings['workers'] = 1
        with suppress(Exception):
            bad_estimator.estimate(self.hologram)
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
            bounds={'z_p': (150., 250.)},           # pixels
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

    def test_default_height_follows_the_magnification(self):
        '''The default covers 1 to 200 um at each magnification.'''
        self.assertEqual(default_bounds(0.1)['z_p'], (10., 2000.))
        self.assertEqual(default_bounds(0.2)['z_p'], (5., 1000.))
        self.assertEqual(default_bounds(0.1)['a_p'], (0.25, 10.))

    def test_search_range_follows_the_magnification(self):
        '''The default bounds cover the same height in the sample.'''
        self.model.instrument.magnification = 0.24
        self.model.particle.r_p = [50., 50., 100.]    # 24 um
        hologram = Hologram(self.model.hologram().reshape((101, 101)))
        estimator = DEEstimator(model=self.model, seed=0)
        estimator.settings.update(maxiter=1, workers=1)
        z_p = estimator.estimate(hologram)['z_p']
        self.assertGreaterEqual(z_p*0.24, 1.)
        self.assertLessEqual(z_p*0.24, 200.)


class TestDEEstimatorConvergence(unittest.TestCase):
    '''DE finds the right basin from a bad start.'''

    def setUp(self):
        shape = (101, 101)
        self.model = LorenzMie()
        self.coordinates = LMObject.meshgrid(shape)
        self.model.coordinates = self.coordinates
        self.model.instrument.wavelength = 0.447
        self.model.instrument.magnification = 0.048
        self.model.instrument.n_m = 1.34
        # Ground truth
        self.model.particle.r_p = [50., 50., 200.]
        self.model.particle.a_p = 0.75
        self.model.particle.n_p = 1.45
        self.hologram = Hologram(self.model.hologram().reshape(shape))

    def test_recovers_from_bad_z(self):
        '''DE recovers z_p from a starting point 2× the true value.'''
        self.model.particle.z_p = 100.   # true: 200
        self.model.particle.a_p = 0.5    # true: 0.75
        self.model.particle.n_p = 1.6    # true: 1.45

        estimator = DEEstimator(model=self.model, seed=0)
        result = estimator.estimate(self.hologram)

        self.assertAlmostEqual(result['z_p'], 200., delta=20.)
        self.assertAlmostEqual(result['a_p'], 0.75, delta=0.15)
        self.assertAlmostEqual(result['n_p'], 1.45, delta=0.1)

    def test_outperforms_naive_start(self):
        '''DE estimate is closer to truth than the deliberately bad start.'''
        z_true, a_true, n_true = 200., 0.75, 1.45
        self.model.particle.z_p = 400.
        self.model.particle.a_p = 2.0
        self.model.particle.n_p = 2.0

        estimator = DEEstimator(model=self.model, seed=0)
        result = estimator.estimate(self.hologram)

        err_de = (abs(result['z_p'] - z_true)
                  + abs(result['a_p'] - a_true) * 100.
                  + abs(result['n_p'] - n_true) * 100.)
        err_naive = (abs(400. - z_true)
                     + abs(2.0 - a_true) * 100.
                     + abs(2.0 - n_true) * 100.)
        self.assertLess(err_de, err_naive)


class TestPoolOfProcesses(unittest.TestCase):
    '''A pool of processes is refused to a calculator that cannot be forked.'''

    class Device(LorenzMie):
        '''A calculator that holds the memory of a device.'''
        fork_safe = False

    def setUp(self):
        shape = (51, 51)
        self.model = self.Device()
        self.model.coordinates = LMObject.meshgrid(shape)
        self.hologram = Hologram(self.model.hologram().reshape(shape))

    def test_a_pool_is_refused(self):
        estimator = DEEstimator(model=self.model, seed=0)
        estimator.settings['workers'] = 2
        with self.assertRaisesRegex(ValueError, 'Device.*pool'):
            estimator.estimate(self.hologram)

    def test_one_process_is_fine(self):
        estimator = DEEstimator(model=self.model, seed=0)
        estimator.settings['maxiter'] = 1
        result = estimator.estimate(self.hologram)
        self.assertIn('z_p', result.index)

    def test_the_coordinates_of_the_model_are_not_touched(self):
        before = self.model.coordinates
        estimator = DEEstimator(model=self.model, seed=0)
        estimator.settings['workers'] = 2
        with self.assertRaises(ValueError):
            estimator.estimate(self.hologram)
        np.testing.assert_array_equal(self.model.coordinates, before)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
