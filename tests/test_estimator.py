import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from pylorenzmie.analysis.Estimator import Estimator
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.analysis.BaseEstimator import BaseEstimator
from pylorenzmie.theory import Instrument


THIS_DIR = Path(__file__).parent
TEST_IMAGE = THIS_DIR / 'data' / 'crop.png'


class TestEstimator(unittest.TestCase):

    def setUp(self):
        import cv2
        self.instrument = Instrument()
        self.estimator = Estimator(instrument=self.instrument)
        data = cv2.imread(str(TEST_IMAGE), cv2.IMREAD_GRAYSCALE)
        self.hologram = Hologram(data.astype(float) / 100.)

    def test_is_base_estimator(self):
        '''Estimator is a subclass of BaseEstimator.'''
        self.assertIsInstance(self.estimator, BaseEstimator)

    def test_properties_keys(self):
        '''properties contains all particle parameter keys.'''
        self.estimator.estimate(self.hologram)
        for key in ('x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'k_p'):
            self.assertIn(key, self.estimator.properties)

    def test_estimate_returns_series(self):
        '''estimate returns a pandas.Series.'''
        result = self.estimator.estimate(self.hologram)
        self.assertIsInstance(result, pd.Series)

    def test_estimate_sets_center(self):
        '''x_p and y_p are set from hologram coordinate means.'''
        result = self.estimator.estimate(self.hologram)
        self.assertAlmostEqual(result['x_p'],
                               float(self.hologram.coordinates[0].mean()))
        self.assertAlmostEqual(result['y_p'],
                               float(self.hologram.coordinates[1].mean()))

    def test_estimate_center_corner_aware(self):
        '''Center shifts correctly for a hologram with a non-zero corner.'''
        corner = (10., 20.)
        h = Hologram(self.hologram.data, corner=corner)
        result = self.estimator.estimate(h)
        self.assertAlmostEqual(result['x_p'], float(h.coordinates[0].mean()))
        self.assertAlmostEqual(result['y_p'], float(h.coordinates[1].mean()))

    def test_estimate_z_positive(self):
        '''Estimated z_p is positive.'''
        result = self.estimator.estimate(self.hologram)
        self.assertGreater(result['z_p'], 0)

    def test_estimate_a_positive(self):
        '''Estimated a_p is positive.'''
        result = self.estimator.estimate(self.hologram)
        self.assertGreater(result['a_p'], 0)

    def test_predict_alias(self):
        '''predict() is a backward-compatibility alias for estimate().'''
        r1 = self.estimator.estimate(self.hologram)
        r2 = self.estimator.predict(self.hologram)
        self.assertAlmostEqual(r1['z_p'], r2['z_p'], places=3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
