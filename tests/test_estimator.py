import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from pylorenzmie.analysis.Estimator import Estimator
from pylorenzmie.theory import Instrument


THIS_DIR = Path(__file__).parent
TEST_IMAGE = THIS_DIR / 'data' / 'crop.png'


class TestEstimator(unittest.TestCase):

    def setUp(self):
        import cv2
        self.instrument = Instrument()
        self.estimator = Estimator(instrument=self.instrument)
        data = cv2.imread(str(TEST_IMAGE), cv2.IMREAD_GRAYSCALE)
        self.data = data.astype(float) / 100.

    def test_properties_keys(self):
        '''properties contains all particle parameter keys'''
        self.estimator.estimate(self.data)
        props = self.estimator.properties
        for key in ('x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'k_p'):
            self.assertIn(key, props)

    def test_estimate_returns_series(self):
        result = self.estimator.estimate(self.data)
        self.assertIsInstance(result, pd.Series)

    def test_estimate_sets_center(self):
        h, w = self.data.shape
        result = self.estimator.estimate(self.data)
        self.assertAlmostEqual(result['x_p'], w / 2.)
        self.assertAlmostEqual(result['y_p'], h / 2.)

    def test_estimate_z_positive(self):
        result = self.estimator.estimate(self.data)
        self.assertGreater(result['z_p'], 0)

    def test_estimate_a_positive(self):
        result = self.estimator.estimate(self.data)
        self.assertGreater(result['a_p'], 0)

    def test_estimate_list(self):
        result = self.estimator.estimate([self.data, self.data])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_predict_alias(self):
        '''predict() is a backward-compatibility alias for estimate()'''
        r1 = self.estimator.estimate(self.data)
        r2 = self.estimator.predict(self.data)
        self.assertAlmostEqual(r1['z_p'], r2['z_p'], places=3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
