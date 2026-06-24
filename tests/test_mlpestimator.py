import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from pylorenzmie.analysis import BaseEstimator, Hologram
from pylorenzmie.theory import LorenzMie
from pylorenzmie.lib import LMObject

WEIGHTS = Path(__file__).parent.parent / 'analysis' / 'mlp_estimator.joblib'


def _make_model_and_hologram():
    shape = (101, 101)
    model = LorenzMie()
    model.coordinates = LMObject.meshgrid(shape)
    model.particle.r_p = [50., 50., 200.]
    model.particle.a_p = 0.75
    model.particle.n_p = 1.45
    hologram = Hologram(model.hologram().reshape(shape))
    return model, hologram


def _make_estimator(model):
    '''Build an MLPEstimator with a mocked pipeline (no real weights needed).'''
    from pylorenzmie.analysis.MLPEstimator import MLPEstimator
    mock_pipe = MagicMock()
    mock_pipe.predict.return_value = np.array([[200., 0.75, 1.45]])
    with patch('pylorenzmie.analysis.MLPEstimator.joblib') as mock_joblib:
        mock_joblib.load.return_value = mock_pipe
        est = MLPEstimator(model=model)
    return est, mock_pipe


class TestMLPEstimatorMocked(unittest.TestCase):
    '''Tests that run against a mocked pipeline — no weights file required.'''

    def setUp(self):
        self.model, self.hologram = _make_model_and_hologram()
        self.estimator, self.mock_pipe = _make_estimator(self.model)

    def test_is_base_estimator(self):
        '''MLPEstimator is a subclass of BaseEstimator.'''
        self.assertIsInstance(self.estimator, BaseEstimator)

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
        '''x_p and y_p are set to hologram coordinate means.'''
        result = self.estimator.estimate(self.hologram)
        self.assertAlmostEqual(
            result['x_p'], float(self.hologram.coordinates[0].mean()), places=3)
        self.assertAlmostEqual(
            result['y_p'], float(self.hologram.coordinates[1].mean()), places=3)

    def test_pipeline_called_with_n_features(self):
        '''Pipeline receives a (1, n_features) array.'''
        n = self.estimator.n_features
        self.estimator.estimate(self.hologram)
        args, _ = self.mock_pipe.predict.call_args
        self.assertEqual(args[0].shape, (1, n))

    def test_short_profile_padded_with_zeros(self):
        '''Profiles shorter than n_features are zero-padded (sentinel for no data).'''
        tiny = Hologram(self.hologram.data[:10, :10])
        self.estimator.estimate(tiny)
        args, _ = self.mock_pipe.predict.call_args
        features = args[0][0]
        n = self.estimator.n_features
        self.assertEqual(len(features), n)
        # A 10x10 crop gives a very short profile; verify the tail is zero.
        # We check from index 15 to be insensitive to the exact profile length.
        self.assertTrue((features[15:] == 0.0).all())

    def test_z_p_clipped_low(self):
        '''z_p predictions below the clip floor are clipped to 10.'''
        self.mock_pipe.predict.return_value = np.array([[-50., 0.75, 1.45]])
        result = self.estimator.estimate(self.hologram)
        self.assertGreaterEqual(result['z_p'], 10.)

    def test_z_p_clipped_high(self):
        '''z_p predictions above the clip ceiling are clipped to 1000.'''
        self.mock_pipe.predict.return_value = np.array([[9999., 0.75, 1.45]])
        result = self.estimator.estimate(self.hologram)
        self.assertLessEqual(result['z_p'], 1000.)

    def test_a_p_clipped(self):
        '''a_p is clipped to [0.1, 20].'''
        self.mock_pipe.predict.return_value = np.array([[200., -1., 1.45]])
        result = self.estimator.estimate(self.hologram)
        self.assertGreaterEqual(result['a_p'], 0.1)

    def test_n_p_clipped(self):
        '''n_p is clipped to [1.0, 3.0].'''
        self.mock_pipe.predict.return_value = np.array([[200., 0.75, 5.0]])
        result = self.estimator.estimate(self.hologram)
        self.assertLessEqual(result['n_p'], 3.0)

    def test_properties_keys(self):
        '''properties contains weights and n_features.'''
        props = self.estimator.properties
        self.assertIn('weights', props)
        self.assertIn('n_features', props)

    def test_model_updated_in_place(self):
        '''estimate() writes predictions back to model.particle.'''
        self.estimator.estimate(self.hologram)
        self.assertAlmostEqual(self.model.particle.z_p, 200., places=3)
        self.assertAlmostEqual(self.model.particle.a_p, 0.75, places=3)
        self.assertAlmostEqual(self.model.particle.n_p, 1.45, places=3)


@unittest.skipUnless(WEIGHTS.exists(), 'pre-trained weights not found')
class TestMLPEstimatorReal(unittest.TestCase):
    '''Accuracy tests against pre-trained weights (skipped without weights).'''

    def setUp(self):
        from pylorenzmie.analysis.MLPEstimator import MLPEstimator
        self.model, self.hologram = _make_model_and_hologram()
        self.estimator = MLPEstimator(model=self.model)

    def test_z_p_in_training_domain(self):
        '''MLP predicts z_p within the training bounds.'''
        result = self.estimator.estimate(self.hologram)
        self.assertGreater(result['z_p'], 10.)
        self.assertLess(result['z_p'], 1000.)

    def test_a_p_in_training_domain(self):
        '''MLP predicts a_p within the training bounds.'''
        result = self.estimator.estimate(self.hologram)
        self.assertGreater(result['a_p'], 0.1)
        self.assertLess(result['a_p'], 20.)

    def test_n_p_in_training_domain(self):
        '''MLP predicts n_p within the training bounds.'''
        result = self.estimator.estimate(self.hologram)
        self.assertGreater(result['n_p'], 1.0)
        self.assertLess(result['n_p'], 3.0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
