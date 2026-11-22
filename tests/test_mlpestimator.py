import importlib.util
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from pylorenzmie.analysis import BaseEstimator, Hologram
from pylorenzmie.theory import LorenzMie
from pylorenzmie.lib import LMObject

WEIGHTS = Path(__file__).parent.parent / 'analysis' / 'mlp_estimator.npz'


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
    '''Build an MLPEstimator with a mocked network (no real weights needed).'''
    import importlib
    from pylorenzmie.analysis.MLPEstimator import MLPEstimator
    _mlp = importlib.import_module('pylorenzmie.analysis.MLPEstimator')
    mock_net = MagicMock()
    mock_net.n_features = 100
    mock_net.predict.return_value = np.array([[200., 0.75, 1.45]])
    with patch.object(_mlp.MLPNetwork, 'load', return_value=mock_net):
        est = MLPEstimator(model=model)
    return est, mock_net


class TestMLPEstimatorMocked(unittest.TestCase):
    '''Tests that run against a mocked pipeline — no weights file required.'''

    def setUp(self):
        self.model, self.hologram = _make_model_and_hologram()
        self.estimator, self.mock_net = _make_estimator(self.model)

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

    def test_network_called_with_n_features(self):
        '''Pipeline receives a (1, n_features) array.'''
        n = self.estimator.n_features
        self.estimator.estimate(self.hologram)
        args, _ = self.mock_net.predict.call_args
        self.assertEqual(args[0].shape, (1, n))

    def test_short_profile_padded_with_zeros(self):
        '''Profiles shorter than n_features are zero-padded (sentinel for no data).'''
        tiny = Hologram(self.hologram.data[:10, :10])
        self.estimator.estimate(tiny)
        args, _ = self.mock_net.predict.call_args
        features = args[0][0]
        n = self.estimator.n_features
        self.assertEqual(len(features), n)
        # A 10x10 crop gives a very short profile; verify the tail is zero.
        # We check from index 15 to be insensitive to the exact profile length.
        self.assertTrue((features[15:] == 0.0).all())

    def test_z_p_clipped_low(self):
        '''z_p predictions below the clip floor are clipped to 10.'''
        self.mock_net.predict.return_value = np.array([[-50., 0.75, 1.45]])
        result = self.estimator.estimate(self.hologram)
        self.assertGreaterEqual(result['z_p'], 10.)

    def test_z_p_clipped_high(self):
        '''z_p predictions above the clip ceiling are clipped to 1000.'''
        self.mock_net.predict.return_value = np.array([[9999., 0.75, 1.45]])
        result = self.estimator.estimate(self.hologram)
        self.assertLessEqual(result['z_p'], 1000.)

    def test_a_p_clipped(self):
        '''a_p is clipped to [0.1, 20].'''
        self.mock_net.predict.return_value = np.array([[200., -1., 1.45]])
        result = self.estimator.estimate(self.hologram)
        self.assertGreaterEqual(result['a_p'], 0.1)

    def test_n_p_clipped(self):
        '''n_p is clipped to [1.0, 3.0].'''
        self.mock_net.predict.return_value = np.array([[200., 0.75, 5.0]])
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


def _network(n_features=6, seed=0):
    from pylorenzmie.analysis.MLPEstimator import MLPNetwork
    rng = np.random.default_rng(seed)
    sizes = (n_features, 8, 5, 3)
    return MLPNetwork(rng.normal(size=n_features),
                      rng.uniform(0.5, 2., n_features),
                      [rng.normal(size=(i, o))
                       for i, o in zip(sizes[:-1], sizes[1:])],
                      [rng.normal(size=o) for o in sizes[1:]],
                      rng.normal(size=3),
                      rng.uniform(0.5, 2., 3))


class TestMLPNetwork(unittest.TestCase):

    def test_predict_has_a_row_of_three_parameters_per_profile(self):
        y = _network().predict(np.ones((4, 6)))
        self.assertEqual(y.shape, (4, 3))

    def test_z_p_and_a_p_are_positive(self):
        y = _network().predict(np.random.default_rng(1).normal(size=(50, 6)))
        self.assertTrue(np.all(y[:, :2] > 0.))

    def test_a_saved_network_is_read_back_unchanged(self):
        import tempfile
        from pylorenzmie.analysis.MLPEstimator import MLPNetwork
        network = _network()
        x = np.random.default_rng(2).normal(size=(5, 6))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'network.npz'
            network.save(path)
            loaded = MLPNetwork.load(path)
        np.testing.assert_array_equal(loaded.predict(x), network.predict(x))
        self.assertEqual(loaded.n_features, 6)

    def test_a_joblib_file_is_refused(self):
        from pylorenzmie.analysis.MLPEstimator import MLPNetwork
        with self.assertRaises(ValueError):
            MLPNetwork.load('mlp_estimator.joblib')

    def test_the_estimator_refuses_a_network_of_another_length(self):
        import tempfile
        from pylorenzmie.analysis.MLPEstimator import MLPEstimator
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'network.npz'
            _network().save(path)
            with self.assertRaises(ValueError):
                MLPEstimator(model=LorenzMie(), weights=path)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),
                         'scikit-learn is not installed')
    def test_it_agrees_with_the_scikit_learn_pipeline_that_trained_it(self):
        import importlib.util
        import sys
        path = Path(__file__).parent.parent / 'tools' / 'train_mlp_estimator.py'
        spec = importlib.util.spec_from_file_location('train_mlp', path)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        rng = np.random.default_rng(3)
        X = rng.uniform(0.5, 1.5, size=(200, 100))
        y = np.column_stack([rng.uniform(20., 500., 200),
                             rng.uniform(0.2, 4., 200),
                             rng.uniform(1.3, 2.5, 200)])
        pipe = tool.build_pipeline()
        pipe.named_steps['regressor'].regressor.set_params(
            hidden_layer_sizes=(8, 4), max_iter=20)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')     # it does not converge
            pipe.fit(X, y)
        network = tool.to_network(pipe)
        np.testing.assert_allclose(network.predict(X),
                                   pipe.predict(X),
                                   rtol=1e-10)


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
