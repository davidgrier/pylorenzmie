import unittest
import pandas as pd
import numpy as np

from pylorenzmie.analysis import BaseEstimator, Hologram
from pylorenzmie.lib.lmtypes import Result


class _ConcreteEstimator(BaseEstimator):
    '''Minimal concrete subclass for testing.'''

    @BaseEstimator.properties.getter
    def properties(self):
        return dict()

    def estimate(self, hologram: Hologram) -> Result:
        return pd.Series({'x_p': 0., 'y_p': 0.})


class TestBaseEstimator(unittest.TestCase):

    def test_cannot_instantiate_abstract(self):
        '''BaseEstimator cannot be instantiated directly.'''
        with self.assertRaises(TypeError):
            BaseEstimator()

    def test_concrete_subclass_instantiates(self):
        '''A subclass implementing estimate and properties can be instantiated.'''
        est = _ConcreteEstimator()
        self.assertIsInstance(est, BaseEstimator)

    def test_estimate_returns_series(self):
        '''estimate returns a pandas.Series.'''
        est = _ConcreteEstimator()
        hologram = Hologram(np.random.rand(50, 50))
        result = est.estimate(hologram)
        self.assertIsInstance(result, pd.Series)

    def test_missing_estimate_raises(self):
        '''Subclass without estimate cannot be instantiated.'''
        class _NoEstimate(BaseEstimator):
            @BaseEstimator.properties.getter
            def properties(self):
                return dict()

        with self.assertRaises(TypeError):
            _NoEstimate()

    def test_missing_properties_raises(self):
        '''Subclass without properties cannot be instantiated.'''
        class _NoProperties(BaseEstimator):
            def estimate(self, hologram):
                return pd.Series()

        with self.assertRaises(TypeError):
            _NoProperties()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
