import unittest
import numpy as np

from pylorenzmie.analysis.DEEstimator import DEEstimator
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import LorenzMie
from pylorenzmie.lib import LMObject


class TestDEEstimatorConvergence(unittest.TestCase):
    '''Convergence tests: verify DE finds the right basin from a bad start.

    These tests are slow (~5–30 s each) and live in tests/slow/ so they
    are excluded from the default pytest run.
    '''

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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
