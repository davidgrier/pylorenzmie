'''Fits with cupyOptimizer, with NumPy standing in for CuPy and the GPU.

The stand-in keeps apart the arrays of the device and those of the host, so
that a residual that never left the device, or data that never reached it,
fail the test.
'''

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pylorenzmie.analysis import Optimizer
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.analysis.Mask import Mask
from pylorenzmie.lib import LMObject
from pylorenzmie.theory.LorenzMie import LorenzMie

SOURCE = Path(__file__).parent.parent / 'analysis' / 'cupyOptimizer.py'
SHAPE = (41, 41)
TRUTH = dict(x_p=20.3, y_p=19.6, z_p=150., a_p=0.75, n_p=1.45)
START = dict(TRUTH, z_p=153., a_p=0.76)


class Device(np.ndarray):
    '''An array on the device: only get() brings it to the host.'''

    def get(self):
        return np.array(self)


class StandIn(LorenzMie):
    '''cupyLorenzMie, computing on the host.'''

    method = 'cupy numpy'
    dtype = np.float64
    double_precision = True

    def hologram(self, device=False):
        hologram = super().hologram()
        return hologram.view(Device) if device else hologram


def load_optimizer():
    '''cupyOptimizer, imported with the stand-ins in place of CuPy.'''
    cupy = types.ModuleType('cupy')
    cupy.asarray = lambda a, dtype=None: np.asarray(a, dtype).view(Device)
    spec = importlib.util.spec_from_file_location('_stand_in', SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'cupy': cupy}):
        spec.loader.exec_module(module)
    return module.cupyOptimizer


class TestCupyOptimizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cupyOptimizer = staticmethod(load_optimizer())
        model = StandIn(coordinates=LMObject.meshgrid(SHAPE))
        model.properties = TRUTH
        clean = model.hologram().reshape(SHAPE)
        noise = np.random.default_rng(0).normal(0., 0.05, SHAPE)
        cls.hologram = Hologram(clean + noise)

    def fit(self, cls, **settings):
        model = StandIn(coordinates=LMObject.meshgrid(SHAPE))
        model.properties = START
        optimizer = cls(model=model, mask=Mask(fraction=0.5, seed=1))
        optimizer.settings.update(settings)
        return optimizer, optimizer.optimize(self.hologram)

    def test_fits_as_the_optimizer_does(self):
        cases = {'lm': {},
                 'robust': dict(method='trf', loss='cauchy', f_scale='auto'),
                 'broyden': dict(method='broyden')}
        for name, settings in cases.items():
            with self.subTest(name):
                optimizer, device = self.fit(self.cupyOptimizer, **settings)
                _, host = self.fit(Optimizer, **settings)
                self.assertIsInstance(optimizer.devicedata, Device)
                self.assertEqual(optimizer.devicedata.size, device['npix'])
                pd.testing.assert_series_equal(device, host)

    def test_a_step_that_the_model_does_not_compute_is_rejected(self):
        optimizer, _ = self.fit(self.cupyOptimizer)
        values = [optimizer.model.properties[p] for p in optimizer.variables]
        values[optimizer.variables.index('a_p')] = 1000.
        residuals = optimizer._residuals(np.array(values))
        self.assertTrue(np.all(residuals == 1e6))

    def test_setting_the_data_copies_them(self):
        optimizer = self.cupyOptimizer(model=StandIn())
        optimizer.data = np.ones(5)
        self.assertIsInstance(optimizer.devicedata, Device)
        np.testing.assert_array_equal(optimizer.data, np.ones(5))
        optimizer.data = None
        self.assertIsNone(optimizer.devicedata)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
