import pickle
import unittest

from pylorenzmie.theory import (AberratedLorenzMie, BACKENDS,
                                DistortedLorenzMie, LorenzMie, model_class)
from pylorenzmie.theory.Aberrated import AberratedLorenzMie as NumpyAberrated
from pylorenzmie.theory.Distorted import DistortedLorenzMie as NumpyDistorted


class TestModelClass(unittest.TestCase):

    def test_default_is_the_exported_class(self):
        self.assertIs(model_class(), AberratedLorenzMie)
        self.assertIs(model_class(distorted=True), DistortedLorenzMie)

    def test_numpy(self):
        self.assertIs(model_class('numpy'), NumpyAberrated)
        self.assertIs(model_class('numpy', distorted=True), NumpyDistorted)

    def test_name_is_not_case_sensitive(self):
        self.assertIs(model_class(' NumPy '), NumpyAberrated)

    def test_same_class_each_time(self):
        for backend in ('numpy', 'numba'):
            for distorted in (False, True):
                try:
                    a = model_class(backend, distorted)
                except ImportError:                  # pragma: no cover
                    continue
                self.assertIs(a, model_class(backend, distorted))

    def test_classes_can_be_pickled(self):
        for backend in (None, 'numpy', 'numba'):
            for distorted in (False, True):
                try:
                    cls = model_class(backend, distorted)
                except ImportError:                  # pragma: no cover
                    continue
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

    def test_distorted_is_distorted(self):
        self.assertTrue(hasattr(model_class('numpy', True), 'warp'))
        self.assertFalse(hasattr(model_class('numpy', False), 'warp'))

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            model_class('fortran')

    def test_distorted_needs_numpy_or_numba(self):
        for backend in ('torch', 'jax', 'cupy'):
            with self.assertRaises((ValueError, ImportError)):
                model_class(backend, distorted=True)

    def test_backends_listed(self):
        self.assertEqual(set(BACKENDS),
                         {'numpy', 'numba', 'torch', 'jax', 'cupy'})


if __name__ == '__main__':
    unittest.main()
