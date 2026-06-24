import unittest
from unittest.mock import patch
from pylorenzmie.theory import best_model, LorenzMie
from pylorenzmie.theory.jaxLorenzMie import jaxLorenzMie, _jax_available


class TestBestModel(unittest.TestCase):

    def test_returns_lorenzmie_instance(self):
        '''best_model() always returns a LorenzMie instance.'''
        model = best_model()
        self.assertIsInstance(model, LorenzMie)

    def test_kwargs_forwarded(self):
        '''Keyword arguments are passed to the chosen constructor.'''
        coords = LorenzMie.meshgrid((32, 32))
        model = best_model(coordinates=coords)
        self.assertIsNotNone(model.coordinates)

    @unittest.skipUnless(_jax_available, 'JAX not installed')
    def test_prefers_jax_when_available(self):
        '''best_model() returns jaxLorenzMie when JAX is present.'''
        model = best_model()
        self.assertIsInstance(model, jaxLorenzMie)

    def test_falls_back_to_numpy(self):
        '''best_model() returns LorenzMie when all accelerated backends fail.'''
        with patch('pylorenzmie.theory.best_model._jax_available', False,
                   create=True):
            with patch.dict('sys.modules',
                            {'cupy': None,
                             'pylorenzmie.theory.cupyLorenzMie': None}):
                with patch('pylorenzmie.theory.numbaLorenzMie._numba_available',
                           False, create=True):
                    model = best_model()
        self.assertIsInstance(model, LorenzMie)

    def test_jax_skipped_when_unavailable(self):
        '''best_model() skips jaxLorenzMie when _jax_available is False.'''
        with patch('pylorenzmie.theory.jaxLorenzMie._jax_available', False):
            model = best_model()
        self.assertNotIsInstance(model, jaxLorenzMie)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
