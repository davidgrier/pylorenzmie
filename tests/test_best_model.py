import unittest
from pylorenzmie.theory import LorenzMie
from pylorenzmie.theory.LorenzMie import LorenzMie as BaseLorenzMie

try:
    from pylorenzmie.theory.jaxLorenzMie import jaxLorenzMie
    _jax_available = True
except Exception:
    _jax_available = False


class TestLorenzMieSelection(unittest.TestCase):
    '''LorenzMie exported from theory: Numba if available, else NumPy.'''

    def test_is_base_subclass(self):
        '''LorenzMie is always the base class or a subclass of it.'''
        self.assertTrue(issubclass(LorenzMie, BaseLorenzMie))

    def test_callable(self):
        '''LorenzMie() creates a usable model instance.'''
        model = LorenzMie()
        self.assertIsInstance(model, BaseLorenzMie)

    def test_kwargs_forwarded(self):
        '''Keyword arguments reach the constructor.'''
        coords = BaseLorenzMie.meshgrid((32, 32))
        model = LorenzMie(coordinates=coords)
        self.assertIsNotNone(model.coordinates)

    def test_default_is_numba_or_numpy(self):
        '''JAX and CuPy are never selected unless requested.'''
        from pylorenzmie.theory import _select_backend
        self.assertIn(_select_backend().__name__,
                      ('numbaLorenzMie', 'LorenzMie'))

    def test_default_prefers_numba_when_available(self):
        try:
            from pylorenzmie.theory.numbaLorenzMie import numbaLorenzMie
        except Exception:
            self.skipTest('Numba not installed')
        from pylorenzmie.theory import _select_backend
        self.assertIs(_select_backend(), numbaLorenzMie)

    def test_numpy_can_be_requested(self):
        from pylorenzmie.theory import _select_backend
        self.assertIs(_select_backend('numpy'), BaseLorenzMie)
        self.assertIs(_select_backend(' NumPy '), BaseLorenzMie)

    def test_unknown_backend_raises(self):
        from pylorenzmie.theory import _select_backend
        with self.assertRaises(ValueError):
            _select_backend('fortran')

    def test_unavailable_backend_raises_instead_of_falling_back(self):
        from pylorenzmie.theory import _select_backend
        from unittest import mock
        with mock.patch.dict('sys.modules',
                             {'pylorenzmie.theory.cupyLorenzMie': None}):
            with self.assertRaises(ImportError):
                _select_backend('cupy')

    @unittest.skipUnless(_jax_available, 'JAX not installed or backend broken')
    def test_jax_can_be_requested(self):
        from pylorenzmie.theory import _select_backend
        self.assertIs(_select_backend('jax'), jaxLorenzMie)


class TestBackendFallback(unittest.TestCase):
    '''What the default selection does when Numba is unusable.'''

    def test_a_missing_numba_gives_the_numpy_calculator(self):
        from unittest import mock
        import pylorenzmie.theory as theory
        with mock.patch.object(theory, '_load', side_effect=ImportError):
            self.assertIs(theory._select_backend(), BaseLorenzMie)

    def test_a_broken_numba_gives_the_numpy_calculator_and_a_warning(self):
        from unittest import mock
        import pylorenzmie.theory as theory
        with mock.patch.object(theory, '_load',
                               side_effect=RuntimeError('broken')):
            with self.assertLogs('pylorenzmie.theory', level='WARNING'):
                self.assertIs(theory._select_backend(), BaseLorenzMie)

    def test_a_requested_backend_that_is_broken_raises(self):
        from unittest import mock
        import pylorenzmie.theory as theory
        with mock.patch.object(theory, '_load',
                               side_effect=RuntimeError('broken')):
            with self.assertRaises(ImportError):
                theory._select_backend('numba')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
