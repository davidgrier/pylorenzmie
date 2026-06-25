import unittest
from pylorenzmie.theory import LorenzMie
from pylorenzmie.theory.LorenzMie import LorenzMie as BaseLorenzMie

try:
    from pylorenzmie.theory.jaxLorenzMie import jaxLorenzMie
    _jax_available = True
except Exception:
    _jax_available = False


class TestLorenzMieSelection(unittest.TestCase):
    '''LorenzMie exported from theory is the best available backend.'''

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

    @unittest.skipUnless(_jax_available, 'JAX not installed or backend broken')
    def test_prefers_jax_when_available(self):
        '''LorenzMie is jaxLorenzMie when JAX is installed and functional.'''
        self.assertIs(LorenzMie, jaxLorenzMie)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
