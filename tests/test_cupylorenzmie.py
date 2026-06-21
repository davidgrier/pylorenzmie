import unittest
import sys

try:
    import cupy  # noqa: F401
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np
from pylorenzmie.theory.LorenzMie import LorenzMie as numpyLorenzMie
from pylorenzmie.lib import LMObject


@unittest.skipUnless(HAS_CUPY, 'cupy not available')
class TestCupyLorenzMie(unittest.TestCase):

    def setUp(self):
        from pylorenzmie.theory.cupyLorenzMie import cupyLorenzMie
        self.method = cupyLorenzMie()
        self.nmethod = numpyLorenzMie()

    def test_method(self):
        self.assertIn('cupy', self.method.method)
        self.assertEqual(self.nmethod.method, 'numpy')

    def test_doubleprecision(self):
        self.method.double_precision = False
        self.assertFalse(self.method.double_precision)
        self.method.double_precision = True
        self.assertTrue(self.method.double_precision)

    def test_field_nocoordinates(self):
        self.method.coordinates = None
        field = self.method.field()
        self.assertIsNotNone(field)

    def test_field(self, bohren=False, cartesian=False):
        p = self.method.particle
        p.a_p = 1.
        p.n_p = 1.4
        p.r_p = [64, 64, 100]
        c = LMObject.meshgrid([128, 128])
        self.method.coordinates = c
        field = self.method.field(bohren=bohren, cartesian=cartesian)
        self.assertEqual(field.shape[1], c.shape[1])

    def test_field_bohren(self):
        self.test_field(bohren=True)

    def test_field_cartesian(self):
        self.test_field(cartesian=True)

    def test_field_both(self):
        self.test_field(bohren=True, cartesian=True)

    def test_compare_methods(self):
        '''CUDA and numpy pipelines must produce identical holograms.'''
        shape = [128, 128]
        c = LMObject.meshgrid(shape)
        self.method.coordinates = c
        self.nmethod.coordinates = c

        p = self.method.particle
        p.a_p = 1.
        p.n_p = 1.4
        p.r_p = [64, 64, 100]
        self.nmethod.particle = p

        holo = self.method.hologram().reshape(shape)
        nholo = self.nmethod.hologram().reshape(shape)
        self.assertTrue(np.allclose(holo, nholo, rtol=1e-5))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
