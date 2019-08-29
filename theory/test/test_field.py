import unittest
import numpy as np
from pylorenzmie.theory.Instrument import coordinates, Instrument
from pylorenzmie.theory.Sphere import Sphere
from pylorenzmie.theory.GeneralizedLorenzMie import GeneralizedLorenzMie
from pylorenzmie.theory.FastGeneralizedLorenzMie import FastGeneralizedLorenzMie
from pylorenzmie.theory.CudaGeneralizedLorenzMie import CudaGeneralizedLorenzMie


class TestField(unittest.TestCase):

    def setUp(self):
        ins = Instrument(magnification=.048,
                         wavelength=.447,
                         n_m=1.335)
        p1 = Sphere(r_p=[150, 150, 200],
                    a_p=1.,
                    n_p=1.45)
        p2 = Sphere(r_p=[75, 75, 150],
                    a_p=1.25,
                    n_p=1.61)
        coords = coordinates((201, 201))
        self.model = GeneralizedLorenzMie(coords, [p1, p2], ins)
        self.fast_model = FastGeneralizedLorenzMie(coords, [p1, p2], ins)
        self.cuda_model = CudaGeneralizedLorenzMie(coords, [p1, p2], ins)

    def test_models_equal(self):
        field = self.model.field()
        fast_field1 = self.fast_model.field()
        cuda_field = self.cuda_model.field()
        self.cuda_model.using_cuda = True
        fast_field2 = self.cuda_model.field()
        self.assertTrue(np.allclose(field, cuda_field))
        self.assertTrue(np.allclose(field, fast_field1))
        self.assertTrue(np.allclose(field, fast_field2))

    def test_precision(self):
        field = self.model.field()
        fast_field = self.fast_model.field()
        double_cuda_field = self.cuda_model.field()
        self.cuda_model.double_precision = False
        single_cuda_field = self.cuda_model.field()
        self.assertTrue(field.dtype == np.complex128)
        self.assertTrue(fast_field.dtype == np.complex128)
        self.assertTrue(double_cuda_field.dtype == np.complex128)
        self.assertTrue(single_cuda_field.dtype == np.complex64)


if __name__ == '__main__':
    unittest.main()
