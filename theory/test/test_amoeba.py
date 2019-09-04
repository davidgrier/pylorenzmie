import unittest
import numpy as np
from pylorenzmie.fitting.cython.cminimizers import amoeba as cymoeba
from pylorenzmie.fitting.minimizers import amoeba
import time


def f(x):
    return x.dot(x)


class TestAmoeba(unittest.TestCase):

    def setUp(self):
        x0 = np.array([1., -5., 1., 2., 3., -5.])
        xmin = np.array([-100., -100., -100., -100., -100., -100.])
        xmax = np.array([100., 100., 100., 100., 100., 100.])
        scale = np.array([.05, .05, .05, .05, .05, .05])
        xtol = np.array([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8])
        ftol = 1e-10
        start1 = time.time()
        self.cysult = cymoeba(f, x0, xmin, xmax, scale, xtol, ftol=ftol)
        self.cyme = time.time() - start1
        start2 = time.time()
        self.result = amoeba(f, x0, xmin, xmax, scale, xtol, ftol=ftol)
        self.time = time.time() - start2

    def test_cy_vs_py(self):
        self.assertTrue(self.cysult.fun == self.result.fun)
        self.assertTrue(self.cysult.nfev == self.result.nfev)
        self.assertTrue(self.cysult.nit == self.result.nit)
        self.assertTrue(self.cysult.success == self.result.success)
        self.assertTrue(all(self.cysult.x == self.result.x))
        self.assertTrue(self.cyme*10 < self.time)


if __name__ == '__main__':
    unittest.main()
