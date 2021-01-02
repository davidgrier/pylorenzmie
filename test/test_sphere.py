import unittest
import numpy as np

from theory.Sphere import (Sphere, wiscombe_yang, mie_coefficients)


class TestSphere(unittest.TestCase):

    def setUp(self):
        self.particle = Sphere()
        self.particle.a_p = 1.
        self.particle.n_p = 1.4
        self.particle.k_p = 0.
        self.n_m = 1.34
        self.wavelength = 0.447

    def test_seta(self):
        value = 2.
        self.particle.a_p = value
        self.assertEqual(self.particle.a_p, value)
        
        self.particle.a_p = [value]
        self.assertEqual(self.particle.a_p, value)

        value = [1., 2.]
        self.particle.a_p = value
        self.assertTrue((self.particle.a_p == value).all())

    def test_ab(self):
        ab = self.particle.ab(self.n_m, self.wavelength)
        self.assertEqual(ab.size, 64)

    def test_properties(self):
        props = self.particle.properties
        value = props['a_p'] + 0.25
        props['a_p'] = value
        self.particle.properties = props
        self.assertEqual(self.particle.a_p, value)

    def test_repr(self):
        s = repr(self.particle)
        self.assertTrue(isinstance(s, str))

    def test_wiscombe_yang(self):
        x = np.atleast_1d(1.)
        m = np.atleast_1d(10.+0.j)
        typestr = str(type(wiscombe_yang))
        if 'numba' in typestr:
            nmax = wiscombe_yang.py_func(x, m)
        else:
            nmax = wiscombe_yang(x, m)
        self.assertEqual(nmax, 10)

    def test_wiscombe_yang_mid(self):
        x = np.atleast_1d(10.)
        m = np.atleast_1d(10.+0.j)
        typestr = str(type(wiscombe_yang))
        if 'numba' in typestr:
            nmax = wiscombe_yang.py_func(x, m)
        else:
            nmax = wiscombe_yang(x, m)
        self.assertEqual(nmax, 100)

    def test_wiscombe_yang_large(self):
        x = np.atleast_1d(4210.)
        m = np.atleast_1d(10.+0.j)
        typestr = str(type(wiscombe_yang))
        if 'numba' in typestr:
            nmax = wiscombe_yang.py_func(x, m)
        else:
            nmax = wiscombe_yang(x, m)
        self.assertEqual(nmax, 42100)

    def test_mie_coefficients(self):
        args = [self.particle.a_p, self.particle.n_p, self.particle.k_p,
                self.n_m, self.wavelength]
        typestr = str(type(mie_coefficients))
        if 'numba' in typestr:
            ab = mie_coefficients.py_func(*args)
        else:
            ab = mie_coefficients(*args)
        self.assertEqual(type(ab), np.ndarray)

    def test_mie_coefficients_coreshell(self):
        args = [[1.0, 1.1], [1.4, 1.5], [0., 0.],
                self.n_m, self.wavelength]
        typestr = str(type(mie_coefficients))
        if 'numba' in typestr:
            ab = mie_coefficients.py_func(*args)
        else:
            ab = mie_coefficients(*args)
        self.assertEqual(type(ab), np.ndarray)        
        

if __name__ == '__main__':
    unittest.main()
