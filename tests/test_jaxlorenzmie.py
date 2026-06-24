from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.jaxLorenzMie import jaxLorenzMie, _jax_available
import unittest
import numpy as np


@unittest.skipUnless(_jax_available, 'JAX not installed')
class TestJaxLorenzMie(unittest.TestCase):

    def setUp(self) -> None:
        self.shape = (64, 64)
        self.model = jaxLorenzMie()
        self.model.particle.r_p = [32., 32., 150.]
        self.model.particle.a_p = 0.75
        self.model.particle.n_p = 1.45
        self.model.coordinates = self.model.meshgrid(self.shape)

        self.ref = LorenzMie()
        self.ref.particle.r_p = [32., 32., 150.]
        self.ref.particle.a_p = 0.75
        self.ref.particle.n_p = 1.45
        self.ref.coordinates = self.model.coordinates

    def test_method(self) -> None:
        self.assertIn('jax', self.model.method)
        self.assertIn('numpy', self.model.method)

    def test_hologram_shape(self) -> None:
        h = self.model.hologram()
        npts = self.shape[0] * self.shape[1]
        self.assertEqual(h.shape, (npts,))

    def test_hologram_dtype(self) -> None:
        h = self.model.hologram()
        self.assertEqual(h.dtype, np.float64)

    def test_hologram_positive(self) -> None:
        h = self.model.hologram()
        self.assertTrue(np.all(h > 0.))

    def test_hologram_matches_numpy(self) -> None:
        h_jax = self.model.hologram()
        h_np = self.ref.hologram()
        np.testing.assert_allclose(h_jax, h_np, rtol=1e-5, atol=1e-8)

    def test_hologram_is_numpy_array(self) -> None:
        h = self.model.hologram()
        self.assertIsInstance(h, np.ndarray)

    def test_jac_keys(self) -> None:
        J = self.model.jac()
        self.assertSetEqual(set(J.keys()),
                            {'x_p', 'y_p', 'z_p', 'a_p', 'n_p'})

    def test_jac_shape(self) -> None:
        J = self.model.jac()
        npts = self.shape[0] * self.shape[1]
        for key, arr in J.items():
            with self.subTest(param=key):
                self.assertEqual(arr.shape, (npts,))

    def test_jac_dtype(self) -> None:
        J = self.model.jac()
        for key, arr in J.items():
            with self.subTest(param=key):
                self.assertEqual(arr.dtype, np.float64)

    def test_jac_finite(self) -> None:
        J = self.model.jac()
        for key, arr in J.items():
            with self.subTest(param=key):
                self.assertTrue(np.all(np.isfinite(arr)))

    def test_jac_matches_finite_differences(self) -> None:
        '''Analytical Jacobian should agree with central-difference FD.'''
        h = 1e-4
        J = self.model.jac()
        coords = self.model.coordinates.copy()

        params = {
            'x_p': (self.model.particle, 'x_p'),
            'y_p': (self.model.particle, 'y_p'),
            'z_p': (self.model.particle, 'z_p'),
            'a_p': (self.model.particle, 'a_p'),
            'n_p': (self.model.particle, 'n_p'),
        }
        for name, (obj, attr) in params.items():
            val0 = float(getattr(obj, attr))
            setattr(obj, attr, val0 + h)
            hf = self.model.hologram()
            setattr(obj, attr, val0 - h)
            hb = self.model.hologram()
            setattr(obj, attr, val0)
            fd = (hf - hb) / (2. * h)
            with self.subTest(param=name):
                np.testing.assert_allclose(J[name], fd, rtol=3e-4, atol=1e-8)

    def test_repr(self) -> None:
        r = repr(self.model)
        self.assertIsInstance(r, str)


@unittest.skipUnless(_jax_available, 'JAX not installed')
class TestJaxFallback(unittest.TestCase):
    '''Verify graceful fallback when JAX acceleration is not applicable.'''

    def setUp(self) -> None:
        from pylorenzmie.theory.Cluster import Cluster
        self.model = jaxLorenzMie()
        self.model.coordinates = self.model.meshgrid((32, 32))

    def test_jac_raises_for_non_sphere(self) -> None:
        from pylorenzmie.theory.Cluster import Cluster
        self.model.particle = Cluster()
        with self.assertRaises(TypeError):
            self.model.jac()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
