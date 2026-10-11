from pylorenzmie.theory.LorenzMie import LorenzMie
import unittest
import numpy as np

try:
    from pylorenzmie.theory.jaxLorenzMie import jaxLorenzMie
    _jax_available = True
except Exception:
    _jax_available = False


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

    def test_jac_params(self) -> None:
        self.assertIsInstance(self.model.jac_params, frozenset)
        self.assertSetEqual(self.model.jac_params,
                            {'x_p', 'y_p', 'z_p', 'a_p', 'n_p'})

    def test_repr(self) -> None:
        r = repr(self.model)
        self.assertIsInstance(r, str)


@unittest.skipUnless(_jax_available, 'JAX not installed')
class TestJaxAberrated(unittest.TestCase):
    '''The JAX kernel must include the aberrations, not ignore them.'''

    ALL = dict(defocus=0.5, spherical=2., coma=0.3, astigmatism=0.2,
               distortion=0.1, x_c=10., y_c=-5.)

    def make(self, base, **aberrations):
        from pylorenzmie.theory import Aberrated, Sphere
        x = np.arange(41) - 20.
        X, Y = np.meshgrid(x, x)
        coordinates = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        # one pixel on the particle's axis and one at the aberration center
        coordinates = np.concatenate(
            [coordinates, [[5., 10.], [3., -5.], [0., 0.]]], axis=1)
        model = Aberrated(base)(coordinates=coordinates, **aberrations)
        model.particle = Sphere(a_p=0.5, n_p=1.5, x_p=5., y_p=3., z_p=150.)
        return model

    def test_hologram_matches_numpy_with_all_aberrations(self) -> None:
        a = self.make(LorenzMie, **self.ALL).hologram()
        b = self.make(jaxLorenzMie, **self.ALL).hologram()
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_each_aberration_matches_numpy(self) -> None:
        for name in ('defocus', 'spherical', 'coma', 'astigmatism',
                     'distortion'):
            kwargs = dict(self.ALL)
            for other in ('defocus', 'spherical', 'coma', 'astigmatism',
                          'distortion'):
                if other != name:
                    kwargs[other] = 0.
            a = self.make(LorenzMie, **kwargs).hologram()
            b = self.make(jaxLorenzMie, **kwargs).hologram()
            np.testing.assert_allclose(a, b, atol=1e-10, err_msg=name)

    def test_aberrations_change_the_hologram(self) -> None:
        plain = self.make(jaxLorenzMie).hologram()
        aberrated = self.make(jaxLorenzMie, spherical=2.).hologram()
        self.assertGreater(np.abs(plain - aberrated).max(), 0.1)

    def test_aberration_center_on_the_particle(self) -> None:
        kwargs = dict(self.ALL, x_c=5., y_c=3.)
        a = self.make(LorenzMie, **kwargs).hologram()
        b = self.make(jaxLorenzMie, **kwargs).hologram()
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_jac_includes_aberrations(self) -> None:
        reference = self.make(LorenzMie, **self.ALL)
        jac = self.make(jaxLorenzMie, **self.ALL).jac()
        for name, step in (('x_p', 1e-3), ('y_p', 1e-3), ('z_p', 1e-3),
                           ('a_p', 1e-5), ('n_p', 1e-6)):
            reference.particle.properties = {name: getattr(
                reference.particle, name) + step}
            up = reference.hologram().copy()
            reference.particle.properties = {name: getattr(
                reference.particle, name) - 2*step}
            down = reference.hologram().copy()
            reference.particle.properties = {name: getattr(
                reference.particle, name) + step}
            fd = (up - down) / (2*step)
            self.assertTrue(np.isfinite(jac[name]).all(), name)
            np.testing.assert_allclose(jac[name], fd, rtol=1e-4,
                                       atol=1e-5 * np.abs(fd).max(),
                                       err_msg=name)

    def test_aberration_coefficients_none_without_aberration(self) -> None:
        self.assertIsNone(self.make(jaxLorenzMie)._aberration_coefficients())

    def test_optimizer_ignores_masked_pixels(self) -> None:
        # corrupt a patch of the data (like saturated pixels) and exclude it:
        # the aberrated JAX model must fit the rest and recover the truth
        from pylorenzmie.analysis import Hologram, Mask, Optimizer
        truth = dict(x_p=60., y_p=60., z_p=150., a_p=0.5, n_p=1.5)
        model = self.make(jaxLorenzMie, spherical=1.5)
        model.coordinates = Hologram(np.zeros((121, 121))).flat_coordinates
        model.particle.properties = truth
        image = model.hologram().reshape(121, 121).copy()
        bad = np.zeros(image.shape, dtype=bool)
        bad[55:75, 55:75] = True
        image[bad] = 5.
        model.particle.properties = dict(truth, z_p=156., a_p=0.52, n_p=1.51)

        def fit(exclude):
            mask = Mask(fraction=1., exclude=exclude)
            optimizer = Optimizer(model=model, mask=mask)
            model.particle.properties = dict(truth, z_p=156., a_p=0.52,
                                             n_p=1.51)
            return optimizer.optimize(Hologram(image))

        good = fit(bad)
        for name, tol in (('x_p', 0.05), ('y_p', 0.05), ('z_p', 1.),
                          ('a_p', 0.01), ('n_p', 0.005)):
            self.assertAlmostEqual(good[name], truth[name], delta=tol,
                                   msg=name)
        spoiled = fit(None)
        self.assertGreater(abs(spoiled['n_p'] - truth['n_p']), 0.005)


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
