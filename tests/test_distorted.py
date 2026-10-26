import subprocess
import sys
import unittest

import numpy as np

from pylorenzmie.theory import Sphere
from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.Aberrated import AberratedLorenzMie
from pylorenzmie.theory.Distorted import Distorted, DistortedLorenzMie
from pylorenzmie.theory.generators import NAMES, generator, flow


class TestDistortedLorenzMie(unittest.TestCase):

    def setUp(self):
        self.n = 61
        self.xp = self.yp = (self.n - 1) / 2.
        self.zp = 300.
        self.coordinates = LorenzMie.meshgrid((self.n, self.n))
        self.particle = Sphere(a_p=0.5, n_p=1.5,
                               x_p=self.xp, y_p=self.yp, z_p=self.zp)

    def model(self, cls=DistortedLorenzMie, **kwargs):
        m = cls(coordinates=self.coordinates, **kwargs)
        m.particle = self.particle
        return m

    def ideal(self):
        return self.model(LorenzMie).hologram()

    def test_no_distortion_matches_lorenzmie(self):
        self.assertTrue(np.allclose(self.model().hologram(),
                                    self.ideal()))

    def test_distortion_changes_hologram(self):
        h = self.model(spherical=5.).hologram()
        self.assertGreater(np.abs(h - self.ideal()).max(), 1e-3)

    def test_small_coefficient_matches_eikonal(self):
        # The exact and eikonal models agree to second order in the
        # coefficient, in the regime where the eikonal approximation
        # applies: a particle far from the focal plane imaged over a
        # wide field.  The difference is a small part of the effect.
        mag = 0.1
        x = np.arange(-300., 301., 6.)
        X, Y = np.meshgrid(x, x)
        coordinates = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        particle = Sphere(a_p=0.337, n_p=1.61, x_p=0., y_p=0., z_p=1400.)

        def hologram(cls, **kwargs):
            m = cls(coordinates=coordinates, **kwargs)
            m.instrument.magnification = mag
            m.particle = particle
            return m.hologram()

        ideal = hologram(LorenzMie)
        for name in ('defocus', 'spherical'):
            exact = hologram(DistortedLorenzMie, **{name: 0.5})
            eikonal = hologram(AberratedLorenzMie, **{name: 0.5})
            effect = np.sqrt(np.mean((exact - ideal)**2))
            error = np.sqrt(np.mean((exact - eikonal)**2))
            self.assertGreater(effect, 1e-4, name)
            self.assertLess(error, 0.1 * effect, name)

    def test_warp_only_moves_points_for_nonzero_coefficients(self):
        m = self.model(coma=0.)
        rho = self.coordinates[:2] - np.array([[self.xp], [self.yp]])
        self.assertTrue(np.array_equal(m.warp(self.particle), rho))

    def test_order_semantics(self):
        # operators act in the listed order; the evaluation points pass
        # through the flows in reverse
        d, s = 4., 8.
        m = self.model(defocus=d, spherical=s, order=('defocus', 'spherical'),
                       nsteps=8)
        inst = m.instrument
        scale = inst.wavelength / inst.magnification
        rho = self.coordinates[:2] - np.array([[self.xp], [self.yp]])

        def g(name):
            return lambda x: generator(name, x, self.zp, inst.n_m)

        expected = flow(flow(rho, s * scale, g('spherical'), 8),
                        d * scale, g('defocus'), 8)
        self.assertTrue(np.allclose(m.warp(self.particle), expected))

    def test_composed_flows_depend_on_order(self):
        kw = dict(defocus=6., spherical=12., nsteps=8)
        a = self.model(order=('defocus', 'spherical'), **kw).hologram()
        b = self.model(order=('spherical', 'defocus'), **kw).hologram()
        self.assertGreater(np.abs(a - b).max(), 1e-5)

    def test_summed_flow_does_not_depend_on_order(self):
        kw = dict(defocus=6., spherical=12., nsteps=8, mode='summed')
        a = self.model(order=('defocus', 'spherical'), **kw).hologram()
        b = self.model(order=('spherical', 'defocus'), **kw).hologram()
        self.assertTrue(np.allclose(a, b))

    def test_single_aberration_is_independent_of_order_and_mode(self):
        a = self.model(spherical=9., mode='composed').hologram()
        b = self.model(spherical=9., mode='summed').hologram()
        self.assertTrue(np.allclose(a, b))

    def test_flow_has_an_exact_inverse(self):
        rho = self.coordinates[:2] - np.array([[self.xp], [self.yp]])
        for name in NAMES:
            q = (0.1, 0.05)

            def g(x, name=name):
                return generator(name, x, self.zp, 1.34, q)

            there = flow(rho, 20., g, 16)
            back = flow(there, -20., g, 16)
            self.assertTrue(np.allclose(back, rho, atol=1e-8), name)

    def test_runge_kutta_steps_converge(self):
        kw = dict(defocus=3., spherical=8., coma=2.,
                  x_c=self.xp - 20., y_c=self.yp)
        ref = self.model(nsteps=16, **kw).hologram()
        coarse = self.model(nsteps=2, **kw).hologram()
        default = self.model(**kw).hologram()
        self.assertLess(np.abs(default - ref).max(), 1e-6)
        self.assertLess(np.abs(coarse - ref).max(), 1e-4)

    def test_field_dependent_aberrations_vanish_at_the_center(self):
        # with the particle on the aberration center q = 0 and the
        # field-dependent generators vanish
        kw = dict(x_c=self.xp, y_c=self.yp)
        for name in ('coma', 'astigmatism', 'fieldcurvature', 'distortion'):
            h = self.model(**{name: 5.}, **kw).hologram()
            self.assertTrue(np.allclose(h, self.ideal()), name)

    def test_options_are_not_fit_variables(self):
        props = self.model().properties
        for name in ('order', 'mode', 'nsteps'):
            self.assertNotIn(name, props)
        for name in NAMES + ('x_c', 'y_c'):
            self.assertIn(name, props)

    def test_unknown_order_and_mode_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model(order=('spherical', 'trefoil'))
        with self.assertRaises(ValueError):
            self.model(mode='parallel')

    def test_cluster_of_two_particles(self):
        p2 = Sphere(a_p=0.4, n_p=1.45, x_p=self.xp + 15., y_p=self.yp,
                    z_p=self.zp + 40.)
        m = DistortedLorenzMie(coordinates=self.coordinates, spherical=6.)
        m.particle = [self.particle, p2]
        m1 = self.model(spherical=6.)
        m2 = DistortedLorenzMie(coordinates=self.coordinates, spherical=6.)
        m2.particle = p2
        field = m1.scattered_field(self.particle) + m2.scattered_field(p2)
        field[0] += 1.
        expected = np.sum(np.abs(field)**2, axis=0)
        self.assertTrue(np.allclose(m.hologram(), expected))

    def test_aberrated_phase_factors_follow_from_the_generators(self):
        # Aberrated keeps closed-form masks for speed. This checks that
        # they equal exp(i k alpha g . rho / R) built from the shared
        # generating functions, for an off-center aberration center.
        xc, yc, c = 10., 20., 0.7
        for name in ('defocus', 'spherical', 'coma', 'astigmatism',
                     'distortion'):
            m = self.model(AberratedLorenzMie, x_c=xc, y_c=yc,
                           **{name: c})
            inst = m.instrument
            r_p = np.array([self.xp, self.yp, self.zp])
            rho = self.coordinates[:2] - r_p[:2, None]
            q = np.array([self.xp - xc, self.yp - yc]) / self.zp
            g = generator(name, rho, self.zp, inst.n_m, q)
            R = np.sqrt(np.sum(rho**2, axis=0) + self.zp**2)
            alpha = c * inst.wavelength / inst.magnification
            phase = alpha * inst.wavenumber() * np.sum(g * rho / R, axis=0)
            self.assertTrue(np.allclose(np.exp(1j * phase),
                                        m._aberration(r_p)), name)

    def kernel_pair(self, **kwargs):
        kw = dict(defocus=3., spherical=8., coma=2., astigmatism=1.,
                  fieldcurvature=1.5, distortion=4.,
                  x_c=self.xp - 20., y_c=self.yp + 5.)
        kw.update(kwargs)
        slow = self.model(kernel=False, **kw)
        fast = self.model(kernel=True, **kw)
        return slow.warp(self.particle), fast.warp(self.particle)

    def test_kernel_matches_numpy_warp(self):
        for mode in ('composed', 'summed'):
            for nsteps in (1, 2, 4, 8):
                slow, fast = self.kernel_pair(mode=mode, nsteps=nsteps)
                self.assertTrue(np.allclose(slow, fast, atol=1e-9),
                                (mode, nsteps))

    def test_kernel_matches_numpy_warp_for_each_aberration(self):
        for name in NAMES:
            zero = {n: 0. for n in NAMES}
            slow, fast = self.kernel_pair(**{**zero, name: 6.})
            self.assertTrue(np.allclose(slow, fast, atol=1e-9), name)

    def test_kernel_respects_the_order(self):
        for order in (('defocus', 'spherical', 'coma'),
                      ('coma', 'spherical', 'defocus')):
            slow, fast = self.kernel_pair(order=order)
            self.assertTrue(np.allclose(slow, fast, atol=1e-9), order)

    def test_parallel_kernel_matches_serial_kernel(self):
        try:
            from pylorenzmie.theory.numbaLorenzMie import numbaLorenzMie
        except Exception:                       # pragma: no cover
            self.skipTest('numba is not available')
        cls = Distorted(numbaLorenzMie)
        kw = dict(defocus=3., spherical=8., coma=2.,
                  x_c=self.xp - 20., y_c=self.yp)
        for mode in ('composed', 'summed'):
            a = self.model(cls, mode=mode, **kw)
            b = self.model(cls, mode=mode, **kw)
            a.parallel = False
            b.parallel = True
            b.parallel_threshold = 1
            self.assertTrue(np.array_equal(a.warp(self.particle),
                                           b.warp(self.particle)), mode)

    def test_a_model_compiles_the_kernels_when_it_is_made(self):
        # Numba compiles a kernel when it is first called, and compiling
        # overflows the stack of a secondary thread of a GUI, which crashed
        # lmtool in its first fit.  So the model must do it when it is made.
        try:
            import numba                        # noqa: F401
        except ImportError:                     # pragma: no cover
            self.skipTest('numba is not available')
        code = (
            'from pylorenzmie.theory import _warp\n'
            'from pylorenzmie.theory.Distorted import DistortedLorenzMie\n'
            'kernels = (_warp.warp_composed, _warp.warp_summed,\n'
            '           _warp.warp_composed_parallel,\n'
            '           _warp.warp_summed_parallel)\n'
            'assert not any(k.signatures for k in kernels)\n'
            'DistortedLorenzMie()\n'
            'assert all(k.signatures for k in kernels)\n')
        result = subprocess.run([sys.executable, '-I', '-c', code],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_kernel_and_fallback_give_the_same_hologram(self):
        kw = dict(defocus=3., spherical=8.)
        a = self.model(kernel=True, **kw).hologram()
        b = self.model(kernel=False, **kw).hologram()
        self.assertTrue(np.allclose(a, b, atol=1e-9))

    def test_piston_is_a_property(self):
        m = self.model(piston=0.7)
        self.assertEqual(m.properties['piston'], 0.7)

    def test_piston_matches_aberrated(self):
        # a constant phase commutes with the warp, so with no distortion
        # the piston acts exactly as in the eikonal model
        a = self.model(piston=0.3).hologram()
        b = self.model(AberratedLorenzMie, piston=0.3).hologram()
        self.assertTrue(np.allclose(a, b))

    def test_options_are_not_properties(self):
        for name in ('order', 'mode', 'nsteps', 'kernel'):
            self.assertNotIn(name, self.model().properties)

    def test_exported_class_is_picklable(self):
        import pickle
        from pylorenzmie.theory import DistortedLorenzMie as exported
        m = pickle.loads(pickle.dumps(exported(spherical=2.)))
        self.assertEqual(m.spherical, 2.)

    def test_unsupported_backend_is_rejected(self):
        class FakeCuPy(LorenzMie):
            method = 'cupy numpy'

        with self.assertRaises(TypeError):
            Distorted(FakeCuPy)

    def test_numba_backend_if_available(self):
        try:
            from pylorenzmie.theory.numbaLorenzMie import numbaLorenzMie
        except Exception:                       # pragma: no cover
            self.skipTest('numba is not available')
        cls = Distorted(numbaLorenzMie)
        a = self.model(cls, spherical=6., defocus=2.).hologram()
        b = self.model(spherical=6., defocus=2.).hologram()
        self.assertTrue(np.allclose(a, b, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
